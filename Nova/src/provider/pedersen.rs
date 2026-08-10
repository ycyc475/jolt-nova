//! This module provides an implementation of a commitment engine
use crate::provider::msm::batch_add;
#[cfg(feature = "io")]
use crate::provider::ptau::{read_points, write_points, PtauFileError};
use crate::traits::evm_serde::EvmCompatSerde;
use crate::{
  errors::NovaError,
  gadgets::utils::to_bignat_repr,
  provider::traits::{DlogGroup, DlogGroupExt},
  traits::{
    commitment::{CommitmentEngineTrait, CommitmentTrait, Len},
    AbsorbInRO2Trait, AbsorbInROTrait, Engine, ROTrait, TranscriptReprTrait,
  },
};
use core::{
  fmt::Debug,
  marker::PhantomData,
  ops::{Add, Mul, MulAssign, Range},
};
use ff::Field;
use num_integer::Integer;
use num_traits::ToPrimitive;
use once_cell::sync::Lazy;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_with::serde_as;
use std::{
  any::{Any, TypeId},
  collections::HashMap,
  sync::{Arc, Mutex},
  time::Instant,
};

#[cfg(feature = "io")]
const KEY_FILE_HEAD: [u8; 12] = *b"PEDERSEN_KEY";

type SetupCacheKey = (TypeId, Vec<u8>, usize);
type ErasedCommitmentKey = Arc<dyn Any + Send + Sync>;

/// Process-local cache for transparent Pedersen generators.
///
/// `DlogGroup::from_label` is deterministic, and a key generated for a larger
/// size has exactly the same prefix as a smaller key with the same curve and
/// label. Reusing that prefix avoids repeating hash-to-curve setup while the
/// returned owned key preserves the existing public API and serialization.
static PEDERSEN_SETUP_CACHE: Lazy<Mutex<HashMap<SetupCacheKey, ErasedCommitmentKey>>> =
  Lazy::new(|| Mutex::new(HashMap::new()));

/// A type that holds commitment generators
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommitmentKey<E: Engine>
where
  E::GE: DlogGroup,
{
  ck: Arc<Vec<<E::GE as DlogGroup>::AffineGroupElement>>,
  h: <E::GE as DlogGroup>::AffineGroupElement,
}

impl<E: Engine> CommitmentKey<E>
where
  E::GE: DlogGroup,
{
  /// Returns a reference to the generator points (affine form).
  pub fn generators(&self) -> &[<E::GE as DlogGroup>::AffineGroupElement] {
    &self.ck
  }
}

impl<E: Engine> Len for CommitmentKey<E>
where
  E::GE: DlogGroup,
{
  fn length(&self) -> usize {
    self.ck.len()
  }
}

/// A type that holds blinding generator
#[serde_as]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct DerandKey<E: Engine>
where
  E::GE: DlogGroup,
{
  #[serde_as(as = "EvmCompatSerde")]
  h: <E::GE as DlogGroup>::AffineGroupElement,
}

/// A type that holds a commitment
#[serde_as]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct Commitment<E: Engine>
where
  E::GE: DlogGroup,
{
  #[serde_as(as = "EvmCompatSerde")]
  pub(crate) comm: E::GE,
}

impl<E: Engine> CommitmentTrait<E> for Commitment<E>
where
  E::GE: DlogGroup,
{
  fn to_coordinates(&self) -> (E::Base, E::Base, bool) {
    self.comm.to_coordinates()
  }
}

impl<E: Engine> Default for Commitment<E>
where
  E::GE: DlogGroup,
{
  fn default() -> Self {
    Commitment {
      comm: E::GE::zero(),
    }
  }
}

impl<E: Engine> TranscriptReprTrait<E::GE> for Commitment<E>
where
  E::GE: DlogGroup,
{
  fn to_transcript_bytes(&self) -> Vec<u8> {
    let (x, y, is_infinity) = self.comm.to_coordinates();
    // Encode the infinity flag as 1 = infinity, matching the RO and in-circuit conventions.
    let is_infinity_byte = is_infinity.into();
    [
      x.to_transcript_bytes(),
      y.to_transcript_bytes(),
      [is_infinity_byte].to_vec(),
    ]
    .concat()
  }
}

impl<E: Engine> AbsorbInROTrait<E> for Commitment<E>
where
  E::GE: DlogGroup,
{
  fn absorb_in_ro(&self, ro: &mut E::RO) {
    let (x, y, is_infinity) = self.comm.to_coordinates();
    // Absorb the affine coordinates and the infinity flag.
    ro.absorb(x);
    ro.absorb(y);
    ro.absorb(if is_infinity {
      E::Base::ONE
    } else {
      E::Base::ZERO
    });
  }
}

impl<E: Engine> AbsorbInRO2Trait<E> for Commitment<E>
where
  E::GE: DlogGroup,
{
  fn absorb_in_ro2(&self, ro: &mut E::RO2) {
    let (x, y, is_infinity) = self.comm.to_coordinates();

    // we have to absorb x and y in big num format
    let limbs_x = to_bignat_repr(&x);
    let limbs_y = to_bignat_repr(&y);

    for limb in limbs_x.iter().chain(limbs_y.iter()) {
      ro.absorb(*limb);
    }
    ro.absorb(if is_infinity {
      E::Scalar::ONE
    } else {
      E::Scalar::ZERO
    });
  }
}

impl<E: Engine> MulAssign<E::Scalar> for Commitment<E>
where
  E::GE: DlogGroup,
{
  fn mul_assign(&mut self, scalar: E::Scalar) {
    *self = Commitment {
      comm: self.comm * scalar,
    };
  }
}

impl<'b, E: Engine> Mul<&'b E::Scalar> for &'_ Commitment<E>
where
  E::GE: DlogGroup,
{
  type Output = Commitment<E>;
  fn mul(self, scalar: &'b E::Scalar) -> Commitment<E> {
    Commitment {
      comm: self.comm * scalar,
    }
  }
}

impl<E: Engine> Mul<E::Scalar> for Commitment<E>
where
  E::GE: DlogGroup,
{
  type Output = Commitment<E>;

  fn mul(self, scalar: E::Scalar) -> Commitment<E> {
    Commitment {
      comm: self.comm * scalar,
    }
  }
}

impl<E: Engine> Add for Commitment<E>
where
  E::GE: DlogGroup,
{
  type Output = Commitment<E>;

  fn add(self, other: Commitment<E>) -> Commitment<E> {
    Commitment {
      comm: self.comm + other.comm,
    }
  }
}

/// Provides a commitment engine
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommitmentEngine<E: Engine> {
  _p: PhantomData<E>,
}

impl<E: Engine> CommitmentKey<E>
where
  E::GE: DlogGroup,
{
  /// Returns the coordinates of the generator points.
  ///
  /// This method extracts the (x, y) coordinates of each generator point
  /// in the commitment key. This is useful for operations that need direct
  /// access to the underlying elliptic curve points.
  ///
  /// # Panics
  ///
  /// Panics if any generator point is the point at infinity.
  pub fn to_coordinates(&self) -> Vec<(E::Base, E::Base)> {
    self
      .ck
      .par_iter()
      .map(|c| {
        let (x, y, is_infinity) = <E::GE as DlogGroup>::group(c).to_coordinates();
        assert!(!is_infinity);
        (x, y)
      })
      .collect()
  }
}

impl<E: Engine + 'static> CommitmentEngineTrait<E> for CommitmentEngine<E>
where
  E::GE: DlogGroupExt,
{
  type CommitmentKey = CommitmentKey<E>;
  type Commitment = Commitment<E>;
  type DerandKey = DerandKey<E>;

  fn setup(label: &'static [u8], n: usize) -> Result<Self::CommitmentKey, NovaError> {
    let started = Instant::now();
    let requested_size = n.next_power_of_two();
    let engine = TypeId::of::<E>();
    let mut cached_size = None;
    let cached = {
      let cache = PEDERSEN_SETUP_CACHE
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
      cache
        .iter()
        .filter(|((cached_engine, cached_label, size), _)| {
          *cached_engine == engine && cached_label.as_slice() == label && *size >= requested_size
        })
        .min_by_key(|((_, _, size), _)| *size)
        .map(|((_, _, size), key)| {
          cached_size = Some(*size);
          Arc::clone(key)
        })
    };
    if let Some(cached) = cached {
      let cached = cached
        .downcast_ref::<CommitmentKey<E>>()
        .expect("Pedersen setup cache type is bound by its engine TypeId");
      if std::env::var_os("JOLT_NOVA_PROFILE_SETUP").is_some() {
        eprintln!(
          "jolt_nova_pedersen_setup engine={} requested={} cached={} hit=true elapsed_micros={}",
          std::any::type_name::<E>(),
          requested_size,
          cached_size.expect("cache hit records its size"),
          started.elapsed().as_micros(),
        );
      }
      return Ok(Self::CommitmentKey {
        ck: if cached.ck.len() == requested_size {
          Arc::clone(&cached.ck)
        } else {
          Arc::new(cached.ck[..requested_size].to_vec())
        },
        h: cached.h,
      });
    }

    let gens = E::GE::from_label(label, requested_size + 1);

    let (h, ck) = gens.split_first().unwrap();
    let commitment_key = Self::CommitmentKey {
      ck: Arc::new(ck.to_vec()),
      h: *h,
    };
    let mut cache = PEDERSEN_SETUP_CACHE
      .lock()
      .unwrap_or_else(|poisoned| poisoned.into_inner());
    let has_larger_key = cache.keys().any(|(cached_engine, cached_label, size)| {
      *cached_engine == engine && cached_label.as_slice() == label && *size >= requested_size
    });
    if !has_larger_key {
      cache.retain(|(cached_engine, cached_label, size), _| {
        *cached_engine != engine || cached_label.as_slice() != label || *size > requested_size
      });
      cache.insert(
        (engine, label.to_vec(), requested_size),
        Arc::new(commitment_key.clone()),
      );
    }
    drop(cache);
    if std::env::var_os("JOLT_NOVA_PROFILE_SETUP").is_some() {
      eprintln!(
        "jolt_nova_pedersen_setup engine={} requested={} cached={} hit=false elapsed_micros={}",
        std::any::type_name::<E>(),
        requested_size,
        requested_size,
        started.elapsed().as_micros(),
      );
    }
    Ok(commitment_key)
  }

  fn derand_key(ck: &Self::CommitmentKey) -> Self::DerandKey {
    Self::DerandKey { h: ck.h }
  }

  fn commit(ck: &Self::CommitmentKey, v: &[E::Scalar], r: &E::Scalar) -> Self::Commitment {
    assert!(ck.ck.len() >= v.len());

    Commitment {
      comm: E::GE::vartime_multiscalar_mul(v, &ck.ck[..v.len()])
        + <E::GE as DlogGroup>::group(&ck.h) * r,
    }
  }

  fn commit_small<T: Integer + Into<u64> + Copy + Sync + ToPrimitive>(
    ck: &Self::CommitmentKey,
    v: &[T],
    r: &E::Scalar,
  ) -> Self::Commitment {
    assert!(ck.ck.len() >= v.len());

    Commitment {
      comm: E::GE::vartime_multiscalar_mul_small(v, &ck.ck[..v.len()])
        + <E::GE as DlogGroup>::group(&ck.h) * r,
    }
  }

  fn commit_small_range<T: Integer + Into<u64> + Copy + Sync + ToPrimitive>(
    ck: &Self::CommitmentKey,
    v: &[T],
    r: &<E as Engine>::Scalar,
    range: Range<usize>,
    max_num_bits: usize,
  ) -> Self::Commitment {
    let bases = &ck.ck[range.clone()];
    let scalars = &v[range];

    assert!(bases.len() == scalars.len());

    let mut res =
      E::GE::vartime_multiscalar_mul_small_with_max_num_bits(scalars, bases, max_num_bits);

    if r != &E::Scalar::ZERO {
      res += <E::GE as DlogGroup>::group(&ck.h) * r;
    }

    Commitment { comm: res }
  }

  fn derandomize(
    dk: &Self::DerandKey,
    commit: &Self::Commitment,
    r: &E::Scalar,
  ) -> Self::Commitment {
    Commitment {
      comm: commit.comm - <E::GE as DlogGroup>::group(&dk.h) * r,
    }
  }

  #[cfg(feature = "io")]
  fn load_setup(
    reader: &mut (impl std::io::Read + std::io::Seek),
    _label: &'static [u8],
    n: usize,
  ) -> Result<Self::CommitmentKey, PtauFileError> {
    let num = n.next_power_of_two();
    {
      let mut head = [0u8; 12];
      reader.read_exact(&mut head)?;
      if head != KEY_FILE_HEAD {
        return Err(PtauFileError::InvalidHead);
      }
    }

    let points = read_points(reader, num + 1)?;

    let (first, second) = points.split_at(1);

    Ok(Self::CommitmentKey {
      ck: Arc::new(second.to_vec()),
      h: first[0],
    })
  }

  fn ck_to_coordinates(ck: &Self::CommitmentKey) -> Vec<(E::Base, E::Base)> {
    ck.to_coordinates()
  }

  fn ck_to_group_elements(ck: &Self::CommitmentKey) -> Vec<E::GE> {
    ck.ck
      .par_iter()
      .map(|g| {
        let ge = E::GE::group(g);
        assert!(
          ge != E::GE::zero(),
          "CommitmentKey contains a generator at infinity"
        );
        ge
      })
      .collect()
  }

  fn ck_derive_by_address(
    ck: &Self::CommitmentKey,
    addresses: &[usize],
    table_size: usize,
  ) -> Result<Self::CommitmentKey, NovaError> {
    let bases = Self::ck_to_group_elements(ck);
    if addresses.len() > bases.len() {
      return Err(NovaError::InvalidCommitmentKeyLength);
    }
    if addresses.iter().any(|&j| j >= table_size) {
      return Err(NovaError::InvalidIndex);
    }
    let mut acc = vec![E::GE::zero(); table_size];
    for (i, &j) in addresses.iter().enumerate() {
      acc[j] += bases[i];
    }
    let ck_affine = acc.par_iter().map(|g| g.affine()).collect();
    Ok(CommitmentKey {
      ck: Arc::new(ck_affine),
      h: ck.h,
    })
  }

  #[cfg(feature = "io")]
  fn save_setup(
    ck: &Self::CommitmentKey,
    writer: &mut impl std::io::Write,
  ) -> Result<(), PtauFileError> {
    writer.write_all(&KEY_FILE_HEAD)?;
    let mut points = Vec::with_capacity(ck.ck.len() + 1);
    points.push(ck.h);
    points.extend(ck.ck.iter().cloned());
    write_points(writer, points)
  }

  fn commit_sparse_binary(
    ck: &Self::CommitmentKey,
    non_zero_indices: &[usize],
    r: &<E as Engine>::Scalar,
  ) -> Self::Commitment {
    let comm = batch_add(&ck.ck, non_zero_indices);
    let mut comm = <E::GE as DlogGroup>::group(&comm.into());

    if r != &E::Scalar::ZERO {
      comm += <E::GE as DlogGroup>::group(&ck.h) * r;
    }

    Commitment { comm }
  }

  fn commit_sparse(
    ck: &Self::CommitmentKey,
    indices: &[usize],
    scalars: &[E::Scalar],
    r: &E::Scalar,
  ) -> Self::Commitment {
    assert_eq!(indices.len(), scalars.len());

    let bases: Vec<_> = indices.par_iter().map(|&i| ck.ck[i]).collect();

    let mut comm = E::GE::vartime_multiscalar_mul(scalars, &bases);

    if r != &E::Scalar::ZERO {
      comm += <E::GE as DlogGroup>::group(&ck.h) * r;
    }

    Commitment { comm }
  }
}

/// A trait listing properties of a commitment key that can be managed in a divide-and-conquer fashion
pub trait CommitmentKeyExtTrait<E: Engine>
where
  E::GE: DlogGroup,
{
  /// Returns the first `n` generators, sharing storage when the full key is requested.
  fn prefix(&self, n: usize) -> Self
  where
    Self: Sized;

  /// Splits the commitment key into two pieces at a specified point
  fn split_at(&self, n: usize) -> (Self, Self)
  where
    Self: Sized;

  /// Commits to `scalars` over a generator range and one extra generator.
  ///
  /// This is the allocation-free equivalent of combining the selected key
  /// range with a one-element key and committing with zero blinding.
  fn commit_range_with_extra(
    &self,
    range: Range<usize>,
    scalars: &[E::Scalar],
    extra: &Self,
    extra_scalar: &E::Scalar,
  ) -> <E::CE as CommitmentEngineTrait<E>>::Commitment;

  /// Combines two commitment keys into one
  fn combine(&self, other: &Self) -> Self;

  /// Folds the two commitment keys into one using the provided weights
  fn fold(&self, w1: &E::Scalar, w2: &E::Scalar) -> Self;

  /// Scales the commitment key using the provided scalar
  fn scale(&self, r: &E::Scalar) -> Self;

  /// Reinterprets commitments as commitment keys
  fn reinterpret_commitments_as_ck(
    c: &[<E::CE as CommitmentEngineTrait<E>>::Commitment],
  ) -> Result<Self, NovaError>
  where
    Self: Sized;
}

impl<E: Engine<CE = CommitmentEngine<E>> + 'static> CommitmentKeyExtTrait<E> for CommitmentKey<E>
where
  E::GE: DlogGroupExt,
{
  fn prefix(&self, n: usize) -> CommitmentKey<E> {
    assert!(n <= self.ck.len());
    CommitmentKey {
      ck: if n == self.ck.len() {
        Arc::clone(&self.ck)
      } else {
        Arc::new(self.ck[..n].to_vec())
      },
      h: self.h,
    }
  }

  fn split_at(&self, n: usize) -> (CommitmentKey<E>, CommitmentKey<E>) {
    (
      CommitmentKey {
        ck: Arc::new(self.ck[0..n].to_vec()),
        h: self.h,
      },
      CommitmentKey {
        ck: Arc::new(self.ck[n..].to_vec()),
        h: self.h,
      },
    )
  }

  fn commit_range_with_extra(
    &self,
    range: Range<usize>,
    scalars: &[E::Scalar],
    extra: &CommitmentKey<E>,
    extra_scalar: &E::Scalar,
  ) -> <E::CE as CommitmentEngineTrait<E>>::Commitment {
    assert_eq!(range.len(), scalars.len());
    assert!(range.end <= self.ck.len());
    assert_eq!(extra.ck.len(), 1);

    Commitment {
      comm: E::GE::vartime_multiscalar_mul(scalars, &self.ck[range])
        + E::GE::group(&extra.ck[0]) * *extra_scalar,
    }
  }

  fn combine(&self, other: &CommitmentKey<E>) -> CommitmentKey<E> {
    let ck = {
      let mut combined = Vec::with_capacity(self.ck.len() + other.ck.len());
      combined.extend_from_slice(&self.ck);
      combined.extend_from_slice(&other.ck);
      Arc::new(combined)
    };
    CommitmentKey { ck, h: self.h }
  }

  // combines the left and right halves of `self` using `w1` and `w2` as the weights
  fn fold(&self, w1: &E::Scalar, w2: &E::Scalar) -> CommitmentKey<E> {
    let half = self.ck.len() / 2;
    let weights = [*w1, *w2];

    let ck = (0..half)
      .into_par_iter()
      .map(|i| {
        let bases = [self.ck[i], self.ck[i + half]];
        E::GE::vartime_multiscalar_mul(&weights, &bases).affine()
      })
      .collect();

    CommitmentKey {
      ck: Arc::new(ck),
      h: self.h,
    }
  }

  /// Scales each element in `self` by `r`
  fn scale(&self, r: &E::Scalar) -> Self {
    let ck_scaled = self
      .ck
      .par_iter()
      .map(|g| E::GE::vartime_multiscalar_mul(&[*r], &[*g]).affine())
      .collect();

    CommitmentKey {
      ck: Arc::new(ck_scaled),
      h: self.h,
    }
  }

  /// reinterprets a vector of commitments as a set of generators
  fn reinterpret_commitments_as_ck(c: &[Commitment<E>]) -> Result<Self, NovaError> {
    let ck = (0..c.len())
      .into_par_iter()
      .map(|i| c[i].comm.affine())
      .collect();

    // cmt is derandomized by the point that this is called
    Ok(CommitmentKey {
      ck: Arc::new(ck),
      h: E::GE::zero().affine(), // this is okay, since this method is used in IPA only,
                                 // and we only use non-blinding commits afterwards
                                 // bc we don't use ZK IPA
    })
  }
}

#[cfg(test)]
mod setup_cache_tests {
  use super::*;
  use crate::{provider::Bn256EngineIPA, traits::commitment::CommitmentEngineTrait};

  type E = Bn256EngineIPA;

  #[test]
  fn deterministic_setup_reuses_exact_key_and_preserves_prefixes() {
    const LABEL: &[u8] = b"pedersen-setup-cache-test";
    let large = CommitmentEngine::<E>::setup(LABEL, 17).unwrap();
    let exact = CommitmentEngine::<E>::setup(LABEL, 20).unwrap();
    assert!(Arc::ptr_eq(&large.ck, &exact.ck));
    assert_eq!(large.h, exact.h);

    let small = CommitmentEngine::<E>::setup(LABEL, 8).unwrap();
    assert_eq!(small.h, large.h);
    assert_eq!(small.ck.as_slice(), &large.ck[..8]);
  }
}

#[cfg(feature = "io")]
#[cfg(test)]
mod tests {
  use super::*;

  use crate::{provider::GrumpkinEngine, CommitmentKey};
  use std::{fs::File, io::BufWriter};

  type E = GrumpkinEngine;

  #[test]
  fn test_key_save_load() {
    let path = "/tmp/pedersen_test.keys";

    const LABEL: &[u8; 4] = b"test";

    let keys = CommitmentEngine::<E>::setup(LABEL, 100).unwrap();

    CommitmentEngine::save_setup(&keys, &mut BufWriter::new(File::create(path).unwrap())).unwrap();

    let keys_read = CommitmentEngine::load_setup(&mut File::open(path).unwrap(), LABEL, 100);

    assert!(keys_read.is_ok());
    let keys_read: CommitmentKey<E> = keys_read.unwrap();
    assert_eq!(keys_read.ck.len(), keys.ck.len());
    assert_eq!(keys_read.h, keys.h);
    assert_eq!(keys_read.ck, keys.ck);
  }
}
