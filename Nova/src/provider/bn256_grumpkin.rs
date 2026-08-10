//! This module implements the Nova traits for `bn256::Point`, `bn256::Scalar`, `grumpkin::Point`, `grumpkin::Scalar`.
use crate::{
  impl_traits,
  provider::{
    msm::{msm, msm_small, msm_small_with_max_num_bits},
    traits::{DlogGroup, DlogGroupExt, PairingGroup},
  },
  traits::{Group, PrimeFieldExt, TranscriptReprTrait},
};
use digest::{ExtendableOutput, Update};
use ff::{Field, FromUniformBytes, PrimeField, WithSmallOrderMulGroup};
use halo2curves::{
  bn256::{Bn256, G1Affine as Bn256Affine, G2Affine, G2Compressed, Gt, G1 as Bn256Point, G2},
  group::{cofactor::CofactorCurveAffine, Curve, Group as AnotherGroup},
  grumpkin::{G1Affine as GrumpkinAffine, G1 as GrumpkinPoint},
  pairing::Engine as H2CEngine,
  CurveAffine, CurveExt,
};
use num_bigint::BigInt;
use num_integer::Integer;
use num_traits::{Num, ToPrimitive};
use rayon::prelude::*;
use sha3::Shake256;

/// Re-exports that give access to the standard aliases used in the code base, for bn256
pub mod bn256 {
  pub use halo2curves::bn256::{Fq as Base, Fr as Scalar, G1Affine as Affine, G1 as Point};
}

/// Re-exports that give access to the standard aliases used in the code base, for grumpkin
pub mod grumpkin {
  pub use halo2curves::grumpkin::{Fq as Base, Fr as Scalar, G1Affine as Affine, G1 as Point};
}

const BN_ENDO_GAMMA_1: [u64; 4] = [0xd91d232ec7e0b3d7, 0x2, 0, 0];
const BN_ENDO_GAMMA_2: [u64; 4] = [0x5398fd0300ff6565, 0x4ccef014a773d2d2, 0x02, 0];
const BN_ENDO_B_1: [u64; 4] = [0x89d3256894d213e3, 0, 0, 0];
const BN_ENDO_B_2: [u64; 4] = [0x0be4e1541221250b, 0x6f4d8248eeb859fd, 0, 0];

fn add_mul_carry(acc: u64, left: u64, right: u64, carry: u64) -> (u64, u64) {
  let result = u128::from(acc) + u128::from(left) * u128::from(right) + u128::from(carry);
  (result as u64, (result >> 64) as u64)
}

fn mul_512(left: [u64; 4], right: [u64; 4]) -> [u64; 8] {
  let mut result = [0u64; 8];
  for (i, left_limb) in left.into_iter().enumerate() {
    let mut carry = 0;
    for (j, right_limb) in right.into_iter().enumerate() {
      let (limb, next_carry) = add_mul_carry(result[i + j], left_limb, right_limb, carry);
      result[i + j] = limb;
      carry = next_carry;
    }
    result[i + 4] = carry;
  }
  result
}

fn scalar_limbs(scalar: &bn256::Scalar) -> [u64; 4] {
  let repr = scalar.to_repr();
  let bytes = repr.as_ref();
  std::array::from_fn(|i| u64::from_le_bytes(bytes[i * 8..(i + 1) * 8].try_into().unwrap()))
}

fn scalar_is_negative(scalar: &bn256::Scalar) -> bool {
  let limbs = scalar_limbs(scalar);
  limbs[3] != 0
}

/// GLV decomposition matching halo2curves' BN256 endomorphism parameters.
fn decompose_bn256_scalar(scalar: &bn256::Scalar) -> (u128, bool, u128, bool) {
  let input = scalar_limbs(scalar);
  let gamma_2_product = mul_512(BN_ENDO_GAMMA_2, input);
  let gamma_1_product = mul_512(BN_ENDO_GAMMA_1, input);
  let c_1: [u64; 4] = gamma_2_product[4..8].try_into().unwrap();
  let c_2: [u64; 4] = gamma_1_product[4..8].try_into().unwrap();
  let q_1_product = mul_512(c_1, BN_ENDO_B_1);
  let q_2_product = mul_512(c_2, BN_ENDO_B_2);
  let q_1 = bn256::Scalar::from_raw(q_1_product[0..4].try_into().unwrap());
  let q_2 = bn256::Scalar::from_raw(q_2_product[0..4].try_into().unwrap());

  let k_2 = q_2 - q_1;
  let k_1 = *scalar + k_2 * bn256::Scalar::ZETA;
  let k_1_negative = scalar_is_negative(&k_1);
  let k_2_negative = scalar_is_negative(&k_2);
  let k_1 = if k_1_negative { -k_1 } else { k_1 };
  let k_2 = if k_2_negative { -k_2 } else { k_2 };
  let k_1_limbs = scalar_limbs(&k_1);
  let k_2_limbs = scalar_limbs(&k_2);

  (
    u128::from(k_1_limbs[0]) | (u128::from(k_1_limbs[1]) << 64),
    k_1_negative,
    u128::from(k_2_limbs[0]) | (u128::from(k_2_limbs[1]) << 64),
    k_2_negative,
  )
}

crate::impl_traits_no_dlog_ext!(
  bn256,
  Bn256Point,
  Bn256Affine,
  "30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001",
  "30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47"
);

impl DlogGroupExt for bn256::Point {
  #[cfg(not(feature = "blitzar"))]
  fn vartime_multiscalar_mul(scalars: &[Self::Scalar], bases: &[Self::AffineGroupElement]) -> Self {
    msm(scalars, bases)
  }

  #[cfg(not(feature = "blitzar"))]
  fn vartime_double_scalar_mul(
    scalars: &[Self::Scalar; 2],
    bases: &[Self::AffineGroupElement; 2],
  ) -> Self {
    super::msm::vartime_double_scalar_mul(scalars, bases)
  }

  #[cfg(not(feature = "blitzar"))]
  fn batch_vartime_double_scalar_mul(
    scalars: &[Self::Scalar; 2],
    left_bases: &[Self::AffineGroupElement],
    right_bases: &[Self::AffineGroupElement],
  ) -> Vec<Self::AffineGroupElement> {
    let decomposed = [
      decompose_bn256_scalar(&scalars[0]),
      decompose_bn256_scalar(&scalars[1]),
    ];
    super::msm::batch_vartime_double_scalar_mul_endo(&decomposed, left_bases, right_bases)
  }

  fn vartime_multiscalar_mul_small<T: Integer + Into<u64> + Copy + Sync + ToPrimitive>(
    scalars: &[T],
    bases: &[Self::AffineGroupElement],
  ) -> Self {
    msm_small(scalars, bases)
  }

  fn vartime_multiscalar_mul_small_with_max_num_bits<
    T: Integer + Into<u64> + Copy + Sync + ToPrimitive,
  >(
    scalars: &[T],
    bases: &[Self::AffineGroupElement],
    max_num_bits: usize,
  ) -> Self {
    msm_small_with_max_num_bits(scalars, bases, max_num_bits)
  }

  #[cfg(feature = "blitzar")]
  fn vartime_multiscalar_mul(scalars: &[Self::Scalar], bases: &[Self::AffineGroupElement]) -> Self {
    super::blitzar::vartime_multiscalar_mul(scalars, bases)
  }

  #[cfg(feature = "blitzar")]
  fn vartime_double_scalar_mul(
    scalars: &[Self::Scalar; 2],
    bases: &[Self::AffineGroupElement; 2],
  ) -> Self {
    super::blitzar::vartime_multiscalar_mul(scalars, bases)
  }

  #[cfg(feature = "blitzar")]
  fn batch_vartime_multiscalar_mul(
    scalars: &[Vec<Self::Scalar>],
    bases: &[Self::AffineGroupElement],
  ) -> Vec<Self> {
    super::blitzar::batch_vartime_multiscalar_mul(scalars, bases)
  }
}

impl_traits!(
  grumpkin,
  GrumpkinPoint,
  GrumpkinAffine,
  "30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47",
  "30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001"
);

impl PairingGroup for Bn256Point {
  type G2 = G2;
  type GT = Gt;

  fn pairing(p: &Self, q: &Self::G2) -> Self::GT {
    <Bn256 as H2CEngine>::pairing(&p.affine(), &q.affine())
  }
}

impl Group for G2 {
  type Base = bn256::Base;
  type Scalar = bn256::Scalar;

  fn group_params() -> (Self::Base, Self::Base, BigInt, BigInt) {
    // G2 uses a quadratic extension field, so A and B are in QuadExtField<Fq>
    // We need to extract the constant terms that can be represented in Fq
    // For BN256 G2, the curve equation is y^2 = x^3 + Ax + B where A and B are in Fq2
    // The constant terms (real parts) are typically 0 for A and 3 for B
    let A = bn256::Base::ZERO; // Constant term of A in Fq2
    let B = bn256::Base::from(3u64); // Constant term of B in Fq2
    let order = BigInt::from_str_radix(
      "30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001",
      16,
    )
    .unwrap();
    let base = BigInt::from_str_radix(
      "30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47",
      16,
    )
    .unwrap();

    (A, B, order, base)
  }
}

impl DlogGroup for G2 {
  type AffineGroupElement = G2Affine;

  fn affine(&self) -> Self::AffineGroupElement {
    self.to_affine()
  }

  fn group(p: &Self::AffineGroupElement) -> Self {
    G2::from(*p)
  }

  fn from_label(_label: &'static [u8], _n: usize) -> Vec<Self::AffineGroupElement> {
    unimplemented!()
  }

  fn zero() -> Self {
    G2::identity()
  }

  fn gen() -> Self {
    G2::generator()
  }

  fn to_coordinates(&self) -> (Self::Base, Self::Base, bool) {
    unimplemented!()
  }
}

impl<G: DlogGroup> TranscriptReprTrait<G> for G2Compressed {
  fn to_transcript_bytes(&self) -> Vec<u8> {
    self.as_ref().to_vec()
  }
}

impl<G: DlogGroup> TranscriptReprTrait<G> for G2Affine {
  fn to_transcript_bytes(&self) -> Vec<u8> {
    unimplemented!()
  }
}

// CustomSerdeTrait implementations for G2
impl crate::traits::evm_serde::CustomSerdeTrait for G2Affine {
  #[cfg(feature = "evm")]
  fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
    use crate::traits::evm_serde::EvmCompatSerde;
    use serde::{Deserialize, Serialize};
    use serde_with::serde_as;

    #[serde_as]
    #[derive(Deserialize, Serialize)]
    struct HelperBase(
      #[serde_as(as = "EvmCompatSerde")] bn256::Base,
      #[serde_as(as = "EvmCompatSerde")] bn256::Base,
    );

    #[derive(Deserialize, Serialize)]
    struct HelperAffine(HelperBase, HelperBase);

    let affine = HelperAffine(
      HelperBase(*self.x.c0(), *self.x.c1()),
      HelperBase(*self.y.c0(), *self.y.c1()),
    );

    affine.serialize(serializer)
  }

  #[cfg(feature = "evm")]
  fn deserialize<'de, D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
    use crate::traits::evm_serde::EvmCompatSerde;
    use halo2curves::bn256::Fq2;
    use halo2curves::group::cofactor::CofactorGroup;
    use serde::de::Error;
    use serde::{Deserialize, Serialize};
    use serde_with::serde_as;

    #[serde_as]
    #[derive(Deserialize, Serialize)]
    struct HelperBase(
      #[serde_as(as = "EvmCompatSerde")] bn256::Base,
      #[serde_as(as = "EvmCompatSerde")] bn256::Base,
    );

    #[derive(Deserialize, Serialize)]
    struct HelperAffine(HelperBase, HelperBase);

    let affine = HelperAffine::deserialize(deserializer)?;
    let x = Fq2::new(affine.0 .0, affine.0 .1);
    let y = Fq2::new(affine.1 .0, affine.1 .1);

    // Validate the decoded coordinates with the checked constructor, which
    // accepts on-curve points (including the canonical identity) and returns a
    // deserialize error otherwise.
    let point: G2Affine = Option::from(G2Affine::from_xy(x, y))
      .ok_or_else(|| D::Error::custom("deserialized G2 point is not on the curve"))?;

    // bn256 G2 has a non-trivial cofactor, so being on the curve does not by
    // itself imply prime-order subgroup membership; additionally require the
    // point to be torsion-free. (G1, with cofactor 1, needs only the on-curve
    // check above.)
    if bool::from(point.to_curve().is_torsion_free()) {
      Ok(point)
    } else {
      Err(D::Error::custom(
        "deserialized G2 point is not in the prime-order subgroup",
      ))
    }
  }

  #[cfg(not(feature = "evm"))]
  fn deserialize<'de, D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
    use halo2curves::group::cofactor::{CofactorCurveAffine, CofactorGroup};
    use serde::de::Error;
    // Non-evm builds decode with the default (compressed) serde, which already
    // enforces on-curve. bn256 G2 has a non-trivial cofactor, so additionally
    // require the point to be torsion-free, matching the evm path above.
    let point = <Self as serde::Deserialize>::deserialize(deserializer)?;
    if bool::from(CofactorCurveAffine::to_curve(&point).is_torsion_free()) {
      Ok(point)
    } else {
      Err(D::Error::custom(
        "deserialized G2 point is not in the prime-order subgroup",
      ))
    }
  }
}

#[cfg(all(test, feature = "evm"))]
mod evm_serde_tests {
  use super::bn256;
  use crate::traits::evm_serde::EvmCompatSerde;
  use halo2curves::group::{cofactor::CofactorCurveAffine, Curve};
  use serde::{Deserialize, Serialize};
  use serde_with::serde_as;

  #[serde_as]
  #[derive(Debug, PartialEq, Serialize, Deserialize)]
  struct WrappedAffine(#[serde_as(as = "EvmCompatSerde")] bn256::Affine);

  #[test]
  fn test_on_curve_point_round_trips() {
    let affine = bn256::Point::generator().to_affine();
    let bytes =
      bincode::serde::encode_to_vec(WrappedAffine(affine), bincode::config::legacy()).unwrap();
    assert_eq!(bytes.len(), 64);
    let (decoded, _): (WrappedAffine, usize) =
      bincode::serde::decode_from_slice(&bytes, bincode::config::legacy()).unwrap();
    assert_eq!(decoded.0, affine);
  }

  #[test]
  fn test_identity_round_trips() {
    let identity = bn256::Affine::identity();
    let bytes =
      bincode::serde::encode_to_vec(WrappedAffine(identity), bincode::config::legacy()).unwrap();
    let (decoded, _): (WrappedAffine, usize) =
      bincode::serde::decode_from_slice(&bytes, bincode::config::legacy()).unwrap();
    assert_eq!(decoded.0, identity);
  }

  #[test]
  fn test_off_curve_point_is_rejected() {
    // x = y = 1 are canonical (< p) but do not satisfy y^2 = x^3 + 3,
    // so they do not lie on the BN254 G1 curve.
    let mut bytes = [0u8; 64];
    bytes[31] = 1; // x = 1 (big-endian)
    bytes[63] = 1; // y = 1 (big-endian)
    let result: Result<(WrappedAffine, usize), _> =
      bincode::serde::decode_from_slice(&bytes, bincode::config::legacy());
    assert!(
      result.is_err(),
      "off-curve point (1, 1) must be rejected on deserialization"
    );
  }

  #[serde_as]
  #[derive(Debug, PartialEq, Serialize, Deserialize)]
  struct WrappedG2(#[serde_as(as = "EvmCompatSerde")] halo2curves::bn256::G2Affine);

  #[test]
  fn test_g2_on_curve_in_subgroup_round_trips() {
    use halo2curves::bn256::G2;
    let affine = G2::generator().to_affine();
    let bytes =
      bincode::serde::encode_to_vec(WrappedG2(affine), bincode::config::legacy()).unwrap();
    assert_eq!(bytes.len(), 128);
    let (decoded, _): (WrappedG2, usize) =
      bincode::serde::decode_from_slice(&bytes, bincode::config::legacy()).unwrap();
    assert_eq!(decoded.0, affine);
  }

  #[test]
  fn test_g2_identity_round_trips() {
    let identity = halo2curves::bn256::G2Affine::identity();
    let bytes =
      bincode::serde::encode_to_vec(WrappedG2(identity), bincode::config::legacy()).unwrap();
    let (decoded, _): (WrappedG2, usize) =
      bincode::serde::decode_from_slice(&bytes, bincode::config::legacy()).unwrap();
    assert_eq!(decoded.0, identity);
  }

  #[test]
  fn test_g2_off_curve_is_rejected() {
    // x = y = 1 (each Fq2 component canonical, < p) do not satisfy the G2 curve
    // equation, so the point is off-curve and must be rejected.
    let mut bytes = [0u8; 128];
    bytes[31] = 1; // x.c0 = 1 (big-endian)
    bytes[95] = 1; // y.c0 = 1 (big-endian)
    let result: Result<(WrappedG2, usize), _> =
      bincode::serde::decode_from_slice(&bytes, bincode::config::legacy());
    assert!(result.is_err(), "off-curve G2 point must be rejected");
  }

  #[test]
  fn test_g2_non_subgroup_is_rejected() {
    use ff::Field;
    use halo2curves::bn256::{Fq2, G2Affine, G2};
    use halo2curves::group::cofactor::CofactorGroup;
    use halo2curves::{CurveAffine, CurveExt};
    // bn256 G2 has a large cofactor, so a point obtained by solving the curve
    // equation for an arbitrary x is overwhelmingly outside the prime-order
    // subgroup. Such an on-curve point serializes fine but must be rejected on
    // deserialization for failing the subgroup check.
    let b = G2::b();
    let mut x = Fq2::ONE;
    let point = loop {
      let rhs = x.square() * x + b;
      if let Some(y) = Option::<Fq2>::from(rhs.sqrt()) {
        let p = G2Affine::from_xy(x, y).unwrap();
        if !bool::from(p.to_curve().is_torsion_free()) {
          break p;
        }
      }
      x += Fq2::ONE;
    };
    let bytes = bincode::serde::encode_to_vec(WrappedG2(point), bincode::config::legacy()).unwrap();
    let result: Result<(WrappedG2, usize), _> =
      bincode::serde::decode_from_slice(&bytes, bincode::config::legacy());
    assert!(
      result.is_err(),
      "on-curve but non-subgroup G2 point must be rejected"
    );
  }
}

#[cfg(test)]
mod endo_double_scalar_tests {
  use super::*;
  use halo2curves::group::Curve;
  use rand_core::OsRng;

  #[test]
  fn test_bn256_batch_endo_double_scalar_mul() {
    let left_bases = (0..16)
      .map(|_| (bn256::Point::generator() * bn256::Scalar::random(OsRng)).to_affine())
      .collect::<Vec<_>>();
    let right_bases = (0..16)
      .map(|_| (bn256::Point::generator() * bn256::Scalar::random(OsRng)).to_affine())
      .collect::<Vec<_>>();

    let mut scalar_cases = vec![
      [bn256::Scalar::ZERO, bn256::Scalar::ZERO],
      [bn256::Scalar::ONE, bn256::Scalar::ZERO],
      [bn256::Scalar::ZERO, bn256::Scalar::ONE],
      [-bn256::Scalar::ONE, bn256::Scalar::ONE],
    ];
    scalar_cases
      .extend((0..64).map(|_| [bn256::Scalar::random(OsRng), bn256::Scalar::random(OsRng)]));

    for scalars in scalar_cases {
      let expected = left_bases
        .iter()
        .zip(&right_bases)
        .map(|(left, right)| {
          crate::provider::msm::vartime_double_scalar_mul(&scalars, &[*left, *right]).to_affine()
        })
        .collect::<Vec<_>>();
      let actual = <bn256::Point as DlogGroupExt>::batch_vartime_double_scalar_mul(
        &scalars,
        &left_bases,
        &right_bases,
      );
      assert_eq!(actual, expected);
    }
  }
}
