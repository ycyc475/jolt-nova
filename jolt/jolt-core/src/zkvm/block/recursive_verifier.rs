//! Typed input object for recursive verification of an original Jolt proof.
//!
//! Earlier block-folding stages retained fixed-width digests emitted after a
//! host-side Jolt verification.  Those digests remain useful identifiers, but
//! they cannot establish that the proof was valid.  This module introduces the
//! Stage 14 boundary that retains every verifier input in its original typed
//! form.  Subsequent gadgets consume this object directly; no `accepted: bool`
//! or opaque receipt is part of the trust boundary.

use ark_serialize::CanonicalSerialize;
use sha3::{Digest, Sha3_256};

use super::RecursiveJoltFieldElement;
use crate::utils::errors::ProofVerifyError;
use crate::{
    curve::JoltCurve,
    field::JoltField,
    poly::commitment::commitment_scheme::CommitmentScheme,
    subprotocols::sumcheck::SumcheckInstanceProof,
    transcripts::Transcript,
    zkvm::{proof_serialization::JoltProof, verifier::JoltVerifierPreprocessing},
};

/// One lossless clear-sumcheck round.  The compressed polynomial omits its
/// linear coefficient exactly as the original Jolt proof does; the verifier
/// reconstructs that coefficient from the running claim.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecursiveClearSumcheckRoundWitness {
    pub coefficients_except_linear: Vec<RecursiveJoltFieldElement>,
    pub challenge: RecursiveJoltFieldElement,
}

/// Complete arithmetic witness for one clear Jolt sumcheck stage.
///
/// Transcript constraints must derive every round challenge, and the stage
/// relation must derive `initial_claim` and `expected_final_claim`.  Keeping
/// those values here as witnesses does not make them trusted inputs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecursiveClearSumcheckStageWitness {
    pub stage_index: usize,
    pub degree_bound: usize,
    pub initial_claim: RecursiveJoltFieldElement,
    pub rounds: Vec<RecursiveClearSumcheckRoundWitness>,
    pub expected_final_claim: RecursiveJoltFieldElement,
}

impl RecursiveClearSumcheckStageWitness {
    pub fn validate_shape(
        &self,
        expected_stage_index: usize,
        expected_rounds: usize,
        expected_degree_bound: usize,
    ) -> Result<(), &'static str> {
        if self.stage_index != expected_stage_index {
            return Err("recursive sumcheck stage index mismatch");
        }
        if self.rounds.len() != expected_rounds {
            return Err("recursive sumcheck round count mismatch");
        }
        if self.degree_bound != expected_degree_bound || self.degree_bound == 0 {
            return Err("recursive sumcheck degree bound mismatch");
        }
        if self.rounds.iter().any(|round| {
            round.coefficients_except_linear.is_empty()
                || round.coefficients_except_linear.len() > self.degree_bound
        }) {
            return Err("recursive sumcheck compressed polynomial has invalid degree");
        }
        Ok(())
    }
}

/// How Stage 8 polynomial-opening verification is linked to the recursive
/// verifier.
///
/// Dory is pairing based.  Its scalar transitions are constrained in the Nova
/// circuit, while the resulting pairing equation is carried by the final
/// folded instance and checked once by the outer verifier.  This is a deferred
/// cryptographic check, not a host assertion or receipt digest.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecursiveVerifierPcsStrategy {
    DeferredDoryPairing,
}

/// A complete, typed PCS equation deferred from the scalar-field Nova circuit.
///
/// The Nova relation constrains the transcript and all scalar inputs that lead
/// to this object.  Final acceptance calls [`Self::verify`] once, which runs the
/// real PCS verifier (Dory's multi-pairing for the production scheme).  There
/// is intentionally no boolean `host_verified` field and no digest-based
/// acceptance path.
#[derive(Clone)]
pub struct RecursiveDeferredPcsOpening<F, PCS, FS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    FS: Transcript,
{
    proof: PCS::Proof,
    verifier_setup: PCS::VerifierSetup,
    transcript_before_pcs: FS,
    opening_point: Vec<F::Challenge>,
    opening: F,
    commitment: PCS::Commitment,
    obligation_id: [u8; 32],
}

impl<F, PCS, FS> RecursiveDeferredPcsOpening<F, PCS, FS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    FS: Transcript,
{
    pub fn new(
        proof: PCS::Proof,
        verifier_setup: PCS::VerifierSetup,
        transcript_before_pcs: FS,
        opening_point: Vec<F::Challenge>,
        opening: F,
        commitment: PCS::Commitment,
        transcript_binding: &[u8],
    ) -> Self {
        let obligation_id = compute_deferred_pcs_obligation_id::<F, PCS>(
            &proof,
            &verifier_setup,
            &opening_point,
            &opening,
            &commitment,
            transcript_binding,
        );
        Self {
            proof,
            verifier_setup,
            transcript_before_pcs,
            opening_point,
            opening,
            commitment,
            obligation_id,
        }
    }

    pub fn proof(&self) -> &PCS::Proof {
        &self.proof
    }

    pub fn verifier_setup(&self) -> &PCS::VerifierSetup {
        &self.verifier_setup
    }

    pub fn transcript_before_pcs(&self) -> &FS {
        &self.transcript_before_pcs
    }

    pub fn opening_point(&self) -> &[F::Challenge] {
        &self.opening_point
    }

    pub fn opening(&self) -> &F {
        &self.opening
    }

    pub fn commitment(&self) -> &PCS::Commitment {
        &self.commitment
    }

    /// Continuity identifier for binding this exact deferred equation into the
    /// recursive verifier public state.  It does not replace [`Self::verify`].
    pub fn obligation_id(&self) -> [u8; 32] {
        self.obligation_id
    }

    /// Executes the cryptographic PCS check.  For Jolt's production
    /// `DoryCommitmentScheme`, this reaches the final BN254 multi-pairing
    /// equality rather than consulting any receipt.
    pub fn verify(&self) -> Result<(), ProofVerifyError> {
        let mut transcript = self.transcript_before_pcs.clone();
        PCS::verify(
            &self.proof,
            &self.verifier_setup,
            &mut transcript,
            &self.opening_point,
            &self.opening,
            &self.commitment,
        )
    }
}

fn compute_deferred_pcs_obligation_id<F, PCS>(
    proof: &PCS::Proof,
    verifier_setup: &PCS::VerifierSetup,
    opening_point: &[F::Challenge],
    opening: &F,
    commitment: &PCS::Commitment,
    transcript_binding: &[u8],
) -> [u8; 32]
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    let mut hasher = Sha3_256::new();
    hasher.update(b"jolt-nova/deferred-pcs-opening/v1");
    append_canonical(&mut hasher, b"proof", proof);
    append_canonical(&mut hasher, b"verifier-setup", verifier_setup);
    for challenge in opening_point {
        let scalar: F = (*challenge).into();
        append_canonical(&mut hasher, b"opening-point-coordinate", &scalar);
    }
    append_canonical(&mut hasher, b"opening", opening);
    append_canonical(&mut hasher, b"commitment", commitment);
    hasher.update((transcript_binding.len() as u64).to_le_bytes());
    hasher.update(transcript_binding);
    hasher.finalize().into()
}

/// Owned parts returned when a recursive verifier object is dismantled.
pub struct RecursiveJoltVerifierObjectParts<F, C, PCS, FS, PublicIo>
where
    F: JoltField,
    C: JoltCurve<F = F>,
    PCS: CommitmentScheme<Field = F>,
    FS: Transcript,
{
    pub preprocessing: JoltVerifierPreprocessing<F, C, PCS>,
    pub proof: JoltProof<F, C, PCS, FS>,
    pub public_io: PublicIo,
    pub trusted_advice_commitment: Option<PCS::Commitment>,
}

/// Complete, lossless input to the Stage 14 recursive Jolt verifier.
///
/// `object_id` is only a commitment/continuity identifier.  Acceptance requires
/// the recursive sumcheck and relation constraints plus the final deferred PCS
/// equation; equality of `object_id` alone is deliberately insufficient.
pub struct RecursiveJoltVerifierObject<F, C, PCS, FS, PublicIo>
where
    F: JoltField,
    C: JoltCurve<F = F>,
    PCS: CommitmentScheme<Field = F>,
    FS: Transcript,
{
    version: u16,
    pcs_strategy: RecursiveVerifierPcsStrategy,
    preprocessing: JoltVerifierPreprocessing<F, C, PCS>,
    proof: JoltProof<F, C, PCS, FS>,
    public_io: PublicIo,
    trusted_advice_commitment: Option<PCS::Commitment>,
    object_id: [u8; 32],
}

impl<F, C, PCS, FS, PublicIo> RecursiveJoltVerifierObject<F, C, PCS, FS, PublicIo>
where
    F: JoltField,
    C: JoltCurve<F = F>,
    PCS: CommitmentScheme<Field = F>,
    FS: Transcript,
    PCS::Commitment: CanonicalSerialize,
    PCS::VerifierSetup: CanonicalSerialize,
    PublicIo: CanonicalSerialize,
{
    pub const VERSION: u16 = 1;
    pub const SUMCHECK_STAGE_COUNT: usize = 8;

    pub fn new(
        preprocessing: JoltVerifierPreprocessing<F, C, PCS>,
        proof: JoltProof<F, C, PCS, FS>,
        public_io: PublicIo,
        trusted_advice_commitment: Option<PCS::Commitment>,
    ) -> Self {
        let pcs_strategy = RecursiveVerifierPcsStrategy::DeferredDoryPairing;
        let object_id = compute_object_id(
            Self::VERSION,
            pcs_strategy,
            &preprocessing,
            &proof,
            &public_io,
            &trusted_advice_commitment,
        );
        Self {
            version: Self::VERSION,
            pcs_strategy,
            preprocessing,
            proof,
            public_io,
            trusted_advice_commitment,
            object_id,
        }
    }

    pub fn version(&self) -> u16 {
        self.version
    }

    pub fn pcs_strategy(&self) -> RecursiveVerifierPcsStrategy {
        self.pcs_strategy
    }

    pub fn preprocessing(&self) -> &JoltVerifierPreprocessing<F, C, PCS> {
        &self.preprocessing
    }

    pub fn proof(&self) -> &JoltProof<F, C, PCS, FS> {
        &self.proof
    }

    pub fn public_io(&self) -> &PublicIo {
        &self.public_io
    }

    pub fn trusted_advice_commitment(&self) -> Option<&PCS::Commitment> {
        self.trusted_advice_commitment.as_ref()
    }

    /// Commitment/continuity identifier; it is not an acceptance certificate.
    pub fn object_id(&self) -> [u8; 32] {
        self.object_id
    }

    pub fn verify_object_id(&self) -> bool {
        self.object_id
            == compute_object_id(
                self.version,
                self.pcs_strategy,
                &self.preprocessing,
                &self.proof,
                &self.public_io,
                &self.trusted_advice_commitment,
            )
    }

    /// Returns all eight standard Jolt sumcheck stages without hashing or
    /// dropping their round polynomials/commitments.
    pub fn sumcheck_stages(&self) -> [&SumcheckInstanceProof<F, C, FS>; 8] {
        [
            &self.proof.stage1_sumcheck_proof,
            &self.proof.stage2_sumcheck_proof,
            &self.proof.stage3_sumcheck_proof,
            &self.proof.stage4_sumcheck_proof,
            &self.proof.stage5_sumcheck_proof,
            &self.proof.stage6a_sumcheck_proof,
            &self.proof.stage6b_sumcheck_proof,
            &self.proof.stage7_sumcheck_proof,
        ]
    }

    pub fn total_sumcheck_rounds(&self) -> usize {
        self.sumcheck_stages()
            .into_iter()
            .map(SumcheckInstanceProof::num_rounds)
            .sum()
    }

    pub fn all_sumchecks_are_clear(&self) -> bool {
        self.sumcheck_stages()
            .into_iter()
            .all(|proof| !proof.is_zk())
    }

    pub fn into_parts(self) -> RecursiveJoltVerifierObjectParts<F, C, PCS, FS, PublicIo> {
        RecursiveJoltVerifierObjectParts {
            preprocessing: self.preprocessing,
            proof: self.proof,
            public_io: self.public_io,
            trusted_advice_commitment: self.trusted_advice_commitment,
        }
    }
}

fn append_canonical<T: CanonicalSerialize>(hasher: &mut Sha3_256, label: &[u8], value: &T) {
    let mut bytes = Vec::new();
    value
        .serialize_compressed(&mut bytes)
        .expect("canonical serialization into memory cannot fail");
    hasher.update((label.len() as u64).to_le_bytes());
    hasher.update(label);
    hasher.update((bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

fn compute_object_id<F, C, PCS, FS, PublicIo>(
    version: u16,
    pcs_strategy: RecursiveVerifierPcsStrategy,
    preprocessing: &JoltVerifierPreprocessing<F, C, PCS>,
    proof: &JoltProof<F, C, PCS, FS>,
    public_io: &PublicIo,
    trusted_advice_commitment: &Option<PCS::Commitment>,
) -> [u8; 32]
where
    F: JoltField,
    C: JoltCurve<F = F>,
    PCS: CommitmentScheme<Field = F>,
    FS: Transcript,
    PCS::Commitment: CanonicalSerialize,
    PCS::VerifierSetup: CanonicalSerialize,
    PublicIo: CanonicalSerialize,
{
    let mut hasher = Sha3_256::new();
    hasher.update(b"jolt-nova/recursive-jolt-verifier-object/v1");
    hasher.update(version.to_le_bytes());
    hasher.update([match pcs_strategy {
        RecursiveVerifierPcsStrategy::DeferredDoryPairing => 0,
    }]);
    append_canonical(&mut hasher, b"preprocessing", preprocessing);
    append_canonical(&mut hasher, b"proof", proof);
    append_canonical(&mut hasher, b"public-io", public_io);
    append_canonical(
        &mut hasher,
        b"trusted-advice-commitment",
        trusted_advice_commitment,
    );
    hasher.finalize().into()
}

#[cfg(test)]
mod tests {
    use super::{append_canonical, RecursiveVerifierPcsStrategy};
    use sha3::{Digest, Sha3_256};

    fn component_commitment(parts: [&Vec<u8>; 4]) -> [u8; 32] {
        let mut hasher = Sha3_256::new();
        hasher.update(b"jolt-nova/recursive-jolt-verifier-object/test");
        hasher.update(1u16.to_le_bytes());
        hasher.update([match RecursiveVerifierPcsStrategy::DeferredDoryPairing {
            RecursiveVerifierPcsStrategy::DeferredDoryPairing => 0,
        }]);
        for (label, part) in [
            b"preprocessing" as &[u8],
            b"proof" as &[u8],
            b"public-io" as &[u8],
            b"advice" as &[u8],
        ]
        .into_iter()
        .zip(parts)
        {
            append_canonical(&mut hasher, label, part);
        }
        hasher.finalize().into()
    }

    #[test]
    fn recursive_verifier_object_identifier_binds_every_lossless_component() {
        let components = [vec![1, 2], vec![3, 4], vec![5, 6], vec![7, 8]];
        let baseline = component_commitment([
            &components[0],
            &components[1],
            &components[2],
            &components[3],
        ]);

        for index in 0..components.len() {
            let mut tampered = components.clone();
            tampered[index][0] ^= 1;
            assert_ne!(
                component_commitment([&tampered[0], &tampered[1], &tampered[2], &tampered[3],]),
                baseline
            );
        }
    }
}

#[cfg(test)]
mod deferred_pcs_tests {
    use ark_bn254::Fr;
    use ark_std::UniformRand;
    use serial_test::serial;

    use crate::{
        field::JoltField,
        poly::{
            commitment::{
                commitment_scheme::CommitmentScheme,
                dory::{bind_opening_inputs, DoryCommitmentScheme, DoryContext, DoryGlobals},
            },
            dense_mlpoly::DensePolynomial,
            multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
        },
        transcripts::{PoseidonTranscript, Transcript},
    };

    use super::RecursiveDeferredPcsOpening;

    #[test]
    #[serial]
    fn deferred_dory_obligation_runs_real_pairing_and_rejects_mutated_opening() {
        let num_vars = 4usize;
        DoryGlobals::reset();
        let _guard =
            DoryGlobals::initialize_context(1, 1usize << num_vars, DoryContext::Main, None);
        let prover_setup = DoryCommitmentScheme::setup_prover(num_vars);
        let verifier_setup = DoryCommitmentScheme::setup_verifier(&prover_setup);
        let mut rng = ark_std::rand::thread_rng();
        let coefficients = (0..1usize << num_vars)
            .map(|_| Fr::rand(&mut rng))
            .collect::<Vec<_>>();
        let polynomial = MultilinearPolynomial::LargeScalars(DensePolynomial::new(coefficients));
        let opening_point = (0..num_vars)
            .map(|_| <Fr as JoltField>::Challenge::random(&mut rng))
            .collect::<Vec<_>>();
        let opening = <MultilinearPolynomial<Fr> as PolynomialEvaluation<Fr>>::evaluate(
            &polynomial,
            &opening_point,
        );
        let (commitment, hint) = DoryCommitmentScheme::commit(&polynomial, &prover_setup);
        let mut prover_transcript = PoseidonTranscript::new(b"stage14-dory");
        bind_opening_inputs::<Fr, _>(&mut prover_transcript, &opening_point, &opening);
        let (proof, _) = DoryCommitmentScheme::prove(
            &prover_setup,
            &polynomial,
            &opening_point,
            Some(hint),
            &mut prover_transcript,
        );
        let mut verifier_transcript = PoseidonTranscript::new(b"stage14-dory");
        bind_opening_inputs::<Fr, _>(&mut verifier_transcript, &opening_point, &opening);
        let mut transcript_binding = verifier_transcript.state.to_vec();
        transcript_binding.extend_from_slice(&verifier_transcript.n_rounds.to_le_bytes());

        let valid = RecursiveDeferredPcsOpening::<Fr, DoryCommitmentScheme, _>::new(
            proof.clone(),
            verifier_setup.clone(),
            verifier_transcript.clone(),
            opening_point.clone(),
            opening,
            commitment,
            &transcript_binding,
        );
        assert!(valid.verify().is_ok());

        let forged = RecursiveDeferredPcsOpening::<Fr, DoryCommitmentScheme, _>::new(
            proof,
            verifier_setup,
            verifier_transcript,
            opening_point,
            opening + Fr::from(1u64),
            commitment,
            &transcript_binding,
        );
        assert!(forged.verify().is_err());
    }
}
