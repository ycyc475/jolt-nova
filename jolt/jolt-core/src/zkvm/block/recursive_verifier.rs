#![cfg_attr(feature = "zk", allow(dead_code))]

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
    poly::opening_proof::{OpeningId, SumcheckId},
    subprotocols::blindfold::VerifierR1CS,
    subprotocols::sumcheck::{ClearSumcheckProof, SumcheckInstanceProof},
    transcripts::{PoseidonTranscript, Transcript},
    zkvm::{proof_serialization::JoltProof, verifier::JoltVerifierPreprocessing},
};
#[cfg(feature = "zk")]
use crate::{
    poly::commitment::pedersen::PedersenGenerators,
    subprotocols::blindfold::{BlindFoldProof, BlindFoldVerifier, BlindFoldVerifierInput},
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

/// Verifier-derived endpoint data needed to adapt one native clear sumcheck.
/// The transcript checkpoint is the exact state immediately before the native
/// `ClearSumcheckProof::verify` call.
#[derive(Clone)]
pub struct RecursiveClearSumcheckStageContext {
    pub transcript_before: PoseidonTranscript,
    pub initial_claim: ark_bn254::Fr,
    pub expected_final_claim: ark_bn254::Fr,
    pub degree_bound: usize,
}

/// Lossless Stage-15 artifact extracted from a real clear Jolt sumcheck.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecursiveClearSumcheckStageArtifact {
    pub transcript_state_before: RecursiveJoltFieldElement,
    pub transcript_round_before: u64,
    pub witness: RecursiveClearSumcheckStageWitness,
}

/// Native Jolt verifier relation represented by one opening variable.
///
/// An opening can participate in its endpoint relation and in the final PCS
/// reduction simultaneously, so bindings retain a set of relation families.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum RecursiveVerifierRelationKind {
    Lasso,
    Register,
    Ram,
    Cpu,
    Pcs,
}

/// Auditable mapping from a native Jolt opening identifier to the exact
/// verifier-R1CS variable that constrains it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecursiveVerifierOpeningBinding {
    pub opening_id: OpeningId,
    pub canonical_opening_id: OpeningId,
    pub variable_index: usize,
    pub relations: Vec<RecursiveVerifierRelationKind>,
}

/// Production artifact emitted only by a successful native clear-Jolt
/// verification. It contains the complete verifier relation and its assigned
/// witness; callers cannot substitute a separately constructed R1CS witness.
#[derive(Clone, Debug)]
pub struct RecursiveJoltVerifierRelationArtifact {
    sumcheck_artifacts: Vec<RecursiveClearSumcheckStageArtifact>,
    verifier_r1cs: VerifierR1CS<ark_bn254::Fr>,
    verifier_witness: Vec<ark_bn254::Fr>,
    opening_bindings: Vec<RecursiveVerifierOpeningBinding>,
}

impl RecursiveJoltVerifierRelationArtifact {
    fn endpoint_relation(opening_id: OpeningId) -> RecursiveVerifierRelationKind {
        let sumcheck_id = match opening_id {
            OpeningId::Polynomial(_, sumcheck_id)
            | OpeningId::TrustedAdvice(sumcheck_id)
            | OpeningId::UntrustedAdvice(sumcheck_id) => sumcheck_id,
        };
        match sumcheck_id {
            SumcheckId::RegistersClaimReduction
            | SumcheckId::RegistersReadWriteChecking
            | SumcheckId::RegistersValEvaluation => RecursiveVerifierRelationKind::Register,
            SumcheckId::RamReadWriteChecking
            | SumcheckId::RamRafEvaluation
            | SumcheckId::RamOutputCheck
            | SumcheckId::RamValCheck
            | SumcheckId::RamRaClaimReduction
            | SumcheckId::RamHammingBooleanity
            | SumcheckId::RamRaVirtualization
            | SumcheckId::AdviceClaimReductionCyclePhase
            | SumcheckId::AdviceClaimReduction
            | SumcheckId::ProgramImageClaimReductionCyclePhase
            | SumcheckId::ProgramImageClaimReduction => RecursiveVerifierRelationKind::Ram,
            SumcheckId::SpartanOuter
            | SumcheckId::SpartanProductVirtualization
            | SumcheckId::SpartanShift
            | SumcheckId::InstructionInputVirtualization => RecursiveVerifierRelationKind::Cpu,
            SumcheckId::InstructionClaimReduction
            | SumcheckId::InstructionReadRaf
            | SumcheckId::InstructionRaVirtualization
            | SumcheckId::BytecodeReadRafAddressPhase
            | SumcheckId::BytecodeReadRaf
            | SumcheckId::BooleanityAddressPhase
            | SumcheckId::Booleanity
            | SumcheckId::BytecodeClaimReductionCyclePhase
            | SumcheckId::BytecodeClaimReduction
            | SumcheckId::IncClaimReduction
            | SumcheckId::HammingWeightClaimReduction => RecursiveVerifierRelationKind::Lasso,
        }
    }

    pub(crate) fn from_verified_parts(
        sumcheck_artifacts: Vec<RecursiveClearSumcheckStageArtifact>,
        verifier_r1cs: VerifierR1CS<ark_bn254::Fr>,
        verifier_witness: Vec<ark_bn254::Fr>,
        pcs_opening_ids: &[OpeningId],
    ) -> Result<Self, &'static str> {
        use std::collections::BTreeSet;

        if sumcheck_artifacts.len() != 8 {
            return Err("recursive Jolt relation requires all eight sumcheck stages");
        }
        if verifier_witness.len() != verifier_r1cs.num_vars {
            return Err("recursive Jolt verifier witness/R1CS dimension mismatch");
        }
        verifier_r1cs
            .check_satisfaction(&verifier_witness)
            .map_err(|_| "native Jolt verifier relation is not satisfied")?;

        let pcs_ids = pcs_opening_ids
            .iter()
            .map(|id| verifier_r1cs.resolve_alias(*id))
            .collect::<BTreeSet<_>>();
        let mut opening_bindings = verifier_r1cs
            .opening_vars
            .iter()
            .map(|(opening_id, variable_index)| {
                let canonical_opening_id = verifier_r1cs.resolve_alias(*opening_id);
                let mut relations = vec![Self::endpoint_relation(*opening_id)];
                if pcs_ids.contains(&canonical_opening_id) {
                    relations.push(RecursiveVerifierRelationKind::Pcs);
                }
                relations.sort();
                relations.dedup();
                RecursiveVerifierOpeningBinding {
                    opening_id: *opening_id,
                    canonical_opening_id,
                    variable_index: *variable_index,
                    relations,
                }
            })
            .collect::<Vec<_>>();
        opening_bindings.sort_by_key(|binding| (binding.variable_index, binding.opening_id));

        let covered = opening_bindings
            .iter()
            .flat_map(|binding| binding.relations.iter().copied())
            .collect::<BTreeSet<_>>();
        if ![
            RecursiveVerifierRelationKind::Lasso,
            RecursiveVerifierRelationKind::Register,
            RecursiveVerifierRelationKind::Ram,
            RecursiveVerifierRelationKind::Cpu,
            RecursiveVerifierRelationKind::Pcs,
        ]
        .into_iter()
        .all(|kind| covered.contains(&kind))
        {
            return Err("recursive Jolt verifier relation is missing an endpoint family");
        }

        Ok(Self {
            sumcheck_artifacts,
            verifier_r1cs,
            verifier_witness,
            opening_bindings,
        })
    }

    pub fn sumcheck_artifacts(&self) -> &[RecursiveClearSumcheckStageArtifact] {
        &self.sumcheck_artifacts
    }

    pub fn verifier_r1cs(&self) -> &VerifierR1CS<ark_bn254::Fr> {
        &self.verifier_r1cs
    }

    pub fn opening_bindings(&self) -> &[RecursiveVerifierOpeningBinding] {
        &self.opening_bindings
    }

    #[cfg(all(test, feature = "prover"))]
    pub(crate) fn test_tamper_verifier_witness(&mut self) {
        if self.verifier_witness.len() > 1 {
            self.verifier_witness[1] += ark_bn254::Fr::from(1u64);
        }
    }

    pub(crate) fn into_circuit_parts(
        self,
    ) -> (
        Vec<RecursiveClearSumcheckStageArtifact>,
        VerifierR1CS<ark_bn254::Fr>,
        Vec<ark_bn254::Fr>,
    ) {
        (
            self.sumcheck_artifacts,
            self.verifier_r1cs,
            self.verifier_witness,
        )
    }
}

impl RecursiveClearSumcheckStageArtifact {
    pub fn from_native_proof(
        stage_index: usize,
        proof: &ClearSumcheckProof<ark_bn254::Fr, PoseidonTranscript>,
        context: RecursiveClearSumcheckStageContext,
    ) -> Result<Self, ProofVerifyError> {
        let mut transcript = context.transcript_before;
        let transcript_state_before = RecursiveJoltFieldElement {
            canonical_le_bytes: transcript.state,
        };
        let transcript_round_before = transcript.n_rounds as u64;
        let (final_claim, challenges) = proof.verify(
            context.initial_claim,
            proof.compressed_polys.len(),
            context.degree_bound,
            &mut transcript,
        )?;
        if final_claim != context.expected_final_claim
            || challenges.len() != proof.compressed_polys.len()
        {
            return Err(ProofVerifyError::SumcheckVerificationError);
        }

        let rounds = proof
            .compressed_polys
            .iter()
            .zip(challenges)
            .map(|(polynomial, challenge)| {
                let challenge_field: ark_bn254::Fr = challenge.into();
                RecursiveClearSumcheckRoundWitness {
                    coefficients_except_linear: polynomial
                        .coeffs_except_linear_term
                        .iter()
                        .copied()
                        .map(RecursiveJoltFieldElement::from_field)
                        .collect::<Result<Vec<_>, _>>()
                        .expect("BN254 Fr has a canonical 32-byte encoding"),
                    challenge: RecursiveJoltFieldElement::from_field(challenge_field)
                        .expect("BN254 challenge has a canonical field encoding"),
                }
            })
            .collect();
        let witness = RecursiveClearSumcheckStageWitness {
            stage_index,
            degree_bound: context.degree_bound,
            initial_claim: RecursiveJoltFieldElement::from_field(context.initial_claim)
                .expect("BN254 Fr has a canonical 32-byte encoding"),
            rounds,
            expected_final_claim: RecursiveJoltFieldElement::from_field(final_claim)
                .expect("BN254 Fr has a canonical 32-byte encoding"),
        };
        witness
            .validate_shape(
                stage_index,
                proof.compressed_polys.len(),
                context.degree_bound,
            )
            .map_err(|_| ProofVerifyError::InternalError)?;
        Ok(Self {
            transcript_state_before,
            transcript_round_before,
            witness,
        })
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

/// A complete BlindFold verifier equation deferred from the scalar Nova step.
///
/// This is the ZK analogue of [`RecursiveDeferredPcsOpening`].  The obligation
/// retains the real proof, commitments, R1CS, generators, and transcript state;
/// final acceptance calls the production BlindFold verifier.  The identifier
/// only binds this exact equation into a recursive statement and is never used
/// as evidence that verification succeeded.
#[cfg(feature = "zk")]
pub struct RecursiveDeferredBlindFoldVerification<F, C, FS>
where
    F: JoltField,
    C: JoltCurve<F = F>,
    FS: Transcript,
{
    proof: BlindFoldProof<F, C>,
    input: BlindFoldVerifierInput<C>,
    generators: PedersenGenerators<C>,
    verifier_r1cs: VerifierR1CS<F>,
    eval_commitment_generators: Option<(C::G1, C::G1)>,
    transcript_before_blindfold: FS,
    obligation_id: [u8; 32],
}

#[cfg(feature = "zk")]
impl<F, C, FS> RecursiveDeferredBlindFoldVerification<F, C, FS>
where
    F: JoltField,
    C: JoltCurve<F = F>,
    FS: Transcript,
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        proof: BlindFoldProof<F, C>,
        input: BlindFoldVerifierInput<C>,
        generators: PedersenGenerators<C>,
        verifier_r1cs: VerifierR1CS<F>,
        eval_commitment_generators: Option<(C::G1, C::G1)>,
        transcript_before_blindfold: FS,
        transcript_binding: &[u8],
    ) -> Self {
        let obligation_id = compute_deferred_blindfold_obligation_id(
            &proof,
            &input,
            &generators,
            &verifier_r1cs,
            &eval_commitment_generators,
            transcript_binding,
        );
        Self {
            proof,
            input,
            generators,
            verifier_r1cs,
            eval_commitment_generators,
            transcript_before_blindfold,
            obligation_id,
        }
    }

    pub fn obligation_id(&self) -> [u8; 32] {
        self.obligation_id
    }

    pub fn verify(&self) -> Result<(), String> {
        let verifier = BlindFoldVerifier::<F, C>::new(
            &self.generators,
            &self.verifier_r1cs,
            self.eval_commitment_generators.clone(),
        );
        let mut transcript = self.transcript_before_blindfold.clone();
        verifier
            .verify(&self.proof, &self.input, &mut transcript)
            .map_err(|error| format!("deferred BlindFold verification failed: {error:?}"))
    }
}

#[cfg(feature = "zk")]
fn append_blindfold_r1cs<F: JoltField>(hasher: &mut Sha3_256, r1cs: &VerifierR1CS<F>) {
    hasher.update((r1cs.num_vars as u64).to_le_bytes());
    hasher.update((r1cs.num_constraints as u64).to_le_bytes());
    for (label, matrix) in [
        (&b"A"[..], &r1cs.a),
        (&b"B"[..], &r1cs.b),
        (&b"C"[..], &r1cs.c),
    ] {
        hasher.update(label);
        hasher.update((matrix.num_rows as u64).to_le_bytes());
        hasher.update((matrix.num_cols as u64).to_le_bytes());
        hasher.update((matrix.entries.len() as u64).to_le_bytes());
        for (row, column, value) in &matrix.entries {
            hasher.update((*row as u64).to_le_bytes());
            hasher.update((*column as u64).to_le_bytes());
            append_canonical(hasher, b"matrix-value", value);
        }
    }
    for value in [
        r1cs.hyrax.C,
        r1cs.hyrax.R_coeff,
        r1cs.hyrax.R_prime,
        r1cs.hyrax.noncoeff_count,
        r1cs.hyrax.total_rounds,
        r1cs.hyrax.output_claims_rows,
    ] {
        hasher.update((value as u64).to_le_bytes());
    }
    hasher.update((r1cs.extra_output_vars.len() as u64).to_le_bytes());
    for variable in &r1cs.extra_output_vars {
        hasher.update((variable.index() as u64).to_le_bytes());
    }
    hasher.update((r1cs.extra_blinding_vars.len() as u64).to_le_bytes());
    for variable in &r1cs.extra_blinding_vars {
        hasher.update((variable.index() as u64).to_le_bytes());
    }
}

#[cfg(feature = "zk")]
fn compute_deferred_blindfold_obligation_id<F, C>(
    proof: &BlindFoldProof<F, C>,
    input: &BlindFoldVerifierInput<C>,
    generators: &PedersenGenerators<C>,
    verifier_r1cs: &VerifierR1CS<F>,
    eval_commitment_generators: &Option<(C::G1, C::G1)>,
    transcript_binding: &[u8],
) -> [u8; 32]
where
    F: JoltField,
    C: JoltCurve<F = F>,
{
    let mut hasher = Sha3_256::new();
    hasher.update(b"jolt-nova/deferred-blindfold-verification/v1");
    append_canonical(&mut hasher, b"proof", proof);
    append_canonical(&mut hasher, b"round-commitments", &input.round_commitments);
    append_canonical(
        &mut hasher,
        b"output-claim-row-commitments",
        &input.output_claims_row_commitments,
    );
    append_canonical(
        &mut hasher,
        b"evaluation-commitments",
        &input.eval_commitments,
    );
    append_canonical(&mut hasher, b"pedersen-generators", generators);
    append_canonical(
        &mut hasher,
        b"evaluation-commitment-generators",
        eval_commitment_generators,
    );
    append_blindfold_r1cs(&mut hasher, verifier_r1cs);
    hasher.update((transcript_binding.len() as u64).to_le_bytes());
    hasher.update(transcript_binding);
    hasher.finalize().into()
}

/// Public continuity statement for the ZK/BlindFold acceptance path.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg(feature = "zk")]
pub struct RecursiveJoltZkStatement {
    pub object_id: [u8; 32],
    pub deferred_pcs_id: [u8; 32],
    pub deferred_blindfold_id: [u8; 32],
}

/// Final ZK envelope. Both deferred equations are verified cryptographically;
/// none of the three identifiers is treated as an acceptance receipt.
#[cfg(feature = "zk")]
pub struct RecursiveJoltZkFinalAcceptance<F, C, PCS, FS>
where
    F: JoltField,
    C: JoltCurve<F = F>,
    PCS: CommitmentScheme<Field = F>,
    FS: Transcript,
{
    pub deferred_pcs_opening: RecursiveDeferredPcsOpening<F, PCS, FS>,
    pub deferred_blindfold_verification: RecursiveDeferredBlindFoldVerification<F, C, FS>,
}

#[cfg(feature = "zk")]
impl<F, C, PCS, FS> RecursiveJoltZkFinalAcceptance<F, C, PCS, FS>
where
    F: JoltField,
    C: JoltCurve<F = F>,
    PCS: CommitmentScheme<Field = F>,
    FS: Transcript,
{
    pub fn verify(
        &self,
        statement: &RecursiveJoltZkStatement,
        expected_object_id: [u8; 32],
    ) -> Result<(), String> {
        if statement.object_id != expected_object_id {
            return Err("recursive ZK verifier object identifier mismatch".to_string());
        }
        if statement.deferred_pcs_id != self.deferred_pcs_opening.obligation_id() {
            return Err("recursive ZK deferred PCS identifier mismatch".to_string());
        }
        if statement.deferred_blindfold_id != self.deferred_blindfold_verification.obligation_id() {
            return Err("recursive ZK deferred BlindFold identifier mismatch".to_string());
        }
        self.deferred_pcs_opening
            .verify()
            .map_err(|error| format!("deferred ZK PCS verification failed: {error}"))?;
        self.deferred_blindfold_verification.verify()
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

impl<C, PCS, PublicIo>
    RecursiveJoltVerifierObject<ark_bn254::Fr, C, PCS, PoseidonTranscript, PublicIo>
where
    C: JoltCurve<F = ark_bn254::Fr>,
    PCS: CommitmentScheme<Field = ark_bn254::Fr>,
    PCS::Commitment: CanonicalSerialize,
    PCS::VerifierSetup: CanonicalSerialize,
    PublicIo: CanonicalSerialize,
{
    /// Automatically extracts all eight native clear sumchecks from this Jolt
    /// proof. Contexts are produced by the verifier stage observer, not by an
    /// application-level receipt.
    pub fn adapt_clear_sumchecks(
        &self,
        contexts: [RecursiveClearSumcheckStageContext; 8],
    ) -> Result<Vec<RecursiveClearSumcheckStageArtifact>, ProofVerifyError> {
        self.sumcheck_stages()
            .into_iter()
            .zip(contexts)
            .enumerate()
            .map(|(stage_index, (proof, context))| match proof {
                SumcheckInstanceProof::Clear(proof) => {
                    RecursiveClearSumcheckStageArtifact::from_native_proof(
                        stage_index,
                        proof,
                        context,
                    )
                }
                SumcheckInstanceProof::Zk(_) => Err(ProofVerifyError::ZkFeatureRequired),
            })
            .collect()
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
    use super::{
        append_canonical, RecursiveClearSumcheckStageArtifact, RecursiveClearSumcheckStageContext,
        RecursiveVerifierPcsStrategy,
    };
    use crate::{
        poly::unipoly::CompressedUniPoly,
        subprotocols::sumcheck::ClearSumcheckProof,
        transcripts::{PoseidonTranscript, Transcript},
    };
    use ark_bn254::Fr;
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

    #[test]
    fn clear_sumcheck_adapter_uses_native_verifier_checkpoint_and_endpoints() {
        let proof = ClearSumcheckProof::new(vec![CompressedUniPoly {
            coeffs_except_linear_term: vec![Fr::from(2u64), Fr::from(6u64)],
        }]);
        let transcript = PoseidonTranscript::new(b"stage15-adapter");
        let mut replay = transcript.clone();
        let (final_claim, _) = proof.verify(Fr::from(10u64), 1, 2, &mut replay).unwrap();
        let artifact = RecursiveClearSumcheckStageArtifact::from_native_proof(
            3,
            &proof,
            RecursiveClearSumcheckStageContext {
                transcript_before: transcript.clone(),
                initial_claim: Fr::from(10u64),
                expected_final_claim: final_claim,
                degree_bound: 2,
            },
        )
        .unwrap();
        assert_eq!(
            artifact.transcript_state_before.canonical_le_bytes,
            transcript.state
        );
        assert_eq!(artifact.transcript_round_before, transcript.n_rounds as u64);
        assert_eq!(artifact.witness.stage_index, 3);
        assert_eq!(artifact.witness.rounds.len(), 1);

        assert!(RecursiveClearSumcheckStageArtifact::from_native_proof(
            3,
            &proof,
            RecursiveClearSumcheckStageContext {
                transcript_before: transcript,
                initial_claim: Fr::from(10u64),
                expected_final_claim: final_claim + Fr::from(1u64),
                degree_bound: 2,
            },
        )
        .is_err());
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

        // Transparent Dory binds the scalar evaluation directly.  In ZK
        // Dory, the evaluation is represented by a blinded commitment inside
        // the proof and the scalar linkage is enforced by BlindFold, so the
        // negative test must instead corrupt that committed evaluation.
        #[cfg(not(feature = "zk"))]
        let (forged_proof, forged_opening) = (proof, opening + Fr::from(1u64));
        #[cfg(feature = "zk")]
        let (forged_proof, forged_opening) = {
            let mut forged_proof = proof;
            if let Some(ref mut y_com) = forged_proof.y_com {
                *y_com = *y_com + verifier_setup.g1_0;
            } else if let Some(ref mut e2) = forged_proof.e2 {
                *e2 = *e2 + verifier_setup.g2_0;
            } else {
                panic!("ZK Dory proof missing committed evaluation fields");
            }
            (forged_proof, opening)
        };
        let forged = RecursiveDeferredPcsOpening::<Fr, DoryCommitmentScheme, _>::new(
            forged_proof,
            verifier_setup,
            verifier_transcript,
            opening_point,
            forged_opening,
            commitment,
            &transcript_binding,
        );
        assert!(forged.verify().is_err());
    }
}

#[cfg(all(test, feature = "zk"))]
mod deferred_blindfold_tests {
    use ark_bn254::Fr;
    use ark_std::Zero;

    use crate::{
        curve::Bn254Curve,
        field::JoltField,
        poly::commitment::pedersen::PedersenGenerators,
        subprotocols::blindfold::{
            BakedPublicInputs, BlindFoldProver, BlindFoldVerifierInput, BlindFoldWitness,
            RelaxedR1CSInstance, RoundWitness, StageConfig, StageWitness, VerifierR1CSBuilder,
        },
        transcripts::{KeccakTranscript, Transcript},
    };

    use super::RecursiveDeferredBlindFoldVerification;

    fn fixture(
        tamper: bool,
    ) -> RecursiveDeferredBlindFoldVerification<Fr, Bn254Curve, KeccakTranscript> {
        let config = StageConfig::new(1, 3);
        let round = RoundWitness::new(
            vec![
                Fr::from_u64(40),
                Fr::from_u64(5),
                Fr::from_u64(10),
                Fr::from_u64(5),
            ],
            Fr::from_u64(3),
        );
        let witness =
            BlindFoldWitness::new(Fr::from_u64(100), vec![StageWitness::new(vec![round])]);
        let baked = BakedPublicInputs::from_witness(&witness, std::slice::from_ref(&config));
        let r1cs = VerifierR1CSBuilder::<Fr>::new(std::slice::from_ref(&config), &baked).build();
        let generators = PedersenGenerators::<Bn254Curve>::deterministic(r1cs.hyrax.C + 1);
        let z = witness.assign(&r1cs);
        r1cs.check_satisfaction(&z).unwrap();
        let w = z[1..].to_vec();
        let mut rng = rand::thread_rng();
        let mut row_blindings = vec![Fr::zero(); r1cs.hyrax.R_prime];
        let mut round_commitments = Vec::new();
        for round_index in 0..r1cs.hyrax.total_rounds {
            let start = round_index * r1cs.hyrax.C;
            let blinding = Fr::random(&mut rng);
            round_commitments.push(generators.commit(&w[start..start + r1cs.hyrax.C], &blinding));
            row_blindings[round_index] = blinding;
        }
        let mut noncoeff_commitments = Vec::new();
        for row in 0..r1cs.hyrax.total_noncoeff_rows() {
            let start = r1cs.hyrax.R_coeff * r1cs.hyrax.C + row * r1cs.hyrax.C;
            let end = (start + r1cs.hyrax.C).min(w.len());
            let blinding = Fr::random(&mut rng);
            noncoeff_commitments.push(generators.commit(&w[start..end], &blinding));
            row_blindings[r1cs.hyrax.R_coeff + row] = blinding;
        }
        let (instance, relaxed_witness) = RelaxedR1CSInstance::<Fr, Bn254Curve>::new_non_relaxed(
            &w,
            r1cs.num_constraints,
            r1cs.hyrax.C,
            round_commitments,
            Vec::new(),
            noncoeff_commitments,
            Vec::new(),
            row_blindings,
        );
        let label = b"stage15-deferred-blindfold";
        let mut prover_transcript = KeccakTranscript::new(label);
        let mut proof = BlindFoldProver::new(&generators, &r1cs, None).prove(
            &instance,
            &relaxed_witness,
            &z,
            &mut prover_transcript,
        );
        if tamper {
            proof.spartan_proof.pop();
        }
        let input = BlindFoldVerifierInput {
            round_commitments: instance.round_commitments,
            output_claims_row_commitments: instance.output_claims_row_commitments,
            eval_commitments: instance.eval_commitments,
        };
        RecursiveDeferredBlindFoldVerification::new(
            proof,
            input,
            generators,
            r1cs,
            None,
            KeccakTranscript::new(label),
            b"stage15-deferred-blindfold-binding",
        )
    }

    #[test]
    fn deferred_blindfold_obligation_runs_real_verifier_and_rejects_tampering() {
        fixture(false).verify().unwrap();
        assert!(fixture(true).verify().is_err());
    }
}
