//! Stage-17 recursive verifier for Jolt's BlindFold/ZK path.
//!
//! BlindFold mixes scalar-field verifier equations with commitment-group
//! equations.  This module deliberately splits those two classes:
//!
//! * [`RecursiveBlindFoldVerifierCircuit`] replays the exact Poseidon
//!   transcript and proves the folding, outer/inner Spartan sumchecks,
//!   relaxed-R1CS endpoint equations, and scalar Hyrax evaluations in Nova.
//! * [`RecursiveBlindFoldGroupObligation`] retains the real curve points and
//!   executes the Pedersen/Hyrax commitment equations during final acceptance.
//!
//! Identifiers only bind the two halves together.  They are never interpreted
//! as evidence that either verifier accepted.

use ark_bn254::Fr;
use ark_ff::PrimeField;
use ark_serialize::CanonicalSerialize;
use ark_std::{One, Zero};
use nova_snark::{
    frontend::{num::AllocatedNum, ConstraintSystem, LinearCombination, SynthesisError},
    traits::circuit::StepCircuit,
};
use sha3::{Digest, Sha3_256};

use crate::{
    curve::JoltCurve,
    field::JoltField,
    poly::{
        commitment::{commitment_scheme::CommitmentScheme, pedersen::PedersenGenerators},
        unipoly::CompressedUniPoly,
    },
    subprotocols::blindfold::{
        compute_L_w_at_ry, BlindFoldProof, BlindFoldVerifier, BlindFoldVerifierInput, VerifierR1CS,
        INNER_SUMCHECK_DEGREE_BOUND, SPARTAN_DEGREE_BOUND,
    },
    transcripts::{PoseidonTranscript, Transcript},
    utils::math::Math,
    zkvm::proof_serialization::VerifiedJoltLookupProofReceipt,
};

use super::{
    nova_hash_bytes_to_scalar, nova_scalar_to_storage,
    recursive_relations::{
        alloc_nova_constant, ark_bn254_scalar_as_nova_scalar,
        synthesize_recursive_clear_sumcheck_stage, synthesize_recursive_poseidon_hash,
        synthesize_recursive_poseidon_transcript_transition,
        AllocatedRecursivePoseidonTranscriptState,
    },
    NovaPrimaryEngine, NovaPrimarySpartanSnark, NovaScalar, NovaSecondaryEngine,
    NovaSecondarySpartanSnark, RecursiveClearSumcheckRoundWitness,
    RecursiveClearSumcheckStageWitness, RecursiveDeferredPcsOpening, RecursiveJoltFieldElement,
};

const ZK_RECURSIVE_Z_ARITY: usize = 8;

type ZkRecursiveSnark = nova_snark::nova::RecursiveSNARK<
    NovaPrimaryEngine,
    NovaSecondaryEngine,
    RecursiveBlindFoldVerifierCircuit,
>;
type ZkCompressedSnark = nova_snark::nova::CompressedSNARK<
    NovaPrimaryEngine,
    NovaSecondaryEngine,
    RecursiveBlindFoldVerifierCircuit,
    NovaPrimarySpartanSnark,
    NovaSecondarySpartanSnark,
>;
type ZkProverKey = nova_snark::nova::ProverKey<
    NovaPrimaryEngine,
    NovaSecondaryEngine,
    RecursiveBlindFoldVerifierCircuit,
    NovaPrimarySpartanSnark,
    NovaSecondarySpartanSnark,
>;
type ZkVerificationKey = nova_snark::nova::VerifierKey<
    NovaPrimaryEngine,
    NovaSecondaryEngine,
    RecursiveBlindFoldVerifierCircuit,
    NovaPrimarySpartanSnark,
    NovaSecondarySpartanSnark,
>;

#[derive(Clone, Debug)]
enum PoseidonOperation {
    Append { chunks: Vec<Fr>, bytes: Vec<u8> },
    Challenge(Fr),
}

/// Transcript observer used only while adapting an already successful native
/// verification.  It forwards every operation to the production Poseidon
/// transcript and records the exact field words consumed by that transcript.
#[derive(Clone)]
struct RecordingPoseidonTranscript {
    inner: PoseidonTranscript,
    operations: Vec<PoseidonOperation>,
}

impl Default for RecordingPoseidonTranscript {
    fn default() -> Self {
        Self::new(b"")
    }
}

impl RecordingPoseidonTranscript {
    fn observe(inner: PoseidonTranscript) -> Self {
        Self {
            inner,
            operations: Vec::new(),
        }
    }

    fn record_bytes(&mut self, bytes: &[u8]) {
        let chunks = if bytes.is_empty() {
            vec![Fr::zero()]
        } else {
            bytes.chunks(32).map(Fr::from_le_bytes_mod_order).collect()
        };
        self.operations.push(PoseidonOperation::Append {
            chunks,
            bytes: bytes.to_vec(),
        });
    }

    fn record_challenge(&mut self, challenge: Fr) {
        self.operations
            .push(PoseidonOperation::Challenge(challenge));
    }
}

impl Transcript for RecordingPoseidonTranscript {
    fn new(label: &'static [u8]) -> Self {
        Self::observe(PoseidonTranscript::new(label))
    }

    #[cfg(test)]
    fn compare_to(&mut self, other: Self) {
        self.inner.compare_to(other.inner);
    }

    fn raw_append_label(&mut self, label: &'static [u8]) {
        let mut packed = [0u8; 32];
        packed[..label.len()].copy_from_slice(label);
        self.record_bytes(&packed);
        self.inner.raw_append_label(label);
    }

    fn raw_append_bytes(&mut self, bytes: &[u8]) {
        self.record_bytes(bytes);
        self.inner.raw_append_bytes(bytes);
    }

    fn raw_append_u64(&mut self, value: u64) {
        let mut packed = [0u8; 32];
        packed[..8].copy_from_slice(&value.to_le_bytes());
        self.record_bytes(&packed);
        self.inner.raw_append_u64(value);
    }

    fn raw_append_scalar<F: JoltField>(&mut self, scalar: &F) {
        let mut bytes = Vec::new();
        scalar
            .serialize_uncompressed(&mut bytes)
            .expect("field serialization into memory cannot fail");
        self.record_bytes(&bytes);
        self.inner.raw_append_scalar(scalar);
    }

    fn append_serializable<T: CanonicalSerialize>(&mut self, label: &'static [u8], data: &T) {
        let mut bytes = Vec::new();
        data.serialize_uncompressed(&mut bytes)
            .expect("serialization into memory cannot fail");
        self.raw_append_label_with_len(label, bytes.len() as u64);
        self.raw_append_bytes(&bytes);
    }

    fn challenge_u128(&mut self) -> u128 {
        let output = self.inner.challenge_u128();
        self.record_challenge(Fr::from_le_bytes_mod_order(&self.inner.state));
        output
    }

    fn challenge_scalar<F: JoltField>(&mut self) -> F {
        let output = self.inner.challenge_scalar::<F>();
        self.record_challenge(Fr::from_le_bytes_mod_order(&self.inner.state));
        output
    }

    fn challenge_scalar_128_bits<F: JoltField>(&mut self) -> F {
        let output = self.inner.challenge_scalar_128_bits::<F>();
        self.record_challenge(Fr::from_le_bytes_mod_order(&self.inner.state));
        output
    }

    fn challenge_vector<F: JoltField>(&mut self, len: usize) -> Vec<F> {
        (0..len).map(|_| self.challenge_scalar()).collect()
    }

    fn challenge_scalar_powers<F: JoltField>(&mut self, len: usize) -> Vec<F> {
        let challenge = self.challenge_scalar::<F>();
        let mut powers = vec![F::one(); len];
        for index in 1..len {
            powers[index] = powers[index - 1] * challenge;
        }
        powers
    }

    fn challenge_scalar_optimized<F: JoltField>(&mut self) -> F::Challenge {
        let challenge = self.challenge_scalar::<F>();
        // Poseidon uses full-width field challenges. This is the same conversion
        // performed by PoseidonTranscript itself.
        unsafe { std::mem::transmute_copy::<F, F::Challenge>(&challenge) }
    }

    fn challenge_vector_optimized<F: JoltField>(&mut self, len: usize) -> Vec<F::Challenge> {
        (0..len)
            .map(|_| self.challenge_scalar_optimized::<F>())
            .collect()
    }

    fn challenge_scalar_powers_optimized<F: JoltField>(&mut self, len: usize) -> Vec<F> {
        let challenge: F = self.challenge_scalar_optimized::<F>().into();
        let mut powers = vec![F::one(); len];
        for index in 1..len {
            powers[index] = powers[index - 1] * challenge;
        }
        powers
    }
}

fn field(value: Fr) -> RecursiveJoltFieldElement {
    let mut bytes = [0u8; 32];
    value
        .serialize_uncompressed(&mut bytes[..])
        .expect("BN254 scalar serialization cannot fail");
    RecursiveJoltFieldElement {
        canonical_le_bytes: bytes,
    }
}

fn compressed_rounds(
    proof: &[CompressedUniPoly<Fr>],
    challenges: &[Fr],
) -> Vec<RecursiveClearSumcheckRoundWitness> {
    proof
        .iter()
        .zip(challenges)
        .map(|(round, challenge)| RecursiveClearSumcheckRoundWitness {
            coefficients_except_linear: round
                .coeffs_except_linear_term
                .iter()
                .copied()
                .map(field)
                .collect(),
            challenge: field(*challenge),
        })
        .collect()
}

#[derive(Clone)]
struct BlindFoldScalarWitness {
    operations: Vec<PoseidonOperation>,
    outer: RecursiveClearSumcheckStageWitness,
    az_r: Fr,
    bz_r: Fr,
    cz_r: Fr,
    inner: RecursiveClearSumcheckStageWitness,
    random_u: Fr,
    e_opening_row: Vec<Fr>,
    w_opening_row: Vec<Fr>,
    folded_eval_outputs: Vec<Fr>,
    folded_eval_blindings: Vec<Fr>,
    folded_eval_output_rows: Vec<Vec<Fr>>,
    folded_eval_blinding_rows: Vec<Vec<Fr>>,
}

/// Public statement verified by a pinned Stage-17 key.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecursiveBlindFoldStatement {
    pub object_id: [u8; 32],
    pub deferred_group_id: [u8; 32],
    pub initial_transcript_state: RecursiveJoltFieldElement,
    pub initial_transcript_round: u64,
    pub final_transcript_state: RecursiveJoltFieldElement,
    pub final_transcript_round: u64,
    pub shape_id: [u8; 32],
    /// Common clear/ZK execution statement.  It binds preprocessing, public
    /// I/O, trace length, and advice identity, but deliberately excludes proof
    /// randomness and the proof mode.
    pub jolt_statement_id: [u8; 32],
}

impl RecursiveBlindFoldStatement {
    fn initial_z(&self) -> Result<Vec<NovaScalar>, String> {
        let decode = |value: &RecursiveJoltFieldElement| {
            Option::from(NovaScalar::from_bytes(&value.canonical_le_bytes))
                .ok_or_else(|| "non-canonical recursive BlindFold field element".to_string())
        };
        Ok(vec![
            nova_hash_bytes_to_scalar("recursive-blindfold", "object-id", &self.object_id),
            nova_hash_bytes_to_scalar("recursive-blindfold", "group-id", &self.deferred_group_id),
            decode(&self.initial_transcript_state)?,
            NovaScalar::from(self.initial_transcript_round),
            decode(&self.final_transcript_state)?,
            NovaScalar::from(self.final_transcript_round),
            NovaScalar::zero(),
            nova_hash_bytes_to_scalar(
                "recursive-blindfold",
                "jolt-statement-id",
                &self.jolt_statement_id,
            ),
        ])
    }
}

/// Complete curve-side obligation paired with the scalar recursive proof.
pub struct RecursiveBlindFoldGroupObligation<C: JoltCurve<F = Fr>> {
    proof: BlindFoldProof<Fr, C>,
    input: BlindFoldVerifierInput<C>,
    generators: PedersenGenerators<C>,
    verifier_r1cs: VerifierR1CS<Fr>,
    eval_commitment_generators: Option<(C::G1, C::G1)>,
    folding_challenge: Fr,
    outer_challenges: Vec<<Fr as JoltField>::Challenge>,
    inner_challenges: Vec<<Fr as JoltField>::Challenge>,
    initial_transcript: PoseidonTranscript,
    final_transcript: PoseidonTranscript,
    transcript_operations: Vec<PoseidonOperation>,
    obligation_id: [u8; 32],
}

impl<C: JoltCurve<F = Fr>> RecursiveBlindFoldGroupObligation<C> {
    pub fn obligation_id(&self) -> [u8; 32] {
        self.obligation_id
    }

    pub fn final_transcript_checkpoint(&self) -> (RecursiveJoltFieldElement, u64) {
        (
            RecursiveJoltFieldElement {
                canonical_le_bytes: self.final_transcript.state,
            },
            self.final_transcript.n_rounds as u64,
        )
    }

    /// Executes the real Pedersen/Hyrax group equations and independently
    /// replays the exact native BlindFold transcript operations to the public
    /// final checkpoint. Scalar acceptance remains exclusively the recursive
    /// circuit's responsibility.
    pub fn verify(&self) -> Result<(), String> {
        let verifier = BlindFoldVerifier::new(
            &self.generators,
            &self.verifier_r1cs,
            self.eval_commitment_generators,
        );
        verifier
            .verify_deferred_group_relations(
                &self.proof,
                &self.input,
                self.folding_challenge,
                &self.outer_challenges,
                &self.inner_challenges,
            )
            .map_err(|error| format!("deferred BlindFold group relation failed: {error:?}"))?;

        let mut transcript = self.initial_transcript.clone();
        for operation in &self.transcript_operations {
            match operation {
                PoseidonOperation::Append { bytes, .. } => transcript.raw_append_bytes(bytes),
                PoseidonOperation::Challenge(expected) => {
                    let challenge = transcript.challenge_scalar::<Fr>();
                    if challenge != *expected {
                        return Err("deferred BlindFold transcript challenge mismatch".to_string());
                    }
                }
            }
        }
        if transcript.state != self.final_transcript.state
            || transcript.n_rounds != self.final_transcript.n_rounds
        {
            return Err("deferred BlindFold transcript checkpoint mismatch".to_string());
        }
        Ok(())
    }
}

/// Verifier-derived artifact. Its constructor is crate-private and requires a
/// successful production BlindFold verification.
pub struct RecursiveBlindFoldRelationArtifact<C: JoltCurve<F = Fr>> {
    circuit: RecursiveBlindFoldVerifierCircuit,
    group_obligation: RecursiveBlindFoldGroupObligation<C>,
}

/// Lossless Stage-18 export from one successful production ZK Jolt verifier.
///
/// Both components originate from the same verifier invocation: the
/// BlindFold scalar/group relation and the exact PCS equation checked before
/// BlindFold.  Keeping them in one typed object prevents a caller from pairing
/// a recursive verifier artifact with an unrelated Dory opening.
pub struct RecursiveJoltZkVerifiedArtifacts<C: JoltCurve<F = Fr>, PCS: CommitmentScheme<Field = Fr>>
{
    blindfold: RecursiveBlindFoldRelationArtifact<C>,
    deferred_pcs_opening: RecursiveDeferredPcsOpening<Fr, PCS, PoseidonTranscript>,
    verified_jolt_receipt: VerifiedJoltLookupProofReceipt,
}

impl<C, PCS> RecursiveJoltZkVerifiedArtifacts<C, PCS>
where
    C: JoltCurve<F = Fr>,
    PCS: CommitmentScheme<Field = Fr>,
{
    pub(crate) fn new(
        blindfold: RecursiveBlindFoldRelationArtifact<C>,
        deferred_pcs_opening: RecursiveDeferredPcsOpening<Fr, PCS, PoseidonTranscript>,
        verified_jolt_receipt: VerifiedJoltLookupProofReceipt,
    ) -> Self {
        Self {
            blindfold,
            deferred_pcs_opening,
            verified_jolt_receipt,
        }
    }

    pub fn statement(&self) -> RecursiveBlindFoldStatement {
        self.blindfold.statement()
    }

    pub fn deferred_pcs_id(&self) -> [u8; 32] {
        self.deferred_pcs_opening.obligation_id()
    }

    /// Exact receipt emitted by the same successful verifier invocation.
    pub fn verified_jolt_receipt(&self) -> &VerifiedJoltLookupProofReceipt {
        &self.verified_jolt_receipt
    }

    pub fn into_parts(
        self,
    ) -> (
        RecursiveBlindFoldRelationArtifact<C>,
        RecursiveDeferredPcsOpening<Fr, PCS, PoseidonTranscript>,
    ) {
        (self.blindfold, self.deferred_pcs_opening)
    }
}

impl<C: JoltCurve<F = Fr>> RecursiveBlindFoldRelationArtifact<C> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_verified_parts(
        proof: BlindFoldProof<Fr, C>,
        input: BlindFoldVerifierInput<C>,
        generators: PedersenGenerators<C>,
        verifier_r1cs: VerifierR1CS<Fr>,
        eval_commitment_generators: Option<(C::G1, C::G1)>,
        transcript_before_blindfold: PoseidonTranscript,
        jolt_statement_id: [u8; 32],
    ) -> Result<Self, String> {
        let mut recorder =
            RecordingPoseidonTranscript::observe(transcript_before_blindfold.clone());
        recorder.append_label(b"BlindFold");
        BlindFoldVerifier::new(&generators, &verifier_r1cs, eval_commitment_generators)
            .verify(&proof, &input, &mut recorder)
            .map_err(|error| format!("verified BlindFold artifact replay failed: {error:?}"))?;

        let challenges = recorder
            .operations
            .iter()
            .filter_map(|operation| match operation {
                PoseidonOperation::Challenge(value) => Some(*value),
                PoseidonOperation::Append { .. } => None,
            })
            .collect::<Vec<_>>();
        let outer_rounds = verifier_r1cs.num_constraints.next_power_of_two().log_2();
        let inner_rounds = (verifier_r1cs.hyrax.R_prime * verifier_r1cs.hyrax.C).log_2();
        let expected_challenges = 1 + outer_rounds + outer_rounds + 3 + inner_rounds;
        if challenges.len() != expected_challenges
            || proof.spartan_proof.len() != outer_rounds
            || proof.inner_sumcheck_proof.len() != inner_rounds
        {
            return Err("BlindFold recursive challenge/round shape mismatch".to_string());
        }
        let folding_challenge = challenges[0];
        let tau = challenges[1..1 + outer_rounds].to_vec();
        let outer_challenges = challenges[1 + outer_rounds..1 + 2 * outer_rounds].to_vec();
        let scalar_challenge_start = 1 + 2 * outer_rounds;
        let ra = challenges[scalar_challenge_start];
        let rb = challenges[scalar_challenge_start + 1];
        let rc = challenges[scalar_challenge_start + 2];
        let inner_challenges = challenges[scalar_challenge_start + 3..].to_vec();
        let folded_u = Fr::one() + folding_challenge * proof.random_instance.u;

        let outer_challenges_native = outer_challenges
            .iter()
            .copied()
            .map(|value| unsafe {
                std::mem::transmute_copy::<Fr, <Fr as JoltField>::Challenge>(&value)
            })
            .collect::<Vec<_>>();
        let inner_challenges_native = inner_challenges
            .iter()
            .copied()
            .map(|value| unsafe {
                std::mem::transmute_copy::<Fr, <Fr as JoltField>::Challenge>(&value)
            })
            .collect::<Vec<_>>();
        let spartan = crate::subprotocols::blindfold::BlindFoldSpartanVerifier::new(
            &verifier_r1cs,
            tau.iter()
                .copied()
                .map(|value| unsafe {
                    std::mem::transmute_copy::<Fr, <Fr as JoltField>::Challenge>(&value)
                })
                .collect(),
            folded_u,
        );
        let (r_e, _) = verifier_r1cs.hyrax.e_grid(verifier_r1cs.num_constraints);
        let rx_col = &outer_challenges[r_e.log_2()..];
        let e_r = crate::poly::commitment::hyrax::evaluate(&proof.e_opening.combined_row, rx_col);
        let expected_outer = spartan.expected_claim(
            &outer_challenges_native,
            proof.az_r,
            proof.bz_r,
            proof.cz_r,
            e_r,
        );
        let (pub_a, pub_b, pub_c) = spartan.public_contributions(&outer_challenges_native);
        let inner_initial =
            ra * (proof.az_r - pub_a) + rb * (proof.bz_r - pub_b) + rc * (proof.cz_r - pub_c);
        let ry_col = &inner_challenges[verifier_r1cs.hyrax.log_R_prime()..];
        let w_ry = crate::poly::commitment::hyrax::evaluate(&proof.w_opening.combined_row, ry_col);
        let l_w = compute_L_w_at_ry(
            &verifier_r1cs,
            &outer_challenges_native,
            &inner_challenges_native,
            ra,
            rb,
            rc,
        );
        let expected_inner = l_w * w_ry;

        let transcript_operations = recorder.operations;
        let scalar_witness = BlindFoldScalarWitness {
            operations: transcript_operations.clone(),
            outer: RecursiveClearSumcheckStageWitness {
                stage_index: 0,
                degree_bound: SPARTAN_DEGREE_BOUND,
                initial_claim: field(Fr::zero()),
                rounds: compressed_rounds(&proof.spartan_proof, &outer_challenges),
                expected_final_claim: field(expected_outer),
            },
            az_r: proof.az_r,
            bz_r: proof.bz_r,
            cz_r: proof.cz_r,
            inner: RecursiveClearSumcheckStageWitness {
                stage_index: 1,
                degree_bound: INNER_SUMCHECK_DEGREE_BOUND,
                initial_claim: field(inner_initial),
                rounds: compressed_rounds(&proof.inner_sumcheck_proof, &inner_challenges),
                expected_final_claim: field(expected_inner),
            },
            random_u: proof.random_instance.u,
            e_opening_row: proof.e_opening.combined_row.clone(),
            w_opening_row: proof.w_opening.combined_row.clone(),
            folded_eval_outputs: proof.folded_eval_outputs.clone(),
            folded_eval_blindings: proof.folded_eval_blindings.clone(),
            folded_eval_output_rows: proof
                .folded_eval_output_openings
                .iter()
                .map(|opening| opening.combined_row.clone())
                .collect(),
            folded_eval_blinding_rows: proof
                .folded_eval_blinding_openings
                .iter()
                .map(|opening| opening.combined_row.clone())
                .collect(),
        };

        let initial = transcript_before_blindfold;
        let final_transcript = recorder.inner;
        let group_id = compute_group_obligation_id(
            &proof,
            &input,
            &generators,
            &verifier_r1cs,
            &eval_commitment_generators,
            &initial,
            &final_transcript,
            &transcript_operations,
        );
        let object_id = compute_scalar_object_id(
            group_id,
            &scalar_witness.operations,
            &verifier_r1cs,
            &initial,
            &final_transcript,
            jolt_statement_id,
        );
        let circuit = RecursiveBlindFoldVerifierCircuit {
            object_id,
            deferred_group_id: group_id,
            jolt_statement_id,
            initial_transcript: initial.clone(),
            final_transcript: final_transcript.clone(),
            verifier_r1cs: verifier_r1cs.clone(),
            witness: scalar_witness,
        };
        let group_obligation = RecursiveBlindFoldGroupObligation {
            proof,
            input,
            generators,
            verifier_r1cs,
            eval_commitment_generators,
            folding_challenge,
            outer_challenges: outer_challenges_native,
            inner_challenges: inner_challenges_native,
            initial_transcript: initial,
            final_transcript,
            transcript_operations,
            obligation_id: group_id,
        };
        Ok(Self {
            circuit,
            group_obligation,
        })
    }

    pub fn statement(&self) -> RecursiveBlindFoldStatement {
        self.circuit.statement()
    }

    pub fn into_parts(
        self,
    ) -> (
        RecursiveBlindFoldVerifierCircuit,
        RecursiveBlindFoldGroupObligation<C>,
    ) {
        (self.circuit, self.group_obligation)
    }
}

fn append_canonical<T: CanonicalSerialize>(hasher: &mut Sha3_256, value: &T) {
    let mut bytes = Vec::new();
    value
        .serialize_compressed(&mut bytes)
        .expect("canonical serialization into memory cannot fail");
    hasher.update((bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

fn append_r1cs(hasher: &mut Sha3_256, r1cs: &VerifierR1CS<Fr>) {
    for matrix in [&r1cs.a, &r1cs.b, &r1cs.c] {
        hasher.update((matrix.num_rows as u64).to_le_bytes());
        hasher.update((matrix.num_cols as u64).to_le_bytes());
        for (row, column, value) in &matrix.entries {
            hasher.update((*row as u64).to_le_bytes());
            hasher.update((*column as u64).to_le_bytes());
            append_canonical(hasher, value);
        }
    }
    for value in [
        r1cs.num_vars,
        r1cs.num_constraints,
        r1cs.hyrax.C,
        r1cs.hyrax.R_coeff,
        r1cs.hyrax.R_prime,
        r1cs.hyrax.output_claims_rows,
    ] {
        hasher.update((value as u64).to_le_bytes());
    }
}

fn append_checkpoint(hasher: &mut Sha3_256, transcript: &PoseidonTranscript) {
    hasher.update(transcript.state);
    hasher.update(transcript.n_rounds.to_le_bytes());
}

fn compute_group_obligation_id<C: JoltCurve<F = Fr>>(
    proof: &BlindFoldProof<Fr, C>,
    input: &BlindFoldVerifierInput<C>,
    generators: &PedersenGenerators<C>,
    r1cs: &VerifierR1CS<Fr>,
    eval_generators: &Option<(C::G1, C::G1)>,
    initial: &PoseidonTranscript,
    final_transcript: &PoseidonTranscript,
    operations: &[PoseidonOperation],
) -> [u8; 32] {
    let mut hasher = Sha3_256::new();
    hasher.update(b"jolt-nova/stage17/blindfold-group-obligation/v1");
    append_canonical(&mut hasher, proof);
    append_canonical(&mut hasher, &input.round_commitments);
    append_canonical(&mut hasher, &input.output_claims_row_commitments);
    append_canonical(&mut hasher, &input.eval_commitments);
    append_canonical(&mut hasher, generators);
    append_canonical(&mut hasher, eval_generators);
    append_r1cs(&mut hasher, r1cs);
    append_checkpoint(&mut hasher, initial);
    append_checkpoint(&mut hasher, final_transcript);
    for operation in operations {
        match operation {
            PoseidonOperation::Append { bytes, .. } => {
                hasher.update([0]);
                hasher.update((bytes.len() as u64).to_le_bytes());
                hasher.update(bytes);
            }
            PoseidonOperation::Challenge(value) => {
                hasher.update([1]);
                append_canonical(&mut hasher, value);
            }
        }
    }
    hasher.finalize().into()
}

fn compute_scalar_object_id(
    group_id: [u8; 32],
    operations: &[PoseidonOperation],
    r1cs: &VerifierR1CS<Fr>,
    initial: &PoseidonTranscript,
    final_transcript: &PoseidonTranscript,
    jolt_statement_id: [u8; 32],
) -> [u8; 32] {
    let mut hasher = Sha3_256::new();
    hasher.update(b"jolt-nova/stage17/blindfold-scalar-object/v1");
    hasher.update(group_id);
    hasher.update(jolt_statement_id);
    for operation in operations {
        match operation {
            PoseidonOperation::Append { chunks, bytes } => {
                hasher.update([0]);
                hasher.update((chunks.len() as u64).to_le_bytes());
                hasher.update((bytes.len() as u64).to_le_bytes());
                hasher.update(bytes);
                for chunk in chunks {
                    append_canonical(&mut hasher, chunk);
                }
            }
            PoseidonOperation::Challenge(value) => {
                hasher.update([1]);
                append_canonical(&mut hasher, value);
            }
        }
    }
    append_r1cs(&mut hasher, r1cs);
    append_checkpoint(&mut hasher, initial);
    append_checkpoint(&mut hasher, final_transcript);
    hasher.finalize().into()
}

/// One-step Nova circuit for the complete BlindFold scalar verifier.
#[derive(Clone)]
pub struct RecursiveBlindFoldVerifierCircuit {
    object_id: [u8; 32],
    deferred_group_id: [u8; 32],
    jolt_statement_id: [u8; 32],
    initial_transcript: PoseidonTranscript,
    final_transcript: PoseidonTranscript,
    verifier_r1cs: VerifierR1CS<Fr>,
    witness: BlindFoldScalarWitness,
}

impl RecursiveBlindFoldVerifierCircuit {
    pub fn statement(&self) -> RecursiveBlindFoldStatement {
        RecursiveBlindFoldStatement {
            object_id: self.object_id,
            deferred_group_id: self.deferred_group_id,
            initial_transcript_state: RecursiveJoltFieldElement {
                canonical_le_bytes: self.initial_transcript.state,
            },
            initial_transcript_round: self.initial_transcript.n_rounds as u64,
            final_transcript_state: RecursiveJoltFieldElement {
                canonical_le_bytes: self.final_transcript.state,
            },
            final_transcript_round: self.final_transcript.n_rounds as u64,
            shape_id: self.shape_id(),
            jolt_statement_id: self.jolt_statement_id,
        }
    }

    fn initial_z(&self) -> Result<Vec<NovaScalar>, SynthesisError> {
        self.statement()
            .initial_z()
            .map_err(SynthesisError::Unsatisfiable)
    }

    fn shape_id(&self) -> [u8; 32] {
        let mut hasher = Sha3_256::new();
        hasher.update(b"jolt-nova/stage17/blindfold-circuit-shape/v1");
        hasher.update((ZK_RECURSIVE_Z_ARITY as u64).to_le_bytes());
        for operation in &self.witness.operations {
            match operation {
                PoseidonOperation::Append { chunks, .. } => {
                    hasher.update([0]);
                    hasher.update((chunks.len() as u64).to_le_bytes());
                }
                PoseidonOperation::Challenge(_) => hasher.update([1]),
            }
        }
        hasher.update((self.witness.outer.rounds.len() as u64).to_le_bytes());
        hasher.update((self.witness.inner.rounds.len() as u64).to_le_bytes());
        hasher.update((self.witness.e_opening_row.len() as u64).to_le_bytes());
        hasher.update((self.witness.w_opening_row.len() as u64).to_le_bytes());
        for rows in [
            &self.witness.folded_eval_output_rows,
            &self.witness.folded_eval_blinding_rows,
        ] {
            hasher.update((rows.len() as u64).to_le_bytes());
            for row in rows {
                hasher.update((row.len() as u64).to_le_bytes());
            }
        }
        append_r1cs(&mut hasher, &self.verifier_r1cs);
        hasher.finalize().into()
    }

    pub fn setup_pinned(
        &self,
    ) -> Result<
        (
            RecursiveBlindFoldVerifierProverParameters,
            RecursiveBlindFoldVerifierVerificationKey,
        ),
        String,
    > {
        let public_params = nova_snark::nova::PublicParams::setup(
            self,
            &*nova_snark::traits::snark::default_ck_hint::<NovaPrimaryEngine>(),
            &*nova_snark::traits::snark::default_ck_hint::<NovaSecondaryEngine>(),
        )
        .map_err(|error| format!("BlindFold Nova setup failed: {error:?}"))?;
        let (prover_key, verifier_key) = ZkCompressedSnark::setup(&public_params)
            .map_err(|error| format!("BlindFold Spartan setup failed: {error:?}"))?;
        let shape_id = self.shape_id();
        Ok((
            RecursiveBlindFoldVerifierProverParameters {
                public_params,
                prover_key,
                shape_id,
            },
            RecursiveBlindFoldVerifierVerificationKey {
                verifier_key,
                shape_id,
            },
        ))
    }

    pub fn baseline(&self, proof: &RecursiveBlindFoldSpartanProof) -> RecursiveBlindFoldBaseline {
        RecursiveBlindFoldBaseline {
            transcript_operations: self.witness.operations.len(),
            transcript_challenges: self
                .witness
                .operations
                .iter()
                .filter(|operation| matches!(operation, PoseidonOperation::Challenge(_)))
                .count(),
            outer_sumcheck_rounds: self.witness.outer.rounds.len(),
            inner_sumcheck_rounds: self.witness.inner.rounds.len(),
            verifier_r1cs_constraints: self.verifier_r1cs.num_constraints,
            proof_bytes: proof.proof_bytes.len(),
        }
    }

    #[cfg(test)]
    pub(crate) fn test_constraints_are_satisfied(&self) -> Result<(), String> {
        use nova_snark::frontend::test_cs::TestConstraintSystem;
        let mut cs = TestConstraintSystem::<NovaScalar>::new();
        let z = self
            .initial_z()
            .map_err(|error| format!("initial state failed: {error:?}"))?
            .into_iter()
            .enumerate()
            .map(|(index, value)| {
                AllocatedNum::alloc(cs.namespace(|| format!("z {index}")), || Ok(value))
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| format!("public state allocation failed: {error:?}"))?;
        self.synthesize(&mut cs, &z)
            .map_err(|error| format!("BlindFold synthesis failed: {error:?}"))?;
        if cs.is_satisfied() {
            Ok(())
        } else {
            Err(format!("unsatisfied: {:?}", cs.which_is_unsatisfied()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        curve::Bn254Curve,
        subprotocols::blindfold::{
            BakedPublicInputs, BlindFoldProver, BlindFoldWitness, RelaxedR1CSInstance,
            RoundWitness, StageConfig, StageWitness, VerifierR1CSBuilder,
        },
    };

    fn fixture() -> RecursiveBlindFoldRelationArtifact<Bn254Curve> {
        let config = StageConfig::new(1, 3);
        let round = RoundWitness::new(
            vec![
                Fr::from(40u64),
                Fr::from(5u64),
                Fr::from(10u64),
                Fr::from(5u64),
            ],
            Fr::from(3u64),
        );
        let witness = BlindFoldWitness::new(Fr::from(100u64), vec![StageWitness::new(vec![round])]);
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
        let initial = PoseidonTranscript::new(b"stage17-blindfold-test");
        let mut prover_transcript = initial.clone();
        prover_transcript.append_label(b"BlindFold");
        let proof = BlindFoldProver::new(&generators, &r1cs, None).prove(
            &instance,
            &relaxed_witness,
            &z,
            &mut prover_transcript,
        );
        let input = BlindFoldVerifierInput {
            round_commitments: instance.round_commitments,
            output_claims_row_commitments: instance.output_claims_row_commitments,
            eval_commitments: instance.eval_commitments,
        };
        RecursiveBlindFoldRelationArtifact::from_verified_parts(
            proof, input, generators, r1cs, None, initial, [17u8; 32],
        )
        .unwrap()
    }

    #[test]
    fn stage17_internalizes_blindfold_scalar_relation_and_defers_real_group_equations() {
        let artifact = fixture();
        artifact.circuit.test_constraints_are_satisfied().unwrap();
        artifact.group_obligation.verify().unwrap();
        let statement = artifact.statement();
        assert_eq!(statement.jolt_statement_id, [17u8; 32]);
        assert_eq!(
            statement.deferred_group_id,
            artifact.group_obligation.obligation_id()
        );
    }

    #[test]
    fn stage17_rejects_scalar_and_group_tampering() {
        let mut scalar = fixture();
        scalar.circuit.witness.az_r += Fr::one();
        assert!(scalar.circuit.test_constraints_are_satisfied().is_err());

        let mut sumcheck = fixture();
        sumcheck.circuit.witness.outer.rounds[0].coefficients_except_linear[0]
            .canonical_le_bytes[0] ^= 1;
        assert!(sumcheck.circuit.test_constraints_are_satisfied().is_err());

        let mut transcript = fixture();
        let first_append = transcript
            .circuit
            .witness
            .operations
            .iter_mut()
            .find_map(|operation| match operation {
                PoseidonOperation::Append { chunks, .. } => Some(chunks),
                PoseidonOperation::Challenge(_) => None,
            })
            .unwrap();
        first_append[0] += Fr::one();
        assert!(transcript.circuit.test_constraints_are_satisfied().is_err());

        let mut group = fixture();
        group.group_obligation.proof.w_opening.combined_row[0] += Fr::one();
        assert!(group.group_obligation.verify().is_err());
    }

    #[cfg(feature = "prover")]
    #[test]
    fn stage17_real_nova_spartan_and_pinned_final_acceptance() {
        let artifact = fixture();
        let statement = artifact.statement();
        let expected_object = statement.object_id;
        let expected_jolt_statement = statement.jolt_statement_id;
        let (circuit, group_obligation) = artifact.into_parts();
        let (prover, verifier) = circuit.setup_pinned().unwrap();
        let recursive_proof = prover.prove(&circuit).unwrap();
        let baseline = circuit.baseline(&recursive_proof);
        assert!(baseline.proof_bytes > 0);
        let acceptance = RecursiveJoltZkRecursiveFinalAcceptance {
            recursive_proof,
            group_obligation,
        };
        acceptance
            .verify(
                &verifier,
                &statement,
                expected_object,
                expected_jolt_statement,
            )
            .unwrap();
        assert!(acceptance
            .verify(&verifier, &statement, [0xff; 32], expected_jolt_statement,)
            .is_err());
        assert!(acceptance
            .verify(&verifier, &statement, expected_object, [0xee; 32],)
            .is_err());

        let mut tampered_statement = statement;
        tampered_statement.final_transcript_round += 1;
        assert!(acceptance
            .verify(
                &verifier,
                &tampered_statement,
                expected_object,
                expected_jolt_statement,
            )
            .is_err());
    }
}

pub struct RecursiveBlindFoldVerifierProverParameters {
    public_params: nova_snark::nova::PublicParams<
        NovaPrimaryEngine,
        NovaSecondaryEngine,
        RecursiveBlindFoldVerifierCircuit,
    >,
    prover_key: ZkProverKey,
    shape_id: [u8; 32],
}

pub struct RecursiveBlindFoldVerifierVerificationKey {
    verifier_key: ZkVerificationKey,
    shape_id: [u8; 32],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecursiveBlindFoldSpartanProof {
    pub proof_bytes: Vec<u8>,
    pub public_output: Vec<[u8; 32]>,
}

impl RecursiveBlindFoldVerifierProverParameters {
    pub fn prove(
        &self,
        circuit: &RecursiveBlindFoldVerifierCircuit,
    ) -> Result<RecursiveBlindFoldSpartanProof, String> {
        if circuit.shape_id() != self.shape_id {
            return Err("BlindFold circuit does not match pinned setup".to_string());
        }
        let z0 = circuit.initial_z().map_err(|error| format!("{error:?}"))?;
        let mut recursive = ZkRecursiveSnark::new(&self.public_params, circuit, &z0)
            .map_err(|error| format!("BlindFold Nova initialization failed: {error:?}"))?;
        recursive
            .prove_step(&self.public_params, circuit)
            .map_err(|error| format!("BlindFold Nova proving failed: {error:?}"))?;
        let output = recursive
            .verify(&self.public_params, 1, &z0)
            .map_err(|error| format!("BlindFold Nova self-check failed: {error:?}"))?;
        if output.len() != ZK_RECURSIVE_Z_ARITY || output[6] != NovaScalar::one() {
            return Err("BlindFold Nova did not derive acceptance".to_string());
        }
        let compressed =
            ZkCompressedSnark::prove(&self.public_params, &self.prover_key, &recursive)
                .map_err(|error| format!("BlindFold Spartan proving failed: {error:?}"))?;
        Ok(RecursiveBlindFoldSpartanProof {
            proof_bytes: postcard::to_stdvec(&compressed)
                .map_err(|error| format!("BlindFold proof serialization failed: {error}"))?,
            public_output: output.into_iter().map(nova_scalar_to_storage).collect(),
        })
    }
}

impl RecursiveBlindFoldVerifierVerificationKey {
    pub fn verify(
        &self,
        statement: &RecursiveBlindFoldStatement,
        proof: &RecursiveBlindFoldSpartanProof,
    ) -> Result<(), String> {
        if statement.shape_id != self.shape_id {
            return Err("BlindFold statement/setup shape mismatch".to_string());
        }
        let compressed: ZkCompressedSnark = postcard::from_bytes(&proof.proof_bytes)
            .map_err(|error| format!("BlindFold proof deserialization failed: {error}"))?;
        let output = compressed
            .verify(&self.verifier_key, 1, &statement.initial_z()?)
            .map_err(|error| format!("BlindFold Spartan verification failed: {error:?}"))?;
        let stored = output
            .iter()
            .copied()
            .map(nova_scalar_to_storage)
            .collect::<Vec<_>>();
        if stored != proof.public_output
            || output.len() != ZK_RECURSIVE_Z_ARITY
            || output[6] != NovaScalar::one()
        {
            return Err("BlindFold Spartan public output mismatch".to_string());
        }
        Ok(())
    }
}

pub struct RecursiveJoltZkRecursiveFinalAcceptance<C: JoltCurve<F = Fr>> {
    pub recursive_proof: RecursiveBlindFoldSpartanProof,
    pub group_obligation: RecursiveBlindFoldGroupObligation<C>,
}

/// Complete Stage-17 acceptance envelope. The recursive scalar proof, the
/// BlindFold commitment-group equations, and Jolt's deferred PCS opening must
/// all verify. `expected_pcs_id` is supplied by the existing trace/folding
/// linkage statement, so a valid PCS proof from another execution cannot be
/// substituted here.
pub struct RecursiveJoltZkCompleteFinalAcceptance<
    C: JoltCurve<F = Fr>,
    PCS: CommitmentScheme<Field = Fr>,
    FS: Transcript,
> {
    pub blindfold: RecursiveJoltZkRecursiveFinalAcceptance<C>,
    pub deferred_pcs_opening: RecursiveDeferredPcsOpening<Fr, PCS, FS>,
}

impl<C, PCS, FS> RecursiveJoltZkCompleteFinalAcceptance<C, PCS, FS>
where
    C: JoltCurve<F = Fr>,
    PCS: CommitmentScheme<Field = Fr>,
    FS: Transcript,
{
    #[allow(clippy::too_many_arguments)]
    pub fn verify(
        &self,
        verification_key: &RecursiveBlindFoldVerifierVerificationKey,
        statement: &RecursiveBlindFoldStatement,
        expected_object_id: [u8; 32],
        expected_jolt_statement_id: [u8; 32],
        expected_pcs_id: [u8; 32],
    ) -> Result<(), String> {
        if self.deferred_pcs_opening.obligation_id() != expected_pcs_id {
            return Err("ZK deferred PCS linkage identifier mismatch".to_string());
        }
        self.blindfold.verify(
            verification_key,
            statement,
            expected_object_id,
            expected_jolt_statement_id,
        )?;
        self.deferred_pcs_opening
            .verify()
            .map_err(|error| format!("ZK deferred PCS verification failed: {error}"))
    }
}

impl<C: JoltCurve<F = Fr>> RecursiveJoltZkRecursiveFinalAcceptance<C> {
    pub fn verify(
        &self,
        verification_key: &RecursiveBlindFoldVerifierVerificationKey,
        statement: &RecursiveBlindFoldStatement,
        expected_object_id: [u8; 32],
        expected_jolt_statement_id: [u8; 32],
    ) -> Result<(), String> {
        if statement.object_id != expected_object_id {
            return Err("BlindFold recursive object identifier mismatch".to_string());
        }
        if statement.jolt_statement_id != expected_jolt_statement_id {
            return Err("clear/ZK Jolt statement identifier mismatch".to_string());
        }
        if statement.deferred_group_id != self.group_obligation.obligation_id() {
            return Err("BlindFold deferred group identifier mismatch".to_string());
        }
        let (final_state, final_round) = self.group_obligation.final_transcript_checkpoint();
        if statement.final_transcript_state != final_state
            || statement.final_transcript_round != final_round
        {
            return Err("BlindFold deferred transcript binding mismatch".to_string());
        }
        verification_key.verify(statement, &self.recursive_proof)?;
        self.group_obligation.verify()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecursiveBlindFoldBaseline {
    pub transcript_operations: usize,
    pub transcript_challenges: usize,
    pub outer_sumcheck_rounds: usize,
    pub inner_sumcheck_rounds: usize,
    pub verifier_r1cs_constraints: usize,
    pub proof_bytes: usize,
}

fn alloc_fr<CS: ConstraintSystem<NovaScalar>>(
    cs: CS,
    value: Fr,
) -> Result<AllocatedNum<NovaScalar>, SynthesisError> {
    AllocatedNum::alloc(cs, || ark_bn254_scalar_as_nova_scalar(&value))
}

fn mul<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    left: &AllocatedNum<NovaScalar>,
    right: &AllocatedNum<NovaScalar>,
) -> Result<AllocatedNum<NovaScalar>, SynthesisError> {
    let product = AllocatedNum::alloc(cs.namespace(|| "product"), || {
        Ok(left.get_value().ok_or(SynthesisError::AssignmentMissing)?
            * right.get_value().ok_or(SynthesisError::AssignmentMissing)?)
    })?;
    cs.enforce(
        || "multiplication",
        |lc| lc + left.get_variable(),
        |lc| lc + right.get_variable(),
        |lc| lc + product.get_variable(),
    );
    Ok(product)
}

fn linear_value<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    terms: &[(NovaScalar, &AllocatedNum<NovaScalar>)],
    constant: NovaScalar,
) -> Result<AllocatedNum<NovaScalar>, SynthesisError> {
    let allocated = AllocatedNum::alloc(cs.namespace(|| "linear value"), || {
        terms
            .iter()
            .try_fold(constant, |accumulator, (coefficient, term)| {
                Ok::<_, SynthesisError>(
                    accumulator
                        + *coefficient
                            * term.get_value().ok_or(SynthesisError::AssignmentMissing)?,
                )
            })
    })?;
    let expression = terms.iter().fold(
        LinearCombination::<NovaScalar>::zero() + (constant, CS::one()),
        |lc, (coefficient, term)| lc + (*coefficient, term.get_variable()),
    );
    cs.enforce(
        || "linear combination",
        |_| expression - allocated.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc,
    );
    Ok(allocated)
}

fn eq_table<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    point: &[AllocatedNum<NovaScalar>],
) -> Result<Vec<AllocatedNum<NovaScalar>>, SynthesisError> {
    let one = alloc_nova_constant(cs.namespace(|| "eq one"), NovaScalar::one())?;
    let mut table = vec![one];
    for (index, coordinate) in point.iter().enumerate() {
        let one_minus = linear_value(
            cs.namespace(|| format!("one minus coordinate {index}")),
            &[(NovaScalar::one(), coordinate)],
            NovaScalar::one().neg(),
        )?;
        // `linear_value` above computes coordinate - 1; negate it explicitly.
        let one_minus = linear_value(
            cs.namespace(|| format!("correct one minus coordinate {index}")),
            &[(NovaScalar::one().neg(), &one_minus)],
            NovaScalar::zero(),
        )?;
        let mut next = Vec::with_capacity(table.len() * 2);
        for (row, prefix) in table.iter().enumerate() {
            next.push(mul(
                cs.namespace(|| format!("eq {index} prefix {row} bit 0")),
                prefix,
                &one_minus,
            )?);
            next.push(mul(
                cs.namespace(|| format!("eq {index} prefix {row} bit 1")),
                prefix,
                coordinate,
            )?);
        }
        table = next;
    }
    Ok(table)
}

fn bind_equal<CS: ConstraintSystem<NovaScalar>>(
    cs: &mut CS,
    name: impl FnOnce() -> String,
    left: &AllocatedNum<NovaScalar>,
    right: &AllocatedNum<NovaScalar>,
) {
    cs.enforce(
        name,
        |lc| lc + left.get_variable() - right.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc,
    );
}

impl StepCircuit<NovaScalar> for RecursiveBlindFoldVerifierCircuit {
    fn arity(&self) -> usize {
        ZK_RECURSIVE_Z_ARITY
    }

    fn synthesize<CS: ConstraintSystem<NovaScalar>>(
        &self,
        cs: &mut CS,
        z: &[AllocatedNum<NovaScalar>],
    ) -> Result<Vec<AllocatedNum<NovaScalar>>, SynthesisError> {
        if z.len() != ZK_RECURSIVE_Z_ARITY {
            return Err(SynthesisError::AssignmentMissing);
        }
        for (index, expected) in self.initial_z()?.into_iter().enumerate() {
            cs.enforce(
                || format!("bind BlindFold public input {index}"),
                |lc| lc + z[index].get_variable() - (expected, CS::one()),
                |lc| lc + CS::one(),
                |lc| lc,
            );
        }

        let mut transcript = AllocatedRecursivePoseidonTranscriptState {
            state: z[2].clone(),
            n_rounds: z[3].clone(),
        };
        let zero = alloc_nova_constant(cs.namespace(|| "transcript zero"), NovaScalar::zero())?;
        let mut transcript_challenges = Vec::new();
        for (operation_index, operation) in self.witness.operations.iter().enumerate() {
            let mut operation_cs =
                cs.namespace(|| format!("transcript operation {operation_index}"));
            match operation {
                PoseidonOperation::Append { chunks, .. } => {
                    let mut current_state = transcript.state.clone();
                    for (chunk_index, chunk) in chunks.iter().enumerate() {
                        let data = alloc_fr(
                            operation_cs.namespace(|| format!("append chunk {chunk_index}")),
                            *chunk,
                        )?;
                        let round = if chunk_index == 0 {
                            &transcript.n_rounds
                        } else {
                            &zero
                        };
                        current_state = synthesize_recursive_poseidon_hash(
                            operation_cs.namespace(|| format!("append hash {chunk_index}")),
                            &current_state,
                            round,
                            &data,
                        )?;
                    }
                    let next_round = linear_value(
                        operation_cs.namespace(|| "increment append round"),
                        &[(NovaScalar::one(), &transcript.n_rounds)],
                        NovaScalar::one(),
                    )?;
                    transcript = AllocatedRecursivePoseidonTranscriptState {
                        state: current_state,
                        n_rounds: next_round,
                    };
                }
                PoseidonOperation::Challenge(expected) => {
                    transcript = synthesize_recursive_poseidon_transcript_transition(
                        operation_cs.namespace(|| "derive challenge"),
                        &transcript,
                        &zero,
                    )?;
                    let expected =
                        alloc_fr(operation_cs.namespace(|| "recorded challenge"), *expected)?;
                    bind_equal(
                        &mut operation_cs,
                        || "challenge equals transcript state".to_string(),
                        &transcript.state,
                        &expected,
                    );
                    transcript_challenges.push(expected);
                }
            }
        }
        bind_equal(
            cs,
            || "final transcript state".to_string(),
            &transcript.state,
            &z[4],
        );
        bind_equal(
            cs,
            || "final transcript round".to_string(),
            &transcript.n_rounds,
            &z[5],
        );

        let outer_rounds = self.witness.outer.rounds.len();
        let inner_rounds = self.witness.inner.rounds.len();
        let expected_challenges = 1 + outer_rounds + outer_rounds + 3 + inner_rounds;
        if transcript_challenges.len() != expected_challenges {
            return Err(SynthesisError::Unsatisfiable(
                "BlindFold transcript challenge layout mismatch".to_string(),
            ));
        }
        let folding = transcript_challenges[0].clone();
        let tau = transcript_challenges[1..1 + outer_rounds].to_vec();
        let outer_fs = transcript_challenges[1 + outer_rounds..1 + 2 * outer_rounds].to_vec();
        let scalar_start = 1 + 2 * outer_rounds;
        let ra = transcript_challenges[scalar_start].clone();
        let rb = transcript_challenges[scalar_start + 1].clone();
        let rc = transcript_challenges[scalar_start + 2].clone();
        let inner_fs = transcript_challenges[scalar_start + 3..].to_vec();

        let outer = synthesize_recursive_clear_sumcheck_stage(
            cs.namespace(|| "BlindFold outer Spartan sumcheck"),
            &self.witness.outer,
            0,
            outer_rounds,
            SPARTAN_DEGREE_BOUND,
        )?;
        let inner = synthesize_recursive_clear_sumcheck_stage(
            cs.namespace(|| "BlindFold inner Spartan sumcheck"),
            &self.witness.inner,
            1,
            inner_rounds,
            INNER_SUMCHECK_DEGREE_BOUND,
        )?;
        for (index, (derived, claimed)) in outer_fs.iter().zip(&outer.challenges).enumerate() {
            bind_equal(cs, || format!("outer challenge {index}"), derived, claimed);
        }
        for (index, (derived, claimed)) in inner_fs.iter().zip(&inner.challenges).enumerate() {
            bind_equal(cs, || format!("inner challenge {index}"), derived, claimed);
        }

        let random_u = alloc_fr(cs.namespace(|| "random relaxed u"), self.witness.random_u)?;
        let r_times_random_u = mul(cs.namespace(|| "fold u product"), &folding, &random_u)?;
        let folded_u = linear_value(
            cs.namespace(|| "folded relaxed u"),
            &[(NovaScalar::one(), &r_times_random_u)],
            NovaScalar::one(),
        )?;

        let az = alloc_fr(cs.namespace(|| "Az(r)"), self.witness.az_r)?;
        let bz = alloc_fr(cs.namespace(|| "Bz(r)"), self.witness.bz_r)?;
        let cz = alloc_fr(cs.namespace(|| "Cz(r)"), self.witness.cz_r)?;
        let e_row = self
            .witness
            .e_opening_row
            .iter()
            .enumerate()
            .map(|(index, value)| alloc_fr(cs.namespace(|| format!("E opening {index}")), *value))
            .collect::<Result<Vec<_>, _>>()?;
        let w_row = self
            .witness
            .w_opening_row
            .iter()
            .enumerate()
            .map(|(index, value)| alloc_fr(cs.namespace(|| format!("W opening {index}")), *value))
            .collect::<Result<Vec<_>, _>>()?;

        let (r_e, _) = self
            .verifier_r1cs
            .hyrax
            .e_grid(self.verifier_r1cs.num_constraints);
        let e_eq = eq_table(
            cs.namespace(|| "E opening column equality table"),
            &outer_fs[r_e.log_2()..],
        )?;
        if e_row.len() != e_eq.len() {
            return Err(SynthesisError::Unsatisfiable(
                "E opening row has wrong width".to_string(),
            ));
        }
        let e_terms = e_row
            .iter()
            .zip(&e_eq)
            .enumerate()
            .map(|(index, (value, selector))| {
                mul(
                    cs.namespace(|| format!("E opening term {index}")),
                    value,
                    selector,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let e_r = linear_value(
            cs.namespace(|| "E(r)"),
            &e_terms
                .iter()
                .map(|term| (NovaScalar::one(), term))
                .collect::<Vec<_>>(),
            NovaScalar::zero(),
        )?;

        let tau_eq_terms = tau
            .iter()
            .zip(&outer_fs)
            .enumerate()
            .map(|(index, (tau_i, r_i))| {
                let product = mul(cs.namespace(|| format!("tau*r {index}")), tau_i, r_i)?;
                let one_minus_tau = linear_value(
                    cs.namespace(|| format!("1-tau {index}")),
                    &[(NovaScalar::one().neg(), tau_i)],
                    NovaScalar::one(),
                )?;
                let one_minus_r = linear_value(
                    cs.namespace(|| format!("1-r {index}")),
                    &[(NovaScalar::one().neg(), r_i)],
                    NovaScalar::one(),
                )?;
                let complement = mul(
                    cs.namespace(|| format!("(1-tau)*(1-r) {index}")),
                    &one_minus_tau,
                    &one_minus_r,
                )?;
                linear_value(
                    cs.namespace(|| format!("eq coordinate {index}")),
                    &[
                        (NovaScalar::one(), &product),
                        (NovaScalar::one(), &complement),
                    ],
                    NovaScalar::zero(),
                )
            })
            .collect::<Result<Vec<_>, SynthesisError>>()?;
        let mut eq_tau_r =
            alloc_nova_constant(cs.namespace(|| "eq tau initial"), NovaScalar::one())?;
        for (index, term) in tau_eq_terms.iter().enumerate() {
            eq_tau_r = mul(
                cs.namespace(|| format!("eq tau product {index}")),
                &eq_tau_r,
                term,
            )?;
        }
        let az_bz = mul(cs.namespace(|| "Az times Bz"), &az, &bz)?;
        let u_cz = mul(cs.namespace(|| "u times Cz"), &folded_u, &cz)?;
        let residual = linear_value(
            cs.namespace(|| "relaxed R1CS residual"),
            &[
                (NovaScalar::one(), &az_bz),
                (NovaScalar::one().neg(), &u_cz),
                (NovaScalar::one().neg(), &e_r),
            ],
            NovaScalar::zero(),
        )?;
        let expected_outer = mul(
            cs.namespace(|| "outer final relation"),
            &eq_tau_r,
            &residual,
        )?;
        bind_equal(
            cs,
            || "outer Spartan final claim".to_string(),
            &outer.final_claim,
            &expected_outer,
        );

        let eq_outer = eq_table(cs.namespace(|| "outer equality table"), &outer_fs)?;
        let matrix_public = |matrix: &crate::subprotocols::blindfold::SparseR1CSMatrix<Fr>,
                             name: &'static str,
                             cs: &mut CS|
         -> Result<AllocatedNum<NovaScalar>, SynthesisError> {
            let terms = matrix
                .entries
                .iter()
                .filter(|(_, column, _)| *column == 0)
                .map(|(row, _, coefficient)| {
                    Ok((
                        ark_bn254_scalar_as_nova_scalar(coefficient)?,
                        &eq_outer[*row],
                    ))
                })
                .collect::<Result<Vec<_>, SynthesisError>>()?;
            let projected = linear_value(cs.namespace(|| name), &terms, NovaScalar::zero())?;
            mul(
                cs.namespace(|| format!("{name} times u")),
                &projected,
                &folded_u,
            )
        };
        let pub_a = matrix_public(&self.verifier_r1cs.a, "public A projection", cs)?;
        let pub_b = matrix_public(&self.verifier_r1cs.b, "public B projection", cs)?;
        let pub_c = matrix_public(&self.verifier_r1cs.c, "public C projection", cs)?;
        let az_private = linear_value(
            cs.namespace(|| "private Az"),
            &[(NovaScalar::one(), &az), (NovaScalar::one().neg(), &pub_a)],
            NovaScalar::zero(),
        )?;
        let bz_private = linear_value(
            cs.namespace(|| "private Bz"),
            &[(NovaScalar::one(), &bz), (NovaScalar::one().neg(), &pub_b)],
            NovaScalar::zero(),
        )?;
        let cz_private = linear_value(
            cs.namespace(|| "private Cz"),
            &[(NovaScalar::one(), &cz), (NovaScalar::one().neg(), &pub_c)],
            NovaScalar::zero(),
        )?;
        let ra_a = mul(cs.namespace(|| "ra private A"), &ra, &az_private)?;
        let rb_b = mul(cs.namespace(|| "rb private B"), &rb, &bz_private)?;
        let rc_c = mul(cs.namespace(|| "rc private C"), &rc, &cz_private)?;
        let expected_inner_initial = linear_value(
            cs.namespace(|| "inner initial relation"),
            &[
                (NovaScalar::one(), &ra_a),
                (NovaScalar::one(), &rb_b),
                (NovaScalar::one(), &rc_c),
            ],
            NovaScalar::zero(),
        )?;
        bind_equal(
            cs,
            || "inner Spartan initial claim".to_string(),
            &inner.initial_claim,
            &expected_inner_initial,
        );

        let w_eq = eq_table(
            cs.namespace(|| "W opening column equality table"),
            &inner_fs[self.verifier_r1cs.hyrax.log_R_prime()..],
        )?;
        if w_row.len() != w_eq.len() {
            return Err(SynthesisError::Unsatisfiable(
                "W opening row has wrong width".to_string(),
            ));
        }
        let w_terms = w_row
            .iter()
            .zip(&w_eq)
            .enumerate()
            .map(|(index, (value, selector))| {
                mul(
                    cs.namespace(|| format!("W opening term {index}")),
                    value,
                    selector,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let w_ry = linear_value(
            cs.namespace(|| "W(ry)"),
            &w_terms
                .iter()
                .map(|term| (NovaScalar::one(), term))
                .collect::<Vec<_>>(),
            NovaScalar::zero(),
        )?;
        let eq_inner = eq_table(cs.namespace(|| "inner equality table"), &inner_fs)?;
        let matrix_bilinear = |matrix: &crate::subprotocols::blindfold::SparseR1CSMatrix<Fr>,
                               name: &'static str,
                               cs: &mut CS|
         -> Result<AllocatedNum<NovaScalar>, SynthesisError> {
            let mut products = Vec::new();
            let mut coefficients = Vec::new();
            for (entry, (row, column, coefficient)) in matrix.entries.iter().enumerate() {
                if *column >= 1 && *column - 1 < eq_inner.len() && *row < eq_outer.len() {
                    products.push(mul(
                        cs.namespace(|| format!("{name} selector product {entry}")),
                        &eq_outer[*row],
                        &eq_inner[*column - 1],
                    )?);
                    coefficients.push(ark_bn254_scalar_as_nova_scalar(coefficient)?);
                }
            }
            let terms = products
                .iter()
                .zip(coefficients)
                .map(|(product, coefficient)| (coefficient, product))
                .collect::<Vec<_>>();
            linear_value(cs.namespace(|| name), &terms, NovaScalar::zero())
        };
        let a_eval = matrix_bilinear(&self.verifier_r1cs.a, "A bilinear", cs)?;
        let b_eval = matrix_bilinear(&self.verifier_r1cs.b, "B bilinear", cs)?;
        let c_eval = matrix_bilinear(&self.verifier_r1cs.c, "C bilinear", cs)?;
        let ra_eval = mul(cs.namespace(|| "ra A bilinear"), &ra, &a_eval)?;
        let rb_eval = mul(cs.namespace(|| "rb B bilinear"), &rb, &b_eval)?;
        let rc_eval = mul(cs.namespace(|| "rc C bilinear"), &rc, &c_eval)?;
        let l_w = linear_value(
            cs.namespace(|| "L_W(ry)"),
            &[
                (NovaScalar::one(), &ra_eval),
                (NovaScalar::one(), &rb_eval),
                (NovaScalar::one(), &rc_eval),
            ],
            NovaScalar::zero(),
        )?;
        let expected_inner_final = mul(cs.namespace(|| "inner final relation"), &l_w, &w_ry)?;
        bind_equal(
            cs,
            || "inner Spartan final claim".to_string(),
            &inner.final_claim,
            &expected_inner_final,
        );

        // Scalar half of the final evaluation-variable bindings. Group
        // commitment equality is checked by RecursiveBlindFoldGroupObligation.
        for (kind, rows, values, variables) in [
            (
                "output",
                &self.witness.folded_eval_output_rows,
                &self.witness.folded_eval_outputs,
                &self.verifier_r1cs.extra_output_vars,
            ),
            (
                "blinding",
                &self.witness.folded_eval_blinding_rows,
                &self.witness.folded_eval_blindings,
                &self.verifier_r1cs.extra_blinding_vars,
            ),
        ] {
            if rows.len() != values.len() || rows.len() != variables.len() {
                return Err(SynthesisError::Unsatisfiable(format!(
                    "folded evaluation {kind} dimensions mismatch"
                )));
            }
            for (index, ((row, value), variable)) in
                rows.iter().zip(values).zip(variables).enumerate()
            {
                if row.len() != self.verifier_r1cs.hyrax.C || variable.index() == 0 {
                    return Err(SynthesisError::Unsatisfiable(format!(
                        "folded evaluation {kind} row shape mismatch"
                    )));
                }
                let column = (variable.index() - 1) % self.verifier_r1cs.hyrax.C;
                let expected = alloc_fr(
                    cs.namespace(|| format!("folded {kind} expected {index}")),
                    *value,
                )?;
                for (slot, scalar) in row.iter().enumerate() {
                    let allocated = alloc_fr(
                        cs.namespace(|| format!("folded {kind} row {index} slot {slot}")),
                        *scalar,
                    )?;
                    if slot == column {
                        bind_equal(
                            cs,
                            || format!("folded {kind} selected value {index}"),
                            &allocated,
                            &expected,
                        );
                    } else {
                        cs.enforce(
                            || format!("folded {kind} sparse zero {index} slot {slot}"),
                            |lc| lc + allocated.get_variable(),
                            |lc| lc + CS::one(),
                            |lc| lc,
                        );
                    }
                }
            }
        }

        let accepted =
            alloc_nova_constant(cs.namespace(|| "BlindFold accepted"), NovaScalar::one())?;
        Ok(vec![
            z[0].clone(),
            z[1].clone(),
            z[2].clone(),
            z[3].clone(),
            z[4].clone(),
            z[5].clone(),
            accepted,
            z[7].clone(),
        ])
    }
}
