//! Unified Nova step for the Stage-14 Jolt verifier relation.
//!
//! This circuit joins exact Poseidon Fiat--Shamir transitions, clear-sumcheck
//! arithmetic, and the verifier R1CS containing Lasso/register/RAM/CPU/PCS
//! endpoint relations.

use ark_serialize::CanonicalSerialize;
use nova_snark::{
    frontend::{num::AllocatedNum, ConstraintSystem, SynthesisError},
    traits::circuit::StepCircuit,
};
use sha3::{Digest, Sha3_256};

use crate::{
    field::JoltField,
    poly::commitment::commitment_scheme::CommitmentScheme,
    subprotocols::blindfold::VerifierR1CS,
    transcripts::{PoseidonTranscript, Transcript},
};

use super::recursive_relations::{
    bind_recursive_sumchecks_to_verifier_r1cs, synthesize_recursive_checkpoint_capsule,
    synthesize_recursive_clear_sumcheck_stage, synthesize_recursive_clear_sumcheck_transcript,
    synthesize_recursive_jolt_verifier_r1cs, AllocatedRecursivePoseidonTranscriptState,
};
use super::{
    nova_hash_bytes_to_scalar, nova_scalar_to_storage, NovaPrimaryEngine, NovaPrimarySpartanSnark,
    NovaScalar, NovaSecondaryEngine, NovaSecondarySpartanSnark,
    RecursiveClearSumcheckStageArtifact, RecursiveClearSumcheckStageWitness,
    RecursiveDeferredPcsOpening, RecursiveJoltFieldElement, RecursiveJoltVerifierRelationArtifact,
};
#[cfg(test)]
use crate::subprotocols::blindfold::BlindFoldWitness;

const RECURSIVE_VERIFIER_Z_ARITY: usize = 8;

type RecursiveVerifierNovaSnark = nova_snark::nova::RecursiveSNARK<
    NovaPrimaryEngine,
    NovaSecondaryEngine,
    RecursiveJoltVerifierCircuit,
>;
type RecursiveVerifierCompressedSnark = nova_snark::nova::CompressedSNARK<
    NovaPrimaryEngine,
    NovaSecondaryEngine,
    RecursiveJoltVerifierCircuit,
    NovaPrimarySpartanSnark,
    NovaSecondarySpartanSnark,
>;
type RecursiveVerifierProverKey = nova_snark::nova::ProverKey<
    NovaPrimaryEngine,
    NovaSecondaryEngine,
    RecursiveJoltVerifierCircuit,
    NovaPrimarySpartanSnark,
    NovaSecondarySpartanSnark,
>;
type RecursiveVerifierNovaVerificationKey = nova_snark::nova::VerifierKey<
    NovaPrimaryEngine,
    NovaSecondaryEngine,
    RecursiveJoltVerifierCircuit,
    NovaPrimarySpartanSnark,
    NovaSecondarySpartanSnark,
>;

/// Public statement accepted by the Stage-15 recursive Jolt verifier.
///
/// Unlike the old API, verification no longer receives a witness-bearing
/// circuit.  The verifier is given only this statement, a pinned verification
/// key, and the proof.  `shape_id` binds the exact verifier R1CS (including its
/// baked coefficients) that was used during setup.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecursiveJoltVerifierStatement {
    pub object_id: [u8; 32],
    pub deferred_pcs_id: [u8; 32],
    pub initial_transcript_state: RecursiveJoltFieldElement,
    pub initial_transcript_round: u64,
    pub transcript_checkpoint_root: RecursiveJoltFieldElement,
    pub shape_id: [u8; 32],
}

impl RecursiveJoltVerifierStatement {
    fn id_scalar(domain: &'static str, id: &[u8; 32]) -> NovaScalar {
        nova_hash_bytes_to_scalar("recursive-jolt-verifier", domain, id)
    }

    fn initial_z(&self) -> Result<Vec<NovaScalar>, String> {
        let transcript_state = Option::from(NovaScalar::from_bytes(
            &self.initial_transcript_state.canonical_le_bytes,
        ))
        .ok_or_else(|| {
            "initial recursive transcript state is not canonical BN254::Fr".to_string()
        })?;
        Ok(vec![
            Self::id_scalar("object-id", &self.object_id),
            Self::id_scalar("deferred-pcs-id", &self.deferred_pcs_id),
            transcript_state,
            NovaScalar::from(self.initial_transcript_round),
            NovaScalar::zero(),
            NovaScalar::zero(),
            NovaScalar::zero(),
            Option::from(NovaScalar::from_bytes(
                &self.transcript_checkpoint_root.canonical_le_bytes,
            ))
            .ok_or_else(|| {
                "recursive transcript checkpoint root is not canonical BN254::Fr".to_string()
            })?,
        ])
    }
}

/// Setup material retained by the prover and reused for every statement with
/// the same verifier shape.
pub struct RecursiveJoltVerifierProverParameters {
    public_params: nova_snark::nova::PublicParams<
        NovaPrimaryEngine,
        NovaSecondaryEngine,
        RecursiveJoltVerifierCircuit,
    >,
    prover_key: RecursiveVerifierProverKey,
    shape_id: [u8; 32],
}

/// Pinned Stage-15 verifier key.  It contains no witness-bearing circuit and
/// can be distributed independently from the prover parameters.
pub struct RecursiveJoltVerifierVerificationKey {
    verifier_key: RecursiveVerifierNovaVerificationKey,
    shape_id: [u8; 32],
}

/// Serialized Spartan-compressed Nova proof for the unified verifier circuit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecursiveJoltVerifierSpartanProof {
    pub proof_bytes: Vec<u8>,
    pub public_output: Vec<[u8; 32]>,
}

/// Stable structural/per-proof baseline for the Stage-14 verifier path.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecursiveJoltVerifierBaseline {
    pub sumcheck_stages: usize,
    pub sumcheck_rounds: usize,
    pub poseidon_transcript_transitions: usize,
    pub verifier_r1cs_variables: usize,
    pub verifier_r1cs_constraints: usize,
    pub spartan_proof_bytes: usize,
    pub public_output_bytes: usize,
}

/// Final Stage-14 envelope. Acceptance requires both cryptographic layers:
/// the Spartan-compressed Nova verifier relation and the deferred PCS pairing.
pub struct RecursiveJoltFinalAcceptance<F, PCS, FS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    FS: Transcript,
{
    pub recursive_verifier_proof: RecursiveJoltVerifierSpartanProof,
    pub deferred_pcs_opening: RecursiveDeferredPcsOpening<F, PCS, FS>,
}

impl<F, PCS, FS> RecursiveJoltFinalAcceptance<F, PCS, FS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    FS: Transcript,
{
    /// Production Stage-15 acceptance path.  The verifier consumes no circuit
    /// witness and never regenerates setup material.
    pub fn verify_pinned(
        &self,
        verification_key: &RecursiveJoltVerifierVerificationKey,
        statement: &RecursiveJoltVerifierStatement,
        expected_object_id: [u8; 32],
    ) -> Result<(), String> {
        if statement.object_id != expected_object_id {
            return Err("recursive verifier object identifier mismatch".to_string());
        }
        if statement.deferred_pcs_id != self.deferred_pcs_opening.obligation_id() {
            return Err("deferred PCS obligation identifier mismatch".to_string());
        }
        verification_key.verify(statement, &self.recursive_verifier_proof)?;
        self.deferred_pcs_opening
            .verify()
            .map_err(|error| format!("deferred PCS verification failed: {error}"))
    }

    /// Legacy Stage-14 compatibility path. New callers should use
    /// [`Self::verify_pinned`] so verification never receives a witness-bearing
    /// circuit or performs per-proof setup.
    pub fn verify(
        &self,
        circuit: &RecursiveJoltVerifierCircuit,
        expected_object_id: [u8; 32],
    ) -> Result<(), String> {
        if circuit.object_id != expected_object_id {
            return Err("recursive verifier object identifier mismatch".to_string());
        }
        if circuit.deferred_pcs_id != self.deferred_pcs_opening.obligation_id() {
            return Err("deferred PCS obligation identifier mismatch".to_string());
        }
        circuit.verify_spartan(&self.recursive_verifier_proof)?;
        self.deferred_pcs_opening
            .verify()
            .map_err(|error| format!("deferred PCS verification failed: {error}"))
    }
}

/// One-step Nova circuit for a complete Jolt verifier relation.
///
/// The object and deferred-PCS IDs are public continuity commitments only.
/// Acceptance (`z[6] = 1`) is produced only after all arithmetic and verifier
/// R1CS constraints have been synthesized.
#[derive(Clone)]
pub struct RecursiveJoltVerifierCircuit {
    object_id: [u8; 32],
    deferred_pcs_id: [u8; 32],
    initial_transcript_state: RecursiveJoltFieldElement,
    initial_transcript_round: u64,
    stage_transcript_checkpoints: Vec<(RecursiveJoltFieldElement, u64)>,
    sumcheck_stages: Vec<RecursiveClearSumcheckStageWitness>,
    verifier_r1cs: VerifierR1CS<ark_bn254::Fr>,
    verifier_witness: Vec<ark_bn254::Fr>,
}

impl RecursiveJoltVerifierCircuit {
    fn checkpoint_root(
        checkpoints: &[(RecursiveJoltFieldElement, u64)],
    ) -> RecursiveJoltFieldElement {
        use ark_ff::PrimeField;

        let mut transcript = PoseidonTranscript::new(b"stage15-checkpoints");
        for (state, round) in checkpoints {
            let state_field = ark_bn254::Fr::from_le_bytes_mod_order(&state.canonical_le_bytes);
            transcript.append_scalars(b"checkpoint", &[state_field, ark_bn254::Fr::from(*round)]);
        }
        RecursiveJoltFieldElement {
            canonical_le_bytes: transcript.state,
        }
    }

    #[cfg(test)]
    pub(crate) fn new(
        object_id: [u8; 32],
        deferred_pcs_id: [u8; 32],
        initial_transcript_state: RecursiveJoltFieldElement,
        initial_transcript_round: u64,
        sumcheck_stages: Vec<RecursiveClearSumcheckStageWitness>,
        verifier_r1cs: VerifierR1CS<ark_bn254::Fr>,
        verifier_witness: Vec<ark_bn254::Fr>,
    ) -> Result<Self, &'static str> {
        if sumcheck_stages.is_empty() {
            return Err("recursive Jolt verifier requires at least one sumcheck stage");
        }
        if verifier_witness.len() != verifier_r1cs.num_vars {
            return Err("recursive Jolt verifier witness/R1CS dimension mismatch");
        }
        if sumcheck_stages.len() != 1 {
            return Err(
                "multi-stage recursive Jolt verifier requires per-stage transcript artifacts",
            );
        }
        Ok(Self {
            object_id,
            deferred_pcs_id,
            initial_transcript_state,
            initial_transcript_round,
            stage_transcript_checkpoints: vec![(
                initial_transcript_state,
                initial_transcript_round,
            )],
            sumcheck_stages,
            verifier_r1cs,
            verifier_witness,
        })
    }

    /// Production adapter from the complete artifact emitted by
    /// `JoltVerifier::verify_with_recursive_relation_artifact`. There is no
    /// public entry point for a caller-supplied verifier witness.
    pub fn from_verified_relation(
        object_id: [u8; 32],
        deferred_pcs_id: [u8; 32],
        artifact: RecursiveJoltVerifierRelationArtifact,
    ) -> Result<Self, &'static str> {
        let (artifacts, verifier_r1cs, verifier_witness) = artifact.into_circuit_parts();
        verifier_r1cs
            .check_satisfaction(&verifier_witness)
            .map_err(|_| "recursive Jolt verifier artifact is not satisfied")?;
        Self::from_relation_parts(
            object_id,
            deferred_pcs_id,
            artifacts,
            verifier_r1cs,
            verifier_witness,
        )
    }

    fn from_relation_parts(
        object_id: [u8; 32],
        deferred_pcs_id: [u8; 32],
        artifacts: Vec<RecursiveClearSumcheckStageArtifact>,
        verifier_r1cs: VerifierR1CS<ark_bn254::Fr>,
        verifier_witness: Vec<ark_bn254::Fr>,
    ) -> Result<Self, &'static str> {
        if artifacts.is_empty() {
            return Err("recursive Jolt verifier requires at least one verified artifact");
        }
        if verifier_witness.len() != verifier_r1cs.num_vars {
            return Err("recursive Jolt verifier witness/R1CS dimension mismatch");
        }
        for (index, artifact) in artifacts.iter().enumerate() {
            artifact.witness.validate_shape(
                index,
                artifact.witness.rounds.len(),
                artifact.witness.degree_bound,
            )?;
        }
        let checkpoints = artifacts
            .iter()
            .map(|artifact| {
                (
                    artifact.transcript_state_before,
                    artifact.transcript_round_before,
                )
            })
            .collect::<Vec<_>>();
        let first = checkpoints[0];
        Ok(Self {
            object_id,
            deferred_pcs_id,
            initial_transcript_state: first.0,
            initial_transcript_round: first.1,
            stage_transcript_checkpoints: checkpoints,
            sumcheck_stages: artifacts
                .into_iter()
                .map(|artifact| artifact.witness)
                .collect(),
            verifier_r1cs,
            verifier_witness,
        })
    }

    #[cfg(test)]
    pub(crate) fn from_verified_artifacts(
        object_id: [u8; 32],
        deferred_pcs_id: [u8; 32],
        artifacts: Vec<RecursiveClearSumcheckStageArtifact>,
        verifier_r1cs: VerifierR1CS<ark_bn254::Fr>,
        verifier_witness: Vec<ark_bn254::Fr>,
    ) -> Result<Self, &'static str> {
        Self::from_relation_parts(
            object_id,
            deferred_pcs_id,
            artifacts,
            verifier_r1cs,
            verifier_witness,
        )
    }

    /// Constructs the recursive circuit directly from Jolt's native
    /// BlindFold verifier-relation witness layout. This adapter preserves the
    /// exact Lasso/register/RAM/CPU endpoint variable ordering.
    #[cfg(test)]
    #[allow(dead_code)]
    pub(crate) fn from_blindfold_witness(
        object_id: [u8; 32],
        deferred_pcs_id: [u8; 32],
        initial_transcript_state: RecursiveJoltFieldElement,
        initial_transcript_round: u64,
        sumcheck_stages: Vec<RecursiveClearSumcheckStageWitness>,
        verifier_r1cs: VerifierR1CS<ark_bn254::Fr>,
        verifier_witness: &BlindFoldWitness<ark_bn254::Fr>,
    ) -> Result<Self, &'static str> {
        let assigned = verifier_witness.assign(&verifier_r1cs);
        Self::new(
            object_id,
            deferred_pcs_id,
            initial_transcript_state,
            initial_transcript_round,
            sumcheck_stages,
            verifier_r1cs,
            assigned,
        )
    }

    /// Returns the exact public statement proved by this circuit.
    pub fn statement(&self) -> RecursiveJoltVerifierStatement {
        RecursiveJoltVerifierStatement {
            object_id: self.object_id,
            deferred_pcs_id: self.deferred_pcs_id,
            initial_transcript_state: self.initial_transcript_state.clone(),
            initial_transcript_round: self.initial_transcript_round,
            transcript_checkpoint_root: Self::checkpoint_root(&self.stage_transcript_checkpoints),
            shape_id: self.shape_id(),
        }
    }

    /// Test-only full synthesis gate used by real-proof integration tests.
    #[cfg(all(test, feature = "prover"))]
    pub(crate) fn test_constraints_are_satisfied(&self) -> Result<(), String> {
        use nova_snark::frontend::test_cs::TestConstraintSystem;

        let mut cs = TestConstraintSystem::<NovaScalar>::new();
        let z = self
            .initial_z()
            .map_err(|error| format!("failed to derive recursive initial state: {error:?}"))?
            .into_iter()
            .enumerate()
            .map(|(index, value)| {
                AllocatedNum::alloc(cs.namespace(|| format!("real verifier z {index}")), || {
                    Ok(value)
                })
                .map_err(|error| format!("failed to allocate recursive state: {error:?}"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.synthesize(&mut cs, &z)
            .map_err(|error| format!("recursive verifier synthesis failed: {error:?}"))?;
        if cs.is_satisfied() {
            Ok(())
        } else {
            Err(format!(
                "recursive verifier constraints are unsatisfied: {:?}",
                cs.which_is_unsatisfied()
            ))
        }
    }

    /// Commits to every value that changes the synthesized verifier relation.
    /// Witness values and public statement values are intentionally excluded.
    fn shape_id(&self) -> [u8; 32] {
        fn append_usize(hasher: &mut Sha3_256, value: usize) {
            hasher.update((value as u64).to_le_bytes());
        }
        fn append_matrix(
            hasher: &mut Sha3_256,
            matrix: &crate::subprotocols::blindfold::SparseR1CSMatrix<ark_bn254::Fr>,
        ) {
            append_usize(hasher, matrix.num_rows);
            append_usize(hasher, matrix.num_cols);
            append_usize(hasher, matrix.entries.len());
            for (row, column, value) in &matrix.entries {
                append_usize(hasher, *row);
                append_usize(hasher, *column);
                let mut encoded = Vec::new();
                value
                    .serialize_compressed(&mut encoded)
                    .expect("field serialization into memory cannot fail");
                hasher.update(encoded);
            }
        }

        let mut hasher = Sha3_256::new();
        hasher.update(b"jolt-nova/recursive-verifier-shape/v1");
        append_usize(&mut hasher, RECURSIVE_VERIFIER_Z_ARITY);
        append_usize(&mut hasher, self.sumcheck_stages.len());
        append_usize(&mut hasher, self.stage_transcript_checkpoints.len());
        for stage in &self.sumcheck_stages {
            append_usize(&mut hasher, stage.stage_index);
            append_usize(&mut hasher, stage.degree_bound);
            append_usize(&mut hasher, stage.rounds.len());
            for round in &stage.rounds {
                append_usize(&mut hasher, round.coefficients_except_linear.len());
            }
        }
        append_usize(&mut hasher, self.verifier_r1cs.num_vars);
        append_usize(&mut hasher, self.verifier_r1cs.num_constraints);
        append_matrix(&mut hasher, &self.verifier_r1cs.a);
        append_matrix(&mut hasher, &self.verifier_r1cs.b);
        append_matrix(&mut hasher, &self.verifier_r1cs.c);
        hasher.finalize().into()
    }

    pub(super) fn initial_z(&self) -> Result<Vec<NovaScalar>, SynthesisError> {
        self.statement()
            .initial_z()
            .map_err(SynthesisError::Unsatisfiable)
    }

    /// Runs Nova/Spartan setup once and returns separate prover/verifier
    /// material.  Subsequent proofs must match this exact circuit shape.
    pub fn setup_pinned(
        &self,
    ) -> Result<
        (
            RecursiveJoltVerifierProverParameters,
            RecursiveJoltVerifierVerificationKey,
        ),
        String,
    > {
        let public_params = self.public_params()?;
        let (prover_key, verifier_key) = RecursiveVerifierCompressedSnark::setup(&public_params)
            .map_err(|error| format!("recursive verifier Spartan setup failed: {error:?}"))?;
        let shape_id = self.shape_id();
        Ok((
            RecursiveJoltVerifierProverParameters {
                public_params,
                prover_key,
                shape_id,
            },
            RecursiveJoltVerifierVerificationKey {
                verifier_key,
                shape_id,
            },
        ))
    }

    fn public_params(
        &self,
    ) -> Result<
        nova_snark::nova::PublicParams<
            NovaPrimaryEngine,
            NovaSecondaryEngine,
            RecursiveJoltVerifierCircuit,
        >,
        String,
    > {
        nova_snark::nova::PublicParams::setup(
            self,
            &*nova_snark::traits::snark::default_ck_hint::<NovaPrimaryEngine>(),
            &*nova_snark::traits::snark::default_ck_hint::<NovaSecondaryEngine>(),
        )
        .map_err(|error| format!("recursive verifier Nova setup failed: {error:?}"))
    }

    /// Proves the complete verifier relation with Nova and compresses the
    /// resulting recursive SNARK using Spartan.
    pub fn prove_spartan(&self) -> Result<RecursiveJoltVerifierSpartanProof, String> {
        let pp = self.public_params()?;
        let z0 = self.initial_z().map_err(|error| format!("{error:?}"))?;
        let mut recursive = RecursiveVerifierNovaSnark::new(&pp, self, &z0)
            .map_err(|error| format!("recursive verifier Nova initialization failed: {error:?}"))?;
        recursive
            .prove_step(&pp, self)
            .map_err(|error| format!("recursive verifier Nova proving failed: {error:?}"))?;
        let recursive_output = recursive
            .verify(&pp, 1, &z0)
            .map_err(|error| format!("recursive verifier Nova self-check failed: {error:?}"))?;
        if recursive_output.len() != RECURSIVE_VERIFIER_Z_ARITY
            || recursive_output[6] != NovaScalar::one()
        {
            return Err("recursive verifier Nova output did not accept".to_string());
        }

        let (pk, vk) = RecursiveVerifierCompressedSnark::setup(&pp)
            .map_err(|error| format!("recursive verifier Spartan setup failed: {error:?}"))?;
        let compressed = RecursiveVerifierCompressedSnark::prove(&pp, &pk, &recursive)
            .map_err(|error| format!("recursive verifier Spartan proving failed: {error:?}"))?;
        let output = compressed
            .verify(&vk, 1, &z0)
            .map_err(|error| format!("recursive verifier Spartan self-check failed: {error:?}"))?;
        if output != recursive_output {
            return Err("recursive verifier Spartan output mismatch".to_string());
        }
        let proof_bytes = postcard::to_stdvec(&compressed)
            .map_err(|error| format!("recursive verifier Spartan serialization failed: {error}"))?;
        Ok(RecursiveJoltVerifierSpartanProof {
            proof_bytes,
            public_output: output.iter().copied().map(nova_scalar_to_storage).collect(),
        })
    }

    /// Verifies the real Spartan-compressed Nova proof and its full public
    /// output, including object/PCS continuity IDs and derived acceptance.
    pub fn verify_spartan(&self, proof: &RecursiveJoltVerifierSpartanProof) -> Result<(), String> {
        let pp = self.public_params()?;
        let (_, vk) = RecursiveVerifierCompressedSnark::setup(&pp)
            .map_err(|error| format!("recursive verifier Spartan setup failed: {error:?}"))?;
        let compressed: RecursiveVerifierCompressedSnark = postcard::from_bytes(&proof.proof_bytes)
            .map_err(|error| {
                format!("recursive verifier Spartan deserialization failed: {error}")
            })?;
        let z0 = self.initial_z().map_err(|error| format!("{error:?}"))?;
        let output = compressed.verify(&vk, 1, &z0).map_err(|error| {
            format!("recursive verifier Spartan verification failed: {error:?}")
        })?;
        let stored_output = output
            .iter()
            .copied()
            .map(nova_scalar_to_storage)
            .collect::<Vec<_>>();
        if stored_output != proof.public_output
            || output.len() != RECURSIVE_VERIFIER_Z_ARITY
            || output[6] != NovaScalar::one()
        {
            return Err("recursive verifier Spartan public output mismatch".to_string());
        }
        Ok(())
    }

    pub fn baseline(
        &self,
        proof: &RecursiveJoltVerifierSpartanProof,
    ) -> RecursiveJoltVerifierBaseline {
        let sumcheck_rounds = self
            .sumcheck_stages
            .iter()
            .map(|stage| stage.rounds.len())
            .sum();
        let poseidon_transcript_transitions = self
            .sumcheck_stages
            .iter()
            .flat_map(|stage| stage.rounds.iter())
            // packed label/count + each compressed coefficient + challenge
            .map(|round| 2 + round.coefficients_except_linear.len())
            .sum();
        RecursiveJoltVerifierBaseline {
            sumcheck_stages: self.sumcheck_stages.len(),
            sumcheck_rounds,
            poseidon_transcript_transitions,
            verifier_r1cs_variables: self.verifier_r1cs.num_vars,
            verifier_r1cs_constraints: self.verifier_r1cs.num_constraints,
            spartan_proof_bytes: proof.proof_bytes.len(),
            public_output_bytes: proof.public_output.len() * 32,
        }
    }
}

impl RecursiveJoltVerifierProverParameters {
    pub fn shape_id(&self) -> [u8; 32] {
        self.shape_id
    }

    /// Proves with setup material fixed before the per-proof statement exists.
    pub fn prove(
        &self,
        circuit: &RecursiveJoltVerifierCircuit,
    ) -> Result<RecursiveJoltVerifierSpartanProof, String> {
        if circuit.shape_id() != self.shape_id {
            return Err("recursive verifier circuit does not match pinned setup shape".to_string());
        }
        let statement = circuit.statement();
        let z0 = statement.initial_z()?;
        let mut recursive = RecursiveVerifierNovaSnark::new(&self.public_params, circuit, &z0)
            .map_err(|error| format!("recursive verifier Nova initialization failed: {error:?}"))?;
        recursive
            .prove_step(&self.public_params, circuit)
            .map_err(|error| format!("recursive verifier Nova proving failed: {error:?}"))?;
        let recursive_output = recursive
            .verify(&self.public_params, 1, &z0)
            .map_err(|error| format!("recursive verifier Nova self-check failed: {error:?}"))?;
        if recursive_output.len() != RECURSIVE_VERIFIER_Z_ARITY
            || recursive_output[6] != NovaScalar::one()
        {
            return Err("recursive verifier Nova output did not accept".to_string());
        }
        let compressed = RecursiveVerifierCompressedSnark::prove(
            &self.public_params,
            &self.prover_key,
            &recursive,
        )
        .map_err(|error| format!("recursive verifier Spartan proving failed: {error:?}"))?;
        let proof_bytes = postcard::to_stdvec(&compressed)
            .map_err(|error| format!("recursive verifier Spartan serialization failed: {error}"))?;
        Ok(RecursiveJoltVerifierSpartanProof {
            proof_bytes,
            public_output: recursive_output
                .iter()
                .copied()
                .map(nova_scalar_to_storage)
                .collect(),
        })
    }
}

impl RecursiveJoltVerifierVerificationKey {
    pub fn shape_id(&self) -> [u8; 32] {
        self.shape_id
    }

    /// Verifies using only a public statement and the setup-pinned key.
    pub fn verify(
        &self,
        statement: &RecursiveJoltVerifierStatement,
        proof: &RecursiveJoltVerifierSpartanProof,
    ) -> Result<(), String> {
        if statement.shape_id != self.shape_id {
            return Err("recursive verifier statement/setup shape mismatch".to_string());
        }
        let compressed: RecursiveVerifierCompressedSnark = postcard::from_bytes(&proof.proof_bytes)
            .map_err(|error| {
                format!("recursive verifier Spartan deserialization failed: {error}")
            })?;
        let z0 = statement.initial_z()?;
        let output = compressed
            .verify(&self.verifier_key, 1, &z0)
            .map_err(|error| {
                format!("recursive verifier Spartan verification failed: {error:?}")
            })?;
        let stored_output = output
            .iter()
            .copied()
            .map(nova_scalar_to_storage)
            .collect::<Vec<_>>();
        if stored_output != proof.public_output
            || output.len() != RECURSIVE_VERIFIER_Z_ARITY
            || output[6] != NovaScalar::one()
        {
            return Err("recursive verifier Spartan public output mismatch".to_string());
        }
        Ok(())
    }
}

impl StepCircuit<NovaScalar> for RecursiveJoltVerifierCircuit {
    fn arity(&self) -> usize {
        RECURSIVE_VERIFIER_Z_ARITY
    }

    fn synthesize<CS: ConstraintSystem<NovaScalar>>(
        &self,
        cs: &mut CS,
        z: &[AllocatedNum<NovaScalar>],
    ) -> Result<Vec<AllocatedNum<NovaScalar>>, SynthesisError> {
        if z.len() != RECURSIVE_VERIFIER_Z_ARITY {
            return Err(SynthesisError::AssignmentMissing);
        }
        for (index, expected) in self.initial_z()?.into_iter().enumerate() {
            cs.enforce(
                || format!("bind recursive verifier initial z word {index}"),
                |lc| lc + z[index].get_variable() - (expected, CS::one()),
                |lc| lc + CS::one(),
                |lc| lc,
            );
        }

        if self.stage_transcript_checkpoints.len() != self.sumcheck_stages.len() {
            return Err(SynthesisError::Unsatisfiable(
                "recursive verifier stage/checkpoint count mismatch".to_string(),
            ));
        }
        let allocated_checkpoints = self
            .stage_transcript_checkpoints
            .iter()
            .enumerate()
            .map(|(index, (state, round))| {
                let state_value = Option::from(NovaScalar::from_bytes(&state.canonical_le_bytes))
                    .ok_or_else(|| {
                    SynthesisError::Unsatisfiable(format!(
                        "recursive transcript checkpoint {index} is not canonical"
                    ))
                })?;
                let allocated_state = AllocatedNum::alloc(
                    cs.namespace(|| format!("allocate transcript checkpoint {index} state")),
                    || Ok(state_value),
                )?;
                let allocated_round = AllocatedNum::alloc(
                    cs.namespace(|| format!("allocate transcript checkpoint {index} round")),
                    || Ok(NovaScalar::from(*round)),
                )?;
                Ok(AllocatedRecursivePoseidonTranscriptState {
                    state: allocated_state,
                    n_rounds: allocated_round,
                })
            })
            .collect::<Result<Vec<_>, SynthesisError>>()?;
        cs.enforce(
            || "first transcript checkpoint state equals public z",
            |lc| lc + allocated_checkpoints[0].state.get_variable() - z[2].get_variable(),
            |lc| lc + CS::one(),
            |lc| lc,
        );
        cs.enforce(
            || "first transcript checkpoint round equals public z",
            |lc| lc + allocated_checkpoints[0].n_rounds.get_variable() - z[3].get_variable(),
            |lc| lc + CS::one(),
            |lc| lc,
        );
        let checkpoint_root = synthesize_recursive_checkpoint_capsule(
            cs.namespace(|| "recursive transcript checkpoint capsule"),
            &allocated_checkpoints,
        )?;
        cs.enforce(
            || "transcript checkpoint capsule equals public root",
            |lc| lc + checkpoint_root.get_variable() - z[7].get_variable(),
            |lc| lc + CS::one(),
            |lc| lc,
        );

        let mut allocated_stages = Vec::with_capacity(self.sumcheck_stages.len());
        let mut final_transcript = None;
        for (stage_position, stage) in self.sumcheck_stages.iter().enumerate() {
            let allocated = synthesize_recursive_clear_sumcheck_stage(
                cs.namespace(|| format!("sumcheck stage {stage_position} arithmetic")),
                stage,
                stage.stage_index,
                stage.rounds.len(),
                stage.degree_bound,
            )?;
            let transcript = synthesize_recursive_clear_sumcheck_transcript(
                cs.namespace(|| format!("sumcheck stage {stage_position} transcript")),
                &allocated_checkpoints[stage_position],
                &allocated,
            )?;
            final_transcript = Some(transcript);
            allocated_stages.push(allocated);
        }
        let final_transcript = final_transcript.ok_or_else(|| {
            SynthesisError::Unsatisfiable("recursive verifier has no transcript stage".to_string())
        })?;

        let allocated_verifier_witness = synthesize_recursive_jolt_verifier_r1cs(
            cs.namespace(|| "complete Jolt verifier relation R1CS"),
            &self.verifier_r1cs,
            &self.verifier_witness,
        )?;
        bind_recursive_sumchecks_to_verifier_r1cs(
            cs.namespace(|| "bind sumchecks to verifier relation R1CS"),
            &allocated_stages,
            &self.verifier_r1cs,
            &allocated_verifier_witness,
        )?;

        let stage_count =
            AllocatedNum::alloc(cs.namespace(|| "verified sumcheck stage count"), || {
                Ok(NovaScalar::from(self.sumcheck_stages.len() as u64))
            })?;
        cs.enforce(
            || "bind verified sumcheck stage count",
            |lc| {
                lc + stage_count.get_variable()
                    - (
                        NovaScalar::from(self.sumcheck_stages.len() as u64),
                        CS::one(),
                    )
            },
            |lc| lc + CS::one(),
            |lc| lc,
        );
        let relation_count =
            AllocatedNum::alloc(cs.namespace(|| "verified relation row count"), || {
                Ok(NovaScalar::from(self.verifier_r1cs.num_constraints as u64))
            })?;
        cs.enforce(
            || "bind verified relation row count",
            |lc| {
                lc + relation_count.get_variable()
                    - (
                        NovaScalar::from(self.verifier_r1cs.num_constraints as u64),
                        CS::one(),
                    )
            },
            |lc| lc + CS::one(),
            |lc| lc,
        );
        let accepted = AllocatedNum::alloc(cs.namespace(|| "recursive verifier accepted"), || {
            Ok(NovaScalar::one())
        })?;
        cs.enforce(
            || "recursive verifier acceptance is derived",
            |lc| lc + accepted.get_variable() - CS::one(),
            |lc| lc + CS::one(),
            |lc| lc,
        );

        Ok(vec![
            z[0].clone(),
            z[1].clone(),
            final_transcript.state,
            final_transcript.n_rounds,
            stage_count,
            relation_count,
            accepted,
            z[7].clone(),
        ])
    }
}

#[cfg(test)]
mod tests {
    use ark_bn254::Fr;
    use ark_std::UniformRand;
    use nova_snark::frontend::{test_cs::TestConstraintSystem, ConstraintSystem};
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
        subprotocols::blindfold::{HyraxParams, SparseR1CSMatrix},
        transcripts::{PoseidonTranscript, Transcript},
        zkvm::block::RecursiveClearSumcheckRoundWitness,
    };

    use super::*;

    fn fixture_with_ids(
        tamper_grid: bool,
        object_id: [u8; 32],
        deferred_pcs_id: [u8; 32],
    ) -> RecursiveJoltVerifierCircuit {
        let mut transcript = PoseidonTranscript::new(b"stage14-unified");
        let coefficients = [Fr::from(2u64), Fr::from(6u64)];
        let initial_state = RecursiveJoltFieldElement {
            canonical_le_bytes: transcript.state,
        };
        transcript.append_scalars(b"sumcheck_poly", &coefficients);
        let challenge = transcript.challenge_scalar::<Fr>();
        let final_claim = Fr::from(2u64) + Fr::from(6u64) * challenge * challenge;
        let stage = RecursiveClearSumcheckStageWitness {
            stage_index: 0,
            degree_bound: 2,
            initial_claim: RecursiveJoltFieldElement::from_field(Fr::from(10u64)).unwrap(),
            rounds: vec![RecursiveClearSumcheckRoundWitness {
                coefficients_except_linear: coefficients
                    .iter()
                    .map(|value| RecursiveJoltFieldElement::from_field(*value).unwrap())
                    .collect(),
                challenge: RecursiveJoltFieldElement::from_field(challenge).unwrap(),
            }],
            expected_final_claim: RecursiveJoltFieldElement::from_field(final_claim).unwrap(),
        };

        // Z = [1 | coefficient grid [c0, c1, c2, padding]].
        let mut a = SparseR1CSMatrix::new(1, 5);
        let mut b = SparseR1CSMatrix::new(1, 5);
        let mut c = SparseR1CSMatrix::new(1, 5);
        a.push(0, 1, Fr::from(1u64));
        b.push(0, 2, Fr::from(1u64));
        c.push(0, 4, Fr::from(1u64));
        let r1cs = VerifierR1CS {
            a,
            b,
            c,
            num_vars: 5,
            num_constraints: 1,
            stage_configs: Vec::new(),
            extra_constraints: Vec::new(),
            extra_output_vars: Vec::new(),
            extra_blinding_vars: Vec::new(),
            hyrax: HyraxParams {
                C: 4,
                R_coeff: 1,
                R_prime: 1,
                noncoeff_count: 0,
                total_rounds: 1,
                output_claims_rows: 0,
            },
            output_claims_opening_ids: Vec::new(),
            opening_aliases: Default::default(),
            opening_vars: Default::default(),
        };
        let verifier_witness = vec![
            Fr::from(1u64),
            if tamper_grid {
                Fr::from(3u64)
            } else {
                Fr::from(2u64)
            },
            Fr::from(0u64),
            Fr::from(6u64),
            Fr::from(0u64),
        ];
        RecursiveJoltVerifierCircuit::new(
            object_id,
            deferred_pcs_id,
            initial_state,
            0,
            vec![stage],
            r1cs,
            verifier_witness,
        )
        .unwrap()
    }

    fn fixture(tamper_grid: bool) -> RecursiveJoltVerifierCircuit {
        fixture_with_ids(tamper_grid, [1u8; 32], [2u8; 32])
    }

    fn two_checkpoint_fixture() -> RecursiveJoltVerifierCircuit {
        let stage_specs = [
            (b"stage15-first".as_slice(), 10u64, 2u64, 6u64),
            (b"stage15-second".as_slice(), 14u64, 3u64, 7u64),
        ];
        let mut artifacts = Vec::with_capacity(stage_specs.len());
        let mut verifier_witness = vec![Fr::from(1u64)];

        for (stage_index, (label, initial_claim, constant, quadratic)) in
            stage_specs.into_iter().enumerate()
        {
            let mut transcript = PoseidonTranscript::new(label);
            let checkpoint = RecursiveJoltFieldElement {
                canonical_le_bytes: transcript.state,
            };
            let compressed = [Fr::from(constant), Fr::from(quadratic)];
            transcript.append_scalars(b"sumcheck_poly", &compressed);
            let challenge = transcript.challenge_scalar::<Fr>();
            let linear =
                Fr::from(initial_claim) - Fr::from(2u64) * Fr::from(constant) - Fr::from(quadratic);
            let final_claim = Fr::from(constant)
                + linear * challenge
                + Fr::from(quadratic) * challenge * challenge;

            artifacts.push(RecursiveClearSumcheckStageArtifact {
                transcript_state_before: checkpoint,
                transcript_round_before: 0,
                witness: RecursiveClearSumcheckStageWitness {
                    stage_index,
                    degree_bound: 2,
                    initial_claim: RecursiveJoltFieldElement::from_field(Fr::from(initial_claim))
                        .unwrap(),
                    rounds: vec![RecursiveClearSumcheckRoundWitness {
                        coefficients_except_linear: compressed
                            .iter()
                            .copied()
                            .map(RecursiveJoltFieldElement::from_field)
                            .collect::<Result<Vec<_>, _>>()
                            .unwrap(),
                        challenge: RecursiveJoltFieldElement::from_field(challenge).unwrap(),
                    }],
                    expected_final_claim: RecursiveJoltFieldElement::from_field(final_claim)
                        .unwrap(),
                },
            });
            verifier_witness.extend([
                Fr::from(constant),
                linear,
                Fr::from(quadratic),
                Fr::from(0u64),
            ]);
        }

        let r1cs = VerifierR1CS {
            a: SparseR1CSMatrix::new(0, 9),
            b: SparseR1CSMatrix::new(0, 9),
            c: SparseR1CSMatrix::new(0, 9),
            num_vars: 9,
            num_constraints: 0,
            stage_configs: Vec::new(),
            extra_constraints: Vec::new(),
            extra_output_vars: Vec::new(),
            extra_blinding_vars: Vec::new(),
            hyrax: HyraxParams {
                C: 4,
                R_coeff: 2,
                R_prime: 2,
                noncoeff_count: 0,
                total_rounds: 2,
                output_claims_rows: 0,
            },
            output_claims_opening_ids: Vec::new(),
            opening_aliases: Default::default(),
            opening_vars: Default::default(),
        };
        RecursiveJoltVerifierCircuit::from_verified_artifacts(
            [3u8; 32],
            [4u8; 32],
            artifacts,
            r1cs,
            verifier_witness,
        )
        .unwrap()
    }

    fn synthesize(circuit: &RecursiveJoltVerifierCircuit) -> TestConstraintSystem<NovaScalar> {
        let mut cs = TestConstraintSystem::<NovaScalar>::new();
        let z = circuit
            .initial_z()
            .unwrap()
            .into_iter()
            .enumerate()
            .map(|(index, value)| {
                AllocatedNum::alloc(cs.namespace(|| format!("z {index}")), || Ok(value)).unwrap()
            })
            .collect::<Vec<_>>();
        let output = circuit.synthesize(&mut cs, &z).unwrap();
        assert_eq!(output[6].get_value(), Some(NovaScalar::one()));
        cs
    }

    #[test]
    fn unified_recursive_jolt_verifier_circuit_accepts_joined_relation() {
        let cs = synthesize(&fixture(false));
        assert!(cs.is_satisfied(), "{:?}", cs.which_is_unsatisfied());
    }

    #[test]
    fn unified_recursive_jolt_verifier_accepts_independent_stage_checkpoints() {
        let circuit = two_checkpoint_fixture();
        assert_ne!(
            circuit.stage_transcript_checkpoints[0],
            circuit.stage_transcript_checkpoints[1]
        );
        let cs = synthesize(&circuit);
        assert!(cs.is_satisfied(), "{:?}", cs.which_is_unsatisfied());
    }

    #[test]
    fn unified_recursive_jolt_verifier_circuit_rejects_disconnected_r1cs_grid() {
        let cs = synthesize(&fixture(true));
        assert!(!cs.is_satisfied());
        assert!(cs
            .which_is_unsatisfied()
            .unwrap_or_default()
            .contains("bind sumcheck artifact round 0 to grid row 0 coefficient 0"));
    }

    #[test]
    fn unified_recursive_jolt_verifier_generates_and_verifies_real_spartan_proof() {
        let circuit = fixture(false);
        let (prover_parameters, verification_key) = circuit.setup_pinned().unwrap();
        let statement = circuit.statement();
        let proof = prover_parameters.prove(&circuit).unwrap();
        assert!(!proof.proof_bytes.is_empty());
        assert_eq!(proof.public_output.len(), RECURSIVE_VERIFIER_Z_ARITY);
        verification_key.verify(&statement, &proof).unwrap();
        let baseline = circuit.baseline(&proof);
        assert_eq!(baseline.sumcheck_stages, 1);
        assert_eq!(baseline.sumcheck_rounds, 1);
        assert_eq!(baseline.poseidon_transcript_transitions, 4);
        assert_eq!(baseline.verifier_r1cs_constraints, 1);
        assert!(baseline.spartan_proof_bytes > 0);

        let mut tampered = proof;
        tampered.public_output[6][0] ^= 1;
        assert!(verification_key.verify(&statement, &tampered).is_err());

        let mut wrong_statement = statement;
        wrong_statement.object_id[0] ^= 1;
        assert!(verification_key
            .verify(&wrong_statement, &tampered)
            .is_err());

        let mut different_shape = fixture(false);
        different_shape.sumcheck_stages[0].rounds[0]
            .coefficients_except_linear
            .push(RecursiveJoltFieldElement::from_field(Fr::from(0u64)).unwrap());
        assert!(prover_parameters.prove(&different_shape).is_err());
    }

    #[test]
    #[serial]
    fn final_acceptance_requires_spartan_relation_and_real_dory_pairing() {
        let num_vars = 4usize;
        DoryGlobals::reset();
        let _guard =
            DoryGlobals::initialize_context(1, 1usize << num_vars, DoryContext::Main, None);
        let prover_setup = DoryCommitmentScheme::setup_prover(num_vars);
        let verifier_setup = DoryCommitmentScheme::setup_verifier(&prover_setup);
        let mut rng = ark_std::rand::thread_rng();
        let polynomial = MultilinearPolynomial::LargeScalars(DensePolynomial::new(
            (0..1usize << num_vars)
                .map(|_| Fr::rand(&mut rng))
                .collect(),
        ));
        let opening_point = (0..num_vars)
            .map(|_| <Fr as JoltField>::Challenge::random(&mut rng))
            .collect::<Vec<_>>();
        let opening = <MultilinearPolynomial<Fr> as PolynomialEvaluation<Fr>>::evaluate(
            &polynomial,
            &opening_point,
        );
        let (commitment, hint) = DoryCommitmentScheme::commit(&polynomial, &prover_setup);
        let mut prover_transcript = PoseidonTranscript::new(b"stage14-final");
        bind_opening_inputs::<Fr, _>(&mut prover_transcript, &opening_point, &opening);
        let (pcs_proof, _) = DoryCommitmentScheme::prove(
            &prover_setup,
            &polynomial,
            &opening_point,
            Some(hint),
            &mut prover_transcript,
        );
        let mut verifier_transcript = PoseidonTranscript::new(b"stage14-final");
        bind_opening_inputs::<Fr, _>(&mut verifier_transcript, &opening_point, &opening);
        let mut transcript_binding = verifier_transcript.state.to_vec();
        transcript_binding.extend_from_slice(&verifier_transcript.n_rounds.to_le_bytes());
        let deferred = RecursiveDeferredPcsOpening::<Fr, DoryCommitmentScheme, _>::new(
            pcs_proof,
            verifier_setup,
            verifier_transcript,
            opening_point,
            opening,
            commitment,
            &transcript_binding,
        );

        let object_id = [9u8; 32];
        let circuit = fixture_with_ids(false, object_id, deferred.obligation_id());
        let statement = circuit.statement();
        let (prover_parameters, verification_key) = circuit.setup_pinned().unwrap();
        let recursive_verifier_proof = prover_parameters.prove(&circuit).unwrap();
        let acceptance = RecursiveJoltFinalAcceptance {
            recursive_verifier_proof,
            deferred_pcs_opening: deferred,
        };
        acceptance
            .verify_pinned(&verification_key, &statement, object_id)
            .unwrap();
        assert!(acceptance
            .verify_pinned(&verification_key, &statement, [8u8; 32])
            .is_err());
    }
}
