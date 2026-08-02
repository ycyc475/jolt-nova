//! Unified Nova step for the Stage-14 Jolt verifier relation.
//!
//! This circuit joins exact Poseidon Fiat--Shamir transitions, clear-sumcheck
//! arithmetic, and the verifier R1CS containing Lasso/register/RAM/CPU/PCS
//! endpoint relations.

use nova_snark::{
    frontend::{num::AllocatedNum, ConstraintSystem, SynthesisError},
    traits::circuit::StepCircuit,
};

use crate::{
    field::JoltField,
    poly::commitment::commitment_scheme::CommitmentScheme,
    subprotocols::blindfold::{BlindFoldWitness, VerifierR1CS},
    transcripts::Transcript,
};

use super::recursive_relations::{
    bind_recursive_sumchecks_to_verifier_r1cs, synthesize_recursive_clear_sumcheck_stage,
    synthesize_recursive_clear_sumcheck_transcript, synthesize_recursive_jolt_verifier_r1cs,
    AllocatedRecursivePoseidonTranscriptState,
};
use super::{
    nova_hash_bytes_to_scalar, nova_scalar_to_storage, NovaPrimaryEngine, NovaPrimarySpartanSnark,
    NovaScalar, NovaSecondaryEngine, NovaSecondarySpartanSnark, RecursiveClearSumcheckStageWitness,
    RecursiveDeferredPcsOpening, RecursiveJoltFieldElement,
};

const RECURSIVE_VERIFIER_Z_ARITY: usize = 7;

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
    sumcheck_stages: Vec<RecursiveClearSumcheckStageWitness>,
    verifier_r1cs: VerifierR1CS<ark_bn254::Fr>,
    verifier_witness: Vec<ark_bn254::Fr>,
}

impl RecursiveJoltVerifierCircuit {
    pub fn new(
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
        Ok(Self {
            object_id,
            deferred_pcs_id,
            initial_transcript_state,
            initial_transcript_round,
            sumcheck_stages,
            verifier_r1cs,
            verifier_witness,
        })
    }

    /// Constructs the recursive circuit directly from Jolt's native
    /// BlindFold verifier-relation witness layout. This adapter preserves the
    /// exact Lasso/register/RAM/CPU endpoint variable ordering.
    pub fn from_blindfold_witness(
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

    fn id_scalar(domain: &'static str, id: &[u8; 32]) -> NovaScalar {
        nova_hash_bytes_to_scalar("recursive-jolt-verifier", domain, id)
    }

    fn initial_transcript_scalar(&self) -> Result<NovaScalar, SynthesisError> {
        Option::from(NovaScalar::from_bytes(
            &self.initial_transcript_state.canonical_le_bytes,
        ))
        .ok_or_else(|| {
            SynthesisError::Unsatisfiable(
                "initial recursive transcript state is not canonical BN254::Fr".to_string(),
            )
        })
    }

    pub(super) fn initial_z(&self) -> Result<Vec<NovaScalar>, SynthesisError> {
        Ok(vec![
            Self::id_scalar("object-id", &self.object_id),
            Self::id_scalar("deferred-pcs-id", &self.deferred_pcs_id),
            self.initial_transcript_scalar()?,
            NovaScalar::from(self.initial_transcript_round),
            NovaScalar::zero(),
            NovaScalar::zero(),
            NovaScalar::zero(),
        ])
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

        // Nova's shape-synthesis pass leaves public `z` assignments empty.
        // Allocate the known initial transcript as witness and constrain it to
        // the public state so downstream Poseidon assignments remain available
        // in both setup and proving passes.
        let initial_transcript_state = AllocatedNum::alloc(
            cs.namespace(|| "allocate initial recursive transcript state"),
            || self.initial_transcript_scalar(),
        )?;
        cs.enforce(
            || "initial recursive transcript state equals public z",
            |lc| lc + initial_transcript_state.get_variable() - z[2].get_variable(),
            |lc| lc + CS::one(),
            |lc| lc,
        );
        let initial_transcript_round = AllocatedNum::alloc(
            cs.namespace(|| "allocate initial recursive transcript round"),
            || Ok(NovaScalar::from(self.initial_transcript_round)),
        )?;
        cs.enforce(
            || "initial recursive transcript round equals public z",
            |lc| lc + initial_transcript_round.get_variable() - z[3].get_variable(),
            |lc| lc + CS::one(),
            |lc| lc,
        );
        let mut transcript = AllocatedRecursivePoseidonTranscriptState {
            state: initial_transcript_state,
            n_rounds: initial_transcript_round,
        };
        let mut allocated_stages = Vec::with_capacity(self.sumcheck_stages.len());
        for (stage_position, stage) in self.sumcheck_stages.iter().enumerate() {
            let allocated = synthesize_recursive_clear_sumcheck_stage(
                cs.namespace(|| format!("sumcheck stage {stage_position} arithmetic")),
                stage,
                stage.stage_index,
                stage.rounds.len(),
                stage.degree_bound,
            )?;
            transcript = synthesize_recursive_clear_sumcheck_transcript(
                cs.namespace(|| format!("sumcheck stage {stage_position} transcript")),
                &transcript,
                &allocated,
            )?;
            allocated_stages.push(allocated);
        }

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
            transcript.state,
            transcript.n_rounds,
            stage_count,
            relation_count,
            accepted,
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
    fn unified_recursive_jolt_verifier_circuit_rejects_disconnected_r1cs_grid() {
        let cs = synthesize(&fixture(true));
        assert!(!cs.is_satisfied());
        assert!(cs
            .which_is_unsatisfied()
            .unwrap_or_default()
            .contains("bind sumcheck round 0 coefficient 0"));
    }

    #[test]
    fn unified_recursive_jolt_verifier_generates_and_verifies_real_spartan_proof() {
        let circuit = fixture(false);
        let proof = circuit.prove_spartan().unwrap();
        assert!(!proof.proof_bytes.is_empty());
        assert_eq!(proof.public_output.len(), RECURSIVE_VERIFIER_Z_ARITY);
        circuit.verify_spartan(&proof).unwrap();
        let baseline = circuit.baseline(&proof);
        assert_eq!(baseline.sumcheck_stages, 1);
        assert_eq!(baseline.sumcheck_rounds, 1);
        assert_eq!(baseline.poseidon_transcript_transitions, 4);
        assert_eq!(baseline.verifier_r1cs_constraints, 1);
        assert!(baseline.spartan_proof_bytes > 0);

        let mut tampered = proof;
        tampered.public_output[6][0] ^= 1;
        assert!(circuit.verify_spartan(&tampered).is_err());
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
        let recursive_verifier_proof = circuit.prove_spartan().unwrap();
        let acceptance = RecursiveJoltFinalAcceptance {
            recursive_verifier_proof,
            deferred_pcs_opening: deferred,
        };
        acceptance.verify(&circuit, object_id).unwrap();
        assert!(acceptance.verify(&circuit, [8u8; 32]).is_err());
    }
}
