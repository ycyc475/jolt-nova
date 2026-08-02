use ark_serialize::CanonicalSerialize;
use light_poseidon::parameters::bn254_x5::get_poseidon_parameters;
use nova_snark::{
    frontend::{
        num::AllocatedNum, AllocatedBit, ConstraintSystem, LinearCombination, SynthesisError,
    },
    gadgets::{nonnative::bignat::BigNat, utils::alloc_bignat_constant},
};
use num::{bigint::Sign, BigInt};

use crate::subprotocols::blindfold::VerifierR1CS;
use crate::transcripts::Transcript;
use crate::zkvm::r1cs::constraints::{LC, R1CS_CONSTRAINTS};

use super::{
    NovaScalar, RecursiveClearSumcheckStageWitness, RecursiveJoltBlockOpeningWitness,
    RecursiveJoltFieldElement, RecursiveJoltLookupOpeningWitness, RecursiveJoltOpeningPoint,
    RecursiveJoltRamOpeningWitness, RecursiveJoltRegisterOpeningWitness,
};

const JOLT_FIELD_LIMB_WIDTH: usize = 64;
const JOLT_FIELD_LIMBS: usize = 4;
const REGISTER_INDEX_BITS: usize = (common::constants::REGISTER_COUNT as usize).ilog2() as usize;

/// Allocates and enforces the verifier R1CS used by Jolt's BlindFold relation.
/// This matrix contains the stage endpoint constraints for Lasso lookups,
/// registers, RAM, CPU/Spartan, and the Stage-8 joint-opening claim.  Enforcing
/// it row-by-row inside Nova avoids replacing those relations with receipt
/// digests.
pub(super) fn synthesize_recursive_jolt_verifier_r1cs<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    verifier_r1cs: &VerifierR1CS<ark_bn254::Fr>,
    witness: &[ark_bn254::Fr],
) -> Result<Vec<AllocatedNum<NovaScalar>>, SynthesisError> {
    if witness.len() != verifier_r1cs.num_vars
        || verifier_r1cs.a.num_rows != verifier_r1cs.num_constraints
        || verifier_r1cs.b.num_rows != verifier_r1cs.num_constraints
        || verifier_r1cs.c.num_rows != verifier_r1cs.num_constraints
        || verifier_r1cs.a.num_cols != verifier_r1cs.num_vars
        || verifier_r1cs.b.num_cols != verifier_r1cs.num_vars
        || verifier_r1cs.c.num_cols != verifier_r1cs.num_vars
    {
        return Err(SynthesisError::Unsatisfiable(
            "recursive Jolt verifier R1CS dimensions are inconsistent".to_string(),
        ));
    }

    let allocated = witness
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let value = ark_bn254_scalar_as_nova_scalar(value)?;
            AllocatedNum::alloc(
                cs.namespace(|| format!("verifier variable {index}")),
                || Ok(value),
            )
        })
        .collect::<Result<Vec<_>, SynthesisError>>()?;
    cs.enforce(
        || "verifier R1CS constant variable equals one",
        |lc| lc + allocated[0].get_variable() - CS::one(),
        |lc| lc + CS::one(),
        |lc| lc,
    );

    let matrix_row = |matrix: &crate::subprotocols::blindfold::SparseR1CSMatrix<ark_bn254::Fr>,
                      row: usize|
     -> Result<LinearCombination<NovaScalar>, SynthesisError> {
        matrix
            .entries
            .iter()
            .filter(|(entry_row, _, _)| *entry_row == row)
            .try_fold(
                LinearCombination::<NovaScalar>::zero(),
                |lc, (_, column, coefficient)| {
                    Ok(lc
                        + (
                            ark_bn254_scalar_as_nova_scalar(coefficient)?,
                            allocated[*column].get_variable(),
                        ))
                },
            )
    };

    for row in 0..verifier_r1cs.num_constraints {
        let a = matrix_row(&verifier_r1cs.a, row)?;
        let b = matrix_row(&verifier_r1cs.b, row)?;
        let c = matrix_row(&verifier_r1cs.c, row)?;
        cs.enforce(
            || format!("Jolt verifier R1CS row {row}"),
            |_| a,
            |_| b,
            |_| c,
        );
    }
    Ok(allocated)
}

/// Binds the transcript-verified sumcheck polynomials to the coefficient rows
/// of the exact Jolt verifier R1CS witness grid.
pub(super) fn bind_recursive_sumchecks_to_verifier_r1cs<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    stages: &[AllocatedRecursiveClearSumcheckStage],
    verifier_r1cs: &VerifierR1CS<ark_bn254::Fr>,
    allocated_verifier_witness: &[AllocatedNum<NovaScalar>],
) -> Result<(), SynthesisError> {
    let rounds = stages
        .iter()
        .flat_map(|stage| stage.full_round_coefficients.iter())
        .collect::<Vec<_>>();
    if rounds.len() != verifier_r1cs.hyrax.total_rounds
        || allocated_verifier_witness.len() != verifier_r1cs.num_vars
        || verifier_r1cs.num_vars < 1 + verifier_r1cs.hyrax.R_prime * verifier_r1cs.hyrax.C
    {
        return Err(SynthesisError::Unsatisfiable(
            "sumcheck/R1CS coefficient-grid dimensions do not match".to_string(),
        ));
    }

    for (round_index, coefficients) in rounds.into_iter().enumerate() {
        if coefficients.len() > verifier_r1cs.hyrax.C {
            return Err(SynthesisError::Unsatisfiable(
                "sumcheck polynomial exceeds verifier R1CS coefficient width".to_string(),
            ));
        }
        for column in 0..verifier_r1cs.hyrax.C {
            let verifier_variable =
                &allocated_verifier_witness[1 + round_index * verifier_r1cs.hyrax.C + column];
            if let Some(coefficient) = coefficients.get(column) {
                cs.enforce(
                    || format!("bind sumcheck round {round_index} coefficient {column}"),
                    |lc| lc + coefficient.get_variable() - verifier_variable.get_variable(),
                    |lc| lc + CS::one(),
                    |lc| lc,
                );
            } else {
                cs.enforce(
                    || format!("zero-pad sumcheck round {round_index} coefficient {column}"),
                    |lc| lc + verifier_variable.get_variable(),
                    |lc| lc + CS::one(),
                    |lc| lc,
                );
            }
        }
    }
    Ok(())
}

/// The exact state carried by Jolt's BN254 Poseidon Fiat--Shamir transcript.
/// Both values are native BN254 scalars after the Nova engine was moved to
/// `Bn256EngineIPA`.
pub(super) struct AllocatedRecursivePoseidonTranscriptState {
    pub state: AllocatedNum<NovaScalar>,
    pub n_rounds: AllocatedNum<NovaScalar>,
}

fn ark_bn254_scalar_as_nova_scalar(value: &ark_bn254::Fr) -> Result<NovaScalar, SynthesisError> {
    let mut bytes = [0u8; 32];
    value
        .serialize_uncompressed(&mut bytes[..])
        .map_err(|error| SynthesisError::Unsatisfiable(error.to_string()))?;
    Option::from(NovaScalar::from_bytes(&bytes)).ok_or_else(|| {
        SynthesisError::Unsatisfiable(
            "arkworks BN254 scalar is not canonical for the Nova BN254 scalar field".to_string(),
        )
    })
}

fn alloc_nova_constant<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    value: NovaScalar,
) -> Result<AllocatedNum<NovaScalar>, SynthesisError> {
    let allocated = AllocatedNum::alloc(cs.namespace(|| "allocate constant"), || Ok(value))?;
    cs.enforce(
        || "bind constant",
        |lc| lc + allocated.get_variable() - (value, CS::one()),
        |lc| lc + CS::one(),
        |lc| lc,
    );
    Ok(allocated)
}

fn poseidon_x5<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    input: &AllocatedNum<NovaScalar>,
) -> Result<AllocatedNum<NovaScalar>, SynthesisError> {
    let square = AllocatedNum::alloc(cs.namespace(|| "x squared"), || {
        input
            .get_value()
            .map(|value| value.square())
            .ok_or(SynthesisError::AssignmentMissing)
    })?;
    cs.enforce(
        || "x squared constraint",
        |lc| lc + input.get_variable(),
        |lc| lc + input.get_variable(),
        |lc| lc + square.get_variable(),
    );
    let fourth = AllocatedNum::alloc(cs.namespace(|| "x fourth"), || {
        input
            .get_value()
            .map(|value| value.square().square())
            .ok_or(SynthesisError::AssignmentMissing)
    })?;
    cs.enforce(
        || "x fourth constraint",
        |lc| lc + square.get_variable(),
        |lc| lc + square.get_variable(),
        |lc| lc + fourth.get_variable(),
    );
    let fifth = AllocatedNum::alloc(cs.namespace(|| "x fifth"), || {
        input
            .get_value()
            .map(|value| value.square().square() * value)
            .ok_or(SynthesisError::AssignmentMissing)
    })?;
    cs.enforce(
        || "x fifth constraint",
        |lc| lc + fourth.get_variable(),
        |lc| lc + input.get_variable(),
        |lc| lc + fifth.get_variable(),
    );
    Ok(fifth)
}

/// Constrains the exact `light_poseidon::Poseidon::<Fr>::new_circom(3)`
/// permutation used by `PoseidonTranscript`.  The domain tag is zero and the
/// three inputs are `(state, n_rounds, data)`.
pub(super) fn synthesize_recursive_poseidon_hash<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    state: &AllocatedNum<NovaScalar>,
    n_rounds: &AllocatedNum<NovaScalar>,
    data: &AllocatedNum<NovaScalar>,
) -> Result<AllocatedNum<NovaScalar>, SynthesisError> {
    let parameters = get_poseidon_parameters::<ark_bn254::Fr>(4)
        .map_err(|error| SynthesisError::Unsatisfiable(error.to_string()))?;
    debug_assert_eq!(parameters.width, 4);
    debug_assert_eq!(parameters.alpha, 5);

    let zero = alloc_nova_constant(cs.namespace(|| "Poseidon domain tag"), NovaScalar::zero())?;
    let mut poseidon_state = vec![zero, state.clone(), n_rounds.clone(), data.clone()];
    let half_full_rounds = parameters.full_rounds / 2;
    let total_rounds = parameters.full_rounds + parameters.partial_rounds;

    for round in 0..total_rounds {
        let mut round_cs = cs.namespace(|| format!("Poseidon round {round}"));
        for (column, state_word) in poseidon_state.iter_mut().enumerate() {
            let round_constant = ark_bn254_scalar_as_nova_scalar(
                &parameters.ark[round * parameters.width + column],
            )?;
            let with_constant = AllocatedNum::alloc(
                round_cs.namespace(|| format!("add round constant {column}")),
                || {
                    state_word
                        .get_value()
                        .map(|value| value + round_constant)
                        .ok_or(SynthesisError::AssignmentMissing)
                },
            )?;
            round_cs.enforce(
                || format!("round constant constraint {column}"),
                |lc| {
                    lc + state_word.get_variable() + (round_constant, CS::one())
                        - with_constant.get_variable()
                },
                |lc| lc + CS::one(),
                |lc| lc,
            );
            *state_word = with_constant;
        }

        let is_full_round =
            round < half_full_rounds || round >= half_full_rounds + parameters.partial_rounds;
        let sbox_columns = if is_full_round { parameters.width } else { 1 };
        for (column, state_word) in poseidon_state.iter_mut().enumerate().take(sbox_columns) {
            *state_word = poseidon_x5(
                round_cs.namespace(|| format!("S-box column {column}")),
                state_word,
            )?;
        }

        let previous_state = poseidon_state;
        poseidon_state = (0..parameters.width)
            .map(|row| {
                let coefficients = parameters.mds[row]
                    .iter()
                    .map(ark_bn254_scalar_as_nova_scalar)
                    .collect::<Result<Vec<_>, _>>()?;
                let mixed = AllocatedNum::alloc(
                    round_cs.namespace(|| format!("MDS output {row}")),
                    || {
                        previous_state.iter().zip(coefficients.iter()).try_fold(
                            NovaScalar::zero(),
                            |accumulator, (word, coefficient)| {
                                Ok::<_, SynthesisError>(
                                    accumulator
                                        + word
                                            .get_value()
                                            .ok_or(SynthesisError::AssignmentMissing)?
                                            * coefficient,
                                )
                            },
                        )
                    },
                )?;
                let input_lc = previous_state.iter().zip(coefficients.iter()).fold(
                    LinearCombination::<NovaScalar>::zero(),
                    |lc, (word, coefficient)| lc + (*coefficient, word.get_variable()),
                );
                round_cs.enforce(
                    || format!("MDS constraint {row}"),
                    |lc| lc + &input_lc - mixed.get_variable(),
                    |lc| lc + CS::one(),
                    |lc| lc,
                );
                Ok(mixed)
            })
            .collect::<Result<Vec<_>, SynthesisError>>()?;
    }

    Ok(poseidon_state.remove(0))
}

/// Performs one exact transcript transition and increments the domain-
/// separation counter.  This primitive is shared by append and challenge
/// operations; challenge generation uses a zero `data` word.
pub(super) fn synthesize_recursive_poseidon_transcript_transition<
    CS: ConstraintSystem<NovaScalar>,
>(
    mut cs: CS,
    current: &AllocatedRecursivePoseidonTranscriptState,
    data: &AllocatedNum<NovaScalar>,
) -> Result<AllocatedRecursivePoseidonTranscriptState, SynthesisError> {
    let state = synthesize_recursive_poseidon_hash(
        cs.namespace(|| "Poseidon transcript hash"),
        &current.state,
        &current.n_rounds,
        data,
    )?;
    let n_rounds = AllocatedNum::alloc(cs.namespace(|| "next transcript round"), || {
        current
            .n_rounds
            .get_value()
            .map(|value| value + NovaScalar::one())
            .ok_or(SynthesisError::AssignmentMissing)
    })?;
    cs.enforce(
        || "increment transcript round",
        |lc| lc + current.n_rounds.get_variable() + CS::one() - n_rounds.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc,
    );
    Ok(AllocatedRecursivePoseidonTranscriptState { state, n_rounds })
}

/// Commits the verifier-observed per-stage transcript checkpoints into one
/// public Poseidon root. Jolt performs other transcript operations between
/// sumcheck stages, so the stages cannot soundly be replayed as one contiguous
/// sequence.
pub(super) fn synthesize_recursive_checkpoint_capsule<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    checkpoints: &[AllocatedRecursivePoseidonTranscriptState],
) -> Result<AllocatedNum<NovaScalar>, SynthesisError> {
    let initial = crate::transcripts::PoseidonTranscript::new(b"stage15-checkpoints");
    let initial_state = Option::from(NovaScalar::from_bytes(&initial.state)).ok_or_else(|| {
        SynthesisError::Unsatisfiable(
            "checkpoint capsule initial state is not canonical".to_string(),
        )
    })?;
    let state = alloc_nova_constant(
        cs.namespace(|| "checkpoint capsule initial state"),
        initial_state,
    )?;
    let rounds = alloc_nova_constant(
        cs.namespace(|| "checkpoint capsule initial round"),
        NovaScalar::from(initial.n_rounds as u64),
    )?;
    let mut capsule = AllocatedRecursivePoseidonTranscriptState {
        state,
        n_rounds: rounds,
    };
    for (index, checkpoint) in checkpoints.iter().enumerate() {
        let mut checkpoint_cs = cs.namespace(|| format!("checkpoint capsule item {index}"));
        let label = alloc_nova_constant(
            checkpoint_cs.namespace(|| "packed checkpoint label and length"),
            packed_transcript_label_and_len(b"checkpoint", 2)?,
        )?;
        capsule = synthesize_recursive_poseidon_transcript_transition(
            checkpoint_cs.namespace(|| "absorb checkpoint label"),
            &capsule,
            &label,
        )?;
        capsule = synthesize_recursive_poseidon_transcript_transition(
            checkpoint_cs.namespace(|| "absorb checkpoint state"),
            &capsule,
            &checkpoint.state,
        )?;
        capsule = synthesize_recursive_poseidon_transcript_transition(
            checkpoint_cs.namespace(|| "absorb checkpoint round"),
            &capsule,
            &checkpoint.n_rounds,
        )?;
    }
    Ok(capsule.state)
}

fn packed_transcript_label_and_len(label: &[u8], len: usize) -> Result<NovaScalar, SynthesisError> {
    if label.len() > 24 {
        return Err(SynthesisError::Unsatisfiable(
            "recursive transcript label exceeds Jolt's 24-byte packed-label limit".to_string(),
        ));
    }
    let mut packed = [0u8; 32];
    packed[..label.len()].copy_from_slice(label);
    packed[24..].copy_from_slice(&(len as u64).to_be_bytes());
    ark_bn254_scalar_as_nova_scalar(
        &<ark_bn254::Fr as ark_ff::PrimeField>::from_le_bytes_mod_order(&packed),
    )
}

/// Binds every compressed-polynomial coefficient and every claimed sumcheck
/// challenge to Jolt's real Poseidon Fiat--Shamir state machine.
pub(super) fn synthesize_recursive_clear_sumcheck_transcript<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    initial_transcript: &AllocatedRecursivePoseidonTranscriptState,
    sumcheck: &AllocatedRecursiveClearSumcheckStage,
) -> Result<AllocatedRecursivePoseidonTranscriptState, SynthesisError> {
    if sumcheck.round_coefficients.len() != sumcheck.challenges.len() {
        return Err(SynthesisError::Unsatisfiable(
            "recursive sumcheck transcript has inconsistent round dimensions".to_string(),
        ));
    }

    let mut transcript = AllocatedRecursivePoseidonTranscriptState {
        state: initial_transcript.state.clone(),
        n_rounds: initial_transcript.n_rounds.clone(),
    };
    let zero = alloc_nova_constant(
        cs.namespace(|| "sumcheck challenge zero"),
        NovaScalar::zero(),
    )?;
    for (round_index, (coefficients, claimed_challenge)) in sumcheck
        .round_coefficients
        .iter()
        .zip(sumcheck.challenges.iter())
        .enumerate()
    {
        let mut round_cs = cs.namespace(|| format!("sumcheck transcript round {round_index}"));
        let label_and_len = alloc_nova_constant(
            round_cs.namespace(|| "packed sumcheck label and coefficient count"),
            packed_transcript_label_and_len(b"sumcheck_poly", coefficients.len())?,
        )?;
        transcript = synthesize_recursive_poseidon_transcript_transition(
            round_cs.namespace(|| "absorb sumcheck label and coefficient count"),
            &transcript,
            &label_and_len,
        )?;
        for (coefficient_index, coefficient) in coefficients.iter().enumerate() {
            transcript = synthesize_recursive_poseidon_transcript_transition(
                round_cs.namespace(|| format!("absorb coefficient {coefficient_index}")),
                &transcript,
                coefficient,
            )?;
        }
        transcript = synthesize_recursive_poseidon_transcript_transition(
            round_cs.namespace(|| "derive sumcheck challenge"),
            &transcript,
            &zero,
        )?;
        round_cs.enforce(
            || "claimed sumcheck challenge equals Poseidon transcript challenge",
            |lc| lc + transcript.state.get_variable() - claimed_challenge.get_variable(),
            |lc| lc + CS::one(),
            |lc| lc,
        );
    }
    Ok(transcript)
}

pub(super) struct AllocatedRecursiveClearSumcheckStage {
    pub initial_claim: AllocatedNum<NovaScalar>,
    pub round_coefficients: Vec<Vec<AllocatedNum<NovaScalar>>>,
    pub full_round_coefficients: Vec<Vec<AllocatedNum<NovaScalar>>>,
    pub challenges: Vec<AllocatedNum<NovaScalar>>,
    pub final_claim: AllocatedNum<NovaScalar>,
    pub expected_final_claim: AllocatedNum<NovaScalar>,
}

fn recursive_jolt_field_as_native_scalar(
    value: &RecursiveJoltFieldElement,
) -> Result<NovaScalar, SynthesisError> {
    Option::from(NovaScalar::from_bytes(&value.canonical_le_bytes)).ok_or_else(|| {
        SynthesisError::Unsatisfiable(
            "recursive Jolt field encoding is not canonical BN254::Fr".to_string(),
        )
    })
}

fn alloc_native_jolt_field<CS: ConstraintSystem<NovaScalar>>(
    cs: CS,
    value: &RecursiveJoltFieldElement,
) -> Result<AllocatedNum<NovaScalar>, SynthesisError> {
    let scalar = recursive_jolt_field_as_native_scalar(value)?;
    AllocatedNum::alloc(cs, || Ok(scalar))
}

/// Synthesizes the complete compressed-polynomial arithmetic for one clear
/// Jolt sumcheck stage.  Challenges and endpoint claims are returned so the
/// Fiat-Shamir and stage-specific relation gadgets can bind them separately.
pub(super) fn synthesize_recursive_clear_sumcheck_stage<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    witness: &RecursiveClearSumcheckStageWitness,
    expected_stage_index: usize,
    expected_rounds: usize,
    expected_degree_bound: usize,
) -> Result<AllocatedRecursiveClearSumcheckStage, SynthesisError> {
    witness
        .validate_shape(expected_stage_index, expected_rounds, expected_degree_bound)
        .map_err(|reason| SynthesisError::Unsatisfiable(reason.to_string()))?;

    let initial_claim = alloc_native_jolt_field(
        cs.namespace(|| "allocate initial sumcheck claim"),
        &witness.initial_claim,
    )?;
    let mut running_claim = initial_claim.clone();
    let mut round_coefficients = Vec::with_capacity(expected_rounds);
    let mut full_round_coefficients = Vec::with_capacity(expected_rounds);
    let mut challenges = Vec::with_capacity(expected_rounds);

    for (round_index, round) in witness.rounds.iter().enumerate() {
        let mut round_cs = cs.namespace(|| format!("sumcheck round {round_index}"));
        let coefficients = round
            .coefficients_except_linear
            .iter()
            .enumerate()
            .map(|(coefficient_index, coefficient)| {
                alloc_native_jolt_field(
                    round_cs.namespace(|| format!("coefficient {coefficient_index}")),
                    coefficient,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let challenge = alloc_native_jolt_field(
            round_cs.namespace(|| "Fiat-Shamir challenge"),
            &round.challenge,
        )?;

        let linear = AllocatedNum::alloc(
            round_cs.namespace(|| "recovered linear coefficient"),
            || {
                let mut value = running_claim
                    .get_value()
                    .ok_or(SynthesisError::AssignmentMissing)?;
                let constant = coefficients[0]
                    .get_value()
                    .ok_or(SynthesisError::AssignmentMissing)?;
                value -= constant;
                value -= constant;
                for coefficient in coefficients.iter().skip(1) {
                    value -= coefficient
                        .get_value()
                        .ok_or(SynthesisError::AssignmentMissing)?;
                }
                Ok(value)
            },
        )?;
        let high_degree_sum = coefficients.iter().skip(1).fold(
            LinearCombination::<NovaScalar>::zero(),
            |lc, coefficient| lc + coefficient.get_variable(),
        );
        round_cs.enforce(
            || "compressed polynomial recovers linear coefficient from claim",
            |lc| {
                lc + running_claim.get_variable()
                    - (NovaScalar::from(2), coefficients[0].get_variable())
                    - &high_degree_sum
                    - linear.get_variable()
            },
            |lc| lc + CS::one(),
            |lc| lc,
        );

        let mut full_coefficients = Vec::with_capacity(coefficients.len() + 1);
        full_coefficients.push(coefficients[0].clone());
        full_coefficients.push(linear);
        full_coefficients.extend(coefficients.iter().skip(1).cloned());

        let mut evaluated = full_coefficients
            .last()
            .expect("validated polynomial has at least two coefficients")
            .clone();
        for (horner_index, coefficient) in full_coefficients[..full_coefficients.len() - 1]
            .iter()
            .rev()
            .enumerate()
        {
            let next = AllocatedNum::alloc(
                round_cs.namespace(|| format!("Horner evaluation {horner_index}")),
                || {
                    Ok(evaluated
                        .get_value()
                        .ok_or(SynthesisError::AssignmentMissing)?
                        * challenge
                            .get_value()
                            .ok_or(SynthesisError::AssignmentMissing)?
                        + coefficient
                            .get_value()
                            .ok_or(SynthesisError::AssignmentMissing)?)
                },
            )?;
            round_cs.enforce(
                || format!("Horner multiplication {horner_index}"),
                |lc| lc + evaluated.get_variable(),
                |lc| lc + challenge.get_variable(),
                |lc| lc + next.get_variable() - coefficient.get_variable(),
            );
            evaluated = next;
        }

        running_claim = evaluated;
        round_coefficients.push(coefficients);
        full_round_coefficients.push(full_coefficients);
        challenges.push(challenge);
    }

    let expected_final_claim = alloc_native_jolt_field(
        cs.namespace(|| "allocate expected final sumcheck claim"),
        &witness.expected_final_claim,
    )?;
    cs.enforce(
        || "sumcheck final claim equals stage relation claim",
        |lc| lc + running_claim.get_variable() - expected_final_claim.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc,
    );

    Ok(AllocatedRecursiveClearSumcheckStage {
        initial_claim,
        round_coefficients,
        full_round_coefficients,
        challenges,
        final_claim: running_claim,
        expected_final_claim,
    })
}

pub(super) struct RecursiveNativeClaimContribution {
    aggregate: BigNat<NovaScalar>,
    target: BigNat<NovaScalar>,
    challenge: BigNat<NovaScalar>,
}

fn jolt_field_modulus() -> BigInt {
    BigInt::parse_bytes(
        b"21888242871839275222246405745257275088548364400416034343698204186575808495617",
        10,
    )
    .expect("the BN254 scalar modulus literal is valid")
}

fn field_element_nat(value: &RecursiveJoltFieldElement) -> BigInt {
    BigInt::from_bytes_le(Sign::Plus, &value.canonical_le_bytes)
}

fn alloc_jolt_modulus<CS: ConstraintSystem<NovaScalar>>(
    cs: CS,
) -> Result<BigNat<NovaScalar>, SynthesisError> {
    alloc_bignat_constant(
        cs,
        &jolt_field_modulus(),
        JOLT_FIELD_LIMB_WIDTH,
        JOLT_FIELD_LIMBS,
    )
}

fn alloc_jolt_constant<CS: ConstraintSystem<NovaScalar>>(
    cs: CS,
    value: u64,
) -> Result<BigNat<NovaScalar>, SynthesisError> {
    alloc_bignat_constant(
        cs,
        &BigInt::from(value),
        JOLT_FIELD_LIMB_WIDTH,
        JOLT_FIELD_LIMBS,
    )
}

fn alloc_jolt_field<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    value: &RecursiveJoltFieldElement,
    modulus: &BigNat<NovaScalar>,
) -> Result<BigNat<NovaScalar>, SynthesisError> {
    let allocated = BigNat::alloc_from_nat(
        cs.namespace(|| "allocate canonical bytes"),
        || Ok(field_element_nat(value)),
        JOLT_FIELD_LIMB_WIDTH,
        JOLT_FIELD_LIMBS,
    )?;
    allocated.assert_well_formed(cs.namespace(|| "limbs fit 64 bits"))?;

    // `alloc_from_nat` proves only that the integer fits in 256 bits. Equality
    // with its reduction additionally proves that the 32-byte encoding is a
    // canonical member of the native Jolt/BN254 scalar field.
    let reduced = allocated.red_mod(cs.namespace(|| "reduce modulo Jolt field"), modulus)?;
    allocated.equal_when_carried_regroup(
        cs.namespace(|| "encoding is strictly below Jolt modulus"),
        &reduced,
    )?;
    Ok(allocated)
}

fn alloc_small_jolt_field<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    value: u64,
) -> Result<BigNat<NovaScalar>, SynthesisError> {
    let allocated = BigNat::alloc_from_nat(
        cs.namespace(|| "allocate small field value"),
        || Ok(BigInt::from(value)),
        JOLT_FIELD_LIMB_WIDTH,
        JOLT_FIELD_LIMBS,
    )?;
    allocated.assert_well_formed(cs.namespace(|| "small value limbs fit"))?;
    Ok(allocated)
}

fn add_mod<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    left: &BigNat<NovaScalar>,
    right: &BigNat<NovaScalar>,
    modulus: &BigNat<NovaScalar>,
) -> Result<BigNat<NovaScalar>, SynthesisError> {
    left.add(right)?
        .red_mod(cs.namespace(|| "reduce sum"), modulus)
}

fn select_bignat<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    selector: &AllocatedBit,
    selector_value: bool,
    when_true: &BigNat<NovaScalar>,
    when_false: &BigNat<NovaScalar>,
) -> Result<BigNat<NovaScalar>, SynthesisError> {
    let when_true_value = when_true.value.clone();
    let when_false_value = when_false.value.clone();
    let selected = BigNat::alloc_from_nat(
        cs.namespace(|| "allocate selected integer"),
        || {
            if selector_value {
                when_true_value.ok_or(SynthesisError::AssignmentMissing)
            } else {
                when_false_value.ok_or(SynthesisError::AssignmentMissing)
            }
        },
        JOLT_FIELD_LIMB_WIDTH,
        JOLT_FIELD_LIMBS,
    )?;
    selected.assert_well_formed(cs.namespace(|| "selected limbs fit"))?;

    for limb_index in 0..JOLT_FIELD_LIMBS {
        cs.enforce(
            || format!("select limb {limb_index}"),
            |lc| lc + &when_true.limbs[limb_index] - &when_false.limbs[limb_index],
            |lc| lc + selector.get_variable(),
            |lc| lc + &selected.limbs[limb_index] - &when_false.limbs[limb_index],
        );
    }
    Ok(selected)
}

fn alloc_boolean<CS: ConstraintSystem<NovaScalar>>(
    cs: CS,
    value: bool,
) -> Result<AllocatedBit, SynthesisError> {
    AllocatedBit::alloc(cs, Some(value))
}

fn alloc_index_bits<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    value: usize,
    bit_len: usize,
) -> Result<Vec<AllocatedBit>, SynthesisError> {
    if bit_len < usize::BITS as usize && value >= (1usize << bit_len) {
        return Err(SynthesisError::Unsatisfiable(
            "recursive opening index exceeds its declared bit width".to_string(),
        ));
    }
    (0..bit_len)
        .map(|bit| {
            AllocatedBit::alloc(
                cs.namespace(|| format!("index bit {bit}")),
                Some(((value >> bit) & 1) == 1),
            )
        })
        .collect()
}

fn alloc_global_cycle_bits<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    global_cycle_start: &AllocatedNum<NovaScalar>,
    active: &AllocatedBit,
    active_value: bool,
    local_cycle: usize,
    global_cycle: usize,
    bit_len: usize,
) -> Result<Vec<AllocatedBit>, SynthesisError> {
    if bit_len >= usize::BITS as usize || global_cycle >= (1usize << bit_len) {
        return Err(SynthesisError::Unsatisfiable(
            "recursive opening cycle exceeds the opening domain".to_string(),
        ));
    }
    let bits = alloc_index_bits(
        cs.namespace(|| "allocate global-cycle bits"),
        global_cycle,
        bit_len,
    )?;
    let mut coefficient = NovaScalar::from(1);
    let packed = bits
        .iter()
        .fold(LinearCombination::<NovaScalar>::zero(), |lc, bit| {
            let current = coefficient;
            coefficient = coefficient + coefficient;
            lc + (current, bit.get_variable())
        });
    let inactive_packed = packed.clone();
    cs.enforce(
        || "active global cycle bits equal public block start plus local offset",
        |_| {
            packed
                - global_cycle_start.get_variable()
                - (NovaScalar::from(local_cycle as u64), CS::one())
        },
        |lc| lc + active.get_variable(),
        |lc| lc,
    );
    cs.enforce(
        || "inactive global cycle is canonical zero",
        |_| inactive_packed,
        |lc| lc + CS::one() - active.get_variable(),
        |lc| lc,
    );
    if !active_value && global_cycle != 0 {
        return Err(SynthesisError::Unsatisfiable(
            "inactive recursive opening cycle has a nonzero global index".to_string(),
        ));
    }
    Ok(bits)
}

fn enforce_bit_implies<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    premise: &AllocatedBit,
    consequence: &AllocatedBit,
) {
    cs.enforce(
        || "boolean implication",
        |lc| lc + premise.get_variable(),
        |lc| lc + CS::one() - consequence.get_variable(),
        |lc| lc,
    );
}

fn enforce_active_cycle_prefix<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    active_bits: &[AllocatedBit],
    active_cycles: &AllocatedNum<NovaScalar>,
) {
    let active_sum = active_bits
        .iter()
        .fold(LinearCombination::<NovaScalar>::zero(), |lc, active| {
            lc + active.get_variable()
        });
    cs.enforce(
        || "active-cycle bit sum equals public active-cycle count",
        |_| active_sum - active_cycles.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc,
    );
    for (index, window) in active_bits.windows(2).enumerate() {
        cs.enforce(
            || format!("active-cycle bits form a prefix at slot {index}"),
            |lc| lc + window[1].get_variable(),
            |lc| lc + CS::one() - window[0].get_variable(),
            |lc| lc,
        );
    }
}

fn packed_bits_lc<CS: ConstraintSystem<NovaScalar>>(
    bits: &[AllocatedBit],
) -> LinearCombination<NovaScalar> {
    let mut coefficient = NovaScalar::from(1);
    bits.iter()
        .fold(LinearCombination::<NovaScalar>::zero(), |lc, bit| {
            let current = coefficient;
            coefficient = coefficient + coefficient;
            lc + (current, bit.get_variable())
        })
}

fn alloc_u64_jolt_field_with_bits<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    value: u64,
) -> Result<(BigNat<NovaScalar>, Vec<AllocatedBit>), SynthesisError> {
    let allocated = alloc_small_jolt_field(cs.namespace(|| "allocate u64 as Jolt field"), value)?;
    let bits = (0..64)
        .map(|bit| {
            AllocatedBit::alloc(
                cs.namespace(|| format!("u64 bit {bit}")),
                Some(((value >> bit) & 1) == 1),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let packed = packed_bits_lc::<CS>(&bits);
    cs.enforce(
        || "u64 bits equal low Jolt-field limb",
        |_| packed - &allocated.limbs[0],
        |lc| lc + CS::one(),
        |lc| lc,
    );
    for limb_index in 1..JOLT_FIELD_LIMBS {
        cs.enforce(
            || format!("u64 upper Jolt-field limb {limb_index} is zero"),
            |lc| lc + &allocated.limbs[limb_index],
            |lc| lc + CS::one(),
            |lc| lc,
        );
    }
    Ok((allocated, bits))
}

fn alloc_u128_jolt_field_with_bits<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    value: u128,
) -> Result<(BigNat<NovaScalar>, Vec<AllocatedBit>), SynthesisError> {
    let allocated = BigNat::alloc_from_nat(
        cs.namespace(|| "allocate u128 as Jolt field"),
        || Ok(BigInt::from(value)),
        JOLT_FIELD_LIMB_WIDTH,
        JOLT_FIELD_LIMBS,
    )?;
    allocated.assert_well_formed(cs.namespace(|| "u128 limbs fit"))?;
    let bits = (0..128)
        .map(|bit| {
            AllocatedBit::alloc(
                cs.namespace(|| format!("u128 bit {bit}")),
                Some(((value >> bit) & 1) == 1),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    for limb_index in 0..2 {
        let packed = packed_bits_lc::<CS>(
            &bits[limb_index * JOLT_FIELD_LIMB_WIDTH..(limb_index + 1) * JOLT_FIELD_LIMB_WIDTH],
        );
        cs.enforce(
            || format!("u128 limb {limb_index} matches bits"),
            |_| packed - &allocated.limbs[limb_index],
            |lc| lc + CS::one(),
            |lc| lc,
        );
    }
    for limb_index in 2..JOLT_FIELD_LIMBS {
        cs.enforce(
            || format!("u128 upper Jolt-field limb {limb_index} is zero"),
            |lc| lc + &allocated.limbs[limb_index],
            |lc| lc + CS::one(),
            |lc| lc,
        );
    }
    Ok((allocated, bits))
}

fn enforce_zero_when_disabled<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    enabled: &AllocatedBit,
    value: &BigNat<NovaScalar>,
) {
    for limb_index in 0..JOLT_FIELD_LIMBS {
        cs.enforce(
            || format!("disabled value limb {limb_index} is zero"),
            |lc| lc + &value.limbs[limb_index],
            |lc| lc + CS::one() - enabled.get_variable(),
            |lc| lc,
        );
    }
}

fn nova_scalar_from_u128(value: u128) -> NovaScalar {
    let low = NovaScalar::from(value as u64);
    let high = NovaScalar::from((value >> 64) as u64);
    let mut two_to_64 = NovaScalar::from(1);
    for _ in 0..64 {
        two_to_64 = two_to_64 + two_to_64;
    }
    low + high * two_to_64
}

fn nova_scalar_from_i128(value: i128) -> NovaScalar {
    if value < 0 {
        -nova_scalar_from_u128(value.unsigned_abs())
    } else {
        nova_scalar_from_u128(value as u128)
    }
}

fn alloc_signed_jolt_input<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    value: i128,
    zero: &BigNat<NovaScalar>,
    modulus: &BigNat<NovaScalar>,
) -> Result<(AllocatedNum<NovaScalar>, BigNat<NovaScalar>), SynthesisError> {
    let negative_value = value < 0;
    let magnitude_value = value.unsigned_abs();
    let sign = alloc_boolean(cs.namespace(|| "signed input sign"), negative_value)?;
    let magnitude = BigNat::alloc_from_nat(
        cs.namespace(|| "signed input magnitude"),
        || Ok(BigInt::from(magnitude_value)),
        JOLT_FIELD_LIMB_WIDTH,
        JOLT_FIELD_LIMBS,
    )?;
    magnitude.assert_well_formed(cs.namespace(|| "signed input magnitude limbs fit"))?;

    let magnitude_bits = (0..128)
        .map(|bit| {
            AllocatedBit::alloc(
                cs.namespace(|| format!("signed input magnitude bit {bit}")),
                Some(((magnitude_value >> bit) & 1) == 1),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    for limb_index in 0..2 {
        let packed = packed_bits_lc::<CS>(
            &magnitude_bits
                [limb_index * JOLT_FIELD_LIMB_WIDTH..(limb_index + 1) * JOLT_FIELD_LIMB_WIDTH],
        );
        cs.enforce(
            || format!("signed input magnitude limb {limb_index} matches bits"),
            |_| packed - &magnitude.limbs[limb_index],
            |lc| lc + CS::one(),
            |lc| lc,
        );
    }
    for limb_index in 2..JOLT_FIELD_LIMBS {
        cs.enforce(
            || format!("signed input magnitude upper limb {limb_index} is zero"),
            |lc| lc + &magnitude.limbs[limb_index],
            |lc| lc + CS::one(),
            |lc| lc,
        );
    }

    let native = AllocatedNum::alloc(cs.namespace(|| "signed input in Nova field"), || {
        Ok(nova_scalar_from_i128(value))
    })?;
    let packed_magnitude = packed_bits_lc::<CS>(&magnitude_bits);
    let signed_left_magnitude = packed_magnitude.clone();
    let signed_right_magnitude = packed_magnitude;
    cs.enforce(
        || "signed Nova value equals sign and magnitude",
        |_| signed_left_magnitude,
        |lc| lc + (NovaScalar::from(2), sign.get_variable()),
        |_| signed_right_magnitude - native.get_variable(),
    );

    let negative = zero.sub_mod(
        cs.namespace(|| "negative signed input modulo Jolt field"),
        &magnitude,
        modulus,
    )?;
    let mapped = select_bignat(
        cs.namespace(|| "select signed Jolt-field encoding"),
        &sign,
        negative_value,
        &negative,
        &magnitude,
    )?;
    Ok((native, mapped))
}

fn jolt_lc_to_nova<CS: ConstraintSystem<NovaScalar>>(
    lc: &LC,
    inputs: &[AllocatedNum<NovaScalar>],
) -> Result<LinearCombination<NovaScalar>, SynthesisError> {
    let mut result = LinearCombination::<NovaScalar>::zero();
    for term_index in 0..lc.num_terms() {
        let term = lc.term(term_index).ok_or_else(|| {
            SynthesisError::Unsatisfiable(
                "Jolt R1CS linear combination omitted a declared term".to_string(),
            )
        })?;
        let input = inputs.get(term.input_index).ok_or_else(|| {
            SynthesisError::Unsatisfiable(
                "Jolt R1CS input index exceeds recursive CPU row".to_string(),
            )
        })?;
        result = result + (nova_scalar_from_i128(term.coeff), input.get_variable());
    }
    if let Some(constant) = lc.const_term() {
        result = result + (nova_scalar_from_i128(constant), CS::one());
    }
    Ok(result)
}

fn evaluate_jolt_lc(lc: &LC, inputs: &[i128]) -> Result<i128, SynthesisError> {
    let mut result = lc.const_term().unwrap_or(0);
    for term_index in 0..lc.num_terms() {
        let term = lc.term(term_index).ok_or_else(|| {
            SynthesisError::Unsatisfiable(
                "Jolt R1CS linear combination omitted a declared term".to_string(),
            )
        })?;
        let input = inputs.get(term.input_index).ok_or_else(|| {
            SynthesisError::Unsatisfiable(
                "Jolt R1CS input index exceeds recursive CPU row".to_string(),
            )
        })?;
        result = result
            .checked_add(term.coeff.checked_mul(*input).ok_or_else(|| {
                SynthesisError::Unsatisfiable(
                    "Jolt R1CS linear combination multiplication overflow".to_string(),
                )
            })?)
            .ok_or_else(|| {
                SynthesisError::Unsatisfiable(
                    "Jolt R1CS linear combination addition overflow".to_string(),
                )
            })?;
    }
    Ok(result)
}

fn enforce_native_zero_when_disabled<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    enabled: &AllocatedBit,
    value: &AllocatedNum<NovaScalar>,
) {
    cs.enforce(
        || "inactive native value is zero",
        |lc| lc + value.get_variable(),
        |lc| lc + CS::one() - enabled.get_variable(),
        |lc| lc,
    );
}

fn enforce_boolean_num<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    value: &AllocatedNum<NovaScalar>,
) {
    cs.enforce(
        || "CPU boolean input",
        |lc| lc + value.get_variable(),
        |lc| lc + value.get_variable() - CS::one(),
        |lc| lc,
    );
}

fn alloc_opening_point<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    point: &RecursiveJoltOpeningPoint,
    modulus: &BigNat<NovaScalar>,
) -> Result<Vec<BigNat<NovaScalar>>, SynthesisError> {
    point
        .coordinates
        .iter()
        .enumerate()
        .map(|(index, coordinate)| {
            alloc_jolt_field(
                cs.namespace(|| format!("opening coordinate {index}")),
                coordinate,
                modulus,
            )
        })
        .collect()
}

fn eq_at_index<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    point: &[BigNat<NovaScalar>],
    index_bits_le: &[AllocatedBit],
    index: usize,
    zero: &BigNat<NovaScalar>,
    one: &BigNat<NovaScalar>,
    modulus: &BigNat<NovaScalar>,
) -> Result<BigNat<NovaScalar>, SynthesisError> {
    if point.len() != index_bits_le.len() {
        return Err(SynthesisError::Unsatisfiable(
            "opening point and index bit dimensions differ".to_string(),
        ));
    }
    let mut weight = one.clone();
    for (coordinate_index, coordinate) in point.iter().enumerate() {
        let bit_position = point.len() - 1 - coordinate_index;
        let bit_value = ((index >> bit_position) & 1) == 1;
        let one_minus_coordinate = one.sub_mod(
            cs.namespace(|| format!("one minus coordinate {coordinate_index}")),
            coordinate,
            modulus,
        )?;
        let factor = select_bignat(
            cs.namespace(|| format!("select eq factor {coordinate_index}")),
            &index_bits_le[bit_position],
            bit_value,
            coordinate,
            &one_minus_coordinate,
        )?;
        let (_, next_weight) = weight.mult_mod(
            cs.namespace(|| format!("multiply eq factor {coordinate_index}")),
            &factor,
            modulus,
        )?;
        weight = next_weight;
    }
    // Keep zero in the signature so both canonical constants are shared by all
    // relation helpers and allocated exactly once by the caller.
    let _ = zero;
    Ok(weight)
}

fn gated_value<CS: ConstraintSystem<NovaScalar>>(
    cs: CS,
    present: &AllocatedBit,
    present_value: bool,
    value: &BigNat<NovaScalar>,
    zero: &BigNat<NovaScalar>,
) -> Result<BigNat<NovaScalar>, SynthesisError> {
    select_bignat(cs, present, present_value, value, zero)
}

fn accumulate_weighted_value<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    accumulator: &BigNat<NovaScalar>,
    weight: &BigNat<NovaScalar>,
    value: &BigNat<NovaScalar>,
    modulus: &BigNat<NovaScalar>,
) -> Result<BigNat<NovaScalar>, SynthesisError> {
    let (_, term) = weight.mult_mod(cs.namespace(|| "weighted value"), value, modulus)?;
    add_mod(
        cs.namespace(|| "accumulate weighted value"),
        accumulator,
        &term,
        modulus,
    )
}

fn enforce_equal<CS: ConstraintSystem<NovaScalar>>(
    cs: CS,
    actual: &BigNat<NovaScalar>,
    expected: &BigNat<NovaScalar>,
) -> Result<(), SynthesisError> {
    actual.equal_when_carried_regroup(cs, expected)
}

fn aggregate_native_claims<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    values: &[BigNat<NovaScalar>],
    target_values: &[BigNat<NovaScalar>],
    challenge_value: &RecursiveJoltFieldElement,
    modulus: &BigNat<NovaScalar>,
    zero: &BigNat<NovaScalar>,
) -> Result<RecursiveNativeClaimContribution, SynthesisError> {
    let challenge = alloc_jolt_field(
        cs.namespace(|| "claim aggregation challenge"),
        challenge_value,
        modulus,
    )?;
    let mut aggregate = zero.clone();
    for (index, value) in values.iter().enumerate() {
        let (_, scaled) = aggregate.mult_mod(
            cs.namespace(|| format!("scale aggregate at claim {index}")),
            &challenge,
            modulus,
        )?;
        aggregate = add_mod(
            cs.namespace(|| format!("absorb claim {index}")),
            &scaled,
            value,
            modulus,
        )?;
    }
    let mut target = zero.clone();
    for (index, value) in target_values.iter().enumerate() {
        let (_, scaled) = target.mult_mod(
            cs.namespace(|| format!("scale target at claim {index}")),
            &challenge,
            modulus,
        )?;
        target = add_mod(
            cs.namespace(|| format!("absorb target claim {index}")),
            &scaled,
            value,
            modulus,
        )?;
    }
    Ok(RecursiveNativeClaimContribution {
        aggregate,
        target,
        challenge,
    })
}

fn packed_bignat_lc<CS: ConstraintSystem<NovaScalar>>(
    value: &BigNat<NovaScalar>,
) -> LinearCombination<NovaScalar> {
    let mut coefficient = NovaScalar::from(1);
    value
        .limbs
        .iter()
        .fold(LinearCombination::zero(), |lc, limb| {
            let current = coefficient;
            for _ in 0..JOLT_FIELD_LIMB_WIDTH {
                coefficient = coefficient + coefficient;
            }
            lc + (current, limb)
        })
}

fn packed_bignat_value(value: &BigNat<NovaScalar>) -> Option<NovaScalar> {
    value.limb_values.as_ref().map(|limbs| {
        let mut coefficient = NovaScalar::from(1);
        limbs.iter().fold(NovaScalar::from(0), |acc, limb| {
            let current = coefficient;
            for _ in 0..JOLT_FIELD_LIMB_WIDTH {
                coefficient = coefficient + coefficient;
            }
            acc + current * limb
        })
    })
}

pub(super) fn accumulate_recursive_native_claim<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    current: &AllocatedNum<NovaScalar>,
    public_challenge: &AllocatedNum<NovaScalar>,
    public_target: &AllocatedNum<NovaScalar>,
    contribution: &RecursiveNativeClaimContribution,
) -> Result<AllocatedNum<NovaScalar>, SynthesisError> {
    let modulus = alloc_jolt_modulus(cs.namespace(|| "Jolt field modulus"))?;
    let current_nat = BigNat::alloc_from_nat(
        cs.namespace(|| "allocate current native claim accumulator"),
        || {
            current
                .get_value()
                .map(|value| BigInt::from_bytes_le(Sign::Plus, &value.to_bytes()))
                .ok_or(SynthesisError::AssignmentMissing)
        },
        JOLT_FIELD_LIMB_WIDTH,
        JOLT_FIELD_LIMBS,
    )?;
    current_nat.assert_well_formed(cs.namespace(|| "current accumulator limbs fit"))?;
    let reduced = current_nat.red_mod(cs.namespace(|| "reduce current accumulator"), &modulus)?;
    enforce_equal(
        cs.namespace(|| "current accumulator is canonical"),
        &current_nat,
        &reduced,
    )?;
    cs.enforce(
        || "current public accumulator packs native limbs",
        |_| packed_bignat_lc::<CS>(&current_nat) - current.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc,
    );
    cs.enforce(
        || "public aggregation challenge packs native limbs",
        |_| packed_bignat_lc::<CS>(&contribution.challenge) - public_challenge.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc,
    );
    cs.enforce(
        || "public closure target packs authenticated native claims",
        |_| packed_bignat_lc::<CS>(&contribution.target) - public_target.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc,
    );
    let next = add_mod(
        cs.namespace(|| "accumulate native block claim"),
        &current_nat,
        &contribution.aggregate,
        &modulus,
    )?;
    let output = AllocatedNum::alloc(cs.namespace(|| "next native claim accumulator"), || {
        packed_bignat_value(&next).ok_or(SynthesisError::AssignmentMissing)
    })?;
    cs.enforce(
        || "next public accumulator packs native limbs",
        |_| packed_bignat_lc::<CS>(&next) - output.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc,
    );
    Ok(output)
}

fn allocate_register_claims<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    register: &RecursiveJoltRegisterOpeningWitness,
    modulus: &BigNat<NovaScalar>,
) -> Result<(), SynthesisError> {
    for (index, claim) in register.value_claims.iter().enumerate() {
        alloc_jolt_field(
            cs.namespace(|| format!("value global claim {index}")),
            claim,
            modulus,
        )?;
    }
    for (index, claim) in register.address_claims.iter().enumerate() {
        alloc_jolt_field(
            cs.namespace(|| format!("address global claim {index}")),
            claim,
            modulus,
        )?;
    }
    alloc_jolt_field(
        cs.namespace(|| "increment global claim"),
        &register.inc_claim,
        modulus,
    )?;
    Ok(())
}

fn allocate_ram_claims<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    ram: &RecursiveJoltRamOpeningWitness,
    modulus: &BigNat<NovaScalar>,
) -> Result<(), SynthesisError> {
    for (index, claim) in ram.ra_claims.iter().enumerate() {
        alloc_jolt_field(
            cs.namespace(|| format!("RamRa global claim {index}")),
            claim,
            modulus,
        )?;
    }
    for (index, claim) in ram.tuple_claims.iter().enumerate() {
        alloc_jolt_field(
            cs.namespace(|| format!("RAM tuple global claim {index}")),
            claim,
            modulus,
        )?;
    }
    alloc_jolt_field(
        cs.namespace(|| "RAM increment global claim"),
        &ram.inc_claim,
        modulus,
    )?;
    Ok(())
}

fn allocate_lookup_claims<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    lookup: &RecursiveJoltLookupOpeningWitness,
    modulus: &BigNat<NovaScalar>,
) -> Result<(), SynthesisError> {
    for (index, claim) in lookup.instruction_claims.iter().enumerate() {
        alloc_jolt_field(
            cs.namespace(|| format!("InstructionRa global claim {index}")),
            claim,
            modulus,
        )?;
    }
    for (index, claim) in lookup.tuple_claims.iter().enumerate() {
        alloc_jolt_field(
            cs.namespace(|| format!("lookup tuple global claim {index}")),
            claim,
            modulus,
        )?;
    }
    Ok(())
}

struct AllocatedRamCycle {
    active: AllocatedBit,
    access: AllocatedBit,
    is_write: AllocatedBit,
    cycle_bits: Vec<AllocatedBit>,
    address: BigNat<NovaScalar>,
    read_value: BigNat<NovaScalar>,
    write_value: BigNat<NovaScalar>,
    offset_bits: Vec<AllocatedBit>,
    offset: usize,
}

/// Verify the complete per-block native Lasso opening decomposition inside
/// the Nova step circuit.
///
/// The relation derives every committed `InstructionRa` chunk from the full
/// lookup index and reconstructs the left-operand, right-operand, and output
/// virtual-polynomial openings from the raw cycle tuple.
pub(super) fn synthesize_recursive_lookup_opening_relation<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    witness: &RecursiveJoltBlockOpeningWitness,
    global_cycle_start: &AllocatedNum<NovaScalar>,
    active_cycles: &AllocatedNum<NovaScalar>,
    opening_present: &AllocatedNum<NovaScalar>,
) -> Result<RecursiveNativeClaimContribution, SynthesisError> {
    witness
        .validate_shape()
        .map_err(|reason| SynthesisError::Unsatisfiable(reason.to_string()))?;

    cs.enforce(
        || "full recursive lookup opening witness requires authenticated opening",
        |lc| lc + opening_present.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc + CS::one(),
    );

    let modulus = alloc_jolt_modulus(cs.namespace(|| "Jolt field modulus"))?;
    let zero = alloc_jolt_constant(cs.namespace(|| "Jolt field zero"), 0)?;
    let one = alloc_jolt_constant(cs.namespace(|| "Jolt field one"), 1)?;
    let lookup = &witness.lookup;
    allocate_lookup_claims(
        cs.namespace(|| "allocate authenticated global lookup claims"),
        lookup,
        &modulus,
    )?;

    let instruction_points = lookup
        .instruction_opening_points
        .iter()
        .enumerate()
        .map(|(index, point)| {
            alloc_opening_point(
                cs.namespace(|| format!("InstructionRa opening point {index}")),
                point,
                &modulus,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let tuple_point = alloc_opening_point(
        cs.namespace(|| "lookup tuple opening point"),
        &lookup.tuple_opening_point,
        &modulus,
    )?;
    if tuple_point.is_empty()
        || lookup.log_k_chunk >= 128
        || instruction_points
            .iter()
            .any(|point| point.len() != lookup.log_k_chunk + tuple_point.len())
        || lookup.log_k_chunk * instruction_points.len() > 128
    {
        return Err(SynthesisError::Unsatisfiable(
            "recursive lookup opening points have inconsistent dimensions".to_string(),
        ));
    }

    let instruction_d = instruction_points.len();
    let chunk_mask = (1u128 << lookup.log_k_chunk) - 1;
    let mut instruction_sums = vec![zero.clone(); instruction_d];
    let mut tuple_sums = [zero.clone(), zero.clone(), zero.clone()];
    let mut active_bits = Vec::with_capacity(witness.cycle_capacity);

    for (local_cycle, cycle) in witness.cycles.iter().enumerate() {
        let active = alloc_boolean(
            cs.namespace(|| format!("cycle {local_cycle} active")),
            cycle.active,
        )?;
        active_bits.push(active.clone());
        let cycle_bits = alloc_global_cycle_bits(
            cs.namespace(|| format!("cycle {local_cycle} global index")),
            global_cycle_start,
            &active,
            cycle.active,
            local_cycle,
            cycle.global_cycle,
            tuple_point.len(),
        )?;
        let tuple_weight = eq_at_index(
            cs.namespace(|| format!("cycle {local_cycle} tuple weight")),
            &tuple_point,
            &cycle_bits,
            cycle.global_cycle,
            &zero,
            &one,
            &modulus,
        )?;
        let (lookup_index, lookup_index_bits) = alloc_u128_jolt_field_with_bits(
            cs.namespace(|| format!("cycle {local_cycle} lookup index")),
            cycle.lookup_index,
        )?;
        enforce_zero_when_disabled(
            cs.namespace(|| format!("cycle {local_cycle} inactive lookup index")),
            &active,
            &lookup_index,
        );

        for (opening_index, point) in instruction_points.iter().enumerate() {
            let (address_point, cycle_point) = point.split_at(lookup.log_k_chunk);
            let shift = lookup.log_k_chunk * (instruction_d - 1 - opening_index);
            let chunk_bits = &lookup_index_bits[shift..shift.saturating_add(lookup.log_k_chunk)];
            let chunk = ((cycle.lookup_index >> shift) & chunk_mask) as usize;
            let address_weight = eq_at_index(
                cs.namespace(|| {
                    format!("cycle {local_cycle} InstructionRa {opening_index} address weight")
                }),
                address_point,
                chunk_bits,
                chunk,
                &zero,
                &one,
                &modulus,
            )?;
            let cycle_weight = eq_at_index(
                cs.namespace(|| {
                    format!("cycle {local_cycle} InstructionRa {opening_index} cycle weight")
                }),
                cycle_point,
                &cycle_bits,
                cycle.global_cycle,
                &zero,
                &one,
                &modulus,
            )?;
            let (_, term) = address_weight.mult_mod(
                cs.namespace(|| format!("cycle {local_cycle} InstructionRa {opening_index} term")),
                &cycle_weight,
                &modulus,
            )?;
            let active_term = gated_value(
                cs.namespace(|| {
                    format!("cycle {local_cycle} gate InstructionRa {opening_index} term")
                }),
                &active,
                cycle.active,
                &term,
                &zero,
            )?;
            instruction_sums[opening_index] = add_mod(
                cs.namespace(|| {
                    format!("cycle {local_cycle} accumulate InstructionRa {opening_index}")
                }),
                &instruction_sums[opening_index],
                &active_term,
                &modulus,
            )?;
        }

        let left = alloc_small_jolt_field(
            cs.namespace(|| format!("cycle {local_cycle} left lookup operand")),
            cycle.left_lookup_operand,
        )?;
        let (right, _) = alloc_u128_jolt_field_with_bits(
            cs.namespace(|| format!("cycle {local_cycle} right lookup operand")),
            cycle.right_lookup_operand,
        )?;
        let output = alloc_small_jolt_field(
            cs.namespace(|| format!("cycle {local_cycle} lookup output")),
            cycle.lookup_output,
        )?;
        for (label, value) in [("left", &left), ("right", &right), ("output", &output)] {
            enforce_zero_when_disabled(
                cs.namespace(|| format!("cycle {local_cycle} inactive {label} lookup value")),
                &active,
                value,
            );
        }
        for (claim_index, value) in [&left, &right, &output].into_iter().enumerate() {
            tuple_sums[claim_index] = accumulate_weighted_value(
                cs.namespace(|| {
                    format!("cycle {local_cycle} accumulate lookup tuple {claim_index}")
                }),
                &tuple_sums[claim_index],
                &tuple_weight,
                value,
                &modulus,
            )?;
        }
    }
    enforce_active_cycle_prefix(
        cs.namespace(|| "lookup active-cycle prefix"),
        &active_bits,
        active_cycles,
    );

    for (opening_index, actual) in instruction_sums.iter().enumerate() {
        let expected = alloc_jolt_field(
            cs.namespace(|| format!("expected InstructionRa contribution {opening_index}")),
            &lookup.instruction_block_contributions[opening_index],
            &modulus,
        )?;
        enforce_equal(
            cs.namespace(|| format!("InstructionRa contribution {opening_index} matches")),
            actual,
            &expected,
        )?;
    }
    for (claim_index, actual) in tuple_sums.iter().enumerate() {
        let expected = alloc_jolt_field(
            cs.namespace(|| format!("expected lookup tuple contribution {claim_index}")),
            &lookup.tuple_block_contributions[claim_index],
            &modulus,
        )?;
        enforce_equal(
            cs.namespace(|| format!("lookup tuple contribution {claim_index} matches")),
            actual,
            &expected,
        )?;
    }
    let challenge = witness
        .claim_aggregation_challenges()
        .map_err(|reason| SynthesisError::Unsatisfiable(reason.to_string()))?[2];
    let aggregate_values = instruction_sums
        .iter()
        .chain(tuple_sums.iter())
        .cloned()
        .collect::<Vec<_>>();
    let mut target_values =
        Vec::with_capacity(lookup.instruction_claims.len() + lookup.tuple_claims.len());
    for (index, (claim, padding)) in lookup
        .instruction_claims
        .iter()
        .zip(lookup.instruction_padding.iter())
        .enumerate()
    {
        let claim = alloc_jolt_field(
            cs.namespace(|| format!("InstructionRa target claim {index}")),
            claim,
            &modulus,
        )?;
        let padding = alloc_jolt_field(
            cs.namespace(|| format!("InstructionRa target padding {index}")),
            padding,
            &modulus,
        )?;
        target_values.push(claim.sub_mod(
            cs.namespace(|| format!("InstructionRa active target {index}")),
            &padding,
            &modulus,
        )?);
    }
    for (index, (claim, padding)) in lookup
        .tuple_claims
        .iter()
        .zip(lookup.tuple_padding.iter())
        .enumerate()
    {
        let claim = alloc_jolt_field(
            cs.namespace(|| format!("lookup tuple target claim {index}")),
            claim,
            &modulus,
        )?;
        let padding = alloc_jolt_field(
            cs.namespace(|| format!("lookup tuple target padding {index}")),
            padding,
            &modulus,
        )?;
        target_values.push(claim.sub_mod(
            cs.namespace(|| format!("lookup tuple active target {index}")),
            &padding,
            &modulus,
        )?);
    }
    aggregate_native_claims(
        cs.namespace(|| "aggregate native lookup block claims"),
        &aggregate_values,
        &target_values,
        &challenge,
        &modulus,
        &zero,
    )
}

/// Verify the complete per-block register opening decomposition inside the
/// Nova step circuit.
///
/// The native Jolt/Dory verifier authenticates the global claims and opening
/// points. This gadget proves that the raw cycles carried by this Nova step
/// produce exactly the block contributions stored in that authenticated
/// receipt. No digest is used as a substitute for native-field arithmetic.
pub(super) fn synthesize_recursive_register_opening_relation<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    witness: &RecursiveJoltBlockOpeningWitness,
    global_cycle_start: &AllocatedNum<NovaScalar>,
    active_cycles: &AllocatedNum<NovaScalar>,
    opening_present: &AllocatedNum<NovaScalar>,
) -> Result<RecursiveNativeClaimContribution, SynthesisError> {
    witness
        .validate_shape()
        .map_err(|reason| SynthesisError::Unsatisfiable(reason.to_string()))?;

    cs.enforce(
        || "full recursive register opening witness requires authenticated opening",
        |lc| lc + opening_present.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc + CS::one(),
    );

    let modulus = alloc_jolt_modulus(cs.namespace(|| "Jolt field modulus"))?;
    let zero = alloc_jolt_constant(cs.namespace(|| "Jolt field zero"), 0)?;
    let one = alloc_jolt_constant(cs.namespace(|| "Jolt field one"), 1)?;
    let register = &witness.register;
    allocate_register_claims(
        cs.namespace(|| "allocate authenticated global register claims"),
        register,
        &modulus,
    )?;

    let value_point = alloc_opening_point(
        cs.namespace(|| "register value opening point"),
        &register.value_opening_point,
        &modulus,
    )?;
    let address_point = alloc_opening_point(
        cs.namespace(|| "register address opening point"),
        &register.address_opening_point,
        &modulus,
    )?;
    let inc_point = alloc_opening_point(
        cs.namespace(|| "register increment opening point"),
        &register.inc_opening_point,
        &modulus,
    )?;
    if address_point.len() != REGISTER_INDEX_BITS + value_point.len()
        || inc_point.len() != value_point.len()
    {
        return Err(SynthesisError::Unsatisfiable(
            "recursive register opening points have inconsistent dimensions".to_string(),
        ));
    }
    let (register_address_point, address_cycle_point) = address_point.split_at(REGISTER_INDEX_BITS);

    let mut value_sums = [zero.clone(), zero.clone(), zero.clone()];
    let mut address_sums = [zero.clone(), zero.clone(), zero.clone()];
    let mut inc_sum = zero.clone();
    let mut active_bits = Vec::with_capacity(witness.cycle_capacity);

    for (local_cycle, cycle) in witness.cycles.iter().enumerate() {
        let active = alloc_boolean(
            cs.namespace(|| format!("cycle {local_cycle} active")),
            cycle.active,
        )?;
        active_bits.push(active.clone());
        let cycle_bits = alloc_global_cycle_bits(
            cs.namespace(|| format!("cycle {local_cycle} global index")),
            global_cycle_start,
            &active,
            cycle.active,
            local_cycle,
            cycle.global_cycle,
            value_point.len(),
        )?;
        let value_weight = eq_at_index(
            cs.namespace(|| format!("cycle {local_cycle} value weight")),
            &value_point,
            &cycle_bits,
            cycle.global_cycle,
            &zero,
            &one,
            &modulus,
        )?;
        let address_cycle_weight = eq_at_index(
            cs.namespace(|| format!("cycle {local_cycle} address-cycle weight")),
            address_cycle_point,
            &cycle_bits,
            cycle.global_cycle,
            &zero,
            &one,
            &modulus,
        )?;
        let inc_weight = eq_at_index(
            cs.namespace(|| format!("cycle {local_cycle} increment weight")),
            &inc_point,
            &cycle_bits,
            cycle.global_cycle,
            &zero,
            &one,
            &modulus,
        )?;

        let accesses = [
            ("rs1", cycle.rs1_present, cycle.rs1_index, cycle.rs1_value),
            ("rs2", cycle.rs2_present, cycle.rs2_index, cycle.rs2_value),
            ("rd", cycle.rd_present, cycle.rd_index, cycle.rd_post_value),
        ];
        for (claim_index, (label, present_value, register_index, value)) in
            accesses.into_iter().enumerate()
        {
            let present = alloc_boolean(
                cs.namespace(|| format!("cycle {local_cycle} {label} present")),
                present_value,
            )?;
            enforce_bit_implies(
                cs.namespace(|| format!("cycle {local_cycle} {label} requires active cycle")),
                &present,
                &active,
            );
            let allocated_value = alloc_small_jolt_field(
                cs.namespace(|| format!("cycle {local_cycle} {label} value")),
                value,
            )?;
            let gated = gated_value(
                cs.namespace(|| format!("cycle {local_cycle} gate {label} value")),
                &present,
                present_value,
                &allocated_value,
                &zero,
            )?;
            value_sums[claim_index] = accumulate_weighted_value(
                cs.namespace(|| format!("cycle {local_cycle} accumulate {label} value")),
                &value_sums[claim_index],
                &value_weight,
                &gated,
                &modulus,
            )?;

            let register_bits = alloc_index_bits(
                cs.namespace(|| format!("cycle {local_cycle} {label} register bits")),
                register_index as usize,
                REGISTER_INDEX_BITS,
            )?;
            let address_weight = eq_at_index(
                cs.namespace(|| format!("cycle {local_cycle} {label} address weight")),
                register_address_point,
                &register_bits,
                register_index as usize,
                &zero,
                &one,
                &modulus,
            )?;
            let gated_address_weight = gated_value(
                cs.namespace(|| format!("cycle {local_cycle} gate {label} address")),
                &present,
                present_value,
                &address_weight,
                &zero,
            )?;
            let (_, address_term) = address_cycle_weight.mult_mod(
                cs.namespace(|| format!("cycle {local_cycle} {label} address term")),
                &gated_address_weight,
                &modulus,
            )?;
            address_sums[claim_index] = add_mod(
                cs.namespace(|| format!("cycle {local_cycle} accumulate {label} address")),
                &address_sums[claim_index],
                &address_term,
                &modulus,
            )?;
        }

        let rd_present = alloc_boolean(
            cs.namespace(|| format!("cycle {local_cycle} rd increment present")),
            cycle.rd_present,
        )?;
        enforce_bit_implies(
            cs.namespace(|| format!("cycle {local_cycle} rd increment requires active cycle")),
            &rd_present,
            &active,
        );
        let rd_pre = alloc_small_jolt_field(
            cs.namespace(|| format!("cycle {local_cycle} rd pre value")),
            cycle.rd_pre_value,
        )?;
        let rd_post = alloc_small_jolt_field(
            cs.namespace(|| format!("cycle {local_cycle} rd post value")),
            cycle.rd_post_value,
        )?;
        let rd_difference = rd_post.sub_mod(
            cs.namespace(|| format!("cycle {local_cycle} rd difference")),
            &rd_pre,
            &modulus,
        )?;
        let gated_difference = gated_value(
            cs.namespace(|| format!("cycle {local_cycle} gate rd difference")),
            &rd_present,
            cycle.rd_present,
            &rd_difference,
            &zero,
        )?;
        inc_sum = accumulate_weighted_value(
            cs.namespace(|| format!("cycle {local_cycle} accumulate rd increment")),
            &inc_sum,
            &inc_weight,
            &gated_difference,
            &modulus,
        )?;
    }
    enforce_active_cycle_prefix(
        cs.namespace(|| "register active-cycle prefix"),
        &active_bits,
        active_cycles,
    );

    for claim_index in 0..3 {
        let expected_value = alloc_jolt_field(
            cs.namespace(|| format!("expected value contribution {claim_index}")),
            &register.value_block_contributions[claim_index],
            &modulus,
        )?;
        enforce_equal(
            cs.namespace(|| format!("register value contribution {claim_index} matches")),
            &value_sums[claim_index],
            &expected_value,
        )?;
        let expected_address = alloc_jolt_field(
            cs.namespace(|| format!("expected address contribution {claim_index}")),
            &register.address_block_contributions[claim_index],
            &modulus,
        )?;
        enforce_equal(
            cs.namespace(|| format!("register address contribution {claim_index} matches")),
            &address_sums[claim_index],
            &expected_address,
        )?;
    }
    let expected_inc = alloc_jolt_field(
        cs.namespace(|| "expected increment contribution"),
        &register.inc_block_contribution,
        &modulus,
    )?;
    enforce_equal(
        cs.namespace(|| "register increment contribution matches"),
        &inc_sum,
        &expected_inc,
    )?;
    let challenge = witness
        .claim_aggregation_challenges()
        .map_err(|reason| SynthesisError::Unsatisfiable(reason.to_string()))?[0];
    let aggregate_values = value_sums
        .iter()
        .chain(address_sums.iter())
        .chain(core::iter::once(&inc_sum))
        .cloned()
        .collect::<Vec<_>>();
    let target_values = register
        .value_claims
        .iter()
        .chain(register.address_claims.iter())
        .chain(core::iter::once(&register.inc_claim))
        .enumerate()
        .map(|(index, claim)| {
            alloc_jolt_field(
                cs.namespace(|| format!("register closure target claim {index}")),
                claim,
                &modulus,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    aggregate_native_claims(
        cs.namespace(|| "aggregate native register block claims"),
        &aggregate_values,
        &target_values,
        &challenge,
        &modulus,
        &zero,
    )
}

/// Verify the complete per-block RAM opening decomposition inside the Nova
/// step circuit.
///
/// This relation derives the RamRa chunks from constrained, aligned RAM
/// addresses and separately reconstructs the RAM tuple and RamInc openings.
/// The native Jolt verifier remains responsible for authenticating the global
/// Dory claims; Nova verifies that this block's raw RAM accesses produce the
/// exact authenticated block contributions.
pub(super) fn synthesize_recursive_ram_opening_relation<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    witness: &RecursiveJoltBlockOpeningWitness,
    global_cycle_start: &AllocatedNum<NovaScalar>,
    active_cycles: &AllocatedNum<NovaScalar>,
    opening_present: &AllocatedNum<NovaScalar>,
) -> Result<RecursiveNativeClaimContribution, SynthesisError> {
    witness
        .validate_shape()
        .map_err(|reason| SynthesisError::Unsatisfiable(reason.to_string()))?;

    cs.enforce(
        || "full recursive RAM opening witness requires authenticated opening",
        |lc| lc + opening_present.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc + CS::one(),
    );

    let ram = &witness.ram;
    if ram.ram_k == 0
        || !ram.ram_k.is_power_of_two()
        || ram.log_k_chunk == 0
        || ram.log_k_chunk >= usize::BITS as usize
    {
        return Err(SynthesisError::Unsatisfiable(
            "recursive RAM opening has invalid one-hot parameters".to_string(),
        ));
    }
    let log_ram_k = ram.ram_k.ilog2() as usize;
    let ram_d = log_ram_k.div_ceil(ram.log_k_chunk);
    let cycle_bits_len = ram.tuple_opening_point.coordinates.len();
    if ram_d != ram.ra_opening_points.len()
        || ram.ra_claims.len() != ram_d
        || ram.ra_block_contributions.len() != ram_d
        || ram.inc_opening_point.coordinates.len() != cycle_bits_len
        || ram
            .ra_opening_points
            .iter()
            .any(|point| point.coordinates.len() != ram.log_k_chunk + cycle_bits_len)
    {
        return Err(SynthesisError::Unsatisfiable(
            "recursive RAM opening points have inconsistent dimensions".to_string(),
        ));
    }

    let modulus = alloc_jolt_modulus(cs.namespace(|| "Jolt field modulus"))?;
    let zero = alloc_jolt_constant(cs.namespace(|| "Jolt field zero"), 0)?;
    let one = alloc_jolt_constant(cs.namespace(|| "Jolt field one"), 1)?;
    allocate_ram_claims(
        cs.namespace(|| "allocate authenticated global RAM claims"),
        ram,
        &modulus,
    )?;

    let tuple_point = alloc_opening_point(
        cs.namespace(|| "RAM tuple opening point"),
        &ram.tuple_opening_point,
        &modulus,
    )?;
    let inc_point = alloc_opening_point(
        cs.namespace(|| "RAM increment opening point"),
        &ram.inc_opening_point,
        &modulus,
    )?;
    let ra_points = ram
        .ra_opening_points
        .iter()
        .enumerate()
        .map(|(opening_index, point)| {
            alloc_opening_point(
                cs.namespace(|| format!("RamRa opening point {opening_index}")),
                point,
                &modulus,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;

    let mut allocated_cycles = Vec::with_capacity(witness.cycle_capacity);
    for (local_cycle, cycle) in witness.cycles.iter().enumerate() {
        let active = alloc_boolean(
            cs.namespace(|| format!("RAM cycle {local_cycle} active")),
            cycle.active,
        )?;
        let cycle_bits = alloc_global_cycle_bits(
            cs.namespace(|| format!("RAM cycle {local_cycle} global index")),
            global_cycle_start,
            &active,
            cycle.active,
            local_cycle,
            cycle.global_cycle,
            cycle_bits_len,
        )?;

        let (is_read_value, is_write_value) = match cycle.ram_kind {
            0 => (false, false),
            1 => (true, false),
            2 => (false, true),
            _ => {
                return Err(SynthesisError::Unsatisfiable(
                    "recursive RAM access kind is not NoOp, Read, or Write".to_string(),
                ));
            }
        };
        let access_value = is_read_value || is_write_value;
        let is_read = alloc_boolean(
            cs.namespace(|| format!("RAM cycle {local_cycle} is read")),
            is_read_value,
        )?;
        let is_write = alloc_boolean(
            cs.namespace(|| format!("RAM cycle {local_cycle} is write")),
            is_write_value,
        )?;
        let access = alloc_boolean(
            cs.namespace(|| format!("RAM cycle {local_cycle} has access")),
            access_value,
        )?;
        cs.enforce(
            || format!("RAM cycle {local_cycle} access kind is exclusive"),
            |lc| lc + is_read.get_variable() + is_write.get_variable() - access.get_variable(),
            |lc| lc + CS::one(),
            |lc| lc,
        );
        enforce_bit_implies(
            cs.namespace(|| format!("RAM cycle {local_cycle} access requires active cycle")),
            &access,
            &active,
        );

        let (address, _) = alloc_u64_jolt_field_with_bits(
            cs.namespace(|| format!("RAM cycle {local_cycle} address")),
            cycle.ram_address,
        )?;
        let (read_value, _) = alloc_u64_jolt_field_with_bits(
            cs.namespace(|| format!("RAM cycle {local_cycle} read value")),
            cycle.ram_read_value,
        )?;
        let (write_value, _) = alloc_u64_jolt_field_with_bits(
            cs.namespace(|| format!("RAM cycle {local_cycle} write value")),
            cycle.ram_write_value,
        )?;
        enforce_zero_when_disabled(
            cs.namespace(|| format!("RAM cycle {local_cycle} NoOp address is zero")),
            &access,
            &address,
        );
        enforce_zero_when_disabled(
            cs.namespace(|| format!("RAM cycle {local_cycle} NoOp read value is zero")),
            &access,
            &read_value,
        );
        enforce_zero_when_disabled(
            cs.namespace(|| format!("RAM cycle {local_cycle} NoOp write value is zero")),
            &access,
            &write_value,
        );
        for limb_index in 0..JOLT_FIELD_LIMBS {
            cs.enforce(
                || {
                    format!(
                        "RAM cycle {local_cycle} read mirrors its write tuple limb {limb_index}"
                    )
                },
                |lc| lc + &read_value.limbs[limb_index] - &write_value.limbs[limb_index],
                |lc| lc + is_read.get_variable(),
                |lc| lc,
            );
        }

        let offset_u64 = if access_value {
            let address_offset = cycle
                .ram_address
                .checked_sub(ram.ram_start_address)
                .ok_or_else(|| {
                    SynthesisError::Unsatisfiable(
                        "recursive RAM address is below the authenticated RAM start".to_string(),
                    )
                })?;
            if address_offset % 8 != 0 {
                return Err(SynthesisError::Unsatisfiable(
                    "recursive RAM address is not word aligned".to_string(),
                ));
            }
            let offset = address_offset / 8;
            if offset >= ram.ram_k as u64 {
                return Err(SynthesisError::Unsatisfiable(
                    "recursive RAM address exceeds the authenticated RAM domain".to_string(),
                ));
            }
            offset
        } else {
            0
        };
        let offset = usize::try_from(offset_u64).map_err(|_| {
            SynthesisError::Unsatisfiable(
                "recursive RAM word offset does not fit the host index".to_string(),
            )
        })?;
        let offset_bits = alloc_index_bits(
            cs.namespace(|| format!("RAM cycle {local_cycle} remapped address bits")),
            offset,
            log_ram_k,
        )?;
        let mut address_relation = LinearCombination::<NovaScalar>::zero() + &address.limbs[0]
            - (
                NovaScalar::from(ram.ram_start_address),
                access.get_variable(),
            );
        let mut coefficient = NovaScalar::from(8);
        for bit in &offset_bits {
            address_relation = address_relation - (coefficient, bit.get_variable());
            coefficient = coefficient + coefficient;
        }
        cs.enforce(
            || format!("RAM cycle {local_cycle} address is aligned remapped word"),
            |_| address_relation,
            |lc| lc + CS::one(),
            |lc| lc,
        );

        allocated_cycles.push(AllocatedRamCycle {
            active,
            access,
            is_write,
            cycle_bits,
            address,
            read_value,
            write_value,
            offset_bits,
            offset,
        });
    }
    enforce_active_cycle_prefix(
        cs.namespace(|| "RAM active-cycle prefix"),
        &allocated_cycles
            .iter()
            .map(|cycle| cycle.active.clone())
            .collect::<Vec<_>>(),
        active_cycles,
    );

    let mut tuple_sums = [zero.clone(), zero.clone(), zero.clone()];
    let mut inc_sum = zero.clone();
    for (local_cycle, allocated) in allocated_cycles.iter().enumerate() {
        let cycle = &witness.cycles[local_cycle];
        let tuple_weight = eq_at_index(
            cs.namespace(|| format!("RAM cycle {local_cycle} tuple weight")),
            &tuple_point,
            &allocated.cycle_bits,
            cycle.global_cycle,
            &zero,
            &one,
            &modulus,
        )?;
        for (claim_index, value) in [
            &allocated.address,
            &allocated.read_value,
            &allocated.write_value,
        ]
        .into_iter()
        .enumerate()
        {
            let gated = gated_value(
                cs.namespace(|| format!("RAM cycle {local_cycle} gate tuple value {claim_index}")),
                &allocated.access,
                cycle.ram_kind != 0,
                value,
                &zero,
            )?;
            tuple_sums[claim_index] = accumulate_weighted_value(
                cs.namespace(|| {
                    format!("RAM cycle {local_cycle} accumulate tuple value {claim_index}")
                }),
                &tuple_sums[claim_index],
                &tuple_weight,
                &gated,
                &modulus,
            )?;
        }

        let inc_weight = eq_at_index(
            cs.namespace(|| format!("RAM cycle {local_cycle} increment weight")),
            &inc_point,
            &allocated.cycle_bits,
            cycle.global_cycle,
            &zero,
            &one,
            &modulus,
        )?;
        let difference = allocated.write_value.sub_mod(
            cs.namespace(|| format!("RAM cycle {local_cycle} write difference")),
            &allocated.read_value,
            &modulus,
        )?;
        let gated_difference = gated_value(
            cs.namespace(|| format!("RAM cycle {local_cycle} gate write difference")),
            &allocated.is_write,
            cycle.ram_kind == 2,
            &difference,
            &zero,
        )?;
        inc_sum = accumulate_weighted_value(
            cs.namespace(|| format!("RAM cycle {local_cycle} accumulate increment")),
            &inc_sum,
            &inc_weight,
            &gated_difference,
            &modulus,
        )?;
    }

    let mut ra_sums = vec![zero.clone(); ram_d];
    for (opening_index, point) in ra_points.iter().enumerate() {
        let (address_point, cycle_point) = point.split_at(ram.log_k_chunk);
        let shift = ram.log_k_chunk * (ram_d - 1 - opening_index);
        for (local_cycle, allocated) in allocated_cycles.iter().enumerate() {
            let cycle = &witness.cycles[local_cycle];
            let chunk =
                (allocated.offset >> shift) & ((1usize << ram.log_k_chunk).saturating_sub(1));
            let mut chunk_bits = Vec::with_capacity(ram.log_k_chunk);
            for chunk_bit in 0..ram.log_k_chunk {
                if let Some(offset_bit) = allocated.offset_bits.get(shift + chunk_bit) {
                    chunk_bits.push(offset_bit.clone());
                } else {
                    chunk_bits.push(alloc_boolean(
                        cs.namespace(|| {
                            format!(
                                "RamRa opening {opening_index} cycle {local_cycle} padded chunk bit {chunk_bit}"
                            )
                        }),
                        false,
                    )?);
                }
            }
            let address_weight = eq_at_index(
                cs.namespace(|| {
                    format!("RamRa opening {opening_index} cycle {local_cycle} address weight")
                }),
                address_point,
                &chunk_bits,
                chunk,
                &zero,
                &one,
                &modulus,
            )?;
            let cycle_weight = eq_at_index(
                cs.namespace(|| {
                    format!("RamRa opening {opening_index} cycle {local_cycle} cycle weight")
                }),
                cycle_point,
                &allocated.cycle_bits,
                cycle.global_cycle,
                &zero,
                &one,
                &modulus,
            )?;
            let gated_address_weight = gated_value(
                cs.namespace(|| {
                    format!("RamRa opening {opening_index} cycle {local_cycle} gate access")
                }),
                &allocated.access,
                cycle.ram_kind != 0,
                &address_weight,
                &zero,
            )?;
            let (_, term) = cycle_weight.mult_mod(
                cs.namespace(|| format!("RamRa opening {opening_index} cycle {local_cycle} term")),
                &gated_address_weight,
                &modulus,
            )?;
            ra_sums[opening_index] = add_mod(
                cs.namespace(|| {
                    format!("RamRa opening {opening_index} cycle {local_cycle} accumulate")
                }),
                &ra_sums[opening_index],
                &term,
                &modulus,
            )?;
        }
    }

    for (opening_index, actual) in ra_sums.iter().enumerate() {
        let expected = alloc_jolt_field(
            cs.namespace(|| format!("expected RamRa contribution {opening_index}")),
            &ram.ra_block_contributions[opening_index],
            &modulus,
        )?;
        enforce_equal(
            cs.namespace(|| format!("RamRa contribution {opening_index} matches")),
            actual,
            &expected,
        )?;
    }
    for (claim_index, actual) in tuple_sums.iter().enumerate() {
        let expected = alloc_jolt_field(
            cs.namespace(|| format!("expected RAM tuple contribution {claim_index}")),
            &ram.tuple_block_contributions[claim_index],
            &modulus,
        )?;
        enforce_equal(
            cs.namespace(|| format!("RAM tuple contribution {claim_index} matches")),
            actual,
            &expected,
        )?;
    }
    let expected_inc = alloc_jolt_field(
        cs.namespace(|| "expected RAM increment contribution"),
        &ram.inc_block_contribution,
        &modulus,
    )?;
    enforce_equal(
        cs.namespace(|| "RAM increment contribution matches"),
        &inc_sum,
        &expected_inc,
    )?;
    let challenge = witness
        .claim_aggregation_challenges()
        .map_err(|reason| SynthesisError::Unsatisfiable(reason.to_string()))?[1];
    let aggregate_values = ra_sums
        .iter()
        .chain(tuple_sums.iter())
        .chain(core::iter::once(&inc_sum))
        .cloned()
        .collect::<Vec<_>>();
    let target_values = ram
        .ra_claims
        .iter()
        .chain(ram.tuple_claims.iter())
        .chain(core::iter::once(&ram.inc_claim))
        .enumerate()
        .map(|(index, claim)| {
            alloc_jolt_field(
                cs.namespace(|| format!("RAM closure target claim {index}")),
                claim,
                &modulus,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    aggregate_native_claims(
        cs.namespace(|| "aggregate native RAM block claims"),
        &aggregate_values,
        &target_values,
        &challenge,
        &modulus,
        &zero,
    )
}

/// Verify the native Jolt CPU/Spartan opening and all uniform CPU R1CS rows
/// inside one Nova step.
///
/// Every one of the 35 CPU inputs is range-bound to a signed 128-bit integer,
/// mapped losslessly into the BN254 scalar field for opening arithmetic, and
/// simultaneously represented in Nova's scalar field for the uniform R1CS
/// constraints. The active-cycle selector gates padded fixed-shape rows.
pub(super) fn synthesize_recursive_cpu_opening_relation<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    witness: &RecursiveJoltBlockOpeningWitness,
    global_cycle_start: &AllocatedNum<NovaScalar>,
    active_cycles: &AllocatedNum<NovaScalar>,
    opening_present: &AllocatedNum<NovaScalar>,
) -> Result<RecursiveNativeClaimContribution, SynthesisError> {
    witness
        .validate_shape()
        .map_err(|reason| SynthesisError::Unsatisfiable(reason.to_string()))?;

    cs.enforce(
        || "full recursive CPU opening witness requires authenticated opening",
        |lc| lc + opening_present.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc + CS::one(),
    );

    let cpu = &witness.cpu;
    if cpu.claims.len() != 35
        || cpu.block_contributions.len() != cpu.claims.len()
        || cpu.opening_point.coordinates.is_empty()
    {
        return Err(SynthesisError::Unsatisfiable(
            "recursive CPU opening has inconsistent dimensions".to_string(),
        ));
    }

    let modulus = alloc_jolt_modulus(cs.namespace(|| "Jolt field modulus"))?;
    let zero = alloc_jolt_constant(cs.namespace(|| "Jolt field zero"), 0)?;
    let one = alloc_jolt_constant(cs.namespace(|| "Jolt field one"), 1)?;
    for (claim_index, claim) in cpu.claims.iter().enumerate() {
        alloc_jolt_field(
            cs.namespace(|| format!("CPU global claim {claim_index}")),
            claim,
            &modulus,
        )?;
    }
    let opening_point = alloc_opening_point(
        cs.namespace(|| "CPU opening point"),
        &cpu.opening_point,
        &modulus,
    )?;

    // Product-virtualization outputs and instruction flags are boolean in the
    // native Jolt relation. Enforcing them here prevents non-boolean guards
    // from satisfying a conditional row through field cancellation.
    const CPU_BOOLEAN_INPUTS: [usize; 18] = [
        3, 17, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
    ];

    let mut opening_sums = vec![zero.clone(); cpu.claims.len()];
    let mut active_bits = Vec::with_capacity(witness.cycle_capacity);
    for (local_cycle, cycle) in witness.cycles.iter().enumerate() {
        let active = alloc_boolean(
            cs.namespace(|| format!("CPU cycle {local_cycle} active")),
            cycle.active,
        )?;
        active_bits.push(active.clone());
        let cycle_bits = alloc_global_cycle_bits(
            cs.namespace(|| format!("CPU cycle {local_cycle} global index")),
            global_cycle_start,
            &active,
            cycle.active,
            local_cycle,
            cycle.global_cycle,
            opening_point.len(),
        )?;
        let opening_weight = eq_at_index(
            cs.namespace(|| format!("CPU cycle {local_cycle} opening weight")),
            &opening_point,
            &cycle_bits,
            cycle.global_cycle,
            &zero,
            &one,
            &modulus,
        )?;

        let mut native_inputs = Vec::with_capacity(cpu.claims.len());
        for (input_index, value) in cycle.cpu_r1cs_inputs.iter().copied().enumerate() {
            let (native, mapped) = alloc_signed_jolt_input(
                cs.namespace(|| format!("CPU cycle {local_cycle} input {input_index}")),
                value,
                &zero,
                &modulus,
            )?;
            enforce_native_zero_when_disabled(
                cs.namespace(|| {
                    format!("CPU cycle {local_cycle} inactive native input {input_index}")
                }),
                &active,
                &native,
            );
            enforce_zero_when_disabled(
                cs.namespace(|| {
                    format!("CPU cycle {local_cycle} inactive Jolt input {input_index}")
                }),
                &active,
                &mapped,
            );
            if CPU_BOOLEAN_INPUTS.contains(&input_index) {
                enforce_boolean_num(
                    cs.namespace(|| format!("CPU cycle {local_cycle} boolean input {input_index}")),
                    &native,
                );
            }

            let gated = gated_value(
                cs.namespace(|| {
                    format!("CPU cycle {local_cycle} gate opening input {input_index}")
                }),
                &active,
                cycle.active,
                &mapped,
                &zero,
            )?;
            opening_sums[input_index] = accumulate_weighted_value(
                cs.namespace(|| {
                    format!("CPU cycle {local_cycle} accumulate opening input {input_index}")
                }),
                &opening_sums[input_index],
                &opening_weight,
                &gated,
                &modulus,
            )?;
            native_inputs.push(native);
        }

        for (constraint_index, named_constraint) in R1CS_CONSTRAINTS.iter().enumerate() {
            let a = jolt_lc_to_nova::<CS>(&named_constraint.cons.a, &native_inputs)?;
            let b = jolt_lc_to_nova::<CS>(&named_constraint.cons.b, &native_inputs)?;
            let a_value = evaluate_jolt_lc(&named_constraint.cons.a, &cycle.cpu_r1cs_inputs)?;
            let gated_a = AllocatedNum::alloc(
                cs.namespace(|| {
                    format!("CPU cycle {local_cycle} constraint {constraint_index} gated A")
                }),
                || {
                    Ok(if cycle.active {
                        nova_scalar_from_i128(a_value)
                    } else {
                        NovaScalar::from(0)
                    })
                },
            )?;
            cs.enforce(
                || {
                    format!(
                        "CPU cycle {local_cycle} constraint {:?} active gate",
                        named_constraint.label
                    )
                },
                |_| a,
                |lc| lc + active.get_variable(),
                |lc| lc + gated_a.get_variable(),
            );
            cs.enforce(
                || {
                    format!(
                        "CPU cycle {local_cycle} constraint {:?}",
                        named_constraint.label
                    )
                },
                |lc| lc + gated_a.get_variable(),
                |_| b,
                |lc| lc,
            );
        }
    }
    enforce_active_cycle_prefix(
        cs.namespace(|| "CPU active-cycle prefix"),
        &active_bits,
        active_cycles,
    );

    for (claim_index, actual) in opening_sums.iter().enumerate() {
        let expected = alloc_jolt_field(
            cs.namespace(|| format!("expected CPU contribution {claim_index}")),
            &cpu.block_contributions[claim_index],
            &modulus,
        )?;
        enforce_equal(
            cs.namespace(|| format!("CPU contribution {claim_index} matches")),
            actual,
            &expected,
        )?;
    }
    let challenge = witness
        .claim_aggregation_challenges()
        .map_err(|reason| SynthesisError::Unsatisfiable(reason.to_string()))?[3];
    let mut target_values = Vec::with_capacity(cpu.claims.len());
    for (index, (claim, padding)) in cpu.claims.iter().zip(cpu.padding.iter()).enumerate() {
        let claim = alloc_jolt_field(
            cs.namespace(|| format!("CPU closure target claim {index}")),
            claim,
            &modulus,
        )?;
        let padding = alloc_jolt_field(
            cs.namespace(|| format!("CPU closure target padding {index}")),
            padding,
            &modulus,
        )?;
        target_values.push(claim.sub_mod(
            cs.namespace(|| format!("CPU active closure target {index}")),
            &padding,
            &modulus,
        )?);
    }
    aggregate_native_claims(
        cs.namespace(|| "aggregate native CPU block claims"),
        &opening_sums,
        &target_values,
        &challenge,
        &modulus,
        &zero,
    )
}

#[cfg(test)]
mod recursive_sumcheck_tests {
    use ark_bn254::Fr;
    use ark_ff::Field;
    use ark_std::One;
    use nova_snark::frontend::test_cs::TestConstraintSystem;

    use crate::transcripts::{PoseidonTranscript, Transcript};

    use super::*;
    use crate::zkvm::block::RecursiveClearSumcheckRoundWitness;

    fn field(value: u64) -> RecursiveJoltFieldElement {
        RecursiveJoltFieldElement::from_field(Fr::from(value)).unwrap()
    }

    fn valid_witness() -> RecursiveClearSumcheckStageWitness {
        // Round 0: h(X) = 2 + 3X + 3X^2, h(0)+h(1)=10,
        // h(4)=62. Round 1: h(X)=5+52X, h(0)+h(1)=62,
        // h(3)=161.
        RecursiveClearSumcheckStageWitness {
            stage_index: 3,
            degree_bound: 2,
            initial_claim: field(10),
            rounds: vec![
                RecursiveClearSumcheckRoundWitness {
                    coefficients_except_linear: vec![field(2), field(3)],
                    challenge: field(4),
                },
                RecursiveClearSumcheckRoundWitness {
                    coefficients_except_linear: vec![field(5)],
                    challenge: field(3),
                },
            ],
            expected_final_claim: field(161),
        }
    }

    fn transcript_bound_witness(
        tamper_first_challenge: bool,
    ) -> (RecursiveClearSumcheckStageWitness, PoseidonTranscript) {
        let mut transcript = PoseidonTranscript::new(b"stage14-sc");
        let first_coefficients = [Fr::from(2u64), Fr::from(3u64)];
        transcript.append_scalars(b"sumcheck_poly", &first_coefficients);
        let native_first_challenge = transcript.challenge_scalar::<Fr>();
        let first_challenge = if tamper_first_challenge {
            native_first_challenge + Fr::one()
        } else {
            native_first_challenge
        };
        let first_claim = Fr::from(2u64)
            + Fr::from(3u64) * first_challenge
            + Fr::from(3u64) * first_challenge.square();

        let second_coefficients = [Fr::from(5u64)];
        transcript.append_scalars(b"sumcheck_poly", &second_coefficients);
        let second_challenge = transcript.challenge_scalar::<Fr>();
        let second_linear = first_claim - Fr::from(10u64);
        let final_claim = Fr::from(5u64) + second_linear * second_challenge;

        (
            RecursiveClearSumcheckStageWitness {
                stage_index: 0,
                degree_bound: 2,
                initial_claim: RecursiveJoltFieldElement::from_field(Fr::from(10u64)).unwrap(),
                rounds: vec![
                    RecursiveClearSumcheckRoundWitness {
                        coefficients_except_linear: first_coefficients
                            .iter()
                            .map(|value| RecursiveJoltFieldElement::from_field(*value).unwrap())
                            .collect(),
                        challenge: RecursiveJoltFieldElement::from_field(first_challenge).unwrap(),
                    },
                    RecursiveClearSumcheckRoundWitness {
                        coefficients_except_linear: second_coefficients
                            .iter()
                            .map(|value| RecursiveJoltFieldElement::from_field(*value).unwrap())
                            .collect(),
                        challenge: RecursiveJoltFieldElement::from_field(second_challenge).unwrap(),
                    },
                ],
                expected_final_claim: RecursiveJoltFieldElement::from_field(final_claim).unwrap(),
            },
            transcript,
        )
    }

    fn alloc_initial_transcript(
        cs: &mut TestConstraintSystem<NovaScalar>,
    ) -> AllocatedRecursivePoseidonTranscriptState {
        let transcript = PoseidonTranscript::new(b"stage14-sc");
        AllocatedRecursivePoseidonTranscriptState {
            state: AllocatedNum::alloc(cs.namespace(|| "initial transcript state"), || {
                Ok(Option::from(NovaScalar::from_bytes(&transcript.state)).unwrap())
            })
            .unwrap(),
            n_rounds: AllocatedNum::alloc(cs.namespace(|| "initial transcript round"), || {
                Ok(NovaScalar::zero())
            })
            .unwrap(),
        }
    }

    #[test]
    fn recursive_clear_sumcheck_gadget_checks_every_round_and_endpoint() {
        let mut cs = TestConstraintSystem::<NovaScalar>::new();
        let allocated = synthesize_recursive_clear_sumcheck_stage(
            cs.namespace(|| "clear sumcheck"),
            &valid_witness(),
            3,
            2,
            2,
        )
        .unwrap();

        assert_eq!(allocated.round_coefficients.len(), 2);
        assert_eq!(allocated.challenges.len(), 2);
        assert_eq!(
            allocated.initial_claim.get_value(),
            Some(NovaScalar::from(10))
        );
        assert_eq!(
            allocated.final_claim.get_value(),
            Some(NovaScalar::from(161))
        );
        assert_eq!(
            allocated.expected_final_claim.get_value(),
            Some(NovaScalar::from(161))
        );
        assert!(cs.is_satisfied(), "{:?}", cs.which_is_unsatisfied());
    }

    #[test]
    fn recursive_clear_sumcheck_gadget_rejects_tampered_final_claim() {
        let mut witness = valid_witness();
        witness.expected_final_claim = field(162);
        let mut cs = TestConstraintSystem::<NovaScalar>::new();
        synthesize_recursive_clear_sumcheck_stage(
            cs.namespace(|| "tampered clear sumcheck"),
            &witness,
            3,
            2,
            2,
        )
        .unwrap();

        assert!(!cs.is_satisfied());
        assert!(cs
            .which_is_unsatisfied()
            .unwrap_or_default()
            .contains("sumcheck final claim equals stage relation claim"));
    }

    #[test]
    fn recursive_clear_sumcheck_gadget_rejects_unexpected_shape() {
        let mut cs = TestConstraintSystem::<NovaScalar>::new();
        let error = synthesize_recursive_clear_sumcheck_stage(
            cs.namespace(|| "wrong-shape clear sumcheck"),
            &valid_witness(),
            3,
            3,
            2,
        )
        .err()
        .expect("wrong round count must be rejected");
        assert!(matches!(error, SynthesisError::Unsatisfiable(_)));
    }

    #[test]
    fn recursive_clear_sumcheck_binds_every_challenge_to_poseidon_transcript() {
        let (witness, native_final_transcript) = transcript_bound_witness(false);
        let mut cs = TestConstraintSystem::<NovaScalar>::new();
        let initial_transcript = alloc_initial_transcript(&mut cs);
        let allocated = synthesize_recursive_clear_sumcheck_stage(
            cs.namespace(|| "clear sumcheck arithmetic"),
            &witness,
            0,
            2,
            2,
        )
        .unwrap();
        let final_transcript = synthesize_recursive_clear_sumcheck_transcript(
            cs.namespace(|| "clear sumcheck Fiat-Shamir"),
            &initial_transcript,
            &allocated,
        )
        .unwrap();

        assert_eq!(
            final_transcript.state.get_value(),
            Some(Option::from(NovaScalar::from_bytes(&native_final_transcript.state)).unwrap())
        );
        assert_eq!(
            final_transcript.n_rounds.get_value(),
            Some(NovaScalar::from(native_final_transcript.n_rounds as u64))
        );
        assert!(cs.is_satisfied(), "{:?}", cs.which_is_unsatisfied());
    }

    #[test]
    fn recursive_clear_sumcheck_rejects_forged_challenge_even_when_arithmetic_is_valid() {
        let (witness, _) = transcript_bound_witness(true);
        let mut cs = TestConstraintSystem::<NovaScalar>::new();
        let initial_transcript = alloc_initial_transcript(&mut cs);
        let allocated = synthesize_recursive_clear_sumcheck_stage(
            cs.namespace(|| "clear sumcheck arithmetic"),
            &witness,
            0,
            2,
            2,
        )
        .unwrap();
        synthesize_recursive_clear_sumcheck_transcript(
            cs.namespace(|| "clear sumcheck Fiat-Shamir"),
            &initial_transcript,
            &allocated,
        )
        .unwrap();

        assert!(!cs.is_satisfied());
        assert!(cs
            .which_is_unsatisfied()
            .unwrap_or_default()
            .contains("claimed sumcheck challenge equals Poseidon transcript challenge"));
    }
}

#[cfg(test)]
mod recursive_poseidon_transcript_tests {
    use ark_bn254::Fr;
    use ark_ff::PrimeField;
    use nova_snark::frontend::test_cs::TestConstraintSystem;

    use crate::transcripts::{PoseidonTranscript, Transcript};

    use super::*;

    fn nova_scalar_from_le_bytes(bytes: &[u8; 32]) -> NovaScalar {
        Option::from(NovaScalar::from_bytes(bytes)).expect("canonical BN254 scalar")
    }

    fn alloc_value(
        cs: &mut TestConstraintSystem<NovaScalar>,
        name: &'static str,
        value: NovaScalar,
    ) -> AllocatedNum<NovaScalar> {
        AllocatedNum::alloc(cs.namespace(|| name), || Ok(value)).unwrap()
    }

    #[test]
    fn recursive_poseidon_hash_matches_jolt_native_poseidon() {
        use light_poseidon::{Poseidon, PoseidonHasher};

        let state_ark = Fr::from(7u64);
        let round_ark = Fr::from(11u64);
        let data_ark = Fr::from(13u64);
        let expected_ark = Poseidon::<Fr>::new_circom(3)
            .unwrap()
            .hash(&[state_ark, round_ark, data_ark])
            .unwrap();
        let expected = ark_bn254_scalar_as_nova_scalar(&expected_ark).unwrap();

        let mut cs = TestConstraintSystem::<NovaScalar>::new();
        let state = alloc_value(&mut cs, "state", NovaScalar::from(7));
        let n_rounds = alloc_value(&mut cs, "round", NovaScalar::from(11));
        let data = alloc_value(&mut cs, "data", NovaScalar::from(13));
        let actual = synthesize_recursive_poseidon_hash(
            cs.namespace(|| "recursive Poseidon"),
            &state,
            &n_rounds,
            &data,
        )
        .unwrap();

        assert_eq!(actual.get_value(), Some(expected));
        assert!(cs.is_satisfied(), "{:?}", cs.which_is_unsatisfied());
    }

    #[test]
    fn recursive_poseidon_transcript_matches_jolt_sumcheck_absorption_and_challenge() {
        let mut native = PoseidonTranscript::new(b"stage14-sumcheck");
        let initial_state = nova_scalar_from_le_bytes(&native.state);
        let coefficients = [Fr::from(7u64), Fr::from(9u64)];
        native.append_scalars(b"sumcheck_poly", &coefficients);
        let native_challenge = native.challenge_scalar::<Fr>();
        let expected_challenge = ark_bn254_scalar_as_nova_scalar(&native_challenge).unwrap();
        let expected_state = nova_scalar_from_le_bytes(&native.state);
        assert_eq!(expected_challenge, expected_state);
        assert_eq!(native.n_rounds, 4);

        let mut packed_label_and_len = [0u8; 32];
        packed_label_and_len[..b"sumcheck_poly".len()].copy_from_slice(b"sumcheck_poly");
        packed_label_and_len[24..].copy_from_slice(&(coefficients.len() as u64).to_be_bytes());

        let mut cs = TestConstraintSystem::<NovaScalar>::new();
        let initial = AllocatedRecursivePoseidonTranscriptState {
            state: alloc_value(&mut cs, "initial transcript state", initial_state),
            n_rounds: alloc_value(&mut cs, "initial transcript round", NovaScalar::zero()),
        };
        let packed = alloc_value(
            &mut cs,
            "packed sumcheck label and length",
            ark_bn254_scalar_as_nova_scalar(&Fr::from_le_bytes_mod_order(&packed_label_and_len))
                .unwrap(),
        );
        let after_label = synthesize_recursive_poseidon_transcript_transition(
            cs.namespace(|| "append packed sumcheck label and length"),
            &initial,
            &packed,
        )
        .unwrap();
        let first_coefficient = alloc_value(&mut cs, "first coefficient", NovaScalar::from(7));
        let after_first = synthesize_recursive_poseidon_transcript_transition(
            cs.namespace(|| "append first coefficient"),
            &after_label,
            &first_coefficient,
        )
        .unwrap();
        let second_coefficient = alloc_value(&mut cs, "second coefficient", NovaScalar::from(9));
        let after_second = synthesize_recursive_poseidon_transcript_transition(
            cs.namespace(|| "append second coefficient"),
            &after_first,
            &second_coefficient,
        )
        .unwrap();
        let zero =
            alloc_nova_constant(cs.namespace(|| "challenge zero"), NovaScalar::zero()).unwrap();
        let after_challenge = synthesize_recursive_poseidon_transcript_transition(
            cs.namespace(|| "derive challenge"),
            &after_second,
            &zero,
        )
        .unwrap();

        assert_eq!(after_challenge.state.get_value(), Some(expected_challenge));
        assert_eq!(
            after_challenge.n_rounds.get_value(),
            Some(NovaScalar::from(4))
        );
        assert!(cs.is_satisfied(), "{:?}", cs.which_is_unsatisfied());
    }

    #[test]
    fn recursive_poseidon_transcript_rejects_tampered_challenge() {
        let mut cs = TestConstraintSystem::<NovaScalar>::new();
        let state = alloc_value(&mut cs, "state", NovaScalar::from(1));
        let n_rounds = alloc_value(&mut cs, "round", NovaScalar::from(2));
        let data = alloc_value(&mut cs, "data", NovaScalar::from(3));
        let actual = synthesize_recursive_poseidon_hash(
            cs.namespace(|| "recursive Poseidon"),
            &state,
            &n_rounds,
            &data,
        )
        .unwrap();
        let tampered = alloc_value(
            &mut cs,
            "tampered challenge",
            actual.get_value().unwrap() + NovaScalar::one(),
        );
        cs.enforce(
            || "challenge must equal transcript output",
            |lc| lc + actual.get_variable() - tampered.get_variable(),
            |lc| lc + <TestConstraintSystem<NovaScalar> as ConstraintSystem<NovaScalar>>::one(),
            |lc| lc,
        );

        assert!(!cs.is_satisfied());
        assert!(cs
            .which_is_unsatisfied()
            .unwrap_or_default()
            .contains("challenge must equal transcript output"));
    }
}

#[cfg(test)]
mod recursive_verifier_r1cs_tests {
    use ark_bn254::Fr;
    use nova_snark::frontend::test_cs::TestConstraintSystem;

    use crate::subprotocols::blindfold::{HyraxParams, SparseR1CSMatrix};

    use super::*;

    fn multiplication_verifier_r1cs() -> VerifierR1CS<Fr> {
        // Z = [1, a, b, c], with the verifier relation a * b = c.
        let mut a = SparseR1CSMatrix::new(1, 4);
        let mut b = SparseR1CSMatrix::new(1, 4);
        let mut c = SparseR1CSMatrix::new(1, 4);
        a.push(0, 1, Fr::from(1u64));
        b.push(0, 2, Fr::from(1u64));
        c.push(0, 3, Fr::from(1u64));
        VerifierR1CS {
            a,
            b,
            c,
            num_vars: 4,
            num_constraints: 1,
            stage_configs: Vec::new(),
            extra_constraints: Vec::new(),
            extra_output_vars: Vec::new(),
            extra_blinding_vars: Vec::new(),
            hyrax: HyraxParams {
                C: 1,
                R_coeff: 0,
                R_prime: 1,
                noncoeff_count: 3,
                total_rounds: 0,
                output_claims_rows: 0,
            },
            output_claims_opening_ids: Vec::new(),
            opening_aliases: Default::default(),
        }
    }

    #[test]
    fn recursive_verifier_r1cs_enforces_every_native_jolt_row() {
        let r1cs = multiplication_verifier_r1cs();
        let witness = [
            Fr::from(1u64),
            Fr::from(3u64),
            Fr::from(4u64),
            Fr::from(12u64),
        ];
        let mut cs = TestConstraintSystem::<NovaScalar>::new();
        synthesize_recursive_jolt_verifier_r1cs(
            cs.namespace(|| "Jolt verifier R1CS"),
            &r1cs,
            &witness,
        )
        .unwrap();
        assert!(cs.is_satisfied(), "{:?}", cs.which_is_unsatisfied());
    }

    #[test]
    fn recursive_verifier_r1cs_rejects_tampered_relation_witness() {
        let r1cs = multiplication_verifier_r1cs();
        let witness = [
            Fr::from(1u64),
            Fr::from(3u64),
            Fr::from(4u64),
            Fr::from(13u64),
        ];
        let mut cs = TestConstraintSystem::<NovaScalar>::new();
        synthesize_recursive_jolt_verifier_r1cs(
            cs.namespace(|| "Jolt verifier R1CS"),
            &r1cs,
            &witness,
        )
        .unwrap();
        assert!(!cs.is_satisfied());
        assert!(cs
            .which_is_unsatisfied()
            .unwrap_or_default()
            .contains("Jolt verifier R1CS row 0"));
    }
}
