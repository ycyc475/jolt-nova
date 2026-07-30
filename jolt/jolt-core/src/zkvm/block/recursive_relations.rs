use nova_snark::{
    frontend::{
        num::AllocatedNum, AllocatedBit, ConstraintSystem, LinearCombination, SynthesisError,
    },
    gadgets::{nonnative::bignat::BigNat, utils::alloc_bignat_constant},
};
use num::{bigint::Sign, BigInt};

use crate::zkvm::r1cs::constraints::{LC, R1CS_CONSTRAINTS};

use super::{
    NovaScalar, RecursiveJoltBlockOpeningWitness, RecursiveJoltFieldElement,
    RecursiveJoltLookupOpeningWitness, RecursiveJoltOpeningPoint, RecursiveJoltRamOpeningWitness,
    RecursiveJoltRegisterOpeningWitness,
};

const JOLT_FIELD_LIMB_WIDTH: usize = 64;
const JOLT_FIELD_LIMBS: usize = 4;
const REGISTER_INDEX_BITS: usize = (common::constants::REGISTER_COUNT as usize).ilog2() as usize;

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
) -> Result<(), SynthesisError> {
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
    Ok(())
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
) -> Result<(), SynthesisError> {
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
    Ok(())
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
) -> Result<(), SynthesisError> {
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
    Ok(())
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
) -> Result<(), SynthesisError> {
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
    Ok(())
}
