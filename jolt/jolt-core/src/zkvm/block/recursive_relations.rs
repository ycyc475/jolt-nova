use nova_snark::{
    frontend::{
        num::AllocatedNum, AllocatedBit, ConstraintSystem, LinearCombination, SynthesisError,
    },
    gadgets::{nonnative::bignat::BigNat, utils::alloc_bignat_constant},
};
use num::{bigint::Sign, BigInt};

use super::{
    NovaScalar, RecursiveJoltBlockOpeningWitness, RecursiveJoltFieldElement,
    RecursiveJoltOpeningPoint, RecursiveJoltRegisterOpeningWitness,
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
    cs.enforce(
        || "global cycle bits equal public block start plus local offset",
        |_| {
            packed
                - global_cycle_start.get_variable()
                - (NovaScalar::from(local_cycle as u64), CS::one())
        },
        |lc| lc + CS::one(),
        |lc| lc,
    );
    Ok(bits)
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

    for (local_cycle, cycle) in witness.cycles.iter().enumerate() {
        let cycle_bits = alloc_global_cycle_bits(
            cs.namespace(|| format!("cycle {local_cycle} global index")),
            global_cycle_start,
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
