#[cfg(feature = "nova")]
use num::{BigUint, Num};
use sha3::{Digest as ShaDigest, Sha3_256};

use crate::field::JoltField;

/// Canonical, lossless representation of one native Jolt field element at the
/// Jolt/Nova recursion boundary.
///
/// This is deliberately not a hash-to-field value. The complete canonical
/// bytes are retained so the Nova step circuit can allocate the value as a
/// non-native `BigNat` and verify arithmetic over the native Jolt field.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RecursiveJoltFieldElement {
    pub canonical_le_bytes: [u8; 32],
}

impl RecursiveJoltFieldElement {
    pub(crate) fn from_field<F: JoltField>(value: F) -> Result<Self, &'static str> {
        let mut encoded = Vec::new();
        value
            .serialize_compressed(&mut encoded)
            .map_err(|_| "failed to serialize recursive Jolt field element")?;
        if encoded.len() > 32 {
            return Err("recursive Jolt field element exceeds 32-byte boundary encoding");
        }
        let mut canonical_le_bytes = [0u8; 32];
        canonical_le_bytes[..encoded.len()].copy_from_slice(&encoded);
        Ok(Self { canonical_le_bytes })
    }

    pub(crate) fn update_digest(&self, hasher: &mut Sha3_256) {
        hasher.update(self.canonical_le_bytes);
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RecursiveJoltOpeningPoint {
    pub coordinates: Vec<RecursiveJoltFieldElement>,
}

impl RecursiveJoltOpeningPoint {
    pub(crate) fn from_challenges<F: JoltField>(
        point: &[F::Challenge],
    ) -> Result<Self, &'static str> {
        let coordinates = point
            .iter()
            .map(|coordinate| RecursiveJoltFieldElement::from_field((*coordinate).into()))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { coordinates })
    }

    fn update_digest(&self, hasher: &mut Sha3_256) {
        update_usize(hasher, self.coordinates.len());
        for coordinate in &self.coordinates {
            coordinate.update_digest(hasher);
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RecursiveJoltLookupOpeningWitness {
    pub log_k_chunk: usize,
    pub instruction_opening_points: Vec<RecursiveJoltOpeningPoint>,
    pub instruction_claims: Vec<RecursiveJoltFieldElement>,
    pub instruction_padding: Vec<RecursiveJoltFieldElement>,
    pub instruction_block_contributions: Vec<RecursiveJoltFieldElement>,
    pub tuple_opening_point: RecursiveJoltOpeningPoint,
    pub tuple_claims: [RecursiveJoltFieldElement; 3],
    pub tuple_padding: [RecursiveJoltFieldElement; 3],
    pub tuple_block_contributions: [RecursiveJoltFieldElement; 3],
}

impl RecursiveJoltLookupOpeningWitness {
    fn update_digest(&self, hasher: &mut Sha3_256) {
        update_usize(hasher, self.log_k_chunk);
        update_usize(hasher, self.instruction_opening_points.len());
        for point in &self.instruction_opening_points {
            point.update_digest(hasher);
        }
        update_field_elements(hasher, &self.instruction_claims);
        update_field_elements(hasher, &self.instruction_padding);
        update_field_elements(hasher, &self.instruction_block_contributions);
        self.tuple_opening_point.update_digest(hasher);
        update_field_elements(hasher, &self.tuple_claims);
        update_field_elements(hasher, &self.tuple_padding);
        update_field_elements(hasher, &self.tuple_block_contributions);
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RecursiveJoltRegisterOpeningWitness {
    pub value_opening_point: RecursiveJoltOpeningPoint,
    pub value_claims: [RecursiveJoltFieldElement; 3],
    pub value_block_contributions: [RecursiveJoltFieldElement; 3],
    pub address_opening_point: RecursiveJoltOpeningPoint,
    pub address_claims: [RecursiveJoltFieldElement; 3],
    pub address_block_contributions: [RecursiveJoltFieldElement; 3],
    pub inc_opening_point: RecursiveJoltOpeningPoint,
    pub inc_claim: RecursiveJoltFieldElement,
    pub inc_block_contribution: RecursiveJoltFieldElement,
}

impl RecursiveJoltRegisterOpeningWitness {
    fn update_digest(&self, hasher: &mut Sha3_256) {
        self.value_opening_point.update_digest(hasher);
        update_field_elements(hasher, &self.value_claims);
        update_field_elements(hasher, &self.value_block_contributions);
        self.address_opening_point.update_digest(hasher);
        update_field_elements(hasher, &self.address_claims);
        update_field_elements(hasher, &self.address_block_contributions);
        self.inc_opening_point.update_digest(hasher);
        self.inc_claim.update_digest(hasher);
        self.inc_block_contribution.update_digest(hasher);
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RecursiveJoltRamOpeningWitness {
    pub ram_start_address: u64,
    pub ram_k: usize,
    pub log_k_chunk: usize,
    pub ra_opening_points: Vec<RecursiveJoltOpeningPoint>,
    pub ra_claims: Vec<RecursiveJoltFieldElement>,
    pub ra_block_contributions: Vec<RecursiveJoltFieldElement>,
    pub tuple_opening_point: RecursiveJoltOpeningPoint,
    pub tuple_claims: [RecursiveJoltFieldElement; 3],
    pub tuple_block_contributions: [RecursiveJoltFieldElement; 3],
    pub inc_opening_point: RecursiveJoltOpeningPoint,
    pub inc_claim: RecursiveJoltFieldElement,
    pub inc_block_contribution: RecursiveJoltFieldElement,
}

impl RecursiveJoltRamOpeningWitness {
    fn update_digest(&self, hasher: &mut Sha3_256) {
        hasher.update(self.ram_start_address.to_le_bytes());
        update_usize(hasher, self.ram_k);
        update_usize(hasher, self.log_k_chunk);
        update_usize(hasher, self.ra_opening_points.len());
        for point in &self.ra_opening_points {
            point.update_digest(hasher);
        }
        update_field_elements(hasher, &self.ra_claims);
        update_field_elements(hasher, &self.ra_block_contributions);
        self.tuple_opening_point.update_digest(hasher);
        update_field_elements(hasher, &self.tuple_claims);
        update_field_elements(hasher, &self.tuple_block_contributions);
        self.inc_opening_point.update_digest(hasher);
        self.inc_claim.update_digest(hasher);
        self.inc_block_contribution.update_digest(hasher);
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RecursiveJoltCpuOpeningWitness {
    pub opening_point: RecursiveJoltOpeningPoint,
    pub claims: Vec<RecursiveJoltFieldElement>,
    pub padding: Vec<RecursiveJoltFieldElement>,
    pub block_contributions: Vec<RecursiveJoltFieldElement>,
}

impl RecursiveJoltCpuOpeningWitness {
    fn update_digest(&self, hasher: &mut Sha3_256) {
        self.opening_point.update_digest(hasher);
        update_field_elements(hasher, &self.claims);
        update_field_elements(hasher, &self.padding);
        update_field_elements(hasher, &self.block_contributions);
    }
}

/// One fixed-shape cycle slot supplied to the recursive opening verifier.
///
/// Inactive slots are all-zero padding. Active slots retain the complete
/// register/RAM/lookup values and the 35 canonical CPU/R1CS inputs needed to
/// recompute the native Jolt opening contributions inside Nova.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecursiveJoltCycleWitness {
    pub active: bool,
    pub global_cycle: usize,
    pub rs1_present: bool,
    pub rs1_index: u8,
    pub rs1_value: u64,
    pub rs2_present: bool,
    pub rs2_index: u8,
    pub rs2_value: u64,
    pub rd_present: bool,
    pub rd_index: u8,
    pub rd_pre_value: u64,
    pub rd_post_value: u64,
    /// 0 = NoOp, 1 = Read, 2 = Write.
    pub ram_kind: u8,
    pub ram_address: u64,
    pub ram_read_value: u64,
    pub ram_write_value: u64,
    pub lookup_index: u128,
    pub left_lookup_operand: u64,
    pub right_lookup_operand: u128,
    pub lookup_output: u64,
    pub cpu_r1cs_inputs: [i128; 35],
}

impl Default for RecursiveJoltCycleWitness {
    fn default() -> Self {
        Self {
            active: false,
            global_cycle: 0,
            rs1_present: false,
            rs1_index: 0,
            rs1_value: 0,
            rs2_present: false,
            rs2_index: 0,
            rs2_value: 0,
            rd_present: false,
            rd_index: 0,
            rd_pre_value: 0,
            rd_post_value: 0,
            ram_kind: 0,
            ram_address: 0,
            ram_read_value: 0,
            ram_write_value: 0,
            lookup_index: 0,
            left_lookup_operand: 0,
            right_lookup_operand: 0,
            lookup_output: 0,
            cpu_r1cs_inputs: [0; 35],
        }
    }
}

impl RecursiveJoltCycleWitness {
    fn update_digest(&self, hasher: &mut Sha3_256) {
        hasher.update([u8::from(self.active)]);
        update_usize(hasher, self.global_cycle);
        hasher.update([u8::from(self.rs1_present), self.rs1_index]);
        hasher.update(self.rs1_value.to_le_bytes());
        hasher.update([u8::from(self.rs2_present), self.rs2_index]);
        hasher.update(self.rs2_value.to_le_bytes());
        hasher.update([u8::from(self.rd_present), self.rd_index]);
        hasher.update(self.rd_pre_value.to_le_bytes());
        hasher.update(self.rd_post_value.to_le_bytes());
        hasher.update([self.ram_kind]);
        hasher.update(self.ram_address.to_le_bytes());
        hasher.update(self.ram_read_value.to_le_bytes());
        hasher.update(self.ram_write_value.to_le_bytes());
        hasher.update(self.lookup_index.to_le_bytes());
        hasher.update(self.left_lookup_operand.to_le_bytes());
        hasher.update(self.right_lookup_operand.to_le_bytes());
        hasher.update(self.lookup_output.to_le_bytes());
        for value in self.cpu_r1cs_inputs {
            hasher.update(value.to_le_bytes());
        }
    }
}

/// Complete private opening witness for one fixed-shape Nova step.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RecursiveJoltBlockOpeningWitness {
    pub version: u16,
    pub block_index: usize,
    pub block_position: usize,
    pub block_count: usize,
    pub active_cycles: usize,
    pub cycle_capacity: usize,
    pub cycles: Vec<RecursiveJoltCycleWitness>,
    pub claim_aggregation_challenges: [RecursiveJoltFieldElement; 4],
    pub claim_closure_targets: [RecursiveJoltFieldElement; 4],
    pub lookup: RecursiveJoltLookupOpeningWitness,
    pub register: RecursiveJoltRegisterOpeningWitness,
    pub ram: RecursiveJoltRamOpeningWitness,
    pub cpu: RecursiveJoltCpuOpeningWitness,
    pub witness_digest: [u8; 32],
}

/// Value-independent description of the recursive verifier's R1CS shape.
///
/// Nova public parameters are tied to one exact step-circuit matrix. Native
/// opening dimensions therefore have to remain identical across all blocks in
/// one fold, even though their witness values differ.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RecursiveJoltOpeningCircuitShape {
    pub cycle_capacity: usize,
    pub lookup_instruction_point_lens: Vec<usize>,
    pub lookup_tuple_point_len: usize,
    pub register_value_point_len: usize,
    pub register_address_point_len: usize,
    pub register_inc_point_len: usize,
    pub ram_ra_point_lens: Vec<usize>,
    pub ram_tuple_point_len: usize,
    pub ram_inc_point_len: usize,
    pub cpu_point_len: usize,
    pub cpu_claim_count: usize,
}

impl RecursiveJoltOpeningCircuitShape {
    pub fn from_witness(witness: &RecursiveJoltBlockOpeningWitness) -> Self {
        Self {
            cycle_capacity: witness.cycle_capacity,
            lookup_instruction_point_lens: witness
                .lookup
                .instruction_opening_points
                .iter()
                .map(|point| point.coordinates.len())
                .collect(),
            lookup_tuple_point_len: witness.lookup.tuple_opening_point.coordinates.len(),
            register_value_point_len: witness.register.value_opening_point.coordinates.len(),
            register_address_point_len: witness.register.address_opening_point.coordinates.len(),
            register_inc_point_len: witness.register.inc_opening_point.coordinates.len(),
            ram_ra_point_lens: witness
                .ram
                .ra_opening_points
                .iter()
                .map(|point| point.coordinates.len())
                .collect(),
            ram_tuple_point_len: witness.ram.tuple_opening_point.coordinates.len(),
            ram_inc_point_len: witness.ram.inc_opening_point.coordinates.len(),
            cpu_point_len: witness.cpu.opening_point.coordinates.len(),
            cpu_claim_count: witness.cpu.claims.len(),
        }
    }
}

impl RecursiveJoltBlockOpeningWitness {
    pub const VERSION: u16 = 3;

    pub(crate) fn compute_digest(&self) -> [u8; 32] {
        let mut hasher = Sha3_256::new();
        hasher.update(b"JOLT_NOVA_RECURSIVE_NATIVE_OPENING_WITNESS_V3");
        hasher.update(self.version.to_le_bytes());
        update_usize(&mut hasher, self.block_index);
        update_usize(&mut hasher, self.block_position);
        update_usize(&mut hasher, self.block_count);
        update_usize(&mut hasher, self.active_cycles);
        update_usize(&mut hasher, self.cycle_capacity);
        update_usize(&mut hasher, self.cycles.len());
        for cycle in &self.cycles {
            cycle.update_digest(&mut hasher);
        }
        update_field_elements(&mut hasher, &self.claim_aggregation_challenges);
        update_field_elements(&mut hasher, &self.claim_closure_targets);
        self.lookup.update_digest(&mut hasher);
        self.register.update_digest(&mut hasher);
        self.ram.update_digest(&mut hasher);
        self.cpu.update_digest(&mut hasher);
        hasher.finalize().into()
    }

    pub(crate) fn seal(mut self) -> Self {
        self.witness_digest = self.compute_digest();
        self
    }

    pub fn validate_shape(&self) -> Result<(), &'static str> {
        if self.version != Self::VERSION {
            return Err("unsupported recursive Jolt opening witness version");
        }
        if self.block_count == 0 {
            return Err("recursive Jolt opening witness has an empty block schedule");
        }
        if self.block_position >= self.block_count {
            return Err("recursive Jolt opening witness has an invalid block position");
        }
        if self.active_cycles == 0 || self.active_cycles > self.cycle_capacity {
            return Err("recursive Jolt opening witness has invalid active-cycle count");
        }
        if self.cycles.len() != self.cycle_capacity {
            return Err("recursive Jolt opening witness does not match fixed cycle capacity");
        }
        if self.cycles[..self.active_cycles]
            .iter()
            .any(|cycle| !cycle.active)
            || self.cycles[self.active_cycles..]
                .iter()
                .any(|cycle| cycle != &RecursiveJoltCycleWitness::default())
        {
            return Err("recursive Jolt opening witness has non-canonical active/padding slots");
        }
        if self.register.value_claims.len() != 3
            || self.register.value_block_contributions.len() != 3
            || self.register.address_claims.len() != 3
            || self.register.address_block_contributions.len() != 3
            || self.ram.tuple_claims.len() != 3
            || self.ram.tuple_block_contributions.len() != 3
            || self.lookup.tuple_claims.len() != 3
            || self.lookup.tuple_block_contributions.len() != 3
        {
            return Err("recursive Jolt opening witness has invalid fixed claim shape");
        }
        if self.lookup.log_k_chunk == 0
            || self.lookup.instruction_opening_points.len() != self.lookup.instruction_claims.len()
            || self.lookup.instruction_claims.len() != self.lookup.instruction_padding.len()
            || self.lookup.instruction_claims.len()
                != self.lookup.instruction_block_contributions.len()
            || self.lookup.instruction_opening_points.is_empty()
        {
            return Err("recursive Jolt lookup opening witness has inconsistent claim shape");
        }
        if self.ram.ra_opening_points.len() != self.ram.ra_claims.len()
            || self.ram.ra_claims.len() != self.ram.ra_block_contributions.len()
        {
            return Err("recursive Jolt RAM opening witness has inconsistent claim shape");
        }
        if self.cpu.claims.len() != 35
            || self.cpu.padding.len() != self.cpu.claims.len()
            || self.cpu.block_contributions.len() != self.cpu.claims.len()
        {
            return Err("recursive Jolt CPU opening witness has inconsistent claim shape");
        }
        if self.compute_digest() != self.witness_digest {
            return Err("recursive Jolt opening witness digest mismatch");
        }
        Ok(())
    }

    pub(crate) fn claim_aggregation_challenges(
        &self,
    ) -> Result<[RecursiveJoltFieldElement; 4], &'static str> {
        Ok(self.claim_aggregation_challenges)
    }

    pub(crate) fn claim_closure_targets(
        &self,
    ) -> Result<[RecursiveJoltFieldElement; 4], &'static str> {
        Ok(self.claim_closure_targets)
    }

    #[cfg(feature = "nova")]
    pub(crate) fn block_claim_aggregates(
        &self,
    ) -> Result<[RecursiveJoltFieldElement; 4], &'static str> {
        let challenges = self.claim_aggregation_challenges()?;
        let register_values = self
            .register
            .value_block_contributions
            .iter()
            .chain(self.register.address_block_contributions.iter())
            .chain(core::iter::once(&self.register.inc_block_contribution));
        let ram_values = self
            .ram
            .ra_block_contributions
            .iter()
            .chain(self.ram.tuple_block_contributions.iter())
            .chain(core::iter::once(&self.ram.inc_block_contribution));
        let lookup_values = self
            .lookup
            .instruction_block_contributions
            .iter()
            .chain(self.lookup.tuple_block_contributions.iter());
        Ok([
            aggregate_claim_values(register_values, &challenges[0])?,
            aggregate_claim_values(ram_values, &challenges[1])?,
            aggregate_claim_values(lookup_values, &challenges[2])?,
            aggregate_claim_values(self.cpu.block_contributions.iter(), &challenges[3])?,
        ])
    }
}

#[cfg(feature = "nova")]
fn aggregate_claim_values<'a>(
    values: impl IntoIterator<Item = &'a RecursiveJoltFieldElement>,
    challenge: &RecursiveJoltFieldElement,
) -> Result<RecursiveJoltFieldElement, &'static str> {
    let modulus = BigUint::from_str_radix(
        "21888242871839275222246405745257275088548364400416034343698204186575808495617",
        10,
    )
    .map_err(|_| "failed to parse the native Jolt field modulus")?;
    let challenge = BigUint::from_bytes_le(&challenge.canonical_le_bytes);
    let aggregate = values.into_iter().fold(BigUint::default(), |acc, value| {
        (acc * &challenge + BigUint::from_bytes_le(&value.canonical_le_bytes)) % &modulus
    });
    let bytes = aggregate.to_bytes_le();
    let mut canonical_le_bytes = [0u8; 32];
    canonical_le_bytes[..bytes.len()].copy_from_slice(&bytes);
    Ok(RecursiveJoltFieldElement { canonical_le_bytes })
}

fn update_usize(hasher: &mut Sha3_256, value: usize) {
    hasher.update((value as u64).to_le_bytes());
}

fn update_field_elements(hasher: &mut Sha3_256, values: &[RecursiveJoltFieldElement]) {
    update_usize(hasher, values.len());
    for value in values {
        value.update_digest(hasher);
    }
}
