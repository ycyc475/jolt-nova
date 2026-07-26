#[cfg(feature = "nova")]
use nova_snark::traits::PrimeFieldExt;
#[cfg(feature = "nova")]
use std::sync::OnceLock;
use std::{collections::HashMap, error::Error, fmt, marker::PhantomData};

use ark_std::Zero;
use common::constants::{REGISTER_COUNT, XLEN};
use sha3::{Digest as ShaDigest, Sha3_256};
use tracer::{instruction::Cycle, MachineBoundaryState, TraceBlock};

use crate::{
    field::JoltField,
    zkvm::{
        bytecode::BytecodePreprocessing,
        instruction::LookupQuery,
        r1cs::{evaluation::R1CSEval, inputs::R1CSCycleInputs, key::UniformSpartanKey},
    },
};

#[derive(Clone, Debug, PartialEq)]
pub struct BlockPublicInput<Digest = [u8; 32]> {
    pub program_digest: Digest,
    pub block_index: usize,
    pub target_size: usize,
    pub global_cycle_start: usize,
    pub active_cycles: usize,
    pub start_state: MachineBoundaryState,
    pub end_state: MachineBoundaryState,
}

impl<Digest> BlockPublicInput<Digest> {
    pub fn from_trace_block(block: &TraceBlock, program_digest: Digest) -> Self {
        Self {
            program_digest,
            block_index: block.block_index,
            target_size: block.target_size,
            global_cycle_start: block.global_cycle_start,
            active_cycles: block.active_cycles,
            start_state: block.start_state.clone(),
            end_state: block.end_state.clone(),
        }
    }

    pub fn global_cycle_end(&self) -> usize {
        self.global_cycle_start + self.active_cycles
    }

    pub fn validate_shape(&self) -> Result<(), BlockPublicInputError> {
        if self.active_cycles == 0 {
            return Err(BlockPublicInputError::EmptyBlock {
                block_index: self.block_index,
            });
        }

        if self.global_cycle_start != self.start_state.global_cycle {
            return Err(BlockPublicInputError::StartCycleMismatch {
                block_index: self.block_index,
                expected: self.global_cycle_start,
                actual: self.start_state.global_cycle,
            });
        }

        let expected_end = self.global_cycle_end();
        if expected_end != self.end_state.global_cycle {
            return Err(BlockPublicInputError::EndCycleMismatch {
                block_index: self.block_index,
                expected: expected_end,
                actual: self.end_state.global_cycle,
            });
        }

        Ok(())
    }

    pub fn validate_contiguous_with(&self, next: &Self) -> Result<(), BlockPublicInputError> {
        if self.block_index + 1 != next.block_index {
            return Err(BlockPublicInputError::BlockIndexGap {
                current: self.block_index,
                next: next.block_index,
            });
        }

        if self.global_cycle_end() != next.global_cycle_start {
            return Err(BlockPublicInputError::CycleGap {
                current_block: self.block_index,
                next_block: next.block_index,
                current_end: self.global_cycle_end(),
                next_start: next.global_cycle_start,
            });
        }

        if self.end_state != next.start_state {
            return Err(BlockPublicInputError::BoundaryStateMismatch {
                current_block: self.block_index,
                next_block: next.block_index,
            });
        }

        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct BlockProof<Digest = [u8; 32], InnerProof = ()> {
    pub public_input: BlockPublicInput<Digest>,
    pub inner_proof: InnerProof,
}

impl<Digest, InnerProof> BlockProof<Digest, InnerProof> {
    pub fn new(public_input: BlockPublicInput<Digest>, inner_proof: InnerProof) -> Self {
        Self {
            public_input,
            inner_proof,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockTraceProof {
    pub cycle_count: usize,
    pub ended_at_tick_boundary: bool,
}

pub type PlaceholderBlockProof<Digest = [u8; 32]> = BlockProof<Digest, BlockTraceProof>;

#[derive(Clone, Debug)]
pub struct BlockTraceProver<Digest = [u8; 32]> {
    program_digest: Digest,
}

impl<Digest: Clone> BlockTraceProver<Digest> {
    pub fn new(program_digest: Digest) -> Self {
        Self { program_digest }
    }

    pub fn prove_block(
        &self,
        block: &TraceBlock,
    ) -> Result<PlaceholderBlockProof<Digest>, BlockTraceError> {
        validate_trace_block_shape(block)?;

        let public_input = BlockPublicInput::from_trace_block(block, self.program_digest.clone());
        public_input.validate_shape()?;

        Ok(BlockProof::new(
            public_input,
            BlockTraceProof {
                cycle_count: block.cycles.len(),
                ended_at_tick_boundary: block.ended_at_tick_boundary,
            },
        ))
    }

    pub fn prove_blocks<'a>(
        &self,
        blocks: impl IntoIterator<Item = &'a TraceBlock>,
    ) -> Result<Vec<PlaceholderBlockProof<Digest>>, BlockTraceError> {
        let proofs = blocks
            .into_iter()
            .map(|block| self.prove_block(block))
            .collect::<Result<Vec<_>, _>>()?;

        let public_inputs = proofs
            .iter()
            .map(|proof| proof.public_input.clone())
            .collect::<Vec<_>>();
        validate_block_chain(&public_inputs)?;

        Ok(proofs)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct BlockCpuProof<F> {
    pub cycle_count: usize,
    pub r1cs_rows_checked: usize,
    pub r1cs_num_steps: usize,
    pub r1cs_vk_digest: F,
    pub used_lookahead_cycle: bool,
    /// Public commitment to the exact CPU cycle used after the final row.
    ///
    /// This is `None` only when the block has no lookahead. In particular,
    /// non-terminal prefix proofs bind the first cycle of the next block here.
    pub lookahead_cycle_digest: Option<[u8; 32]>,
}

pub type CpuBlockProof<Digest, F> = BlockProof<Digest, BlockCpuProof<F>>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RegisterReadKind {
    Rs1,
    Rs2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RegisterReadClaim {
    pub local_cycle: usize,
    pub global_cycle: usize,
    pub register_index: u8,
    pub value: u64,
    pub kind: RegisterReadKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RegisterWriteClaim {
    pub local_cycle: usize,
    pub global_cycle: usize,
    pub register_index: u8,
    pub pre_value: u64,
    pub post_value: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RamAccessClaim {
    Read {
        local_cycle: usize,
        global_cycle: usize,
        address: u64,
        value: u64,
    },
    Write {
        local_cycle: usize,
        global_cycle: usize,
        address: u64,
        pre_value: u64,
        post_value: u64,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LookupClaim {
    pub local_cycle: usize,
    pub global_cycle: usize,
    pub left_instruction_input: u64,
    pub right_instruction_input: i128,
    pub left_lookup_operand: u64,
    pub right_lookup_operand: u128,
    pub lookup_index: u128,
    pub lookup_output: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockIOClaims {
    pub block_index: usize,
    pub global_cycle_start: usize,
    pub active_cycles: usize,
    pub register_reads: Vec<RegisterReadClaim>,
    pub register_writes: Vec<RegisterWriteClaim>,
    pub ram_accesses: Vec<RamAccessClaim>,
    pub lookup_claims: Vec<LookupClaim>,
}

impl BlockIOClaims {
    pub fn validate_shape(&self, block: &TraceBlock) -> Result<(), BlockTraceError> {
        if self.block_index != block.block_index {
            return Err(BlockTraceError::BlockIOClaimShapeMismatch {
                block_index: block.block_index,
                reason: "block index mismatch",
            });
        }

        if self.global_cycle_start != block.global_cycle_start {
            return Err(BlockTraceError::BlockIOClaimShapeMismatch {
                block_index: block.block_index,
                reason: "global cycle start mismatch",
            });
        }

        if self.active_cycles != block.active_cycles {
            return Err(BlockTraceError::BlockIOClaimShapeMismatch {
                block_index: block.block_index,
                reason: "active cycle count mismatch",
            });
        }

        if self.lookup_claims.len() != block.active_cycles {
            return Err(BlockTraceError::BlockIOClaimShapeMismatch {
                block_index: block.block_index,
                reason: "lookup claim count mismatch",
            });
        }

        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockRegisterClaim {
    pub block_index: usize,
    pub global_cycle_start: usize,
    pub active_cycles: usize,
    pub start_register_digest: [u8; 32],
    pub end_register_digest: [u8; 32],
    pub reads_digest: [u8; 32],
    pub writes_digest: [u8; 32],
    pub read_count: usize,
    pub write_count: usize,
}

impl BlockRegisterClaim {
    pub fn validate_shape(
        &self,
        block: &TraceBlock,
        io_claims: &BlockIOClaims,
    ) -> Result<(), BlockTraceError> {
        if self.block_index != block.block_index {
            return Err(BlockTraceError::BlockRegisterClaimShapeMismatch {
                block_index: block.block_index,
                reason: "block index mismatch",
            });
        }

        if self.global_cycle_start != block.global_cycle_start {
            return Err(BlockTraceError::BlockRegisterClaimShapeMismatch {
                block_index: block.block_index,
                reason: "global cycle start mismatch",
            });
        }

        if self.active_cycles != block.active_cycles {
            return Err(BlockTraceError::BlockRegisterClaimShapeMismatch {
                block_index: block.block_index,
                reason: "active cycle count mismatch",
            });
        }

        if self.read_count != io_claims.register_reads.len() {
            return Err(BlockTraceError::BlockRegisterClaimShapeMismatch {
                block_index: block.block_index,
                reason: "register read count mismatch",
            });
        }

        if self.write_count != io_claims.register_writes.len() {
            return Err(BlockTraceError::BlockRegisterClaimShapeMismatch {
                block_index: block.block_index,
                reason: "register write count mismatch",
            });
        }

        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RamAddressSummary {
    pub address: u64,
    pub first_value: u64,
    pub final_value: u64,
    pub read_count: usize,
    pub write_count: usize,
    pub first_global_cycle: usize,
    pub last_global_cycle: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockRamClaim {
    pub block_index: usize,
    pub global_cycle_start: usize,
    pub active_cycles: usize,
    pub access_count: usize,
    pub touched_address_count: usize,
    pub accesses_digest: [u8; 32],
    pub touched_addresses_digest: [u8; 32],
}

impl BlockRamClaim {
    pub fn validate_shape(
        &self,
        block: &TraceBlock,
        io_claims: &BlockIOClaims,
    ) -> Result<(), BlockTraceError> {
        if self.block_index != block.block_index {
            return Err(BlockTraceError::BlockRamClaimShapeMismatch {
                block_index: block.block_index,
                reason: "block index mismatch",
            });
        }

        if self.global_cycle_start != block.global_cycle_start {
            return Err(BlockTraceError::BlockRamClaimShapeMismatch {
                block_index: block.block_index,
                reason: "global cycle start mismatch",
            });
        }

        if self.active_cycles != block.active_cycles {
            return Err(BlockTraceError::BlockRamClaimShapeMismatch {
                block_index: block.block_index,
                reason: "active cycle count mismatch",
            });
        }

        if self.access_count != io_claims.ram_accesses.len() {
            return Err(BlockTraceError::BlockRamClaimShapeMismatch {
                block_index: block.block_index,
                reason: "RAM access count mismatch",
            });
        }

        if self.touched_address_count != ram_address_summaries(&io_claims.ram_accesses).len() {
            return Err(BlockTraceError::BlockRamClaimShapeMismatch {
                block_index: block.block_index,
                reason: "touched RAM address count mismatch",
            });
        }

        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LookupEntrySummary {
    pub lookup_index: u128,
    pub left_lookup_operand: u64,
    pub right_lookup_operand: u128,
    pub lookup_output: u64,
    pub count: usize,
    pub first_global_cycle: usize,
    pub last_global_cycle: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockLogUpProof {
    /// Fiat-Shamir challenge used to compress a lookup tuple into one field element.
    pub tuple_challenge: ark_bn254::Fr,
    /// Fiat-Shamir challenge used as the LogUp denominator offset.
    pub denominator_challenge: ark_bn254::Fr,
    /// Number of deterministic increments needed to avoid a zero denominator.
    pub denominator_retry_count: usize,
    /// `sum_i 1 / (beta + query_i)`.
    pub query_sum: ark_bn254::Fr,
    /// `sum_j multiplicity_j / (beta + table_entry_j)`.
    pub table_sum: ark_bn254::Fr,
    pub query_count: usize,
    pub table_distinct_entry_count: usize,
    /// Domain-separated binding of the challenges, sums, and cardinalities.
    pub proof_digest: [u8; 32],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockLookupClaim {
    pub block_index: usize,
    pub global_cycle_start: usize,
    pub active_cycles: usize,
    pub lookup_count: usize,
    pub distinct_lookup_entry_count: usize,
    pub claims_digest: [u8; 32],
    pub entry_summaries_digest: [u8; 32],
    pub logup_proof: BlockLogUpProof,
}

impl BlockLookupClaim {
    pub fn validate_shape(
        &self,
        block: &TraceBlock,
        io_claims: &BlockIOClaims,
    ) -> Result<(), BlockTraceError> {
        if self.block_index != block.block_index {
            return Err(BlockTraceError::BlockLookupClaimShapeMismatch {
                block_index: block.block_index,
                reason: "block index mismatch",
            });
        }

        if self.global_cycle_start != block.global_cycle_start {
            return Err(BlockTraceError::BlockLookupClaimShapeMismatch {
                block_index: block.block_index,
                reason: "global cycle start mismatch",
            });
        }

        if self.active_cycles != block.active_cycles {
            return Err(BlockTraceError::BlockLookupClaimShapeMismatch {
                block_index: block.block_index,
                reason: "active cycle count mismatch",
            });
        }

        if self.lookup_count != io_claims.lookup_claims.len() {
            return Err(BlockTraceError::BlockLookupClaimShapeMismatch {
                block_index: block.block_index,
                reason: "lookup count mismatch",
            });
        }

        if self.lookup_count != block.active_cycles {
            return Err(BlockTraceError::BlockLookupClaimShapeMismatch {
                block_index: block.block_index,
                reason: "lookup count must equal active cycle count",
            });
        }

        if self.distinct_lookup_entry_count
            != lookup_entry_summaries(&io_claims.lookup_claims).len()
        {
            return Err(BlockTraceError::BlockLookupClaimShapeMismatch {
                block_index: block.block_index,
                reason: "distinct lookup entry count mismatch",
            });
        }

        if self.logup_proof.query_count != self.lookup_count {
            return Err(BlockTraceError::BlockLookupClaimShapeMismatch {
                block_index: block.block_index,
                reason: "LogUp query count mismatch",
            });
        }

        if self.logup_proof.table_distinct_entry_count != self.distinct_lookup_entry_count {
            return Err(BlockTraceError::BlockLookupClaimShapeMismatch {
                block_index: block.block_index,
                reason: "LogUp table distinct-entry count mismatch",
            });
        }

        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct BlockProofBundle<Digest = [u8; 32], F = ark_bn254::Fr> {
    pub cpu_proof: CpuBlockProof<Digest, F>,
    pub io_claims: BlockIOClaims,
    pub register_claim: BlockRegisterClaim,
    pub ram_claim: BlockRamClaim,
    pub lookup_claim: BlockLookupClaim,
}

impl<Digest, F> BlockProofBundle<Digest, F> {
    pub fn public_input(&self) -> &BlockPublicInput<Digest> {
        &self.cpu_proof.public_input
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FoldableBlockState<F = ark_bn254::Fr> {
    pub block_index: usize,
    pub global_cycle_start: usize,
    pub global_cycle_end: usize,
    pub active_cycles: usize,
    pub start_state_digest: [u8; 32],
    pub end_state_digest: [u8; 32],
    pub start_register_digest: [u8; 32],
    pub end_register_digest: [u8; 32],
    pub register_reads_digest: [u8; 32],
    pub register_writes_digest: [u8; 32],
    pub register_read_count: usize,
    pub register_write_count: usize,
    pub ram_accesses_digest: [u8; 32],
    pub ram_touched_addresses_digest: [u8; 32],
    pub ram_access_count: usize,
    pub ram_touched_address_count: usize,
    pub lookup_claims_digest: [u8; 32],
    pub lookup_entry_summaries_digest: [u8; 32],
    pub lookup_count: usize,
    pub lookup_distinct_entry_count: usize,
    pub lookup_logup_proof_digest: [u8; 32],
    pub lookup_logup_tuple_challenge: ark_bn254::Fr,
    pub lookup_logup_denominator_challenge: ark_bn254::Fr,
    pub lookup_logup_denominator_retry_count: usize,
    pub lookup_logup_query_sum: ark_bn254::Fr,
    pub lookup_logup_table_sum: ark_bn254::Fr,
    pub r1cs_rows_checked: usize,
    pub r1cs_num_steps: usize,
    pub r1cs_vk_digest: F,
    pub used_lookahead_cycle: bool,
    pub lookahead_cycle_digest: Option<[u8; 32]>,
    pub state_digest: [u8; 32],
}

impl<F> FoldableBlockState<F> {
    pub fn validate_contiguous_with(&self, next: &Self) -> Result<(), BlockTraceError> {
        if self.block_index + 1 != next.block_index {
            return Err(BlockTraceError::BlockFoldInputBoundaryMismatch {
                current_block: self.block_index,
                next_block: next.block_index,
                reason: "block index gap",
            });
        }

        if self.global_cycle_end != next.global_cycle_start {
            return Err(BlockTraceError::BlockFoldInputBoundaryMismatch {
                current_block: self.block_index,
                next_block: next.block_index,
                reason: "global cycle gap",
            });
        }

        if self.end_state_digest != next.start_state_digest {
            return Err(BlockTraceError::BlockFoldInputBoundaryMismatch {
                current_block: self.block_index,
                next_block: next.block_index,
                reason: "machine boundary state digest mismatch",
            });
        }

        if self.end_register_digest != next.start_register_digest {
            return Err(BlockTraceError::BlockFoldInputBoundaryMismatch {
                current_block: self.block_index,
                next_block: next.block_index,
                reason: "register boundary digest mismatch",
            });
        }

        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockFoldInput<Digest = [u8; 32], F = ark_bn254::Fr> {
    pub program_digest: Digest,
    pub state: FoldableBlockState<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockFoldAccumulator<Digest = [u8; 32]> {
    pub program_digest: Option<Digest>,
    pub absorbed_blocks: usize,
    pub first_block_index: Option<usize>,
    pub last_block_index: Option<usize>,
    pub global_cycle_start: Option<usize>,
    pub global_cycle_end: Option<usize>,
    pub latest_state_digest: Option<[u8; 32]>,
    pub latest_machine_state_digest: Option<[u8; 32]>,
    pub latest_register_digest: Option<[u8; 32]>,
    pub total_active_cycles: usize,
    pub total_register_reads: usize,
    pub total_register_writes: usize,
    pub total_ram_accesses: usize,
    pub total_lookup_claims: usize,
    pub accumulator_digest: [u8; 32],
}

impl<Digest> Default for BlockFoldAccumulator<Digest> {
    fn default() -> Self {
        Self {
            program_digest: None,
            absorbed_blocks: 0,
            first_block_index: None,
            last_block_index: None,
            global_cycle_start: None,
            global_cycle_end: None,
            latest_state_digest: None,
            latest_machine_state_digest: None,
            latest_register_digest: None,
            total_active_cycles: 0,
            total_register_reads: 0,
            total_register_writes: 0,
            total_ram_accesses: 0,
            total_lookup_claims: 0,
            accumulator_digest: digest_empty_fold_accumulator(),
        }
    }
}

impl<Digest> BlockFoldAccumulator<Digest>
where
    Digest: Clone + PartialEq + AsRef<[u8]>,
{
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.absorbed_blocks == 0
    }

    pub fn absorb<F>(
        &mut self,
        fold_input: &BlockFoldInput<Digest, F>,
    ) -> Result<(), BlockTraceError>
    where
        F: JoltField,
    {
        let state = &fold_input.state;

        if state.state_digest != digest_foldable_block_state(state) {
            return Err(BlockTraceError::BlockFoldAccumulatorAbsorbMismatch {
                block_index: state.block_index,
                reason: "foldable state digest mismatch",
            });
        }

        if let Some(program_digest) = self.program_digest.as_ref() {
            if program_digest != &fold_input.program_digest {
                return Err(BlockTraceError::BlockFoldAccumulatorProgramDigestMismatch {
                    block_index: state.block_index,
                });
            }

            let current_block = self
                .last_block_index
                .expect("non-empty fold accumulator should have a last block index");
            if current_block + 1 != state.block_index {
                return Err(BlockTraceError::BlockFoldAccumulatorBoundaryMismatch {
                    current_block,
                    next_block: state.block_index,
                    reason: "block index gap",
                });
            }

            if self.global_cycle_end != Some(state.global_cycle_start) {
                return Err(BlockTraceError::BlockFoldAccumulatorBoundaryMismatch {
                    current_block,
                    next_block: state.block_index,
                    reason: "global cycle gap",
                });
            }

            if self.latest_machine_state_digest != Some(state.start_state_digest) {
                return Err(BlockTraceError::BlockFoldAccumulatorBoundaryMismatch {
                    current_block,
                    next_block: state.block_index,
                    reason: "machine boundary state digest mismatch",
                });
            }

            if self.latest_register_digest != Some(state.start_register_digest) {
                return Err(BlockTraceError::BlockFoldAccumulatorBoundaryMismatch {
                    current_block,
                    next_block: state.block_index,
                    reason: "register boundary digest mismatch",
                });
            }
        } else {
            self.program_digest = Some(fold_input.program_digest.clone());
            self.first_block_index = Some(state.block_index);
            self.global_cycle_start = Some(state.global_cycle_start);
        }

        self.accumulator_digest = digest_fold_accumulator_step(self.accumulator_digest, fold_input);
        self.absorbed_blocks += 1;
        self.last_block_index = Some(state.block_index);
        self.global_cycle_end = Some(state.global_cycle_end);
        self.latest_state_digest = Some(state.state_digest);
        self.latest_machine_state_digest = Some(state.end_state_digest);
        self.latest_register_digest = Some(state.end_register_digest);
        self.total_active_cycles += state.active_cycles;
        self.total_register_reads += state.register_read_count;
        self.total_register_writes += state.register_write_count;
        self.total_ram_accesses += state.ram_access_count;
        self.total_lookup_claims += state.lookup_count;

        Ok(())
    }
}

pub trait BlockFoldingBackend<Digest, F>
where
    Digest: Clone + PartialEq + AsRef<[u8]>,
    F: JoltField,
{
    type Accumulator: Clone + PartialEq;

    fn name(&self) -> &'static str;

    fn relation_name(&self) -> &'static str {
        self.name()
    }

    fn new_accumulator(&self) -> Self::Accumulator;

    fn absorb(
        &self,
        accumulator: &mut Self::Accumulator,
        fold_input: &BlockFoldInput<Digest, F>,
    ) -> Result<(), BlockTraceError>;

    fn absorbed_blocks(&self, accumulator: &Self::Accumulator) -> usize;

    fn fold(
        &self,
        fold_inputs: &[BlockFoldInput<Digest, F>],
    ) -> Result<Self::Accumulator, BlockTraceError> {
        let mut accumulator = self.new_accumulator();
        for fold_input in fold_inputs {
            self.absorb(&mut accumulator, fold_input)?;
        }
        Ok(accumulator)
    }

    fn verify(
        &self,
        fold_inputs: &[BlockFoldInput<Digest, F>],
        accumulator: &Self::Accumulator,
    ) -> Result<(), BlockTraceError> {
        let expected = self.fold(fold_inputs)?;
        if &expected != accumulator {
            return Err(BlockTraceError::BlockFoldAccumulatorMismatch {
                expected_blocks: self.absorbed_blocks(&expected),
                actual_blocks: self.absorbed_blocks(accumulator),
            });
        }

        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MockFoldingBackend;

impl<Digest, F> BlockFoldingBackend<Digest, F> for MockFoldingBackend
where
    Digest: Clone + PartialEq + AsRef<[u8]>,
    F: JoltField,
{
    type Accumulator = BlockFoldAccumulator<Digest>;

    fn name(&self) -> &'static str {
        "mock-hash-chain"
    }

    fn new_accumulator(&self) -> Self::Accumulator {
        BlockFoldAccumulator::new()
    }

    fn absorb(
        &self,
        accumulator: &mut Self::Accumulator,
        fold_input: &BlockFoldInput<Digest, F>,
    ) -> Result<(), BlockTraceError> {
        accumulator.absorb(fold_input)
    }

    fn absorbed_blocks(&self, accumulator: &Self::Accumulator) -> usize {
        accumulator.absorbed_blocks
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NovaFoldConfig {
    pub backend_name: &'static str,
    pub relation_name: &'static str,
    pub subclaim_backend_name: &'static str,
    /// Selects the final folded proof backend used after Nova folding.
    ///
    /// The default `spartan-placeholder` keeps the stage-6 proof envelope path
    /// lightweight. Use `spartan-final-proof` to compress the Nova recursive
    /// SNARK with a real Spartan `CompressedSNARK` in the end-to-end pipeline.
    pub final_proof_backend_name: &'static str,
    pub use_zero_knowledge: bool,
}

pub const NOVA_BLOCK_FOLD_RELATION_NAME: &str = "jolt-nova-block-fold-v1";
pub const NOVA_TRANSCRIPT_SUBCLAIM_BACKEND_NAME: &str = "transcript-subclaim-fingerprints";
pub const NOVA_LOGUP_SUBCLAIM_BACKEND_NAME: &str = "logup-subclaim-v1";

impl Default for NovaFoldConfig {
    fn default() -> Self {
        Self {
            backend_name: {
                #[cfg(feature = "nova")]
                {
                    "nova-recursive-snark"
                }
                #[cfg(not(feature = "nova"))]
                {
                    "nova-placeholder"
                }
            },
            relation_name: NOVA_BLOCK_FOLD_RELATION_NAME,
            subclaim_backend_name: NOVA_TRANSCRIPT_SUBCLAIM_BACKEND_NAME,
            final_proof_backend_name: SPARTAN_PLACEHOLDER_PROOF_SYSTEM_NAME,
            use_zero_knowledge: true,
        }
    }
}

pub const NOVA_Z_ARITY: usize = 7;
pub const NOVA_SEMANTIC_ACCUMULATOR_INDEX: usize = 0;
pub const NOVA_NEXT_BLOCK_INDEX_INDEX: usize = 1;
pub const NOVA_TOTAL_ACTIVE_CYCLES_INDEX: usize = 2;
pub const NOVA_REGISTER_ACCUMULATOR_INDEX: usize = 3;
pub const NOVA_RAM_ACCUMULATOR_INDEX: usize = 4;
pub const NOVA_LOOKUP_ACCUMULATOR_INDEX: usize = 5;
pub const NOVA_CPU_ACCUMULATOR_INDEX: usize = 6;
pub type NovaFoldZState = [[u8; 32]; NOVA_Z_ARITY];

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NovaFoldAccumulator<Digest = [u8; 32]> {
    pub config: NovaFoldConfig,
    pub metadata: BlockFoldAccumulator<Digest>,
    pub recursive_snark_bytes: Option<Vec<u8>>,
    pub recursive_snark_output_digest: Option<[u8; 32]>,
    pub recursive_z_state: Option<NovaFoldZState>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FinalFoldedInstance<Digest = [u8; 32]> {
    pub config: NovaFoldConfig,
    pub metadata: BlockFoldAccumulator<Digest>,
    pub recursive_snark_output_digest: [u8; 32],
    pub recursive_z_state: NovaFoldZState,
    pub instance_digest: [u8; 32],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FinalFoldedProof<Digest = [u8; 32]> {
    pub proof_system: &'static str,
    pub instance: FinalFoldedInstance<Digest>,
    pub spartan_encoding_digest: Option<[u8; 32]>,
    pub proof_digest: [u8; 32],
    pub spartan_proof_bytes: Option<Vec<u8>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpartanFinalInstanceEncoding {
    pub version: &'static str,
    pub public_input_bytes: Vec<u8>,
    pub witness_bytes: Vec<u8>,
    pub public_input_digest: [u8; 32],
    pub witness_digest: [u8; 32],
    pub encoding_digest: [u8; 32],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JoltNovaPhase6Baseline {
    pub absorbed_blocks: usize,
    pub total_active_cycles: usize,
    pub total_register_reads: usize,
    pub total_register_writes: usize,
    pub total_ram_accesses: usize,
    pub total_lookup_claims: usize,
    pub recursive_snark_bytes_len: Option<usize>,
    pub recursive_z_state_words: usize,
    pub final_public_input_bytes_len: usize,
    pub final_witness_bytes_len: usize,
    pub final_instance_digest: [u8; 32],
    pub spartan_encoding_digest: [u8; 32],
    pub final_proof_system: Option<&'static str>,
    pub final_proof_digest: Option<[u8; 32]>,
    pub final_proof_bytes_len: Option<usize>,
}

/// Size-oriented summary for one final folded proof envelope.
///
/// This intentionally separates the proof envelope bytes from the backend proof
/// payload bytes. Placeholder final proofs have no payload, while real Spartan
/// final proofs carry serialized compressed proof bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JoltNovaFinalProofSizeBaseline {
    pub configured_backend_name: &'static str,
    pub proof_system: &'static str,
    pub absorbed_blocks: usize,
    pub total_active_cycles: usize,
    pub recursive_snark_bytes_len: Option<usize>,
    pub final_public_input_bytes_len: usize,
    pub final_witness_bytes_len: usize,
    pub proof_envelope_bytes_len: usize,
    pub proof_payload_bytes_len: usize,
    pub proof_total_bytes_len: usize,
    pub final_instance_digest: [u8; 32],
    pub spartan_encoding_digest: [u8; 32],
    pub proof_digest: [u8; 32],
}

/// Side-by-side final proof size comparison for the same Nova accumulator.
///
/// The two baselines are generated by reusing the same folded accumulator and
/// switching only the final proof backend between `spartan-placeholder` and
/// `spartan-final-proof`. This keeps the folded execution state fixed and
/// isolates the cost of replacing the final proof backend.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JoltNovaFinalProofSizeComparison {
    pub folded_accumulator_digest: [u8; 32],
    pub absorbed_blocks: usize,
    pub total_active_cycles: usize,
    pub recursive_snark_bytes_len: Option<usize>,
    pub placeholder: JoltNovaFinalProofSizeBaseline,
    pub spartan: JoltNovaFinalProofSizeBaseline,
    pub spartan_payload_extra_bytes: usize,
    pub spartan_total_extra_bytes: i128,
}

/// Stable output format selector for Jolt-Nova benchmark/report artifacts.
///
/// JSON is the canonical format because the stage-7 reports are nested. CSV is
/// reserved for later table-oriented exports used in papers or spreadsheets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JoltNovaReportOutputFormat {
    Json,
    Csv,
}

impl JoltNovaReportOutputFormat {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Json => "json",
            Self::Csv => "csv",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "json" => Some(Self::Json),
            "csv" => Some(Self::Csv),
            _ => None,
        }
    }

    pub fn is_canonical(self) -> bool {
        self == JOLT_NOVA_REPORT_CANONICAL_OUTPUT_FORMAT
    }
}

impl fmt::Display for JoltNovaReportOutputFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

pub const JOLT_NOVA_REPORT_SCHEMA_VERSION: &str = "jolt-nova-report-v1";
pub const JOLT_NOVA_REPORT_CANONICAL_OUTPUT_FORMAT: JoltNovaReportOutputFormat =
    JoltNovaReportOutputFormat::Json;
pub const JOLT_NOVA_FINAL_PROOF_SIZE_SCALING_REPORT_KIND: &str = "final-proof-size-scaling";

pub const FINAL_FOLDED_INSTANCE_VERSION: &str = "jolt-nova-final-folded-instance-v1";
pub const SPARTAN_FINAL_INSTANCE_ENCODING_VERSION: &str =
    "jolt-nova-spartan-final-instance-encoding-v1";
pub const SPARTAN_PLACEHOLDER_PROOF_SYSTEM_NAME: &str = "spartan-placeholder";
pub const SPARTAN_FINAL_PROOF_SYSTEM_NAME: &str = "spartan-final-proof";

pub trait FinalFoldedProofBackend<Digest = [u8; 32]>
where
    Digest: Clone + PartialEq + AsRef<[u8]>,
{
    fn name(&self) -> &'static str;

    fn prove(
        &self,
        instance: FinalFoldedInstance<Digest>,
    ) -> Result<FinalFoldedProof<Digest>, BlockTraceError>;

    fn prove_from_accumulator(
        &self,
        accumulator: &NovaFoldAccumulator<Digest>,
    ) -> Result<FinalFoldedProof<Digest>, BlockTraceError> {
        let instance = build_final_folded_instance(accumulator)?;
        self.prove(instance)
    }

    fn verify(
        &self,
        accumulator: &NovaFoldAccumulator<Digest>,
        proof: &FinalFoldedProof<Digest>,
    ) -> Result<(), BlockTraceError>;
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SpartanPlaceholderFinalProofBackend;

impl<Digest> FinalFoldedProofBackend<Digest> for SpartanPlaceholderFinalProofBackend
where
    Digest: Clone + PartialEq + AsRef<[u8]>,
{
    fn name(&self) -> &'static str {
        SPARTAN_PLACEHOLDER_PROOF_SYSTEM_NAME
    }

    fn prove(
        &self,
        instance: FinalFoldedInstance<Digest>,
    ) -> Result<FinalFoldedProof<Digest>, BlockTraceError> {
        let spartan_encoding = encode_final_folded_instance_for_spartan(&instance)?;
        let spartan_encoding_digest = Some(spartan_encoding.encoding_digest);
        let proof_digest = digest_final_folded_proof(
            SPARTAN_PLACEHOLDER_PROOF_SYSTEM_NAME,
            &instance,
            spartan_encoding_digest,
            None,
        );
        Ok(FinalFoldedProof {
            proof_system: SPARTAN_PLACEHOLDER_PROOF_SYSTEM_NAME,
            instance,
            spartan_encoding_digest,
            proof_digest,
            spartan_proof_bytes: None,
        })
    }

    fn verify(
        &self,
        accumulator: &NovaFoldAccumulator<Digest>,
        proof: &FinalFoldedProof<Digest>,
    ) -> Result<(), BlockTraceError> {
        if proof.proof_system != SPARTAN_PLACEHOLDER_PROOF_SYSTEM_NAME {
            return Err(BlockTraceError::NovaFoldingBackendError {
                block_index: proof.instance.metadata.last_block_index.unwrap_or(0),
                reason: "unsupported final folded proof system",
            });
        }
        if proof.spartan_proof_bytes.is_some() {
            return Err(BlockTraceError::NovaFoldingBackendError {
                block_index: proof.instance.metadata.last_block_index.unwrap_or(0),
                reason: "Spartan placeholder proof must not contain proof bytes",
            });
        }

        verify_final_folded_proof_envelope(accumulator, proof)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SpartanFinalProofBackend;

impl<Digest> FinalFoldedProofBackend<Digest> for SpartanFinalProofBackend
where
    Digest: Clone + PartialEq + AsRef<[u8]>,
{
    fn name(&self) -> &'static str {
        SPARTAN_FINAL_PROOF_SYSTEM_NAME
    }

    fn prove(
        &self,
        instance: FinalFoldedInstance<Digest>,
    ) -> Result<FinalFoldedProof<Digest>, BlockTraceError> {
        let block_index = instance.metadata.last_block_index.unwrap_or(0);
        let _encoding = encode_final_folded_instance_for_spartan(&instance)?;

        Err(BlockTraceError::NovaFoldingBackendUnavailable {
            block_index,
            reason: "Spartan final proof proving requires a Nova accumulator",
        })
    }

    fn prove_from_accumulator(
        &self,
        accumulator: &NovaFoldAccumulator<Digest>,
    ) -> Result<FinalFoldedProof<Digest>, BlockTraceError> {
        #[cfg(feature = "nova")]
        {
            prove_spartan_compressed_final_proof_from_accumulator(accumulator)
        }
        #[cfg(not(feature = "nova"))]
        {
            let block_index = accumulator.metadata.last_block_index.unwrap_or(0);
            Err(BlockTraceError::NovaFoldingBackendUnavailable {
                block_index,
                reason: "Spartan final proof backend requires the nova feature",
            })
        }
    }

    fn verify(
        &self,
        accumulator: &NovaFoldAccumulator<Digest>,
        proof: &FinalFoldedProof<Digest>,
    ) -> Result<(), BlockTraceError> {
        if proof.proof_system != SPARTAN_FINAL_PROOF_SYSTEM_NAME {
            return Err(BlockTraceError::NovaFoldingBackendError {
                block_index: proof.instance.metadata.last_block_index.unwrap_or(0),
                reason: "unsupported final folded proof system",
            });
        }

        if proof.spartan_proof_bytes.is_none() {
            return Err(BlockTraceError::NovaFoldingBackendError {
                block_index: proof.instance.metadata.last_block_index.unwrap_or(0),
                reason: "Spartan final proof is missing proof bytes",
            });
        }

        #[cfg(feature = "nova")]
        {
            verify_spartan_compressed_final_proof(accumulator, proof)
        }
        #[cfg(not(feature = "nova"))]
        {
            let _ = accumulator;
            let block_index = proof.instance.metadata.last_block_index.unwrap_or(0);
            Err(BlockTraceError::NovaFoldingBackendUnavailable {
                block_index,
                reason: "Spartan final proof backend requires the nova feature",
            })
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NovaFoldingBackend {
    pub config: NovaFoldConfig,
}

impl NovaFoldingBackend {
    pub fn new(config: NovaFoldConfig) -> Self {
        Self { config }
    }
}

impl Default for NovaFoldingBackend {
    fn default() -> Self {
        Self::new(NovaFoldConfig::default())
    }
}

pub fn build_final_folded_instance<Digest>(
    accumulator: &NovaFoldAccumulator<Digest>,
) -> Result<FinalFoldedInstance<Digest>, BlockTraceError>
where
    Digest: Clone + PartialEq + AsRef<[u8]>,
{
    let block_index = accumulator.metadata.last_block_index.unwrap_or(0);
    if accumulator.metadata.is_empty() {
        return Err(BlockTraceError::NovaFoldingBackendError {
            block_index,
            reason: "final folded instance requires a non-empty Nova accumulator",
        });
    }

    let recursive_snark_output_digest = accumulator.recursive_snark_output_digest.ok_or(
        BlockTraceError::NovaFoldingBackendError {
            block_index,
            reason: "final folded instance is missing Nova recursive output digest",
        },
    )?;
    let recursive_z_state =
        accumulator
            .recursive_z_state
            .ok_or(BlockTraceError::NovaFoldingBackendError {
                block_index,
                reason: "final folded instance is missing Nova recursive z-state",
            })?;

    let mut instance = FinalFoldedInstance {
        config: accumulator.config.clone(),
        metadata: accumulator.metadata.clone(),
        recursive_snark_output_digest,
        recursive_z_state,
        instance_digest: [0u8; 32],
    };
    instance.instance_digest = digest_final_folded_instance(&instance);
    Ok(instance)
}

pub fn verify_final_folded_instance<Digest>(
    accumulator: &NovaFoldAccumulator<Digest>,
    instance: &FinalFoldedInstance<Digest>,
) -> Result<(), BlockTraceError>
where
    Digest: Clone + PartialEq + AsRef<[u8]>,
{
    let expected = build_final_folded_instance(accumulator)?;
    if &expected != instance {
        return Err(BlockTraceError::NovaFoldingBackendError {
            block_index: accumulator.metadata.last_block_index.unwrap_or(0),
            reason: "final folded instance mismatch",
        });
    }

    Ok(())
}

pub fn encode_final_folded_instance_for_spartan<Digest>(
    instance: &FinalFoldedInstance<Digest>,
) -> Result<SpartanFinalInstanceEncoding, BlockTraceError>
where
    Digest: AsRef<[u8]>,
{
    let expected_instance_digest = digest_final_folded_instance(instance);
    if instance.instance_digest != expected_instance_digest {
        return Err(BlockTraceError::NovaFoldingBackendError {
            block_index: instance.metadata.last_block_index.unwrap_or(0),
            reason: "final folded instance digest mismatch",
        });
    }

    let public_input_bytes = encode_spartan_final_public_input_bytes(instance);
    let witness_bytes = encode_spartan_final_witness_bytes(instance);
    let public_input_digest =
        digest_spartan_final_encoding_component("public-inputs", &public_input_bytes);
    let witness_digest = digest_spartan_final_encoding_component("witness", &witness_bytes);
    let encoding_digest =
        digest_spartan_final_instance_encoding(public_input_digest, witness_digest);

    Ok(SpartanFinalInstanceEncoding {
        version: SPARTAN_FINAL_INSTANCE_ENCODING_VERSION,
        public_input_bytes,
        witness_bytes,
        public_input_digest,
        witness_digest,
        encoding_digest,
    })
}

pub fn verify_spartan_final_instance_encoding<Digest>(
    instance: &FinalFoldedInstance<Digest>,
    encoding: &SpartanFinalInstanceEncoding,
) -> Result<(), BlockTraceError>
where
    Digest: AsRef<[u8]>,
{
    if encoding.version != SPARTAN_FINAL_INSTANCE_ENCODING_VERSION {
        return Err(BlockTraceError::NovaFoldingBackendError {
            block_index: instance.metadata.last_block_index.unwrap_or(0),
            reason: "unsupported Spartan final instance encoding version",
        });
    }

    let expected = encode_final_folded_instance_for_spartan(instance)?;
    if encoding != &expected {
        return Err(BlockTraceError::NovaFoldingBackendError {
            block_index: instance.metadata.last_block_index.unwrap_or(0),
            reason: "Spartan final instance encoding mismatch",
        });
    }

    Ok(())
}

pub fn verify_final_folded_proof_envelope<Digest>(
    accumulator: &NovaFoldAccumulator<Digest>,
    proof: &FinalFoldedProof<Digest>,
) -> Result<(), BlockTraceError>
where
    Digest: Clone + PartialEq + AsRef<[u8]>,
{
    verify_final_folded_instance(accumulator, &proof.instance)?;
    let spartan_encoding = encode_final_folded_instance_for_spartan(&proof.instance)?;
    if proof.spartan_encoding_digest != Some(spartan_encoding.encoding_digest) {
        return Err(BlockTraceError::NovaFoldingBackendError {
            block_index: proof.instance.metadata.last_block_index.unwrap_or(0),
            reason: "Spartan final instance encoding digest mismatch",
        });
    }

    let expected_digest = digest_final_folded_proof(
        proof.proof_system,
        &proof.instance,
        proof.spartan_encoding_digest,
        proof.spartan_proof_bytes.as_deref(),
    );
    if proof.proof_digest != expected_digest {
        return Err(BlockTraceError::NovaFoldingBackendError {
            block_index: proof.instance.metadata.last_block_index.unwrap_or(0),
            reason: "final folded proof digest mismatch",
        });
    }

    Ok(())
}

pub fn summarize_jolt_nova_phase6_baseline<Digest>(
    accumulator: &NovaFoldAccumulator<Digest>,
    proof: Option<&FinalFoldedProof<Digest>>,
) -> Result<JoltNovaPhase6Baseline, BlockTraceError>
where
    Digest: Clone + PartialEq + AsRef<[u8]>,
{
    let instance = build_final_folded_instance(accumulator)?;
    let encoding = encode_final_folded_instance_for_spartan(&instance)?;

    if let Some(proof) = proof {
        verify_final_folded_proof_envelope(accumulator, proof)?;
    }

    Ok(JoltNovaPhase6Baseline {
        absorbed_blocks: accumulator.metadata.absorbed_blocks,
        total_active_cycles: accumulator.metadata.total_active_cycles,
        total_register_reads: accumulator.metadata.total_register_reads,
        total_register_writes: accumulator.metadata.total_register_writes,
        total_ram_accesses: accumulator.metadata.total_ram_accesses,
        total_lookup_claims: accumulator.metadata.total_lookup_claims,
        recursive_snark_bytes_len: accumulator.recursive_snark_bytes.as_ref().map(Vec::len),
        recursive_z_state_words: NOVA_Z_ARITY,
        final_public_input_bytes_len: encoding.public_input_bytes.len(),
        final_witness_bytes_len: encoding.witness_bytes.len(),
        final_instance_digest: instance.instance_digest,
        spartan_encoding_digest: encoding.encoding_digest,
        final_proof_system: proof.map(|proof| proof.proof_system),
        final_proof_digest: proof.map(|proof| proof.proof_digest),
        final_proof_bytes_len: proof
            .and_then(|proof| proof.spartan_proof_bytes.as_ref().map(Vec::len)),
    })
}

/// Summarizes the encoded size boundary of one final folded proof.
///
/// The proof envelope is verified against the accumulator before sizes are
/// reported, so callers can use this as a checked measurement primitive.
pub fn summarize_jolt_nova_final_proof_size_baseline<Digest>(
    accumulator: &NovaFoldAccumulator<Digest>,
    proof: &FinalFoldedProof<Digest>,
) -> Result<JoltNovaFinalProofSizeBaseline, BlockTraceError>
where
    Digest: Clone + PartialEq + AsRef<[u8]>,
{
    verify_final_folded_proof_envelope(accumulator, proof)?;
    let instance = build_final_folded_instance(accumulator)?;
    let encoding = encode_final_folded_instance_for_spartan(&instance)?;
    let proof_envelope_bytes_len = encode_final_folded_proof_envelope_size_bytes(proof).len();
    let proof_payload_bytes_len = proof
        .spartan_proof_bytes
        .as_ref()
        .map(Vec::len)
        .unwrap_or(0);

    Ok(JoltNovaFinalProofSizeBaseline {
        configured_backend_name: accumulator.config.final_proof_backend_name,
        proof_system: proof.proof_system,
        absorbed_blocks: accumulator.metadata.absorbed_blocks,
        total_active_cycles: accumulator.metadata.total_active_cycles,
        recursive_snark_bytes_len: accumulator.recursive_snark_bytes.as_ref().map(Vec::len),
        final_public_input_bytes_len: encoding.public_input_bytes.len(),
        final_witness_bytes_len: encoding.witness_bytes.len(),
        proof_envelope_bytes_len,
        proof_payload_bytes_len,
        proof_total_bytes_len: proof_envelope_bytes_len + proof_payload_bytes_len,
        final_instance_digest: instance.instance_digest,
        spartan_encoding_digest: encoding.encoding_digest,
        proof_digest: proof.proof_digest,
    })
}

/// Generates placeholder-vs-Spartan final proof size baselines for one folded
/// Nova accumulator.
///
/// This is a measurement helper: it does not change the accumulator, and it
/// produces both proof variants solely to compare the final proof boundary.
pub fn summarize_jolt_nova_final_proof_size_comparison<Digest>(
    accumulator: &NovaFoldAccumulator<Digest>,
) -> Result<JoltNovaFinalProofSizeComparison, BlockTraceError>
where
    Digest: Clone + PartialEq + AsRef<[u8]>,
{
    let placeholder_accumulator = clone_nova_accumulator_with_final_proof_backend(
        accumulator,
        SPARTAN_PLACEHOLDER_PROOF_SYSTEM_NAME,
    );
    let placeholder_proof = prove_configured_final_folded_accumulator(&placeholder_accumulator)?;
    let placeholder = summarize_jolt_nova_final_proof_size_baseline(
        &placeholder_accumulator,
        &placeholder_proof,
    )?;

    let spartan_accumulator = clone_nova_accumulator_with_final_proof_backend(
        accumulator,
        SPARTAN_FINAL_PROOF_SYSTEM_NAME,
    );
    let spartan_proof = prove_configured_final_folded_accumulator(&spartan_accumulator)?;
    let spartan =
        summarize_jolt_nova_final_proof_size_baseline(&spartan_accumulator, &spartan_proof)?;

    if placeholder.absorbed_blocks != spartan.absorbed_blocks
        || placeholder.total_active_cycles != spartan.total_active_cycles
        || placeholder.recursive_snark_bytes_len != spartan.recursive_snark_bytes_len
    {
        return Err(BlockTraceError::NovaFoldingBackendError {
            block_index: accumulator.metadata.last_block_index.unwrap_or(0),
            reason: "final proof size comparison metadata mismatch",
        });
    }

    Ok(JoltNovaFinalProofSizeComparison {
        folded_accumulator_digest: accumulator.metadata.accumulator_digest,
        absorbed_blocks: accumulator.metadata.absorbed_blocks,
        total_active_cycles: accumulator.metadata.total_active_cycles,
        recursive_snark_bytes_len: accumulator.recursive_snark_bytes.as_ref().map(Vec::len),
        spartan_payload_extra_bytes: spartan
            .proof_payload_bytes_len
            .saturating_sub(placeholder.proof_payload_bytes_len),
        spartan_total_extra_bytes: spartan.proof_total_bytes_len as i128
            - placeholder.proof_total_bytes_len as i128,
        placeholder,
        spartan,
    })
}

fn clone_nova_accumulator_with_final_proof_backend<Digest>(
    accumulator: &NovaFoldAccumulator<Digest>,
    final_proof_backend_name: &'static str,
) -> NovaFoldAccumulator<Digest>
where
    Digest: Clone,
{
    let mut configured = accumulator.clone();
    configured.config.final_proof_backend_name = final_proof_backend_name;
    configured
}

pub fn assemble_spartan_final_proof<Digest>(
    instance: FinalFoldedInstance<Digest>,
    spartan_proof_bytes: Vec<u8>,
) -> Result<FinalFoldedProof<Digest>, BlockTraceError>
where
    Digest: AsRef<[u8]>,
{
    if spartan_proof_bytes.is_empty() {
        return Err(BlockTraceError::NovaFoldingBackendError {
            block_index: instance.metadata.last_block_index.unwrap_or(0),
            reason: "Spartan final proof bytes must not be empty",
        });
    }

    let spartan_encoding = encode_final_folded_instance_for_spartan(&instance)?;
    let spartan_encoding_digest = Some(spartan_encoding.encoding_digest);
    let proof_digest = digest_final_folded_proof(
        SPARTAN_FINAL_PROOF_SYSTEM_NAME,
        &instance,
        spartan_encoding_digest,
        Some(&spartan_proof_bytes),
    );

    Ok(FinalFoldedProof {
        proof_system: SPARTAN_FINAL_PROOF_SYSTEM_NAME,
        instance,
        spartan_encoding_digest,
        proof_digest,
        spartan_proof_bytes: Some(spartan_proof_bytes),
    })
}

pub fn prove_final_folded_instance_with_backend<Digest, Backend>(
    backend: &Backend,
    instance: FinalFoldedInstance<Digest>,
) -> Result<FinalFoldedProof<Digest>, BlockTraceError>
where
    Digest: Clone + PartialEq + AsRef<[u8]>,
    Backend: FinalFoldedProofBackend<Digest> + ?Sized,
{
    backend.prove(instance)
}

pub fn prove_final_folded_accumulator_with_backend<Digest, Backend>(
    backend: &Backend,
    accumulator: &NovaFoldAccumulator<Digest>,
) -> Result<FinalFoldedProof<Digest>, BlockTraceError>
where
    Digest: Clone + PartialEq + AsRef<[u8]>,
    Backend: FinalFoldedProofBackend<Digest> + ?Sized,
{
    backend.prove_from_accumulator(accumulator)
}

pub fn verify_final_folded_proof_with_backend<Digest, Backend>(
    backend: &Backend,
    accumulator: &NovaFoldAccumulator<Digest>,
    proof: &FinalFoldedProof<Digest>,
) -> Result<(), BlockTraceError>
where
    Digest: Clone + PartialEq + AsRef<[u8]>,
    Backend: FinalFoldedProofBackend<Digest> + ?Sized,
{
    backend.verify(accumulator, proof)
}

pub fn prove_configured_final_folded_instance<Digest>(
    instance: FinalFoldedInstance<Digest>,
) -> Result<FinalFoldedProof<Digest>, BlockTraceError>
where
    Digest: Clone + PartialEq + AsRef<[u8]>,
{
    match instance.config.final_proof_backend_name {
        SPARTAN_PLACEHOLDER_PROOF_SYSTEM_NAME => {
            prove_final_folded_instance_with_backend(&SpartanPlaceholderFinalProofBackend, instance)
        }
        SPARTAN_FINAL_PROOF_SYSTEM_NAME => {
            prove_final_folded_instance_with_backend(&SpartanFinalProofBackend, instance)
        }
        _ => Err(BlockTraceError::NovaFoldingBackendError {
            block_index: instance.metadata.last_block_index.unwrap_or(0),
            reason: "unsupported final folded proof backend",
        }),
    }
}

pub fn prove_configured_final_folded_accumulator<Digest>(
    accumulator: &NovaFoldAccumulator<Digest>,
) -> Result<FinalFoldedProof<Digest>, BlockTraceError>
where
    Digest: Clone + PartialEq + AsRef<[u8]>,
{
    match accumulator.config.final_proof_backend_name {
        SPARTAN_PLACEHOLDER_PROOF_SYSTEM_NAME => prove_final_folded_accumulator_with_backend(
            &SpartanPlaceholderFinalProofBackend,
            accumulator,
        ),
        SPARTAN_FINAL_PROOF_SYSTEM_NAME => {
            prove_final_folded_accumulator_with_backend(&SpartanFinalProofBackend, accumulator)
        }
        _ => Err(BlockTraceError::NovaFoldingBackendError {
            block_index: accumulator.metadata.last_block_index.unwrap_or(0),
            reason: "unsupported final folded proof backend",
        }),
    }
}

pub fn verify_configured_final_folded_proof<Digest>(
    accumulator: &NovaFoldAccumulator<Digest>,
    proof: &FinalFoldedProof<Digest>,
) -> Result<(), BlockTraceError>
where
    Digest: Clone + PartialEq + AsRef<[u8]>,
{
    if proof.proof_system != proof.instance.config.final_proof_backend_name {
        return Err(BlockTraceError::NovaFoldingBackendError {
            block_index: proof.instance.metadata.last_block_index.unwrap_or(0),
            reason: "final folded proof system does not match configured backend",
        });
    }

    match proof.instance.config.final_proof_backend_name {
        SPARTAN_PLACEHOLDER_PROOF_SYSTEM_NAME => verify_final_folded_proof_with_backend(
            &SpartanPlaceholderFinalProofBackend,
            accumulator,
            proof,
        ),
        SPARTAN_FINAL_PROOF_SYSTEM_NAME => {
            verify_final_folded_proof_with_backend(&SpartanFinalProofBackend, accumulator, proof)
        }
        _ => Err(BlockTraceError::NovaFoldingBackendError {
            block_index: proof.instance.metadata.last_block_index.unwrap_or(0),
            reason: "unsupported final folded proof backend",
        }),
    }
}

pub fn prove_spartan_placeholder_final_instance<Digest>(
    instance: FinalFoldedInstance<Digest>,
) -> FinalFoldedProof<Digest>
where
    Digest: Clone + PartialEq + AsRef<[u8]>,
{
    prove_final_folded_instance_with_backend(&SpartanPlaceholderFinalProofBackend, instance)
        .expect("Spartan placeholder final proof construction should not fail")
}

pub fn verify_spartan_placeholder_final_proof<Digest>(
    accumulator: &NovaFoldAccumulator<Digest>,
    proof: &FinalFoldedProof<Digest>,
) -> Result<(), BlockTraceError>
where
    Digest: Clone + PartialEq + AsRef<[u8]>,
{
    verify_final_folded_proof_with_backend(&SpartanPlaceholderFinalProofBackend, accumulator, proof)
}

fn ensure_supported_nova_relation(
    config: &NovaFoldConfig,
    block_index: usize,
) -> Result<(), BlockTraceError> {
    if config.relation_name == NOVA_BLOCK_FOLD_RELATION_NAME {
        Ok(())
    } else {
        Err(BlockTraceError::NovaFoldingBackendError {
            block_index,
            reason: "unsupported Nova fold relation",
        })
    }
}

fn ensure_supported_nova_subclaim_backend(
    config: &NovaFoldConfig,
    block_index: usize,
) -> Result<(), BlockTraceError> {
    if config.subclaim_backend_name == NOVA_TRANSCRIPT_SUBCLAIM_BACKEND_NAME
        || config.subclaim_backend_name == NOVA_LOGUP_SUBCLAIM_BACKEND_NAME
    {
        Ok(())
    } else {
        Err(BlockTraceError::NovaFoldingBackendError {
            block_index,
            reason: "unsupported Nova subclaim folding backend",
        })
    }
}

fn ensure_supported_nova_config(
    config: &NovaFoldConfig,
    block_index: usize,
) -> Result<(), BlockTraceError> {
    ensure_supported_nova_relation(config, block_index)?;
    ensure_supported_nova_subclaim_backend(config, block_index)
}

impl<Digest, F> BlockFoldingBackend<Digest, F> for NovaFoldingBackend
where
    Digest: Clone + PartialEq + AsRef<[u8]>,
    F: JoltField,
{
    type Accumulator = NovaFoldAccumulator<Digest>;

    fn name(&self) -> &'static str {
        self.config.backend_name
    }

    fn relation_name(&self) -> &'static str {
        self.config.relation_name
    }

    fn new_accumulator(&self) -> Self::Accumulator {
        NovaFoldAccumulator {
            config: self.config.clone(),
            metadata: BlockFoldAccumulator::new(),
            recursive_snark_bytes: None,
            recursive_snark_output_digest: None,
            recursive_z_state: None,
        }
    }

    fn absorb(
        &self,
        accumulator: &mut Self::Accumulator,
        fold_input: &BlockFoldInput<Digest, F>,
    ) -> Result<(), BlockTraceError> {
        ensure_supported_nova_config(&self.config, fold_input.state.block_index)?;

        #[cfg(feature = "nova")]
        {
            let mut next_metadata = accumulator.metadata.clone();
            next_metadata.absorb(fold_input)?;

            let (recursive_snark_bytes, recursive_snark_output_digest, recursive_z_state) =
                prove_nova_recursive_snark_step(
                    accumulator.recursive_snark_bytes.as_deref(),
                    next_metadata.absorbed_blocks,
                    fold_input.state.block_index,
                    &self.config,
                    match accumulator.recursive_z_state.as_ref() {
                        Some(stored_z_state) => {
                            nova_z_state_from_storage(stored_z_state, fold_input.state.block_index)?
                        }
                        None => nova_initial_z_state(),
                    },
                    fold_input,
                )?;

            accumulator.metadata = next_metadata;
            accumulator.recursive_snark_bytes = Some(recursive_snark_bytes);
            accumulator.recursive_snark_output_digest = Some(recursive_snark_output_digest);
            accumulator.recursive_z_state = Some(recursive_z_state);

            return Ok(());
        }

        #[cfg(not(feature = "nova"))]
        {
            let _ = accumulator;
            Err(BlockTraceError::NovaFoldingBackendUnavailable {
                block_index: fold_input.state.block_index,
                reason: "compile jolt-core with the `nova` feature to enable Nova folding",
            })
        }
    }

    fn absorbed_blocks(&self, accumulator: &Self::Accumulator) -> usize {
        accumulator.metadata.absorbed_blocks
    }

    fn verify(
        &self,
        fold_inputs: &[BlockFoldInput<Digest, F>],
        accumulator: &Self::Accumulator,
    ) -> Result<(), BlockTraceError> {
        ensure_supported_nova_config(
            &self.config,
            fold_inputs
                .last()
                .map(|fold_input| fold_input.state.block_index)
                .or(accumulator.metadata.last_block_index)
                .unwrap_or(0),
        )?;

        if accumulator.config != self.config {
            return Err(BlockTraceError::NovaFoldingBackendError {
                block_index: accumulator.metadata.last_block_index.unwrap_or(0),
                reason: "Nova fold config mismatch",
            });
        }

        let mut expected_metadata = BlockFoldAccumulator::new();
        validate_block_fold_input_chain(fold_inputs)?;
        for fold_input in fold_inputs {
            expected_metadata.absorb(fold_input)?;
        }

        if expected_metadata != accumulator.metadata {
            return Err(BlockTraceError::BlockFoldAccumulatorMismatch {
                expected_blocks: expected_metadata.absorbed_blocks,
                actual_blocks: accumulator.metadata.absorbed_blocks,
            });
        }

        #[cfg(feature = "nova")]
        {
            verify_nova_recursive_snark_accumulator(fold_inputs, accumulator)
        }

        #[cfg(not(feature = "nova"))]
        {
            if fold_inputs.is_empty() {
                Ok(())
            } else {
                Err(BlockTraceError::NovaFoldingBackendUnavailable {
                    block_index: fold_inputs
                        .last()
                        .map(|fold_input| fold_input.state.block_index)
                        .unwrap_or(0),
                    reason: "compile jolt-core with the `nova` feature to enable Nova folding",
                })
            }
        }
    }
}

#[cfg(feature = "nova")]
type NovaPrimaryEngine = nova_snark::provider::PallasEngine;

#[cfg(feature = "nova")]
type NovaSecondaryEngine = nova_snark::provider::VestaEngine;

#[cfg(feature = "nova")]
type NovaScalar = <NovaPrimaryEngine as nova_snark::traits::Engine>::Scalar;

#[cfg(feature = "nova")]
const NOVA_TRANSCRIPT_DOMAIN_SEMANTIC: &str = "semantic";
#[cfg(feature = "nova")]
const NOVA_TRANSCRIPT_DOMAIN_REGISTER: &str = "register";
#[cfg(feature = "nova")]
const NOVA_TRANSCRIPT_DOMAIN_RAM: &str = "ram";
#[cfg(feature = "nova")]
const NOVA_TRANSCRIPT_DOMAIN_LOOKUP: &str = "lookup";
#[cfg(feature = "nova")]
const NOVA_TRANSCRIPT_DOMAIN_LOOKUP_LOGUP: &str = "lookup-logup";
#[cfg(feature = "nova")]
const NOVA_TRANSCRIPT_DOMAIN_CPU: &str = "cpu";
#[cfg(feature = "nova")]
const NOVA_TRANSCRIPT_DOMAIN_STATEMENT: &str = "statement";

#[cfg(feature = "nova")]
type NovaZState = [NovaScalar; NOVA_Z_ARITY];

#[cfg(feature = "nova")]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct BlockFoldStatement {
    program_digest: NovaScalar,
    block_index: NovaScalar,
    global_cycle_start: NovaScalar,
    global_cycle_end: NovaScalar,
    active_cycles: NovaScalar,
    start_state_digest: NovaScalar,
    end_state_digest: NovaScalar,
    start_register_digest: NovaScalar,
    end_register_digest: NovaScalar,
    register_reads_digest: NovaScalar,
    register_writes_digest: NovaScalar,
    state_digest: NovaScalar,
    register_read_count: NovaScalar,
    register_write_count: NovaScalar,
    ram_accesses_digest: NovaScalar,
    ram_touched_addresses_digest: NovaScalar,
    ram_access_count: NovaScalar,
    ram_touched_address_count: NovaScalar,
    lookup_claims_digest: NovaScalar,
    lookup_entry_summaries_digest: NovaScalar,
    lookup_count: NovaScalar,
    lookup_distinct_entry_count: NovaScalar,
    lookup_logup_proof_digest: NovaScalar,
    lookup_logup_tuple_challenge: NovaScalar,
    lookup_logup_denominator_challenge: NovaScalar,
    lookup_logup_denominator_retry_count: NovaScalar,
    lookup_logup_query_sum: NovaScalar,
    lookup_logup_table_sum: NovaScalar,
    r1cs_rows_checked: NovaScalar,
    r1cs_num_steps: NovaScalar,
    r1cs_vk_digest: NovaScalar,
    used_lookahead_cycle: NovaScalar,
}

#[cfg(feature = "nova")]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct BlockFoldSubclaimFingerprints {
    register: NovaScalar,
    ram: NovaScalar,
    lookup: NovaScalar,
    cpu: NovaScalar,
}

#[cfg(feature = "nova")]
trait NovaSubclaimFoldingBackend {
    fn name(&self) -> &'static str;

    fn lookup_backend_selector(&self) -> NovaScalar {
        NovaScalar::zero()
    }

    fn subclaim_fingerprints(
        &self,
        statement: &BlockFoldStatement,
    ) -> BlockFoldSubclaimFingerprints;
}

#[cfg(feature = "nova")]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct TranscriptSubclaimFoldingBackend;

#[cfg(feature = "nova")]
impl NovaSubclaimFoldingBackend for TranscriptSubclaimFoldingBackend {
    fn name(&self) -> &'static str {
        "transcript-subclaim-fingerprints"
    }

    fn subclaim_fingerprints(
        &self,
        statement: &BlockFoldStatement,
    ) -> BlockFoldSubclaimFingerprints {
        statement.subclaim_fingerprints()
    }
}

#[cfg(feature = "nova")]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct LogUpSubclaimFoldingBackend;

#[cfg(feature = "nova")]
impl NovaSubclaimFoldingBackend for LogUpSubclaimFoldingBackend {
    fn name(&self) -> &'static str {
        NOVA_LOGUP_SUBCLAIM_BACKEND_NAME
    }

    fn lookup_backend_selector(&self) -> NovaScalar {
        NovaScalar::from(1)
    }

    fn subclaim_fingerprints(
        &self,
        statement: &BlockFoldStatement,
    ) -> BlockFoldSubclaimFingerprints {
        let mut subclaims = statement.subclaim_fingerprints();
        subclaims.lookup = statement.lookup_logup_fingerprint();
        subclaims
    }
}

#[cfg(feature = "nova")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ConfiguredSubclaimFoldingBackend {
    Transcript(TranscriptSubclaimFoldingBackend),
    LogUp(LogUpSubclaimFoldingBackend),
}

#[cfg(feature = "nova")]
impl NovaSubclaimFoldingBackend for ConfiguredSubclaimFoldingBackend {
    fn name(&self) -> &'static str {
        match self {
            Self::Transcript(backend) => backend.name(),
            Self::LogUp(backend) => backend.name(),
        }
    }

    fn lookup_backend_selector(&self) -> NovaScalar {
        match self {
            Self::Transcript(backend) => backend.lookup_backend_selector(),
            Self::LogUp(backend) => backend.lookup_backend_selector(),
        }
    }

    fn subclaim_fingerprints(
        &self,
        statement: &BlockFoldStatement,
    ) -> BlockFoldSubclaimFingerprints {
        match self {
            Self::Transcript(backend) => backend.subclaim_fingerprints(statement),
            Self::LogUp(backend) => backend.subclaim_fingerprints(statement),
        }
    }
}

#[cfg(feature = "nova")]
impl BlockFoldStatement {
    /// Block-level folding relation used by the staged Jolt-Nova backend.
    ///
    /// For each block `i`, the Nova step relation consumes a private
    /// `BlockFoldStatement_i` derived from the verified Jolt block bundle and
    /// transforms public recursive state `z_i` into `z_{i+1}`:
    ///
    /// `z = [semantic, next_block_index, total_cycles, register, ram, lookup, cpu]`
    ///
    /// The transition enforced in-circuit is:
    ///
    /// - `next_block_index' = next_block_index + 1`
    /// - `total_cycles' = total_cycles + active_cycles_i`
    /// - claim accumulators advance by domain-separated transcript deltas
    /// - semantic accumulator binds the full statement digest and all subclaims
    ///
    /// Statement digest values are encoded as full-width Nova scalar field
    /// elements. The in-circuit statement digest is a domain-separated
    /// transcript fingerprint over the statement fields; `digest()` remains a
    /// native SHA3 summary for host-side diagnostics and future hash gadgets.
    fn from_fold_input<Digest, F>(fold_input: &BlockFoldInput<Digest, F>) -> Self
    where
        Digest: AsRef<[u8]>,
        F: JoltField,
    {
        let state = &fold_input.state;
        Self {
            program_digest: nova_hash_bytes_to_scalar(
                "statement-field",
                "program_digest",
                fold_input.program_digest.as_ref(),
            ),
            block_index: NovaScalar::from(state.block_index as u64),
            global_cycle_start: NovaScalar::from(state.global_cycle_start as u64),
            global_cycle_end: NovaScalar::from(state.global_cycle_end as u64),
            active_cycles: NovaScalar::from(state.active_cycles as u64),
            start_state_digest: nova_hash_bytes_to_scalar(
                "statement-field",
                "start_state_digest",
                &state.start_state_digest,
            ),
            end_state_digest: nova_hash_bytes_to_scalar(
                "statement-field",
                "end_state_digest",
                &state.end_state_digest,
            ),
            start_register_digest: nova_hash_bytes_to_scalar(
                "statement-field",
                "start_register_digest",
                &state.start_register_digest,
            ),
            end_register_digest: nova_hash_bytes_to_scalar(
                "statement-field",
                "end_register_digest",
                &state.end_register_digest,
            ),
            register_reads_digest: nova_hash_bytes_to_scalar(
                "statement-field",
                "register_reads_digest",
                &state.register_reads_digest,
            ),
            register_writes_digest: nova_hash_bytes_to_scalar(
                "statement-field",
                "register_writes_digest",
                &state.register_writes_digest,
            ),
            state_digest: nova_hash_bytes_to_scalar(
                "statement-field",
                "state_digest",
                &state.state_digest,
            ),
            register_read_count: NovaScalar::from(state.register_read_count as u64),
            register_write_count: NovaScalar::from(state.register_write_count as u64),
            ram_accesses_digest: nova_hash_bytes_to_scalar(
                "statement-field",
                "ram_accesses_digest",
                &state.ram_accesses_digest,
            ),
            ram_touched_addresses_digest: nova_hash_bytes_to_scalar(
                "statement-field",
                "ram_touched_addresses_digest",
                &state.ram_touched_addresses_digest,
            ),
            ram_access_count: NovaScalar::from(state.ram_access_count as u64),
            ram_touched_address_count: NovaScalar::from(state.ram_touched_address_count as u64),
            lookup_claims_digest: nova_hash_bytes_to_scalar(
                "statement-field",
                "lookup_claims_digest",
                &state.lookup_claims_digest,
            ),
            lookup_entry_summaries_digest: nova_hash_bytes_to_scalar(
                "statement-field",
                "lookup_entry_summaries_digest",
                &state.lookup_entry_summaries_digest,
            ),
            lookup_count: NovaScalar::from(state.lookup_count as u64),
            lookup_distinct_entry_count: NovaScalar::from(state.lookup_distinct_entry_count as u64),
            lookup_logup_proof_digest: nova_hash_bytes_to_scalar(
                "statement-field",
                "lookup_logup_proof_digest",
                &state.lookup_logup_proof_digest,
            ),
            lookup_logup_tuple_challenge: nova_jolt_field_to_scalar(
                "lookup_logup_tuple_challenge",
                state.lookup_logup_tuple_challenge,
            ),
            lookup_logup_denominator_challenge: nova_jolt_field_to_scalar(
                "lookup_logup_denominator_challenge",
                state.lookup_logup_denominator_challenge,
            ),
            lookup_logup_denominator_retry_count: NovaScalar::from(
                state.lookup_logup_denominator_retry_count as u64,
            ),
            lookup_logup_query_sum: nova_jolt_field_to_scalar(
                "lookup_logup_balanced_sum",
                state.lookup_logup_query_sum,
            ),
            lookup_logup_table_sum: nova_jolt_field_to_scalar(
                "lookup_logup_balanced_sum",
                state.lookup_logup_table_sum,
            ),
            r1cs_rows_checked: NovaScalar::from(state.r1cs_rows_checked as u64),
            r1cs_num_steps: NovaScalar::from(state.r1cs_num_steps as u64),
            r1cs_vk_digest: nova_jolt_field_to_scalar("r1cs_vk_digest", state.r1cs_vk_digest),
            used_lookahead_cycle: NovaScalar::from(u64::from(state.used_lookahead_cycle)),
        }
    }

    fn digest(&self) -> [u8; 32] {
        let mut hasher = Sha3_256::new();
        hasher.update(b"JOLT_NOVA_BLOCK_FOLD_STATEMENT_V1");
        for value in [
            self.program_digest,
            self.block_index,
            self.global_cycle_start,
            self.global_cycle_end,
            self.active_cycles,
            self.start_state_digest,
            self.end_state_digest,
            self.start_register_digest,
            self.end_register_digest,
            self.register_reads_digest,
            self.register_writes_digest,
            self.state_digest,
            self.register_read_count,
            self.register_write_count,
            self.ram_accesses_digest,
            self.ram_touched_addresses_digest,
            self.ram_access_count,
            self.ram_touched_address_count,
            self.lookup_claims_digest,
            self.lookup_entry_summaries_digest,
            self.lookup_count,
            self.lookup_distinct_entry_count,
            self.lookup_logup_proof_digest,
            self.lookup_logup_tuple_challenge,
            self.lookup_logup_denominator_challenge,
            self.lookup_logup_denominator_retry_count,
            self.lookup_logup_query_sum,
            self.lookup_logup_table_sum,
            self.r1cs_rows_checked,
            self.r1cs_num_steps,
            self.r1cs_vk_digest,
            self.used_lookahead_cycle,
        ] {
            update_nova_scalar(&mut hasher, value);
        }
        finalize_digest(hasher)
    }

    fn statement_digest_scalar(&self) -> NovaScalar {
        nova_transcript_delta(
            NOVA_TRANSCRIPT_DOMAIN_STATEMENT,
            [
                ("program_digest", self.program_digest),
                ("block_index", self.block_index),
                ("global_cycle_start", self.global_cycle_start),
                ("global_cycle_end", self.global_cycle_end),
                ("active_cycles", self.active_cycles),
                ("start_state_digest", self.start_state_digest),
                ("end_state_digest", self.end_state_digest),
                ("start_register_digest", self.start_register_digest),
                ("end_register_digest", self.end_register_digest),
                ("register_reads_digest", self.register_reads_digest),
                ("register_writes_digest", self.register_writes_digest),
                ("state_digest", self.state_digest),
                ("register_read_count", self.register_read_count),
                ("register_write_count", self.register_write_count),
                ("ram_accesses_digest", self.ram_accesses_digest),
                (
                    "ram_touched_addresses_digest",
                    self.ram_touched_addresses_digest,
                ),
                ("ram_access_count", self.ram_access_count),
                ("ram_touched_address_count", self.ram_touched_address_count),
                ("lookup_claims_digest", self.lookup_claims_digest),
                (
                    "lookup_entry_summaries_digest",
                    self.lookup_entry_summaries_digest,
                ),
                ("lookup_count", self.lookup_count),
                (
                    "lookup_distinct_entry_count",
                    self.lookup_distinct_entry_count,
                ),
                ("lookup_logup_proof_digest", self.lookup_logup_proof_digest),
                (
                    "lookup_logup_tuple_challenge",
                    self.lookup_logup_tuple_challenge,
                ),
                (
                    "lookup_logup_denominator_challenge",
                    self.lookup_logup_denominator_challenge,
                ),
                (
                    "lookup_logup_denominator_retry_count",
                    self.lookup_logup_denominator_retry_count,
                ),
                ("lookup_logup_query_sum", self.lookup_logup_query_sum),
                ("lookup_logup_table_sum", self.lookup_logup_table_sum),
                ("r1cs_rows_checked", self.r1cs_rows_checked),
                ("r1cs_num_steps", self.r1cs_num_steps),
                ("r1cs_vk_digest", self.r1cs_vk_digest),
                ("used_lookahead_cycle", self.used_lookahead_cycle),
            ],
        )
    }

    fn semantic_delta_with_subclaims(
        &self,
        subclaims: BlockFoldSubclaimFingerprints,
    ) -> NovaScalar {
        nova_transcript_delta(
            NOVA_TRANSCRIPT_DOMAIN_SEMANTIC,
            [
                ("statement_digest", self.statement_digest_scalar()),
                ("program_digest", self.program_digest),
                ("block_index", self.block_index),
                ("global_cycle_start", self.global_cycle_start),
                ("global_cycle_end", self.global_cycle_end),
                ("active_cycles", self.active_cycles),
                ("start_state_digest", self.start_state_digest),
                ("end_state_digest", self.end_state_digest),
                ("state_digest", self.state_digest),
                ("register_claim_fingerprint", subclaims.register),
                ("ram_claim_fingerprint", subclaims.ram),
                ("lookup_claim_fingerprint", subclaims.lookup),
                ("cpu_claim_fingerprint", subclaims.cpu),
            ],
        )
    }

    fn semantic_delta(&self) -> NovaScalar {
        self.semantic_delta_with_subclaims(self.subclaim_fingerprints())
    }

    fn register_fingerprint(&self) -> NovaScalar {
        nova_transcript_delta(
            NOVA_TRANSCRIPT_DOMAIN_REGISTER,
            [
                ("start_register_digest", self.start_register_digest),
                ("end_register_digest", self.end_register_digest),
                ("register_reads_digest", self.register_reads_digest),
                ("register_writes_digest", self.register_writes_digest),
                ("register_read_count", self.register_read_count),
                ("register_write_count", self.register_write_count),
            ],
        )
    }

    fn register_delta(&self) -> NovaScalar {
        self.register_fingerprint()
    }

    fn ram_fingerprint(&self) -> NovaScalar {
        nova_transcript_delta(
            NOVA_TRANSCRIPT_DOMAIN_RAM,
            [
                ("ram_accesses_digest", self.ram_accesses_digest),
                (
                    "ram_touched_addresses_digest",
                    self.ram_touched_addresses_digest,
                ),
                ("ram_access_count", self.ram_access_count),
                ("ram_touched_address_count", self.ram_touched_address_count),
            ],
        )
    }

    fn ram_delta(&self) -> NovaScalar {
        self.ram_fingerprint()
    }

    fn lookup_fingerprint(&self) -> NovaScalar {
        nova_transcript_delta(
            NOVA_TRANSCRIPT_DOMAIN_LOOKUP,
            [
                ("lookup_claims_digest", self.lookup_claims_digest),
                (
                    "lookup_entry_summaries_digest",
                    self.lookup_entry_summaries_digest,
                ),
                ("lookup_count", self.lookup_count),
                (
                    "lookup_distinct_entry_count",
                    self.lookup_distinct_entry_count,
                ),
            ],
        )
    }

    fn lookup_delta(&self) -> NovaScalar {
        self.lookup_fingerprint()
    }

    fn lookup_logup_fingerprint(&self) -> NovaScalar {
        nova_transcript_delta(
            NOVA_TRANSCRIPT_DOMAIN_LOOKUP_LOGUP,
            [
                ("lookup_claims_digest", self.lookup_claims_digest),
                (
                    "lookup_entry_summaries_digest",
                    self.lookup_entry_summaries_digest,
                ),
                ("lookup_count", self.lookup_count),
                (
                    "lookup_distinct_entry_count",
                    self.lookup_distinct_entry_count,
                ),
                ("lookup_logup_proof_digest", self.lookup_logup_proof_digest),
                (
                    "lookup_logup_tuple_challenge",
                    self.lookup_logup_tuple_challenge,
                ),
                (
                    "lookup_logup_denominator_challenge",
                    self.lookup_logup_denominator_challenge,
                ),
                (
                    "lookup_logup_denominator_retry_count",
                    self.lookup_logup_denominator_retry_count,
                ),
                ("lookup_logup_query_sum", self.lookup_logup_query_sum),
                ("lookup_logup_table_sum", self.lookup_logup_table_sum),
            ],
        )
    }

    fn cpu_fingerprint(&self) -> NovaScalar {
        nova_transcript_delta(
            NOVA_TRANSCRIPT_DOMAIN_CPU,
            [
                ("r1cs_rows_checked", self.r1cs_rows_checked),
                ("r1cs_num_steps", self.r1cs_num_steps),
                ("r1cs_vk_digest", self.r1cs_vk_digest),
                ("used_lookahead_cycle", self.used_lookahead_cycle),
            ],
        )
    }

    fn subclaim_fingerprints(&self) -> BlockFoldSubclaimFingerprints {
        BlockFoldSubclaimFingerprints {
            register: self.register_fingerprint(),
            ram: self.ram_fingerprint(),
            lookup: self.lookup_fingerprint(),
            cpu: self.cpu_fingerprint(),
        }
    }

    fn cpu_delta(&self) -> NovaScalar {
        self.cpu_fingerprint()
    }
}

#[cfg(feature = "nova")]
fn nova_hash_bytes_to_scalar(
    domain: &'static str,
    label: &'static str,
    bytes: &[u8],
) -> NovaScalar {
    let mut hasher = Sha3_256::new();
    hasher.update(b"JOLT_NOVA_HASH_TO_FIELD_V1");
    hasher.update(NOVA_BLOCK_FOLD_RELATION_NAME.as_bytes());
    hasher.update(domain.as_bytes());
    hasher.update(label.as_bytes());
    update_usize(&mut hasher, bytes.len());
    hasher.update(bytes);
    let digest = finalize_digest(hasher);
    let mut uniform = [0u8; 64];
    uniform[..32].copy_from_slice(&digest);

    let mut hasher = Sha3_256::new();
    hasher.update(b"JOLT_NOVA_HASH_TO_FIELD_WIDE_V1");
    hasher.update(&digest);
    let wide_digest = finalize_digest(hasher);
    uniform[32..].copy_from_slice(&wide_digest);

    NovaScalar::from_uniform(&uniform)
}

#[cfg(feature = "nova")]
fn nova_jolt_field_to_scalar<F>(label: &'static str, value: F) -> NovaScalar
where
    F: JoltField,
{
    let mut bytes = Vec::new();
    value
        .serialize_compressed(&mut bytes)
        .expect("serializing a field element into Vec<u8> should not fail");
    nova_hash_bytes_to_scalar("statement-field", label, &bytes)
}

#[cfg(feature = "nova")]
fn nova_transcript_challenge_scalar(domain: &'static str, label: &'static str) -> NovaScalar {
    nova_hash_bytes_to_scalar("transcript-challenge", domain, label.as_bytes())
}

#[cfg(feature = "nova")]
fn nova_transcript_delta<const N: usize>(
    domain: &'static str,
    fields: [(&'static str, NovaScalar); N],
) -> NovaScalar {
    fields
        .iter()
        .fold(NovaScalar::zero(), |acc, (label, value)| {
            acc + (*value * nova_transcript_challenge_scalar(domain, label))
        })
}

#[cfg(feature = "nova")]
#[derive(Clone, Debug, PartialEq, Eq)]
struct JoltNovaStepCircuit {
    witness: JoltNovaStepWitness,
}

#[cfg(feature = "nova")]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct JoltNovaStepWitness {
    statement_digest: NovaScalar,
    program_digest: NovaScalar,
    block_index: NovaScalar,
    global_cycle_start: NovaScalar,
    global_cycle_end: NovaScalar,
    active_cycles: NovaScalar,
    start_state_digest: NovaScalar,
    end_state_digest: NovaScalar,
    start_register_digest: NovaScalar,
    end_register_digest: NovaScalar,
    register_reads_digest: NovaScalar,
    register_writes_digest: NovaScalar,
    state_digest: NovaScalar,
    register_read_count: NovaScalar,
    register_write_count: NovaScalar,
    register_claim_fingerprint: NovaScalar,
    ram_accesses_digest: NovaScalar,
    ram_touched_addresses_digest: NovaScalar,
    ram_access_count: NovaScalar,
    ram_touched_address_count: NovaScalar,
    ram_claim_fingerprint: NovaScalar,
    lookup_claims_digest: NovaScalar,
    lookup_entry_summaries_digest: NovaScalar,
    lookup_count: NovaScalar,
    lookup_distinct_entry_count: NovaScalar,
    lookup_logup_proof_digest: NovaScalar,
    lookup_logup_tuple_challenge: NovaScalar,
    lookup_logup_denominator_challenge: NovaScalar,
    lookup_logup_denominator_retry_count: NovaScalar,
    lookup_logup_query_sum: NovaScalar,
    lookup_logup_table_sum: NovaScalar,
    lookup_backend_selector: NovaScalar,
    lookup_claim_fingerprint: NovaScalar,
    r1cs_rows_checked: NovaScalar,
    r1cs_num_steps: NovaScalar,
    r1cs_vk_digest: NovaScalar,
    used_lookahead_cycle: NovaScalar,
    cpu_claim_fingerprint: NovaScalar,
}

#[cfg(feature = "nova")]
impl JoltNovaStepWitness {
    fn from_fold_input<Digest, F>(fold_input: &BlockFoldInput<Digest, F>) -> Self
    where
        Digest: AsRef<[u8]>,
        F: JoltField,
    {
        Self::from_fold_input_with_subclaim_backend(fold_input, &TranscriptSubclaimFoldingBackend)
    }

    fn from_fold_input_with_subclaim_backend<Digest, F, Backend>(
        fold_input: &BlockFoldInput<Digest, F>,
        subclaim_backend: &Backend,
    ) -> Self
    where
        Digest: AsRef<[u8]>,
        F: JoltField,
        Backend: NovaSubclaimFoldingBackend,
    {
        let statement = BlockFoldStatement::from_fold_input(fold_input);
        Self::from_statement_with_subclaim_backend(statement, subclaim_backend)
    }

    fn from_statement_with_subclaim_backend<Backend>(
        statement: BlockFoldStatement,
        subclaim_backend: &Backend,
    ) -> Self
    where
        Backend: NovaSubclaimFoldingBackend,
    {
        let subclaims = subclaim_backend.subclaim_fingerprints(&statement);
        Self {
            statement_digest: statement.statement_digest_scalar(),
            program_digest: statement.program_digest,
            block_index: statement.block_index,
            global_cycle_start: statement.global_cycle_start,
            global_cycle_end: statement.global_cycle_end,
            active_cycles: statement.active_cycles,
            start_state_digest: statement.start_state_digest,
            end_state_digest: statement.end_state_digest,
            start_register_digest: statement.start_register_digest,
            end_register_digest: statement.end_register_digest,
            register_reads_digest: statement.register_reads_digest,
            register_writes_digest: statement.register_writes_digest,
            state_digest: statement.state_digest,
            register_read_count: statement.register_read_count,
            register_write_count: statement.register_write_count,
            register_claim_fingerprint: subclaims.register,
            ram_accesses_digest: statement.ram_accesses_digest,
            ram_touched_addresses_digest: statement.ram_touched_addresses_digest,
            ram_access_count: statement.ram_access_count,
            ram_touched_address_count: statement.ram_touched_address_count,
            ram_claim_fingerprint: subclaims.ram,
            lookup_claims_digest: statement.lookup_claims_digest,
            lookup_entry_summaries_digest: statement.lookup_entry_summaries_digest,
            lookup_count: statement.lookup_count,
            lookup_distinct_entry_count: statement.lookup_distinct_entry_count,
            lookup_logup_proof_digest: statement.lookup_logup_proof_digest,
            lookup_logup_tuple_challenge: statement.lookup_logup_tuple_challenge,
            lookup_logup_denominator_challenge: statement.lookup_logup_denominator_challenge,
            lookup_logup_denominator_retry_count: statement.lookup_logup_denominator_retry_count,
            lookup_logup_query_sum: statement.lookup_logup_query_sum,
            lookup_logup_table_sum: statement.lookup_logup_table_sum,
            lookup_backend_selector: subclaim_backend.lookup_backend_selector(),
            lookup_claim_fingerprint: subclaims.lookup,
            r1cs_rows_checked: statement.r1cs_rows_checked,
            r1cs_num_steps: statement.r1cs_num_steps,
            r1cs_vk_digest: statement.r1cs_vk_digest,
            used_lookahead_cycle: statement.used_lookahead_cycle,
            cpu_claim_fingerprint: subclaims.cpu,
        }
    }

    fn statement(&self) -> BlockFoldStatement {
        BlockFoldStatement {
            program_digest: self.program_digest,
            block_index: self.block_index,
            global_cycle_start: self.global_cycle_start,
            global_cycle_end: self.global_cycle_end,
            active_cycles: self.active_cycles,
            start_state_digest: self.start_state_digest,
            end_state_digest: self.end_state_digest,
            start_register_digest: self.start_register_digest,
            end_register_digest: self.end_register_digest,
            register_reads_digest: self.register_reads_digest,
            register_writes_digest: self.register_writes_digest,
            state_digest: self.state_digest,
            register_read_count: self.register_read_count,
            register_write_count: self.register_write_count,
            ram_accesses_digest: self.ram_accesses_digest,
            ram_touched_addresses_digest: self.ram_touched_addresses_digest,
            ram_access_count: self.ram_access_count,
            ram_touched_address_count: self.ram_touched_address_count,
            lookup_claims_digest: self.lookup_claims_digest,
            lookup_entry_summaries_digest: self.lookup_entry_summaries_digest,
            lookup_count: self.lookup_count,
            lookup_distinct_entry_count: self.lookup_distinct_entry_count,
            lookup_logup_proof_digest: self.lookup_logup_proof_digest,
            lookup_logup_tuple_challenge: self.lookup_logup_tuple_challenge,
            lookup_logup_denominator_challenge: self.lookup_logup_denominator_challenge,
            lookup_logup_denominator_retry_count: self.lookup_logup_denominator_retry_count,
            lookup_logup_query_sum: self.lookup_logup_query_sum,
            lookup_logup_table_sum: self.lookup_logup_table_sum,
            r1cs_rows_checked: self.r1cs_rows_checked,
            r1cs_num_steps: self.r1cs_num_steps,
            r1cs_vk_digest: self.r1cs_vk_digest,
            used_lookahead_cycle: self.used_lookahead_cycle,
        }
    }

    fn semantic_delta(&self) -> NovaScalar {
        self.statement()
            .semantic_delta_with_subclaims(self.subclaim_fingerprints())
    }

    fn semantic_delta_scalar(&self) -> NovaScalar {
        self.semantic_delta()
    }

    fn register_delta(&self) -> NovaScalar {
        self.subclaim_fingerprints().register
    }

    fn register_delta_scalar(&self) -> NovaScalar {
        self.register_delta()
    }

    fn ram_delta(&self) -> NovaScalar {
        self.subclaim_fingerprints().ram
    }

    fn ram_delta_scalar(&self) -> NovaScalar {
        self.ram_delta()
    }

    fn lookup_delta(&self) -> NovaScalar {
        self.subclaim_fingerprints().lookup
    }

    fn lookup_delta_scalar(&self) -> NovaScalar {
        self.lookup_delta()
    }

    fn cpu_delta(&self) -> NovaScalar {
        self.subclaim_fingerprints().cpu
    }

    fn cpu_delta_scalar(&self) -> NovaScalar {
        self.cpu_delta()
    }

    fn subclaim_fingerprints(&self) -> BlockFoldSubclaimFingerprints {
        BlockFoldSubclaimFingerprints {
            register: self.register_claim_fingerprint,
            ram: self.ram_claim_fingerprint,
            lookup: self.lookup_claim_fingerprint,
            cpu: self.cpu_claim_fingerprint,
        }
    }
}

#[cfg(feature = "nova")]
impl JoltNovaStepCircuit {
    fn for_fold_input<Digest, F>(fold_input: &BlockFoldInput<Digest, F>) -> Self
    where
        Digest: AsRef<[u8]>,
        F: JoltField,
    {
        Self::for_fold_input_with_subclaim_backend(fold_input, &TranscriptSubclaimFoldingBackend)
    }

    fn for_fold_input_with_subclaim_backend<Digest, F, Backend>(
        fold_input: &BlockFoldInput<Digest, F>,
        subclaim_backend: &Backend,
    ) -> Self
    where
        Digest: AsRef<[u8]>,
        F: JoltField,
        Backend: NovaSubclaimFoldingBackend,
    {
        Self {
            witness: JoltNovaStepWitness::from_fold_input_with_subclaim_backend(
                fold_input,
                subclaim_backend,
            ),
        }
    }
}

#[cfg(feature = "nova")]
impl Default for JoltNovaStepCircuit {
    fn default() -> Self {
        Self {
            witness: JoltNovaStepWitness::default(),
        }
    }
}

#[cfg(feature = "nova")]
impl nova_snark::traits::circuit::StepCircuit<NovaScalar> for JoltNovaStepCircuit {
    fn arity(&self) -> usize {
        NOVA_Z_ARITY
    }

    fn synthesize<CS: nova_snark::frontend::ConstraintSystem<NovaScalar>>(
        &self,
        cs: &mut CS,
        z: &[nova_snark::frontend::num::AllocatedNum<NovaScalar>],
    ) -> Result<
        Vec<nova_snark::frontend::num::AllocatedNum<NovaScalar>>,
        nova_snark::frontend::SynthesisError,
    > {
        if z.len() != NOVA_Z_ARITY {
            return Err(nova_snark::frontend::SynthesisError::AssignmentMissing);
        }

        let accumulator = &z[0];
        let next_block_index = &z[1];
        let total_active_cycles = &z[2];
        let register_claim_accumulator = &z[3];
        let ram_claim_accumulator = &z[4];
        let lookup_claim_accumulator = &z[5];
        let cpu_claim_accumulator = &z[6];

        let statement_digest =
            alloc_nova_witness(cs, "statement digest", self.witness.statement_digest)?;
        let program_digest = alloc_nova_witness(cs, "program digest", self.witness.program_digest)?;
        let block_index = alloc_nova_witness(cs, "block index", self.witness.block_index)?;
        let global_cycle_start =
            alloc_nova_witness(cs, "global cycle start", self.witness.global_cycle_start)?;
        let global_cycle_end =
            alloc_nova_witness(cs, "global cycle end", self.witness.global_cycle_end)?;
        let active_cycles = alloc_nova_witness(cs, "active cycles", self.witness.active_cycles)?;
        let start_state_digest =
            alloc_nova_witness(cs, "start state digest", self.witness.start_state_digest)?;
        let end_state_digest =
            alloc_nova_witness(cs, "end state digest", self.witness.end_state_digest)?;
        let start_register_digest = alloc_nova_witness(
            cs,
            "start register digest",
            self.witness.start_register_digest,
        )?;
        let end_register_digest =
            alloc_nova_witness(cs, "end register digest", self.witness.end_register_digest)?;
        let register_reads_digest = alloc_nova_witness(
            cs,
            "register reads digest",
            self.witness.register_reads_digest,
        )?;
        let register_writes_digest = alloc_nova_witness(
            cs,
            "register writes digest",
            self.witness.register_writes_digest,
        )?;
        let state_digest = alloc_nova_witness(cs, "state digest", self.witness.state_digest)?;
        let register_read_count =
            alloc_nova_witness(cs, "register read count", self.witness.register_read_count)?;
        let register_write_count = alloc_nova_witness(
            cs,
            "register write count",
            self.witness.register_write_count,
        )?;
        let register_claim_fingerprint = alloc_nova_witness(
            cs,
            "register claim fingerprint",
            self.witness.register_claim_fingerprint,
        )?;
        let ram_accesses_digest =
            alloc_nova_witness(cs, "ram accesses digest", self.witness.ram_accesses_digest)?;
        let ram_touched_addresses_digest = alloc_nova_witness(
            cs,
            "ram touched addresses digest",
            self.witness.ram_touched_addresses_digest,
        )?;
        let ram_access_count =
            alloc_nova_witness(cs, "ram access count", self.witness.ram_access_count)?;
        let ram_touched_address_count = alloc_nova_witness(
            cs,
            "ram touched address count",
            self.witness.ram_touched_address_count,
        )?;
        let ram_claim_fingerprint = alloc_nova_witness(
            cs,
            "ram claim fingerprint",
            self.witness.ram_claim_fingerprint,
        )?;
        let lookup_claims_digest = alloc_nova_witness(
            cs,
            "lookup claims digest",
            self.witness.lookup_claims_digest,
        )?;
        let lookup_entry_summaries_digest = alloc_nova_witness(
            cs,
            "lookup entry summaries digest",
            self.witness.lookup_entry_summaries_digest,
        )?;
        let lookup_count = alloc_nova_witness(cs, "lookup count", self.witness.lookup_count)?;
        let lookup_distinct_entry_count = alloc_nova_witness(
            cs,
            "lookup distinct entry count",
            self.witness.lookup_distinct_entry_count,
        )?;
        let lookup_logup_proof_digest = alloc_nova_witness(
            cs,
            "lookup LogUp proof digest",
            self.witness.lookup_logup_proof_digest,
        )?;
        let lookup_logup_tuple_challenge = alloc_nova_witness(
            cs,
            "lookup LogUp tuple challenge",
            self.witness.lookup_logup_tuple_challenge,
        )?;
        let lookup_logup_denominator_challenge = alloc_nova_witness(
            cs,
            "lookup LogUp denominator challenge",
            self.witness.lookup_logup_denominator_challenge,
        )?;
        let lookup_logup_denominator_retry_count = alloc_nova_witness(
            cs,
            "lookup LogUp denominator retry count",
            self.witness.lookup_logup_denominator_retry_count,
        )?;
        let lookup_logup_query_sum = alloc_nova_witness(
            cs,
            "lookup LogUp query sum",
            self.witness.lookup_logup_query_sum,
        )?;
        let lookup_logup_table_sum = alloc_nova_witness(
            cs,
            "lookup LogUp table sum",
            self.witness.lookup_logup_table_sum,
        )?;
        let lookup_backend_selector = alloc_nova_witness(
            cs,
            "lookup backend selector",
            self.witness.lookup_backend_selector,
        )?;
        let lookup_claim_fingerprint = alloc_nova_witness(
            cs,
            "lookup claim fingerprint",
            self.witness.lookup_claim_fingerprint,
        )?;
        let r1cs_rows_checked =
            alloc_nova_witness(cs, "r1cs rows checked", self.witness.r1cs_rows_checked)?;
        let r1cs_num_steps = alloc_nova_witness(cs, "r1cs num steps", self.witness.r1cs_num_steps)?;
        let r1cs_vk_digest = alloc_nova_witness(cs, "r1cs vk digest", self.witness.r1cs_vk_digest)?;
        let used_lookahead_cycle = alloc_nova_witness(
            cs,
            "used lookahead cycle",
            self.witness.used_lookahead_cycle,
        )?;
        let cpu_claim_fingerprint = alloc_nova_witness(
            cs,
            "CPU R1CS claim fingerprint",
            self.witness.cpu_claim_fingerprint,
        )?;

        let output_accumulator = nova_snark::frontend::num::AllocatedNum::alloc(
            cs.namespace(|| "next semantic fold accumulator"),
            || {
                accumulator
                    .get_value()
                    .map(|current| current + self.witness.semantic_delta_scalar())
                    .ok_or(nova_snark::frontend::SynthesisError::AssignmentMissing)
            },
        )?;
        let output_next_block_index = nova_snark::frontend::num::AllocatedNum::alloc(
            cs.namespace(|| "next expected block index"),
            || {
                next_block_index
                    .get_value()
                    .map(|current| current + NovaScalar::from(1))
                    .ok_or(nova_snark::frontend::SynthesisError::AssignmentMissing)
            },
        )?;
        let output_total_active_cycles = nova_snark::frontend::num::AllocatedNum::alloc(
            cs.namespace(|| "next total active cycles"),
            || {
                total_active_cycles
                    .get_value()
                    .map(|current| current + self.witness.active_cycles)
                    .ok_or(nova_snark::frontend::SynthesisError::AssignmentMissing)
            },
        )?;
        let output_register_claim_accumulator = nova_snark::frontend::num::AllocatedNum::alloc(
            cs.namespace(|| "next register claim accumulator"),
            || {
                register_claim_accumulator
                    .get_value()
                    .map(|current| current + self.witness.register_delta_scalar())
                    .ok_or(nova_snark::frontend::SynthesisError::AssignmentMissing)
            },
        )?;
        let output_ram_claim_accumulator = nova_snark::frontend::num::AllocatedNum::alloc(
            cs.namespace(|| "next RAM claim accumulator"),
            || {
                ram_claim_accumulator
                    .get_value()
                    .map(|current| current + self.witness.ram_delta_scalar())
                    .ok_or(nova_snark::frontend::SynthesisError::AssignmentMissing)
            },
        )?;
        let output_lookup_claim_accumulator = nova_snark::frontend::num::AllocatedNum::alloc(
            cs.namespace(|| "next lookup claim accumulator"),
            || {
                lookup_claim_accumulator
                    .get_value()
                    .map(|current| current + self.witness.lookup_delta_scalar())
                    .ok_or(nova_snark::frontend::SynthesisError::AssignmentMissing)
            },
        )?;
        let output_cpu_claim_accumulator = nova_snark::frontend::num::AllocatedNum::alloc(
            cs.namespace(|| "next CPU R1CS claim accumulator"),
            || {
                cpu_claim_accumulator
                    .get_value()
                    .map(|current| current + self.witness.cpu_delta_scalar())
                    .ok_or(nova_snark::frontend::SynthesisError::AssignmentMissing)
            },
        )?;

        cs.enforce(
            || "block index matches running state",
            |lc| lc + block_index.get_variable(),
            |lc| lc + CS::one(),
            |lc| lc + next_block_index.get_variable(),
        );

        cs.enforce(
            || "statement digest binds statement fields",
            |lc| {
                lc + (
                    nova_transcript_challenge_scalar(
                        NOVA_TRANSCRIPT_DOMAIN_STATEMENT,
                        "program_digest",
                    ),
                    program_digest.get_variable(),
                ) + (
                    nova_transcript_challenge_scalar(
                        NOVA_TRANSCRIPT_DOMAIN_STATEMENT,
                        "block_index",
                    ),
                    block_index.get_variable(),
                ) + (
                    nova_transcript_challenge_scalar(
                        NOVA_TRANSCRIPT_DOMAIN_STATEMENT,
                        "global_cycle_start",
                    ),
                    global_cycle_start.get_variable(),
                ) + (
                    nova_transcript_challenge_scalar(
                        NOVA_TRANSCRIPT_DOMAIN_STATEMENT,
                        "global_cycle_end",
                    ),
                    global_cycle_end.get_variable(),
                ) + (
                    nova_transcript_challenge_scalar(
                        NOVA_TRANSCRIPT_DOMAIN_STATEMENT,
                        "active_cycles",
                    ),
                    active_cycles.get_variable(),
                ) + (
                    nova_transcript_challenge_scalar(
                        NOVA_TRANSCRIPT_DOMAIN_STATEMENT,
                        "start_state_digest",
                    ),
                    start_state_digest.get_variable(),
                ) + (
                    nova_transcript_challenge_scalar(
                        NOVA_TRANSCRIPT_DOMAIN_STATEMENT,
                        "end_state_digest",
                    ),
                    end_state_digest.get_variable(),
                ) + (
                    nova_transcript_challenge_scalar(
                        NOVA_TRANSCRIPT_DOMAIN_STATEMENT,
                        "start_register_digest",
                    ),
                    start_register_digest.get_variable(),
                ) + (
                    nova_transcript_challenge_scalar(
                        NOVA_TRANSCRIPT_DOMAIN_STATEMENT,
                        "end_register_digest",
                    ),
                    end_register_digest.get_variable(),
                ) + (
                    nova_transcript_challenge_scalar(
                        NOVA_TRANSCRIPT_DOMAIN_STATEMENT,
                        "register_reads_digest",
                    ),
                    register_reads_digest.get_variable(),
                ) + (
                    nova_transcript_challenge_scalar(
                        NOVA_TRANSCRIPT_DOMAIN_STATEMENT,
                        "register_writes_digest",
                    ),
                    register_writes_digest.get_variable(),
                ) + (
                    nova_transcript_challenge_scalar(
                        NOVA_TRANSCRIPT_DOMAIN_STATEMENT,
                        "state_digest",
                    ),
                    state_digest.get_variable(),
                ) + (
                    nova_transcript_challenge_scalar(
                        NOVA_TRANSCRIPT_DOMAIN_STATEMENT,
                        "register_read_count",
                    ),
                    register_read_count.get_variable(),
                ) + (
                    nova_transcript_challenge_scalar(
                        NOVA_TRANSCRIPT_DOMAIN_STATEMENT,
                        "register_write_count",
                    ),
                    register_write_count.get_variable(),
                ) + (
                    nova_transcript_challenge_scalar(
                        NOVA_TRANSCRIPT_DOMAIN_STATEMENT,
                        "ram_accesses_digest",
                    ),
                    ram_accesses_digest.get_variable(),
                ) + (
                    nova_transcript_challenge_scalar(
                        NOVA_TRANSCRIPT_DOMAIN_STATEMENT,
                        "ram_touched_addresses_digest",
                    ),
                    ram_touched_addresses_digest.get_variable(),
                ) + (
                    nova_transcript_challenge_scalar(
                        NOVA_TRANSCRIPT_DOMAIN_STATEMENT,
                        "ram_access_count",
                    ),
                    ram_access_count.get_variable(),
                ) + (
                    nova_transcript_challenge_scalar(
                        NOVA_TRANSCRIPT_DOMAIN_STATEMENT,
                        "ram_touched_address_count",
                    ),
                    ram_touched_address_count.get_variable(),
                ) + (
                    nova_transcript_challenge_scalar(
                        NOVA_TRANSCRIPT_DOMAIN_STATEMENT,
                        "lookup_claims_digest",
                    ),
                    lookup_claims_digest.get_variable(),
                ) + (
                    nova_transcript_challenge_scalar(
                        NOVA_TRANSCRIPT_DOMAIN_STATEMENT,
                        "lookup_entry_summaries_digest",
                    ),
                    lookup_entry_summaries_digest.get_variable(),
                ) + (
                    nova_transcript_challenge_scalar(
                        NOVA_TRANSCRIPT_DOMAIN_STATEMENT,
                        "lookup_count",
                    ),
                    lookup_count.get_variable(),
                ) + (
                    nova_transcript_challenge_scalar(
                        NOVA_TRANSCRIPT_DOMAIN_STATEMENT,
                        "lookup_distinct_entry_count",
                    ),
                    lookup_distinct_entry_count.get_variable(),
                ) + (
                    nova_transcript_challenge_scalar(
                        NOVA_TRANSCRIPT_DOMAIN_STATEMENT,
                        "lookup_logup_proof_digest",
                    ),
                    lookup_logup_proof_digest.get_variable(),
                ) + (
                    nova_transcript_challenge_scalar(
                        NOVA_TRANSCRIPT_DOMAIN_STATEMENT,
                        "lookup_logup_tuple_challenge",
                    ),
                    lookup_logup_tuple_challenge.get_variable(),
                ) + (
                    nova_transcript_challenge_scalar(
                        NOVA_TRANSCRIPT_DOMAIN_STATEMENT,
                        "lookup_logup_denominator_challenge",
                    ),
                    lookup_logup_denominator_challenge.get_variable(),
                ) + (
                    nova_transcript_challenge_scalar(
                        NOVA_TRANSCRIPT_DOMAIN_STATEMENT,
                        "lookup_logup_denominator_retry_count",
                    ),
                    lookup_logup_denominator_retry_count.get_variable(),
                ) + (
                    nova_transcript_challenge_scalar(
                        NOVA_TRANSCRIPT_DOMAIN_STATEMENT,
                        "lookup_logup_query_sum",
                    ),
                    lookup_logup_query_sum.get_variable(),
                ) + (
                    nova_transcript_challenge_scalar(
                        NOVA_TRANSCRIPT_DOMAIN_STATEMENT,
                        "lookup_logup_table_sum",
                    ),
                    lookup_logup_table_sum.get_variable(),
                ) + (
                    nova_transcript_challenge_scalar(
                        NOVA_TRANSCRIPT_DOMAIN_STATEMENT,
                        "r1cs_rows_checked",
                    ),
                    r1cs_rows_checked.get_variable(),
                ) + (
                    nova_transcript_challenge_scalar(
                        NOVA_TRANSCRIPT_DOMAIN_STATEMENT,
                        "r1cs_num_steps",
                    ),
                    r1cs_num_steps.get_variable(),
                ) + (
                    nova_transcript_challenge_scalar(
                        NOVA_TRANSCRIPT_DOMAIN_STATEMENT,
                        "r1cs_vk_digest",
                    ),
                    r1cs_vk_digest.get_variable(),
                ) + (
                    nova_transcript_challenge_scalar(
                        NOVA_TRANSCRIPT_DOMAIN_STATEMENT,
                        "used_lookahead_cycle",
                    ),
                    used_lookahead_cycle.get_variable(),
                )
            },
            |lc| lc + CS::one(),
            |lc| lc + statement_digest.get_variable(),
        );

        cs.enforce(
            || "semantic fold accumulator transition",
            |lc| {
                lc + accumulator.get_variable()
                    + (
                        nova_transcript_challenge_scalar(
                            NOVA_TRANSCRIPT_DOMAIN_SEMANTIC,
                            "statement_digest",
                        ),
                        statement_digest.get_variable(),
                    )
                    + (
                        nova_transcript_challenge_scalar(
                            NOVA_TRANSCRIPT_DOMAIN_SEMANTIC,
                            "program_digest",
                        ),
                        program_digest.get_variable(),
                    )
                    + (
                        nova_transcript_challenge_scalar(
                            NOVA_TRANSCRIPT_DOMAIN_SEMANTIC,
                            "block_index",
                        ),
                        block_index.get_variable(),
                    )
                    + (
                        nova_transcript_challenge_scalar(
                            NOVA_TRANSCRIPT_DOMAIN_SEMANTIC,
                            "global_cycle_start",
                        ),
                        global_cycle_start.get_variable(),
                    )
                    + (
                        nova_transcript_challenge_scalar(
                            NOVA_TRANSCRIPT_DOMAIN_SEMANTIC,
                            "global_cycle_end",
                        ),
                        global_cycle_end.get_variable(),
                    )
                    + (
                        nova_transcript_challenge_scalar(
                            NOVA_TRANSCRIPT_DOMAIN_SEMANTIC,
                            "active_cycles",
                        ),
                        active_cycles.get_variable(),
                    )
                    + (
                        nova_transcript_challenge_scalar(
                            NOVA_TRANSCRIPT_DOMAIN_SEMANTIC,
                            "start_state_digest",
                        ),
                        start_state_digest.get_variable(),
                    )
                    + (
                        nova_transcript_challenge_scalar(
                            NOVA_TRANSCRIPT_DOMAIN_SEMANTIC,
                            "end_state_digest",
                        ),
                        end_state_digest.get_variable(),
                    )
                    + (
                        nova_transcript_challenge_scalar(
                            NOVA_TRANSCRIPT_DOMAIN_SEMANTIC,
                            "state_digest",
                        ),
                        state_digest.get_variable(),
                    )
                    + (
                        nova_transcript_challenge_scalar(
                            NOVA_TRANSCRIPT_DOMAIN_SEMANTIC,
                            "register_claim_fingerprint",
                        ),
                        register_claim_fingerprint.get_variable(),
                    )
                    + (
                        nova_transcript_challenge_scalar(
                            NOVA_TRANSCRIPT_DOMAIN_SEMANTIC,
                            "ram_claim_fingerprint",
                        ),
                        ram_claim_fingerprint.get_variable(),
                    )
                    + (
                        nova_transcript_challenge_scalar(
                            NOVA_TRANSCRIPT_DOMAIN_SEMANTIC,
                            "lookup_claim_fingerprint",
                        ),
                        lookup_claim_fingerprint.get_variable(),
                    )
                    + (
                        nova_transcript_challenge_scalar(
                            NOVA_TRANSCRIPT_DOMAIN_SEMANTIC,
                            "cpu_claim_fingerprint",
                        ),
                        cpu_claim_fingerprint.get_variable(),
                    )
            },
            |lc| lc + CS::one(),
            |lc| lc + output_accumulator.get_variable(),
        );

        cs.enforce(
            || "next block index increments by one",
            |lc| lc + next_block_index.get_variable() + (NovaScalar::from(1), CS::one()),
            |lc| lc + CS::one(),
            |lc| lc + output_next_block_index.get_variable(),
        );

        cs.enforce(
            || "total active cycles accumulate",
            |lc| lc + total_active_cycles.get_variable() + active_cycles.get_variable(),
            |lc| lc + CS::one(),
            |lc| lc + output_total_active_cycles.get_variable(),
        );

        cs.enforce(
            || "register claim fingerprint binds register fields",
            |lc| {
                lc + (
                    nova_transcript_challenge_scalar(
                        NOVA_TRANSCRIPT_DOMAIN_REGISTER,
                        "start_register_digest",
                    ),
                    start_register_digest.get_variable(),
                ) + (
                    nova_transcript_challenge_scalar(
                        NOVA_TRANSCRIPT_DOMAIN_REGISTER,
                        "end_register_digest",
                    ),
                    end_register_digest.get_variable(),
                ) + (
                    nova_transcript_challenge_scalar(
                        NOVA_TRANSCRIPT_DOMAIN_REGISTER,
                        "register_reads_digest",
                    ),
                    register_reads_digest.get_variable(),
                ) + (
                    nova_transcript_challenge_scalar(
                        NOVA_TRANSCRIPT_DOMAIN_REGISTER,
                        "register_writes_digest",
                    ),
                    register_writes_digest.get_variable(),
                ) + (
                    nova_transcript_challenge_scalar(
                        NOVA_TRANSCRIPT_DOMAIN_REGISTER,
                        "register_read_count",
                    ),
                    register_read_count.get_variable(),
                ) + (
                    nova_transcript_challenge_scalar(
                        NOVA_TRANSCRIPT_DOMAIN_REGISTER,
                        "register_write_count",
                    ),
                    register_write_count.get_variable(),
                )
            },
            |lc| lc + CS::one(),
            |lc| lc + register_claim_fingerprint.get_variable(),
        );

        cs.enforce(
            || "register claim accumulator transition",
            |lc| {
                lc + register_claim_accumulator.get_variable()
                    + register_claim_fingerprint.get_variable()
            },
            |lc| lc + CS::one(),
            |lc| lc + output_register_claim_accumulator.get_variable(),
        );

        cs.enforce(
            || "ram claim fingerprint binds RAM fields",
            |lc| {
                lc + (
                    nova_transcript_challenge_scalar(
                        NOVA_TRANSCRIPT_DOMAIN_RAM,
                        "ram_accesses_digest",
                    ),
                    ram_accesses_digest.get_variable(),
                ) + (
                    nova_transcript_challenge_scalar(
                        NOVA_TRANSCRIPT_DOMAIN_RAM,
                        "ram_touched_addresses_digest",
                    ),
                    ram_touched_addresses_digest.get_variable(),
                ) + (
                    nova_transcript_challenge_scalar(
                        NOVA_TRANSCRIPT_DOMAIN_RAM,
                        "ram_access_count",
                    ),
                    ram_access_count.get_variable(),
                ) + (
                    nova_transcript_challenge_scalar(
                        NOVA_TRANSCRIPT_DOMAIN_RAM,
                        "ram_touched_address_count",
                    ),
                    ram_touched_address_count.get_variable(),
                )
            },
            |lc| lc + CS::one(),
            |lc| lc + ram_claim_fingerprint.get_variable(),
        );

        cs.enforce(
            || "RAM claim accumulator transition",
            |lc| lc + ram_claim_accumulator.get_variable() + ram_claim_fingerprint.get_variable(),
            |lc| lc + CS::one(),
            |lc| lc + output_ram_claim_accumulator.get_variable(),
        );

        cs.enforce(
            || "lookup backend selector is boolean",
            |lc| lc + lookup_backend_selector.get_variable(),
            |lc| lc + lookup_backend_selector.get_variable() - (NovaScalar::from(1), CS::one()),
            |lc| lc,
        );

        cs.enforce(
            || "LogUp query and table sums balance",
            |lc| lc + lookup_logup_query_sum.get_variable() - lookup_logup_table_sum.get_variable(),
            |lc| lc + lookup_backend_selector.get_variable(),
            |lc| lc,
        );

        cs.enforce(
            || "lookup claim fingerprint binds lookup fields",
            |lc| {
                lc + (
                    nova_transcript_challenge_scalar(
                        NOVA_TRANSCRIPT_DOMAIN_LOOKUP_LOGUP,
                        "lookup_claims_digest",
                    ) - nova_transcript_challenge_scalar(
                        NOVA_TRANSCRIPT_DOMAIN_LOOKUP,
                        "lookup_claims_digest",
                    ),
                    lookup_claims_digest.get_variable(),
                ) + (
                    nova_transcript_challenge_scalar(
                        NOVA_TRANSCRIPT_DOMAIN_LOOKUP_LOGUP,
                        "lookup_entry_summaries_digest",
                    ) - nova_transcript_challenge_scalar(
                        NOVA_TRANSCRIPT_DOMAIN_LOOKUP,
                        "lookup_entry_summaries_digest",
                    ),
                    lookup_entry_summaries_digest.get_variable(),
                ) + (
                    nova_transcript_challenge_scalar(
                        NOVA_TRANSCRIPT_DOMAIN_LOOKUP_LOGUP,
                        "lookup_count",
                    ) - nova_transcript_challenge_scalar(
                        NOVA_TRANSCRIPT_DOMAIN_LOOKUP,
                        "lookup_count",
                    ),
                    lookup_count.get_variable(),
                ) + (
                    nova_transcript_challenge_scalar(
                        NOVA_TRANSCRIPT_DOMAIN_LOOKUP_LOGUP,
                        "lookup_distinct_entry_count",
                    ) - nova_transcript_challenge_scalar(
                        NOVA_TRANSCRIPT_DOMAIN_LOOKUP,
                        "lookup_distinct_entry_count",
                    ),
                    lookup_distinct_entry_count.get_variable(),
                ) + (
                    nova_transcript_challenge_scalar(
                        NOVA_TRANSCRIPT_DOMAIN_LOOKUP_LOGUP,
                        "lookup_logup_proof_digest",
                    ),
                    lookup_logup_proof_digest.get_variable(),
                ) + (
                    nova_transcript_challenge_scalar(
                        NOVA_TRANSCRIPT_DOMAIN_LOOKUP_LOGUP,
                        "lookup_logup_tuple_challenge",
                    ),
                    lookup_logup_tuple_challenge.get_variable(),
                ) + (
                    nova_transcript_challenge_scalar(
                        NOVA_TRANSCRIPT_DOMAIN_LOOKUP_LOGUP,
                        "lookup_logup_denominator_challenge",
                    ),
                    lookup_logup_denominator_challenge.get_variable(),
                ) + (
                    nova_transcript_challenge_scalar(
                        NOVA_TRANSCRIPT_DOMAIN_LOOKUP_LOGUP,
                        "lookup_logup_denominator_retry_count",
                    ),
                    lookup_logup_denominator_retry_count.get_variable(),
                ) + (
                    nova_transcript_challenge_scalar(
                        NOVA_TRANSCRIPT_DOMAIN_LOOKUP_LOGUP,
                        "lookup_logup_query_sum",
                    ),
                    lookup_logup_query_sum.get_variable(),
                ) + (
                    nova_transcript_challenge_scalar(
                        NOVA_TRANSCRIPT_DOMAIN_LOOKUP_LOGUP,
                        "lookup_logup_table_sum",
                    ),
                    lookup_logup_table_sum.get_variable(),
                )
            },
            |lc| lc + lookup_backend_selector.get_variable(),
            |lc| {
                lc + lookup_claim_fingerprint.get_variable()
                    - (
                        nova_transcript_challenge_scalar(
                            NOVA_TRANSCRIPT_DOMAIN_LOOKUP,
                            "lookup_claims_digest",
                        ),
                        lookup_claims_digest.get_variable(),
                    )
                    - (
                        nova_transcript_challenge_scalar(
                            NOVA_TRANSCRIPT_DOMAIN_LOOKUP,
                            "lookup_entry_summaries_digest",
                        ),
                        lookup_entry_summaries_digest.get_variable(),
                    )
                    - (
                        nova_transcript_challenge_scalar(
                            NOVA_TRANSCRIPT_DOMAIN_LOOKUP,
                            "lookup_count",
                        ),
                        lookup_count.get_variable(),
                    )
                    - (
                        nova_transcript_challenge_scalar(
                            NOVA_TRANSCRIPT_DOMAIN_LOOKUP,
                            "lookup_distinct_entry_count",
                        ),
                        lookup_distinct_entry_count.get_variable(),
                    )
            },
        );

        cs.enforce(
            || "lookup claim accumulator transition",
            |lc| {
                lc + lookup_claim_accumulator.get_variable()
                    + lookup_claim_fingerprint.get_variable()
            },
            |lc| lc + CS::one(),
            |lc| lc + output_lookup_claim_accumulator.get_variable(),
        );

        cs.enforce(
            || "CPU R1CS claim fingerprint binds CPU fields",
            |lc| {
                lc + (
                    nova_transcript_challenge_scalar(
                        NOVA_TRANSCRIPT_DOMAIN_CPU,
                        "r1cs_rows_checked",
                    ),
                    r1cs_rows_checked.get_variable(),
                ) + (
                    nova_transcript_challenge_scalar(NOVA_TRANSCRIPT_DOMAIN_CPU, "r1cs_num_steps"),
                    r1cs_num_steps.get_variable(),
                ) + (
                    nova_transcript_challenge_scalar(NOVA_TRANSCRIPT_DOMAIN_CPU, "r1cs_vk_digest"),
                    r1cs_vk_digest.get_variable(),
                ) + (
                    nova_transcript_challenge_scalar(
                        NOVA_TRANSCRIPT_DOMAIN_CPU,
                        "used_lookahead_cycle",
                    ),
                    used_lookahead_cycle.get_variable(),
                )
            },
            |lc| lc + CS::one(),
            |lc| lc + cpu_claim_fingerprint.get_variable(),
        );

        cs.enforce(
            || "CPU R1CS claim accumulator transition",
            |lc| lc + cpu_claim_accumulator.get_variable() + cpu_claim_fingerprint.get_variable(),
            |lc| lc + CS::one(),
            |lc| lc + output_cpu_claim_accumulator.get_variable(),
        );

        Ok(vec![
            output_accumulator,
            output_next_block_index,
            output_total_active_cycles,
            output_register_claim_accumulator,
            output_ram_claim_accumulator,
            output_lookup_claim_accumulator,
            output_cpu_claim_accumulator,
        ])
    }
}

#[cfg(feature = "nova")]
type NovaStepCircuit = JoltNovaStepCircuit;

#[cfg(feature = "nova")]
type NovaRecursiveSnark =
    nova_snark::nova::RecursiveSNARK<NovaPrimaryEngine, NovaSecondaryEngine, NovaStepCircuit>;

#[cfg(feature = "nova")]
type NovaEvaluationEngine<E> = nova_snark::provider::ipa_pc::EvaluationEngine<E>;

#[cfg(feature = "nova")]
type NovaPrimarySpartanSnark = nova_snark::spartan::snark::RelaxedR1CSSNARK<
    NovaPrimaryEngine,
    NovaEvaluationEngine<NovaPrimaryEngine>,
>;

#[cfg(feature = "nova")]
type NovaSecondarySpartanSnark = nova_snark::spartan::snark::RelaxedR1CSSNARK<
    NovaSecondaryEngine,
    NovaEvaluationEngine<NovaSecondaryEngine>,
>;

#[cfg(feature = "nova")]
type NovaCompressedSnark = nova_snark::nova::CompressedSNARK<
    NovaPrimaryEngine,
    NovaSecondaryEngine,
    NovaStepCircuit,
    NovaPrimarySpartanSnark,
    NovaSecondarySpartanSnark,
>;

#[cfg(feature = "nova")]
type NovaCompressedProverKey = nova_snark::nova::ProverKey<
    NovaPrimaryEngine,
    NovaSecondaryEngine,
    NovaStepCircuit,
    NovaPrimarySpartanSnark,
    NovaSecondarySpartanSnark,
>;

#[cfg(feature = "nova")]
type NovaCompressedVerifierKey = nova_snark::nova::VerifierKey<
    NovaPrimaryEngine,
    NovaSecondaryEngine,
    NovaStepCircuit,
    NovaPrimarySpartanSnark,
    NovaSecondarySpartanSnark,
>;

#[cfg(feature = "nova")]
type NovaPublicParams =
    nova_snark::nova::PublicParams<NovaPrimaryEngine, NovaSecondaryEngine, NovaStepCircuit>;

#[cfg(feature = "nova")]
static NOVA_PUBLIC_PARAMS: OnceLock<Result<NovaPublicParams, &'static str>> = OnceLock::new();

#[cfg(feature = "nova")]
static NOVA_COMPRESSED_KEYS: OnceLock<
    Result<(NovaCompressedProverKey, NovaCompressedVerifierKey), &'static str>,
> = OnceLock::new();

#[cfg(feature = "nova")]
fn nova_setup_circuit() -> NovaStepCircuit {
    NovaStepCircuit::default()
}

#[cfg(feature = "nova")]
fn nova_step_circuit_for_fold_input<Digest, F>(
    fold_input: &BlockFoldInput<Digest, F>,
) -> NovaStepCircuit
where
    Digest: AsRef<[u8]>,
    F: JoltField,
{
    JoltNovaStepCircuit::for_fold_input(fold_input)
}

#[cfg(feature = "nova")]
fn nova_step_circuit_for_fold_input_with_subclaim_backend<Digest, F, Backend>(
    fold_input: &BlockFoldInput<Digest, F>,
    subclaim_backend: &Backend,
) -> NovaStepCircuit
where
    Digest: AsRef<[u8]>,
    F: JoltField,
    Backend: NovaSubclaimFoldingBackend,
{
    JoltNovaStepCircuit::for_fold_input_with_subclaim_backend(fold_input, subclaim_backend)
}

#[cfg(feature = "nova")]
fn nova_initial_z_state() -> NovaZState {
    [NovaScalar::zero(); NOVA_Z_ARITY]
}

#[cfg(feature = "nova")]
fn nova_initial_input() -> NovaZState {
    nova_initial_z_state()
}

#[cfg(feature = "nova")]
fn nova_scalar_to_storage(value: NovaScalar) -> [u8; 32] {
    value.to_bytes()
}

#[cfg(feature = "nova")]
fn nova_scalar_from_storage(
    bytes: &[u8; 32],
    block_index: usize,
) -> Result<NovaScalar, BlockTraceError> {
    Option::from(NovaScalar::from_bytes(bytes)).ok_or({
        BlockTraceError::NovaFoldingBackendError {
            block_index,
            reason: "Nova recursive z-state deserialization failed",
        }
    })
}

#[cfg(feature = "nova")]
fn nova_z_state_to_storage(z_state: NovaZState) -> NovaFoldZState {
    z_state.map(nova_scalar_to_storage)
}

#[cfg(feature = "nova")]
fn nova_z_state_from_storage(
    storage: &NovaFoldZState,
    block_index: usize,
) -> Result<NovaZState, BlockTraceError> {
    Ok([
        nova_scalar_from_storage(&storage[0], block_index)?,
        nova_scalar_from_storage(&storage[1], block_index)?,
        nova_scalar_from_storage(&storage[2], block_index)?,
        nova_scalar_from_storage(&storage[3], block_index)?,
        nova_scalar_from_storage(&storage[4], block_index)?,
        nova_scalar_from_storage(&storage[5], block_index)?,
        nova_scalar_from_storage(&storage[6], block_index)?,
    ])
}

#[cfg(feature = "nova")]
fn nova_initial_z_state_storage() -> NovaFoldZState {
    nova_z_state_to_storage(nova_initial_z_state())
}

#[cfg(feature = "nova")]
fn nova_next_z_state<Digest, F>(
    current_z_state: NovaZState,
    fold_input: &BlockFoldInput<Digest, F>,
) -> NovaZState
where
    Digest: AsRef<[u8]>,
    F: JoltField,
{
    nova_next_z_state_with_subclaim_backend(
        current_z_state,
        fold_input,
        &TranscriptSubclaimFoldingBackend,
    )
}

#[cfg(feature = "nova")]
fn nova_next_z_state_with_subclaim_backend<Digest, F, Backend>(
    current_z_state: NovaZState,
    fold_input: &BlockFoldInput<Digest, F>,
    subclaim_backend: &Backend,
) -> NovaZState
where
    Digest: AsRef<[u8]>,
    F: JoltField,
    Backend: NovaSubclaimFoldingBackend,
{
    let witness =
        JoltNovaStepWitness::from_fold_input_with_subclaim_backend(fold_input, subclaim_backend);
    let delta = nova_step_delta_vector(&witness);
    [
        current_z_state[NOVA_SEMANTIC_ACCUMULATOR_INDEX] + delta[0],
        current_z_state[NOVA_NEXT_BLOCK_INDEX_INDEX] + delta[1],
        current_z_state[NOVA_TOTAL_ACTIVE_CYCLES_INDEX] + delta[2],
        current_z_state[NOVA_REGISTER_ACCUMULATOR_INDEX] + delta[3],
        current_z_state[NOVA_RAM_ACCUMULATOR_INDEX] + delta[4],
        current_z_state[NOVA_LOOKUP_ACCUMULATOR_INDEX] + delta[5],
        current_z_state[NOVA_CPU_ACCUMULATOR_INDEX] + delta[6],
    ]
}

#[cfg(feature = "nova")]
fn nova_step_delta_vector(witness: &JoltNovaStepWitness) -> NovaZState {
    [
        witness.semantic_delta(),
        NovaScalar::from(1),
        witness.active_cycles,
        witness.register_delta(),
        witness.ram_delta(),
        witness.lookup_delta(),
        witness.cpu_delta(),
    ]
}

#[cfg(feature = "nova")]
fn nova_expected_z_state<Digest, F>(fold_inputs: &[BlockFoldInput<Digest, F>]) -> NovaZState
where
    Digest: AsRef<[u8]>,
    F: JoltField,
{
    nova_expected_z_state_with_subclaim_backend(fold_inputs, &TranscriptSubclaimFoldingBackend)
}

#[cfg(feature = "nova")]
fn nova_expected_z_state_with_subclaim_backend<Digest, F, Backend>(
    fold_inputs: &[BlockFoldInput<Digest, F>],
    subclaim_backend: &Backend,
) -> NovaZState
where
    Digest: AsRef<[u8]>,
    F: JoltField,
    Backend: NovaSubclaimFoldingBackend,
{
    fold_inputs
        .iter()
        .fold(nova_initial_z_state(), |z_state, fold_input| {
            nova_next_z_state_with_subclaim_backend(z_state, fold_input, subclaim_backend)
        })
}

#[cfg(feature = "nova")]
fn alloc_nova_witness<CS>(
    cs: &mut CS,
    label: &'static str,
    value: NovaScalar,
) -> Result<nova_snark::frontend::num::AllocatedNum<NovaScalar>, nova_snark::frontend::SynthesisError>
where
    CS: nova_snark::frontend::ConstraintSystem<NovaScalar>,
{
    nova_snark::frontend::num::AllocatedNum::alloc(cs.namespace(|| label), || Ok(value))
}

#[cfg(feature = "nova")]
fn nova_subclaim_backend_from_config(
    config: &NovaFoldConfig,
    block_index: usize,
) -> Result<ConfiguredSubclaimFoldingBackend, BlockTraceError> {
    ensure_supported_nova_subclaim_backend(config, block_index)?;
    match config.subclaim_backend_name {
        NOVA_TRANSCRIPT_SUBCLAIM_BACKEND_NAME => Ok(ConfiguredSubclaimFoldingBackend::Transcript(
            TranscriptSubclaimFoldingBackend,
        )),
        NOVA_LOGUP_SUBCLAIM_BACKEND_NAME => Ok(ConfiguredSubclaimFoldingBackend::LogUp(
            LogUpSubclaimFoldingBackend,
        )),
        _ => unreachable!("subclaim backend was validated above"),
    }
}

#[cfg(feature = "nova")]
fn nova_public_params(block_index: usize) -> Result<&'static NovaPublicParams, BlockTraceError> {
    match NOVA_PUBLIC_PARAMS.get_or_init(|| {
        let circuit = nova_setup_circuit();
        NovaPublicParams::setup(
            &circuit,
            &*nova_snark::traits::snark::default_ck_hint::<NovaPrimaryEngine>(),
            &*nova_snark::traits::snark::default_ck_hint::<NovaSecondaryEngine>(),
        )
        .map_err(|_| "Nova public parameter setup failed")
    }) {
        Ok(pp) => Ok(pp),
        Err(reason) => Err(BlockTraceError::NovaFoldingBackendError {
            block_index,
            reason,
        }),
    }
}

#[cfg(feature = "nova")]
fn nova_compressed_keys(
    block_index: usize,
) -> Result<&'static (NovaCompressedProverKey, NovaCompressedVerifierKey), BlockTraceError> {
    let pp = nova_public_params(block_index)?;
    match NOVA_COMPRESSED_KEYS.get_or_init(|| {
        NovaCompressedSnark::setup(pp).map_err(|_| "Nova compressed Spartan key setup failed")
    }) {
        Ok(keys) => Ok(keys),
        Err(reason) => Err(BlockTraceError::NovaFoldingBackendError {
            block_index,
            reason,
        }),
    }
}

#[cfg(feature = "nova")]
fn prove_spartan_compressed_final_proof_from_accumulator<Digest>(
    accumulator: &NovaFoldAccumulator<Digest>,
) -> Result<FinalFoldedProof<Digest>, BlockTraceError>
where
    Digest: Clone + PartialEq + AsRef<[u8]>,
{
    let block_index = accumulator.metadata.last_block_index.unwrap_or(0);
    if accumulator.config.final_proof_backend_name != SPARTAN_FINAL_PROOF_SYSTEM_NAME {
        return Err(BlockTraceError::NovaFoldingBackendError {
            block_index,
            reason: "final folded proof backend config mismatch",
        });
    }

    let instance = build_final_folded_instance(accumulator)?;
    let recursive_snark_bytes = accumulator.recursive_snark_bytes.as_deref().ok_or(
        BlockTraceError::NovaFoldingBackendError {
            block_index,
            reason: "Nova recursive SNARK is missing",
        },
    )?;
    let recursive_snark = postcard::from_bytes::<NovaRecursiveSnark>(recursive_snark_bytes)
        .map_err(|_| BlockTraceError::NovaFoldingBackendError {
            block_index,
            reason: "Nova recursive SNARK deserialization failed",
        })?;

    let pp = nova_public_params(block_index)?;
    let (pk, vk) = nova_compressed_keys(block_index)?;
    let compressed_snark = NovaCompressedSnark::prove(pp, pk, &recursive_snark).map_err(|_| {
        BlockTraceError::NovaFoldingBackendError {
            block_index,
            reason: "Spartan compressed SNARK proving failed",
        }
    })?;

    let z0 = nova_initial_input();
    let output = compressed_snark
        .verify(vk, accumulator.metadata.absorbed_blocks, &z0)
        .map_err(|_| BlockTraceError::NovaFoldingBackendError {
            block_index,
            reason: "Spartan compressed SNARK self-verification failed",
        })?;
    let expected_z_state = nova_z_state_from_storage(&instance.recursive_z_state, block_index)?;
    verify_nova_recursive_snark_output(&output, expected_z_state, block_index)?;
    let output_digest = digest_nova_recursive_snark_output(&output, block_index)?;
    if output_digest != instance.recursive_snark_output_digest {
        return Err(BlockTraceError::NovaFoldingBackendError {
            block_index,
            reason: "Spartan compressed SNARK output digest mismatch",
        });
    }

    let proof_bytes = postcard::to_stdvec(&compressed_snark).map_err(|_| {
        BlockTraceError::NovaFoldingBackendError {
            block_index,
            reason: "Spartan compressed SNARK serialization failed",
        }
    })?;

    assemble_spartan_final_proof(instance, proof_bytes)
}

#[cfg(feature = "nova")]
fn verify_spartan_compressed_final_proof<Digest>(
    accumulator: &NovaFoldAccumulator<Digest>,
    proof: &FinalFoldedProof<Digest>,
) -> Result<(), BlockTraceError>
where
    Digest: Clone + PartialEq + AsRef<[u8]>,
{
    let block_index = proof.instance.metadata.last_block_index.unwrap_or(0);
    if proof.instance.config.final_proof_backend_name != SPARTAN_FINAL_PROOF_SYSTEM_NAME {
        return Err(BlockTraceError::NovaFoldingBackendError {
            block_index,
            reason: "final folded proof backend config mismatch",
        });
    }

    verify_final_folded_proof_envelope(accumulator, proof)?;
    let proof_bytes =
        proof
            .spartan_proof_bytes
            .as_deref()
            .ok_or(BlockTraceError::NovaFoldingBackendError {
                block_index,
                reason: "Spartan final proof is missing proof bytes",
            })?;
    let compressed_snark =
        postcard::from_bytes::<NovaCompressedSnark>(proof_bytes).map_err(|_| {
            BlockTraceError::NovaFoldingBackendError {
                block_index,
                reason: "Spartan compressed SNARK deserialization failed",
            }
        })?;

    let (_, vk) = nova_compressed_keys(block_index)?;
    let z0 = nova_initial_input();
    let output = compressed_snark
        .verify(vk, proof.instance.metadata.absorbed_blocks, &z0)
        .map_err(|_| BlockTraceError::NovaFoldingBackendError {
            block_index,
            reason: "Spartan compressed SNARK verification failed",
        })?;
    let expected_z_state =
        nova_z_state_from_storage(&proof.instance.recursive_z_state, block_index)?;
    verify_nova_recursive_snark_output(&output, expected_z_state, block_index)?;
    let output_digest = digest_nova_recursive_snark_output(&output, block_index)?;
    if output_digest != proof.instance.recursive_snark_output_digest {
        return Err(BlockTraceError::NovaFoldingBackendError {
            block_index,
            reason: "Spartan compressed SNARK output digest mismatch",
        });
    }

    Ok(())
}

#[cfg(feature = "nova")]
fn prove_nova_recursive_snark_step<Digest, F>(
    existing_recursive_snark_bytes: Option<&[u8]>,
    next_num_steps: usize,
    block_index: usize,
    config: &NovaFoldConfig,
    current_z_state: NovaZState,
    fold_input: &BlockFoldInput<Digest, F>,
) -> Result<(Vec<u8>, [u8; 32], NovaFoldZState), BlockTraceError>
where
    Digest: AsRef<[u8]>,
    F: JoltField,
{
    let pp = nova_public_params(block_index)?;
    let subclaim_backend = nova_subclaim_backend_from_config(config, block_index)?;
    let circuit =
        nova_step_circuit_for_fold_input_with_subclaim_backend(fold_input, &subclaim_backend);
    let z0 = nova_initial_input();
    let next_z_state =
        nova_next_z_state_with_subclaim_backend(current_z_state, fold_input, &subclaim_backend);

    let mut recursive_snark = match existing_recursive_snark_bytes {
        Some(bytes) => postcard::from_bytes::<NovaRecursiveSnark>(bytes).map_err(|_| {
            BlockTraceError::NovaFoldingBackendError {
                block_index,
                reason: "Nova recursive SNARK deserialization failed",
            }
        })?,
        None => NovaRecursiveSnark::new(pp, &circuit, &z0).map_err(|_| {
            BlockTraceError::NovaFoldingBackendError {
                block_index,
                reason: "Nova recursive SNARK initialization failed",
            }
        })?,
    };

    recursive_snark.prove_step(pp, &circuit).map_err(|_| {
        BlockTraceError::NovaFoldingBackendError {
            block_index,
            reason: "Nova recursive SNARK step proving failed",
        }
    })?;

    let output = recursive_snark
        .verify(pp, next_num_steps, &z0)
        .map_err(|_| BlockTraceError::NovaFoldingBackendError {
            block_index,
            reason: "Nova recursive SNARK self-verification failed",
        })?;
    verify_nova_recursive_snark_output(&output, next_z_state, block_index)?;
    let output_digest = digest_nova_recursive_snark_output(&output, block_index)?;

    let recursive_snark_bytes = postcard::to_stdvec(&recursive_snark).map_err(|_| {
        BlockTraceError::NovaFoldingBackendError {
            block_index,
            reason: "Nova recursive SNARK serialization failed",
        }
    })?;

    Ok((
        recursive_snark_bytes,
        output_digest,
        nova_z_state_to_storage(next_z_state),
    ))
}

#[cfg(feature = "nova")]
fn verify_nova_recursive_snark_accumulator<Digest, F>(
    fold_inputs: &[BlockFoldInput<Digest, F>],
    accumulator: &NovaFoldAccumulator<Digest>,
) -> Result<(), BlockTraceError>
where
    Digest: Clone + PartialEq + AsRef<[u8]>,
    F: JoltField,
{
    let block_index = accumulator.metadata.last_block_index.unwrap_or(0);

    if accumulator.metadata.absorbed_blocks == 0 {
        if accumulator.recursive_snark_bytes.is_some()
            || accumulator.recursive_snark_output_digest.is_some()
            || accumulator.recursive_z_state.is_some()
        {
            return Err(BlockTraceError::NovaFoldingBackendError {
                block_index,
                reason: "empty Nova accumulator must not contain recursive state",
            });
        }

        return Ok(());
    }

    let subclaim_backend = nova_subclaim_backend_from_config(&accumulator.config, block_index)?;
    let expected_z_state =
        nova_expected_z_state_with_subclaim_backend(fold_inputs, &subclaim_backend);
    let expected_z_state_storage = nova_z_state_to_storage(expected_z_state);
    if accumulator.recursive_z_state != Some(expected_z_state_storage) {
        return Err(BlockTraceError::NovaFoldingBackendError {
            block_index,
            reason: "Nova recursive z-state mismatch",
        });
    }

    let recursive_snark_bytes = accumulator.recursive_snark_bytes.as_deref().ok_or(
        BlockTraceError::NovaFoldingBackendError {
            block_index,
            reason: "Nova recursive SNARK is missing",
        },
    )?;

    let recursive_snark = postcard::from_bytes::<NovaRecursiveSnark>(recursive_snark_bytes)
        .map_err(|_| BlockTraceError::NovaFoldingBackendError {
            block_index,
            reason: "Nova recursive SNARK deserialization failed",
        })?;

    let pp = nova_public_params(block_index)?;
    let z0 = nova_initial_input();
    let output = recursive_snark
        .verify(pp, accumulator.metadata.absorbed_blocks, &z0)
        .map_err(|_| BlockTraceError::NovaFoldingBackendError {
            block_index,
            reason: "Nova recursive SNARK verification failed",
        })?;
    verify_nova_recursive_snark_output(&output, expected_z_state, block_index)?;
    let output_digest = digest_nova_recursive_snark_output(&output, block_index)?;

    if accumulator.recursive_snark_output_digest != Some(output_digest) {
        Err(BlockTraceError::NovaFoldingBackendError {
            block_index,
            reason: "Nova recursive SNARK output digest mismatch",
        })
    } else {
        Ok(())
    }
}

#[cfg(feature = "nova")]
fn verify_nova_recursive_snark_output(
    output: &[NovaScalar],
    expected_z_state: NovaZState,
    block_index: usize,
) -> Result<(), BlockTraceError> {
    if output.len() != NOVA_Z_ARITY || output != expected_z_state.as_slice() {
        Err(BlockTraceError::NovaFoldingBackendError {
            block_index,
            reason: "Nova recursive SNARK z-state mismatch",
        })
    } else {
        Ok(())
    }
}

#[cfg(feature = "nova")]
fn digest_nova_recursive_snark_output<T>(
    output: &T,
    block_index: usize,
) -> Result<[u8; 32], BlockTraceError>
where
    T: serde::Serialize + ?Sized,
{
    let output_bytes =
        postcard::to_stdvec(output).map_err(|_| BlockTraceError::NovaFoldingBackendError {
            block_index,
            reason: "Nova recursive SNARK output serialization failed",
        })?;
    let digest = Sha3_256::digest(output_bytes);
    let mut output_digest = [0u8; 32];
    output_digest.copy_from_slice(&digest);
    Ok(output_digest)
}

#[derive(Clone, Debug)]
pub struct BlockCpuProver<Digest = [u8; 32], F = ark_bn254::Fr> {
    program_digest: Digest,
    _field: PhantomData<F>,
}

impl<Digest, F> BlockCpuProver<Digest, F>
where
    Digest: Clone,
    F: JoltField,
{
    pub fn new(program_digest: Digest) -> Self {
        Self {
            program_digest,
            _field: PhantomData,
        }
    }

    pub fn prove_block(
        &self,
        bytecode_preprocessing: &BytecodePreprocessing,
        block: &TraceBlock,
        lookahead_cycle: Option<&Cycle>,
    ) -> Result<CpuBlockProof<Digest, F>, BlockTraceError> {
        validate_trace_block_shape(block)?;
        validate_cpu_r1cs_block::<F>(bytecode_preprocessing, block, lookahead_cycle)?;
        let lookahead_cycle_digest =
            digest_cpu_lookahead_cycle(block.block_index, lookahead_cycle)?;

        let public_input = BlockPublicInput::from_trace_block(block, self.program_digest.clone());
        public_input.validate_shape()?;

        let r1cs_num_steps = r1cs_num_steps_for_block(block.active_cycles);
        let spartan_key = UniformSpartanKey::<F>::new(r1cs_num_steps);

        Ok(BlockProof::new(
            public_input,
            BlockCpuProof {
                cycle_count: block.cycles.len(),
                r1cs_rows_checked: block.cycles.len(),
                r1cs_num_steps,
                r1cs_vk_digest: spartan_key.vk_digest,
                used_lookahead_cycle: lookahead_cycle.is_some(),
                lookahead_cycle_digest,
            },
        ))
    }

    pub fn prove_blocks(
        &self,
        bytecode_preprocessing: &BytecodePreprocessing,
        blocks: &[TraceBlock],
    ) -> Result<Vec<CpuBlockProof<Digest, F>>, BlockTraceError> {
        self.prove_blocks_with_external_lookahead(bytecode_preprocessing, blocks, None)
    }

    /// Proves a block sequence whose final non-terminal block is followed by
    /// `external_lookahead_cycle`.
    ///
    /// The exact cycle is committed in the CPU proof, so a verifier cannot
    /// replace or omit it while reusing the same proof.
    pub fn prove_blocks_with_external_lookahead(
        &self,
        bytecode_preprocessing: &BytecodePreprocessing,
        blocks: &[TraceBlock],
        external_lookahead_cycle: Option<&Cycle>,
    ) -> Result<Vec<CpuBlockProof<Digest, F>>, BlockTraceError> {
        let proofs = blocks
            .iter()
            .enumerate()
            .map(|(index, block)| {
                let lookahead = block_lookahead_cycle(blocks, index, external_lookahead_cycle);
                self.prove_block(bytecode_preprocessing, block, lookahead)
            })
            .collect::<Result<Vec<_>, _>>()?;

        let public_inputs = proofs
            .iter()
            .map(|proof| proof.public_input.clone())
            .collect::<Vec<_>>();
        validate_block_chain(&public_inputs)?;

        Ok(proofs)
    }
}

static TERMINAL_LOOKAHEAD_CYCLE: Cycle = Cycle::NoOp;

fn block_lookahead_cycle<'a>(
    blocks: &'a [TraceBlock],
    index: usize,
    external_lookahead_cycle: Option<&'a Cycle>,
) -> Option<&'a Cycle> {
    blocks
        .get(index + 1)
        .and_then(|next_block| next_block.cycles.first())
        .or_else(|| {
            blocks
                .get(index)
                .is_some_and(|block| block.end_state.terminated)
                .then_some(&TERMINAL_LOOKAHEAD_CYCLE)
        })
        .or_else(|| {
            (index + 1 == blocks.len())
                .then_some(external_lookahead_cycle)
                .flatten()
        })
}

#[derive(Clone, Debug)]
pub struct BlockProofBundleProver<Digest = [u8; 32], F = ark_bn254::Fr> {
    cpu_prover: BlockCpuProver<Digest, F>,
}

impl<Digest, F> BlockProofBundleProver<Digest, F>
where
    Digest: Clone,
    F: JoltField,
{
    pub fn new(program_digest: Digest) -> Self {
        Self {
            cpu_prover: BlockCpuProver::new(program_digest),
        }
    }

    pub fn prove_block(
        &self,
        bytecode_preprocessing: &BytecodePreprocessing,
        block: &TraceBlock,
        lookahead_cycle: Option<&Cycle>,
    ) -> Result<BlockProofBundle<Digest, F>, BlockTraceError> {
        let cpu_proof =
            self.cpu_prover
                .prove_block(bytecode_preprocessing, block, lookahead_cycle)?;
        let io_claims = extract_block_io_claims(block)?;
        let register_claim = build_block_register_claim(block, &io_claims)?;
        let ram_claim = build_block_ram_claim(block, &io_claims)?;
        let lookup_claim = build_block_lookup_claim(block, &io_claims)?;

        Ok(BlockProofBundle {
            cpu_proof,
            io_claims,
            register_claim,
            ram_claim,
            lookup_claim,
        })
    }
}

impl<Digest, F> BlockProofBundleProver<Digest, F>
where
    Digest: Clone + PartialEq,
    F: JoltField,
{
    pub fn prove_blocks(
        &self,
        bytecode_preprocessing: &BytecodePreprocessing,
        blocks: &[TraceBlock],
    ) -> Result<Vec<BlockProofBundle<Digest, F>>, BlockTraceError> {
        self.prove_blocks_with_external_lookahead(bytecode_preprocessing, blocks, None)
    }

    pub fn prove_blocks_with_external_lookahead(
        &self,
        bytecode_preprocessing: &BytecodePreprocessing,
        blocks: &[TraceBlock],
        external_lookahead_cycle: Option<&Cycle>,
    ) -> Result<Vec<BlockProofBundle<Digest, F>>, BlockTraceError> {
        let bundles = blocks
            .iter()
            .enumerate()
            .map(|(index, block)| {
                let lookahead = block_lookahead_cycle(blocks, index, external_lookahead_cycle);
                self.prove_block(bytecode_preprocessing, block, lookahead)
            })
            .collect::<Result<Vec<_>, _>>()?;

        verify_block_proof_bundle_chain_with_external_lookahead(
            bytecode_preprocessing,
            blocks,
            external_lookahead_cycle,
            &bundles,
        )?;
        Ok(bundles)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct BlockProofPipelineOutput<
    Digest = [u8; 32],
    F = ark_bn254::Fr,
    Accumulator = BlockFoldAccumulator<Digest>,
> {
    pub bundles: Vec<BlockProofBundle<Digest, F>>,
    pub fold_inputs: Vec<BlockFoldInput<Digest, F>>,
    pub accumulator: Accumulator,
    /// Optional final folded proof for Nova-backed pipelines.
    ///
    /// Plain `prove_blocks` leaves this empty and should be verified with
    /// `verify_block_proof_pipeline_with_backend`. Nova callers that want a
    /// final folded proof should use `prove_blocks_with_final_proof` and verify
    /// with `verify_nova_block_proof_pipeline_with_final_proof`, otherwise the
    /// final proof would not be checked by the generic pipeline verifier.
    pub final_proof: Option<FinalFoldedProof<Digest>>,
}

/// Nova pipeline output paired with a final proof size comparison.
///
/// The embedded pipeline output intentionally leaves `final_proof` empty: this
/// report is for measurement, not for carrying the verifier-facing final proof.
#[derive(Clone, Debug, PartialEq)]
pub struct NovaBlockProofPipelineFinalProofSizeReport<Digest = [u8; 32], F = ark_bn254::Fr> {
    pub output: BlockProofPipelineOutput<Digest, F, NovaFoldAccumulator<Digest>>,
    pub final_proof_size_comparison: JoltNovaFinalProofSizeComparison,
}

/// One row in a final proof size scaling table over block prefixes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NovaBlockProofPipelineFinalProofSizeScalingRow {
    pub block_count: usize,
    pub first_block_index: Option<usize>,
    pub last_block_index: Option<usize>,
    pub total_active_cycles: usize,
    pub recursive_snark_bytes_len: Option<usize>,
    pub final_proof_size_comparison: JoltNovaFinalProofSizeComparison,
}

/// Final proof size scaling report over increasing block-prefix lengths.
///
/// This is the stage-7 experiment surface: callers can request rows such as
/// `[1, 2, 4, 8]` and compare how the recursive accumulator and final proof
/// boundary change as more blocks are folded.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NovaBlockProofPipelineFinalProofSizeScalingReport {
    pub rows: Vec<NovaBlockProofPipelineFinalProofSizeScalingRow>,
}

/// Serialized benchmark artifact for final proof size scaling.
///
/// This is the stage-8 runner surface: the same call returns the structured
/// scaling report and a serialized representation ready to write to disk.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NovaBlockProofPipelineFinalProofSizeBenchmarkArtifact {
    pub output_format: JoltNovaReportOutputFormat,
    pub report: NovaBlockProofPipelineFinalProofSizeScalingReport,
    pub serialized_report: String,
}

impl NovaBlockProofPipelineFinalProofSizeBenchmarkArtifact {
    /// Returns the canonical filename extension for this serialized artifact.
    pub fn file_extension(&self) -> &'static str {
        self.output_format.as_str()
    }

    /// Returns the serialized artifact bytes ready for file output.
    pub fn serialized_bytes(&self) -> &[u8] {
        self.serialized_report.as_bytes()
    }

    /// Writes the serialized artifact to disk, creating parent directories when
    /// the target path includes them.
    pub fn write_to_path(&self, path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }

        std::fs::write(path, self.serialized_bytes())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NovaBlockProofPipelineBenchmarkArtifactError {
    Pipeline(BlockTraceError),
    ArtifactIo { path: String, reason: String },
}

impl From<BlockTraceError> for NovaBlockProofPipelineBenchmarkArtifactError {
    fn from(error: BlockTraceError) -> Self {
        Self::Pipeline(error)
    }
}

impl fmt::Display for NovaBlockProofPipelineBenchmarkArtifactError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pipeline(error) => error.fmt(f),
            Self::ArtifactIo { path, reason } => {
                write!(
                    f,
                    "failed to write Jolt-Nova benchmark artifact to {path}: {reason}"
                )
            }
        }
    }
}

impl Error for NovaBlockProofPipelineBenchmarkArtifactError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Pipeline(error) => Some(error),
            Self::ArtifactIo { .. } => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct BlockProofPipeline<Digest = [u8; 32], F = ark_bn254::Fr, Backend = MockFoldingBackend> {
    bundle_prover: BlockProofBundleProver<Digest, F>,
    folding_backend: Backend,
}

impl<Digest, F> BlockProofPipeline<Digest, F, MockFoldingBackend>
where
    Digest: Clone + PartialEq + AsRef<[u8]>,
    F: JoltField,
{
    pub fn new(program_digest: Digest) -> Self {
        Self::with_backend(program_digest, MockFoldingBackend)
    }
}

impl<Digest, F, Backend> BlockProofPipeline<Digest, F, Backend>
where
    Digest: Clone + PartialEq + AsRef<[u8]>,
    F: JoltField,
    Backend: BlockFoldingBackend<Digest, F>,
{
    pub fn with_backend(program_digest: Digest, folding_backend: Backend) -> Self {
        Self {
            bundle_prover: BlockProofBundleProver::new(program_digest),
            folding_backend,
        }
    }

    pub fn prove_blocks(
        &self,
        bytecode_preprocessing: &BytecodePreprocessing,
        blocks: &[TraceBlock],
    ) -> Result<BlockProofPipelineOutput<Digest, F, Backend::Accumulator>, BlockTraceError> {
        self.prove_blocks_with_external_lookahead(bytecode_preprocessing, blocks, None)
    }

    pub fn prove_blocks_with_external_lookahead(
        &self,
        bytecode_preprocessing: &BytecodePreprocessing,
        blocks: &[TraceBlock],
        external_lookahead_cycle: Option<&Cycle>,
    ) -> Result<BlockProofPipelineOutput<Digest, F, Backend::Accumulator>, BlockTraceError> {
        let bundles = self.bundle_prover.prove_blocks_with_external_lookahead(
            bytecode_preprocessing,
            blocks,
            external_lookahead_cycle,
        )?;
        let fold_inputs = build_block_fold_inputs(&bundles);
        verify_block_fold_input_chain_with_external_lookahead(
            bytecode_preprocessing,
            blocks,
            external_lookahead_cycle,
            &bundles,
            &fold_inputs,
        )?;
        let accumulator = self.folding_backend.fold(&fold_inputs)?;

        Ok(BlockProofPipelineOutput {
            bundles,
            fold_inputs,
            accumulator,
            final_proof: None,
        })
    }
}

impl<Digest, F> BlockProofPipeline<Digest, F, NovaFoldingBackend>
where
    Digest: Clone + PartialEq + AsRef<[u8]>,
    F: JoltField,
{
    /// Proves blocks with the Nova folding backend and automatically attaches
    /// the configured final folded proof.
    ///
    /// With the default config this attaches a `spartan-placeholder` envelope.
    /// With `final_proof_backend_name = spartan-final-proof`, this compresses
    /// the Nova recursive SNARK into a real Spartan `CompressedSNARK`.
    pub fn prove_blocks_with_final_proof(
        &self,
        bytecode_preprocessing: &BytecodePreprocessing,
        blocks: &[TraceBlock],
    ) -> Result<BlockProofPipelineOutput<Digest, F, NovaFoldAccumulator<Digest>>, BlockTraceError>
    {
        self.prove_blocks_with_final_proof_and_external_lookahead(
            bytecode_preprocessing,
            blocks,
            None,
        )
    }

    pub fn prove_blocks_with_final_proof_and_external_lookahead(
        &self,
        bytecode_preprocessing: &BytecodePreprocessing,
        blocks: &[TraceBlock],
        external_lookahead_cycle: Option<&Cycle>,
    ) -> Result<BlockProofPipelineOutput<Digest, F, NovaFoldAccumulator<Digest>>, BlockTraceError>
    {
        let mut output = self.prove_blocks_with_external_lookahead(
            bytecode_preprocessing,
            blocks,
            external_lookahead_cycle,
        )?;
        let final_proof = prove_configured_final_folded_accumulator(&output.accumulator)?;
        output.final_proof = Some(final_proof);
        Ok(output)
    }

    /// Proves blocks with the Nova folding backend and emits a final proof size
    /// comparison report for the folded accumulator.
    ///
    /// This keeps `output.final_proof` empty and treats the final proofs as
    /// report artifacts: one `spartan-placeholder` envelope and one real
    /// `spartan-final-proof` compressed proof are generated only for measuring
    /// the final proof size boundary.
    pub fn prove_blocks_with_final_proof_size_report(
        &self,
        bytecode_preprocessing: &BytecodePreprocessing,
        blocks: &[TraceBlock],
    ) -> Result<NovaBlockProofPipelineFinalProofSizeReport<Digest, F>, BlockTraceError> {
        self.prove_blocks_with_final_proof_size_report_and_external_lookahead(
            bytecode_preprocessing,
            blocks,
            None,
        )
    }

    pub fn prove_blocks_with_final_proof_size_report_and_external_lookahead(
        &self,
        bytecode_preprocessing: &BytecodePreprocessing,
        blocks: &[TraceBlock],
        external_lookahead_cycle: Option<&Cycle>,
    ) -> Result<NovaBlockProofPipelineFinalProofSizeReport<Digest, F>, BlockTraceError> {
        let output = self.prove_blocks_with_external_lookahead(
            bytecode_preprocessing,
            blocks,
            external_lookahead_cycle,
        )?;
        let final_proof_size_comparison =
            summarize_nova_block_proof_pipeline_final_proof_size_comparison(&output)?;

        Ok(NovaBlockProofPipelineFinalProofSizeReport {
            output,
            final_proof_size_comparison,
        })
    }

    /// Proves increasing block prefixes and emits one final proof size row for
    /// each requested prefix length.
    ///
    /// `block_counts` must be non-empty, strictly increasing, and each count
    /// must be in `1..=blocks.len()`. For example, `[1, 2, 4, 8]` produces a
    /// scaling table over progressively larger prefixes of the same block
    /// sequence.
    pub fn prove_block_prefixes_with_final_proof_size_scaling_report(
        &self,
        bytecode_preprocessing: &BytecodePreprocessing,
        blocks: &[TraceBlock],
        block_counts: &[usize],
    ) -> Result<NovaBlockProofPipelineFinalProofSizeScalingReport, BlockTraceError> {
        validate_final_proof_size_scaling_block_counts(blocks.len(), block_counts)?;

        let rows = block_counts
            .iter()
            .map(|&block_count| {
                let external_lookahead_cycle = blocks
                    .get(block_count)
                    .and_then(|next_block| next_block.cycles.first());
                let report = self
                    .prove_blocks_with_final_proof_size_report_and_external_lookahead(
                        bytecode_preprocessing,
                        &blocks[..block_count],
                        external_lookahead_cycle,
                    )?;
                let accumulator = &report.output.accumulator;

                Ok(NovaBlockProofPipelineFinalProofSizeScalingRow {
                    block_count,
                    first_block_index: accumulator.metadata.first_block_index,
                    last_block_index: accumulator.metadata.last_block_index,
                    total_active_cycles: accumulator.metadata.total_active_cycles,
                    recursive_snark_bytes_len: accumulator
                        .recursive_snark_bytes
                        .as_ref()
                        .map(Vec::len),
                    final_proof_size_comparison: report.final_proof_size_comparison,
                })
            })
            .collect::<Result<Vec<_>, BlockTraceError>>()?;

        Ok(NovaBlockProofPipelineFinalProofSizeScalingReport { rows })
    }

    /// Runs the final proof size scaling benchmark and serializes the result.
    ///
    /// This is a lightweight runner/helper rather than a wall-clock benchmark:
    /// it executes the proving/reporting path for the requested block prefixes
    /// and returns both the structured scaling report and the requested output
    /// artifact. JSON is currently the canonical supported format.
    pub fn prove_block_prefixes_with_final_proof_size_benchmark_artifact(
        &self,
        bytecode_preprocessing: &BytecodePreprocessing,
        blocks: &[TraceBlock],
        block_counts: &[usize],
        output_format: JoltNovaReportOutputFormat,
    ) -> Result<NovaBlockProofPipelineFinalProofSizeBenchmarkArtifact, BlockTraceError> {
        let report = self.prove_block_prefixes_with_final_proof_size_scaling_report(
            bytecode_preprocessing,
            blocks,
            block_counts,
        )?;
        let serialized_report =
            export_nova_final_proof_size_scaling_report(&report, output_format)?;

        Ok(NovaBlockProofPipelineFinalProofSizeBenchmarkArtifact {
            output_format,
            report,
            serialized_report,
        })
    }

    /// Runs the final proof size scaling benchmark and writes the serialized
    /// artifact to disk.
    pub fn prove_block_prefixes_and_write_final_proof_size_benchmark_artifact(
        &self,
        bytecode_preprocessing: &BytecodePreprocessing,
        blocks: &[TraceBlock],
        block_counts: &[usize],
        output_format: JoltNovaReportOutputFormat,
        output_path: impl AsRef<std::path::Path>,
    ) -> Result<
        NovaBlockProofPipelineFinalProofSizeBenchmarkArtifact,
        NovaBlockProofPipelineBenchmarkArtifactError,
    > {
        let output_path = output_path.as_ref();
        let artifact = self.prove_block_prefixes_with_final_proof_size_benchmark_artifact(
            bytecode_preprocessing,
            blocks,
            block_counts,
            output_format,
        )?;

        artifact.write_to_path(output_path).map_err(|error| {
            NovaBlockProofPipelineBenchmarkArtifactError::ArtifactIo {
                path: output_path.display().to_string(),
                reason: error.to_string(),
            }
        })?;

        Ok(artifact)
    }
}

/// Summarizes the final proof size comparison for an already-produced Nova
/// pipeline output.
///
/// This helper deliberately ignores `output.final_proof`; it measures both
/// placeholder and real Spartan final proof variants from the folded
/// accumulator so the comparison is independent from verifier-facing output.
pub fn summarize_nova_block_proof_pipeline_final_proof_size_comparison<Digest, F>(
    output: &BlockProofPipelineOutput<Digest, F, NovaFoldAccumulator<Digest>>,
) -> Result<JoltNovaFinalProofSizeComparison, BlockTraceError>
where
    Digest: Clone + PartialEq + AsRef<[u8]>,
    F: JoltField,
{
    summarize_jolt_nova_final_proof_size_comparison(&output.accumulator)
}

fn validate_final_proof_size_scaling_block_counts(
    blocks_len: usize,
    block_counts: &[usize],
) -> Result<(), BlockTraceError> {
    if block_counts.is_empty() {
        return Err(BlockTraceError::NovaFoldingBackendError {
            block_index: 0,
            reason: "final proof size scaling requires at least one block count",
        });
    }

    let mut previous = 0;
    for &block_count in block_counts {
        if block_count == 0 || block_count > blocks_len {
            return Err(BlockTraceError::NovaFoldingBackendError {
                block_index: block_count.saturating_sub(1),
                reason: "final proof size scaling block count is out of range",
            });
        }
        if block_count <= previous {
            return Err(BlockTraceError::NovaFoldingBackendError {
                block_index: block_count.saturating_sub(1),
                reason: "final proof size scaling block counts must be strictly increasing",
            });
        }
        previous = block_count;
    }

    Ok(())
}

pub fn export_nova_final_proof_size_scaling_report(
    report: &NovaBlockProofPipelineFinalProofSizeScalingReport,
    format: JoltNovaReportOutputFormat,
) -> Result<String, BlockTraceError> {
    match format {
        JoltNovaReportOutputFormat::Json => {
            Ok(export_nova_final_proof_size_scaling_report_json(report))
        }
        JoltNovaReportOutputFormat::Csv => Err(BlockTraceError::NovaFoldingBackendError {
            block_index: 0,
            reason: "CSV report export is not implemented yet",
        }),
    }
}

/// Exports a final proof size scaling report as stable, deterministic JSON.
///
/// Field order is intentionally fixed so small benchmark artifacts are easy to
/// diff in Git and across runs.
pub fn export_nova_final_proof_size_scaling_report_json(
    report: &NovaBlockProofPipelineFinalProofSizeScalingReport,
) -> String {
    let mut output = String::new();
    output.push('{');
    append_json_string_field(
        &mut output,
        "schema_version",
        JOLT_NOVA_REPORT_SCHEMA_VERSION,
    );
    output.push(',');
    append_json_string_field(
        &mut output,
        "format",
        JoltNovaReportOutputFormat::Json.as_str(),
    );
    output.push(',');
    append_json_string_field(
        &mut output,
        "report_kind",
        JOLT_NOVA_FINAL_PROOF_SIZE_SCALING_REPORT_KIND,
    );
    output.push(',');
    append_json_usize_field(&mut output, "row_count", report.rows.len());
    output.push(',');
    append_json_string(&mut output, "rows");
    output.push_str(":[");
    for (index, row) in report.rows.iter().enumerate() {
        if index > 0 {
            output.push(',');
        }
        append_final_proof_size_scaling_row_json(&mut output, row);
    }
    output.push_str("]}");
    output
}

fn append_final_proof_size_scaling_row_json(
    output: &mut String,
    row: &NovaBlockProofPipelineFinalProofSizeScalingRow,
) {
    output.push('{');
    append_json_usize_field(output, "block_count", row.block_count);
    output.push(',');
    append_json_optional_usize_field(output, "first_block_index", row.first_block_index);
    output.push(',');
    append_json_optional_usize_field(output, "last_block_index", row.last_block_index);
    output.push(',');
    append_json_usize_field(output, "total_active_cycles", row.total_active_cycles);
    output.push(',');
    append_json_optional_usize_field(
        output,
        "recursive_snark_bytes_len",
        row.recursive_snark_bytes_len,
    );
    output.push(',');
    append_json_string(output, "final_proof_size_comparison");
    output.push(':');
    append_final_proof_size_comparison_json(output, &row.final_proof_size_comparison);
    output.push('}');
}

fn append_final_proof_size_comparison_json(
    output: &mut String,
    comparison: &JoltNovaFinalProofSizeComparison,
) {
    output.push('{');
    append_json_digest_field(
        output,
        "folded_accumulator_digest",
        &comparison.folded_accumulator_digest,
    );
    output.push(',');
    append_json_usize_field(output, "absorbed_blocks", comparison.absorbed_blocks);
    output.push(',');
    append_json_usize_field(
        output,
        "total_active_cycles",
        comparison.total_active_cycles,
    );
    output.push(',');
    append_json_optional_usize_field(
        output,
        "recursive_snark_bytes_len",
        comparison.recursive_snark_bytes_len,
    );
    output.push(',');
    append_json_string(output, "placeholder");
    output.push(':');
    append_final_proof_size_baseline_json(output, &comparison.placeholder);
    output.push(',');
    append_json_string(output, "spartan");
    output.push(':');
    append_final_proof_size_baseline_json(output, &comparison.spartan);
    output.push(',');
    append_json_usize_field(
        output,
        "spartan_payload_extra_bytes",
        comparison.spartan_payload_extra_bytes,
    );
    output.push(',');
    append_json_i128_field(
        output,
        "spartan_total_extra_bytes",
        comparison.spartan_total_extra_bytes,
    );
    output.push('}');
}

fn append_final_proof_size_baseline_json(
    output: &mut String,
    baseline: &JoltNovaFinalProofSizeBaseline,
) {
    output.push('{');
    append_json_string_field(
        output,
        "configured_backend_name",
        baseline.configured_backend_name,
    );
    output.push(',');
    append_json_string_field(output, "proof_system", baseline.proof_system);
    output.push(',');
    append_json_usize_field(output, "absorbed_blocks", baseline.absorbed_blocks);
    output.push(',');
    append_json_usize_field(output, "total_active_cycles", baseline.total_active_cycles);
    output.push(',');
    append_json_optional_usize_field(
        output,
        "recursive_snark_bytes_len",
        baseline.recursive_snark_bytes_len,
    );
    output.push(',');
    append_json_usize_field(
        output,
        "final_public_input_bytes_len",
        baseline.final_public_input_bytes_len,
    );
    output.push(',');
    append_json_usize_field(
        output,
        "final_witness_bytes_len",
        baseline.final_witness_bytes_len,
    );
    output.push(',');
    append_json_usize_field(
        output,
        "proof_envelope_bytes_len",
        baseline.proof_envelope_bytes_len,
    );
    output.push(',');
    append_json_usize_field(
        output,
        "proof_payload_bytes_len",
        baseline.proof_payload_bytes_len,
    );
    output.push(',');
    append_json_usize_field(
        output,
        "proof_total_bytes_len",
        baseline.proof_total_bytes_len,
    );
    output.push(',');
    append_json_digest_field(
        output,
        "final_instance_digest",
        &baseline.final_instance_digest,
    );
    output.push(',');
    append_json_digest_field(
        output,
        "spartan_encoding_digest",
        &baseline.spartan_encoding_digest,
    );
    output.push(',');
    append_json_digest_field(output, "proof_digest", &baseline.proof_digest);
    output.push('}');
}

fn append_json_string_field(output: &mut String, name: &str, value: &str) {
    append_json_string(output, name);
    output.push(':');
    append_json_string(output, value);
}

fn append_json_usize_field(output: &mut String, name: &str, value: usize) {
    append_json_string(output, name);
    output.push(':');
    output.push_str(&value.to_string());
}

fn append_json_i128_field(output: &mut String, name: &str, value: i128) {
    append_json_string(output, name);
    output.push(':');
    output.push_str(&value.to_string());
}

fn append_json_optional_usize_field(output: &mut String, name: &str, value: Option<usize>) {
    append_json_string(output, name);
    output.push(':');
    match value {
        Some(value) => output.push_str(&value.to_string()),
        None => output.push_str("null"),
    }
}

fn append_json_digest_field(output: &mut String, name: &str, digest: &[u8; 32]) {
    append_json_string(output, name);
    output.push(':');
    output.push('"');
    append_hex_digest(output, digest);
    output.push('"');
}

fn append_json_string(output: &mut String, value: &str) {
    output.push('"');
    for ch in value.chars() {
        match ch {
            '"' => output.push_str("\\\""),
            '\\' => output.push_str("\\\\"),
            '\u{08}' => output.push_str("\\b"),
            '\u{0C}' => output.push_str("\\f"),
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            ch if ch.is_control() => append_json_unicode_escape(output, ch as u32),
            ch => output.push(ch),
        }
    }
    output.push('"');
}

fn append_json_unicode_escape(output: &mut String, codepoint: u32) {
    output.push_str("\\u");
    for shift in [12, 8, 4, 0] {
        let nibble = ((codepoint >> shift) & 0xF) as usize;
        output.push(HEX_CHARS[nibble] as char);
    }
}

fn append_hex_digest(output: &mut String, digest: &[u8; 32]) {
    for byte in digest {
        output.push(HEX_CHARS[(byte >> 4) as usize] as char);
        output.push(HEX_CHARS[(byte & 0x0F) as usize] as char);
    }
}

const HEX_CHARS: &[u8; 16] = b"0123456789abcdef";

pub fn verify_block_proof_pipeline<Digest, F>(
    bytecode_preprocessing: &BytecodePreprocessing,
    blocks: &[TraceBlock],
    output: &BlockProofPipelineOutput<Digest, F>,
) -> Result<(), BlockTraceError>
where
    Digest: Clone + PartialEq + AsRef<[u8]>,
    F: JoltField,
{
    verify_block_proof_pipeline_with_external_lookahead(
        bytecode_preprocessing,
        blocks,
        None,
        output,
    )
}

pub fn verify_block_proof_pipeline_with_external_lookahead<Digest, F>(
    bytecode_preprocessing: &BytecodePreprocessing,
    blocks: &[TraceBlock],
    external_lookahead_cycle: Option<&Cycle>,
    output: &BlockProofPipelineOutput<Digest, F>,
) -> Result<(), BlockTraceError>
where
    Digest: Clone + PartialEq + AsRef<[u8]>,
    F: JoltField,
{
    verify_block_proof_pipeline_with_backend_and_external_lookahead(
        bytecode_preprocessing,
        blocks,
        external_lookahead_cycle,
        output,
        &MockFoldingBackend,
    )
}

pub fn verify_block_proof_pipeline_with_backend<Digest, F, Backend>(
    bytecode_preprocessing: &BytecodePreprocessing,
    blocks: &[TraceBlock],
    output: &BlockProofPipelineOutput<Digest, F, Backend::Accumulator>,
    folding_backend: &Backend,
) -> Result<(), BlockTraceError>
where
    Digest: Clone + PartialEq + AsRef<[u8]>,
    F: JoltField,
    Backend: BlockFoldingBackend<Digest, F>,
{
    verify_block_proof_pipeline_with_backend_and_external_lookahead(
        bytecode_preprocessing,
        blocks,
        None,
        output,
        folding_backend,
    )
}

pub fn verify_block_proof_pipeline_with_backend_and_external_lookahead<Digest, F, Backend>(
    bytecode_preprocessing: &BytecodePreprocessing,
    blocks: &[TraceBlock],
    external_lookahead_cycle: Option<&Cycle>,
    output: &BlockProofPipelineOutput<Digest, F, Backend::Accumulator>,
    folding_backend: &Backend,
) -> Result<(), BlockTraceError>
where
    Digest: Clone + PartialEq + AsRef<[u8]>,
    F: JoltField,
    Backend: BlockFoldingBackend<Digest, F>,
{
    if let Some(final_proof) = &output.final_proof {
        return Err(BlockTraceError::NovaFoldingBackendError {
            block_index: final_proof.instance.metadata.last_block_index.unwrap_or(0),
            reason:
                "pipeline output carries a final folded proof; use the Nova final-proof verifier",
        });
    }

    verify_block_proof_pipeline_core_with_backend(
        bytecode_preprocessing,
        blocks,
        external_lookahead_cycle,
        output,
        folding_backend,
    )
}

fn verify_block_proof_pipeline_core_with_backend<Digest, F, Backend>(
    bytecode_preprocessing: &BytecodePreprocessing,
    blocks: &[TraceBlock],
    external_lookahead_cycle: Option<&Cycle>,
    output: &BlockProofPipelineOutput<Digest, F, Backend::Accumulator>,
    folding_backend: &Backend,
) -> Result<(), BlockTraceError>
where
    Digest: Clone + PartialEq + AsRef<[u8]>,
    F: JoltField,
    Backend: BlockFoldingBackend<Digest, F>,
{
    verify_block_fold_input_chain_with_external_lookahead(
        bytecode_preprocessing,
        blocks,
        external_lookahead_cycle,
        &output.bundles,
        &output.fold_inputs,
    )?;
    folding_backend.verify(&output.fold_inputs, &output.accumulator)
}

/// Verifies a Nova-backed block proof pipeline output that includes a final
/// folded proof.
///
/// This first verifies the block/bundle/fold accumulator pipeline, then verifies
/// the configured final proof backend. Use this for outputs produced by
/// `prove_blocks_with_final_proof`.
pub fn verify_nova_block_proof_pipeline_with_final_proof<Digest, F>(
    bytecode_preprocessing: &BytecodePreprocessing,
    blocks: &[TraceBlock],
    output: &BlockProofPipelineOutput<Digest, F, NovaFoldAccumulator<Digest>>,
    folding_backend: &NovaFoldingBackend,
) -> Result<(), BlockTraceError>
where
    Digest: Clone + PartialEq + AsRef<[u8]>,
    F: JoltField,
{
    verify_nova_block_proof_pipeline_with_final_proof_and_external_lookahead(
        bytecode_preprocessing,
        blocks,
        None,
        output,
        folding_backend,
    )
}

pub fn verify_nova_block_proof_pipeline_with_final_proof_and_external_lookahead<Digest, F>(
    bytecode_preprocessing: &BytecodePreprocessing,
    blocks: &[TraceBlock],
    external_lookahead_cycle: Option<&Cycle>,
    output: &BlockProofPipelineOutput<Digest, F, NovaFoldAccumulator<Digest>>,
    folding_backend: &NovaFoldingBackend,
) -> Result<(), BlockTraceError>
where
    Digest: Clone + PartialEq + AsRef<[u8]>,
    F: JoltField,
{
    verify_block_proof_pipeline_core_with_backend(
        bytecode_preprocessing,
        blocks,
        external_lookahead_cycle,
        output,
        folding_backend,
    )?;
    let final_proof =
        output
            .final_proof
            .as_ref()
            .ok_or(BlockTraceError::NovaFoldingBackendError {
                block_index: output.accumulator.metadata.last_block_index.unwrap_or(0),
                reason: "final folded proof is missing from pipeline output",
            })?;
    verify_configured_final_folded_proof(&output.accumulator, final_proof)
}

pub fn verify_placeholder_block_proof<Digest>(
    proof: &PlaceholderBlockProof<Digest>,
) -> Result<(), BlockTraceError> {
    proof.public_input.validate_shape()?;

    if proof.public_input.active_cycles != proof.inner_proof.cycle_count {
        return Err(BlockTraceError::ProofCycleCountMismatch {
            block_index: proof.public_input.block_index,
            public_input_cycles: proof.public_input.active_cycles,
            proof_cycles: proof.inner_proof.cycle_count,
        });
    }

    if !proof.inner_proof.ended_at_tick_boundary {
        return Err(BlockTraceError::BlockDidNotEndAtTickBoundary {
            block_index: proof.public_input.block_index,
        });
    }

    Ok(())
}

pub fn verify_placeholder_block_proof_chain<Digest>(
    proofs: &[PlaceholderBlockProof<Digest>],
) -> Result<(), BlockTraceError>
where
    Digest: Clone,
{
    for proof in proofs {
        verify_placeholder_block_proof(proof)?;
    }

    let public_inputs = proofs
        .iter()
        .map(|proof| proof.public_input.clone())
        .collect::<Vec<_>>();
    validate_block_chain(&public_inputs)?;

    Ok(())
}

pub fn verify_cpu_block_proof<Digest, F>(
    proof: &CpuBlockProof<Digest, F>,
) -> Result<(), BlockTraceError>
where
    F: JoltField,
{
    proof.public_input.validate_shape()?;

    let expected_cycles = proof.public_input.active_cycles;
    if expected_cycles != proof.inner_proof.cycle_count {
        return Err(BlockTraceError::ProofCycleCountMismatch {
            block_index: proof.public_input.block_index,
            public_input_cycles: expected_cycles,
            proof_cycles: proof.inner_proof.cycle_count,
        });
    }

    if expected_cycles != proof.inner_proof.r1cs_rows_checked {
        return Err(BlockTraceError::CpuR1CSRowsCheckedMismatch {
            block_index: proof.public_input.block_index,
            expected: expected_cycles,
            actual: proof.inner_proof.r1cs_rows_checked,
        });
    }

    let expected_num_steps = r1cs_num_steps_for_block(expected_cycles);
    if expected_num_steps != proof.inner_proof.r1cs_num_steps {
        return Err(BlockTraceError::CpuR1CSNumStepsMismatch {
            block_index: proof.public_input.block_index,
            expected: expected_num_steps,
            actual: proof.inner_proof.r1cs_num_steps,
        });
    }

    let expected_key = UniformSpartanKey::<F>::new(expected_num_steps);
    if expected_key.vk_digest != proof.inner_proof.r1cs_vk_digest {
        return Err(BlockTraceError::CpuR1CSShapeDigestMismatch {
            block_index: proof.public_input.block_index,
        });
    }
    if proof.inner_proof.used_lookahead_cycle != proof.inner_proof.lookahead_cycle_digest.is_some()
    {
        return Err(BlockTraceError::CpuLookaheadMismatch {
            block_index: proof.public_input.block_index,
            proof_used_lookahead: proof.inner_proof.used_lookahead_cycle,
            actual_used_lookahead: proof.inner_proof.lookahead_cycle_digest.is_some(),
        });
    }

    Ok(())
}

pub fn verify_cpu_block_witness<Digest, F>(
    bytecode_preprocessing: &BytecodePreprocessing,
    block: &TraceBlock,
    lookahead_cycle: Option<&Cycle>,
    proof: &CpuBlockProof<Digest, F>,
) -> Result<(), BlockTraceError>
where
    Digest: Clone + PartialEq,
    F: JoltField,
{
    verify_cpu_block_proof(proof)?;
    validate_trace_block_shape(block)?;

    let expected_public_input =
        BlockPublicInput::from_trace_block(block, proof.public_input.program_digest.clone());
    if proof.public_input != expected_public_input {
        return Err(BlockTraceError::CpuPublicInputMismatch {
            block_index: block.block_index,
        });
    }

    if proof.inner_proof.used_lookahead_cycle != lookahead_cycle.is_some() {
        return Err(BlockTraceError::CpuLookaheadMismatch {
            block_index: block.block_index,
            proof_used_lookahead: proof.inner_proof.used_lookahead_cycle,
            actual_used_lookahead: lookahead_cycle.is_some(),
        });
    }
    let expected_lookahead_cycle_digest =
        digest_cpu_lookahead_cycle(block.block_index, lookahead_cycle)?;
    if proof.inner_proof.lookahead_cycle_digest != expected_lookahead_cycle_digest {
        return Err(BlockTraceError::CpuLookaheadDigestMismatch {
            block_index: block.block_index,
        });
    }

    validate_cpu_r1cs_block::<F>(bytecode_preprocessing, block, lookahead_cycle)
}

pub fn verify_block_proof_bundle<Digest, F>(
    bytecode_preprocessing: &BytecodePreprocessing,
    block: &TraceBlock,
    lookahead_cycle: Option<&Cycle>,
    bundle: &BlockProofBundle<Digest, F>,
) -> Result<(), BlockTraceError>
where
    Digest: Clone + PartialEq,
    F: JoltField,
{
    verify_cpu_block_witness(
        bytecode_preprocessing,
        block,
        lookahead_cycle,
        &bundle.cpu_proof,
    )?;
    verify_block_io_claims(block, &bundle.io_claims)?;
    verify_block_register_claim(block, &bundle.io_claims, &bundle.register_claim)?;
    verify_block_ram_claim(block, &bundle.io_claims, &bundle.ram_claim)?;
    verify_block_lookup_claim(block, &bundle.io_claims, &bundle.lookup_claim)
}

pub fn verify_block_proof_bundle_chain<Digest, F>(
    bytecode_preprocessing: &BytecodePreprocessing,
    blocks: &[TraceBlock],
    bundles: &[BlockProofBundle<Digest, F>],
) -> Result<(), BlockTraceError>
where
    Digest: Clone + PartialEq,
    F: JoltField,
{
    verify_block_proof_bundle_chain_with_external_lookahead(
        bytecode_preprocessing,
        blocks,
        None,
        bundles,
    )
}

pub fn verify_block_proof_bundle_chain_with_external_lookahead<Digest, F>(
    bytecode_preprocessing: &BytecodePreprocessing,
    blocks: &[TraceBlock],
    external_lookahead_cycle: Option<&Cycle>,
    bundles: &[BlockProofBundle<Digest, F>],
) -> Result<(), BlockTraceError>
where
    Digest: Clone + PartialEq,
    F: JoltField,
{
    if blocks.len() != bundles.len() {
        return Err(BlockTraceError::BlockProofBundleChainLengthMismatch {
            blocks: blocks.len(),
            bundles: bundles.len(),
        });
    }

    let public_inputs = bundles
        .iter()
        .map(|bundle| bundle.public_input().clone())
        .collect::<Vec<_>>();
    validate_block_chain(&public_inputs)?;

    for (index, (block, bundle)) in blocks.iter().zip(bundles).enumerate() {
        let lookahead = block_lookahead_cycle(blocks, index, external_lookahead_cycle);
        verify_block_proof_bundle(bytecode_preprocessing, block, lookahead, bundle)?;
    }

    let io_claims = bundles
        .iter()
        .map(|bundle| bundle.io_claims.clone())
        .collect::<Vec<_>>();
    let register_claims = bundles
        .iter()
        .map(|bundle| bundle.register_claim.clone())
        .collect::<Vec<_>>();
    let ram_claims = bundles
        .iter()
        .map(|bundle| bundle.ram_claim.clone())
        .collect::<Vec<_>>();
    let lookup_claims = bundles
        .iter()
        .map(|bundle| bundle.lookup_claim.clone())
        .collect::<Vec<_>>();

    verify_block_register_claim_chain(blocks, &io_claims, &register_claims)?;
    verify_block_ram_claim_chain(blocks, &io_claims, &ram_claims)?;
    verify_block_lookup_claim_chain(blocks, &io_claims, &lookup_claims)
}

pub fn build_block_fold_input<Digest, F>(
    bundle: &BlockProofBundle<Digest, F>,
) -> BlockFoldInput<Digest, F>
where
    Digest: Clone,
    F: JoltField,
{
    BlockFoldInput {
        program_digest: bundle.public_input().program_digest.clone(),
        state: build_foldable_block_state(bundle),
    }
}

pub fn build_block_fold_inputs<Digest, F>(
    bundles: &[BlockProofBundle<Digest, F>],
) -> Vec<BlockFoldInput<Digest, F>>
where
    Digest: Clone,
    F: JoltField,
{
    bundles.iter().map(build_block_fold_input).collect()
}

pub fn verify_block_fold_input<Digest, F>(
    bytecode_preprocessing: &BytecodePreprocessing,
    block: &TraceBlock,
    lookahead_cycle: Option<&Cycle>,
    bundle: &BlockProofBundle<Digest, F>,
    fold_input: &BlockFoldInput<Digest, F>,
) -> Result<(), BlockTraceError>
where
    Digest: Clone + PartialEq,
    F: JoltField,
{
    verify_block_proof_bundle(bytecode_preprocessing, block, lookahead_cycle, bundle)?;

    let expected = build_block_fold_input(bundle);
    if fold_input != &expected {
        return Err(BlockTraceError::BlockFoldInputMismatch {
            block_index: block.block_index,
        });
    }

    Ok(())
}

pub fn verify_block_fold_input_chain<Digest, F>(
    bytecode_preprocessing: &BytecodePreprocessing,
    blocks: &[TraceBlock],
    bundles: &[BlockProofBundle<Digest, F>],
    fold_inputs: &[BlockFoldInput<Digest, F>],
) -> Result<(), BlockTraceError>
where
    Digest: Clone + PartialEq,
    F: JoltField,
{
    verify_block_fold_input_chain_with_external_lookahead(
        bytecode_preprocessing,
        blocks,
        None,
        bundles,
        fold_inputs,
    )
}

pub fn verify_block_fold_input_chain_with_external_lookahead<Digest, F>(
    bytecode_preprocessing: &BytecodePreprocessing,
    blocks: &[TraceBlock],
    external_lookahead_cycle: Option<&Cycle>,
    bundles: &[BlockProofBundle<Digest, F>],
    fold_inputs: &[BlockFoldInput<Digest, F>],
) -> Result<(), BlockTraceError>
where
    Digest: Clone + PartialEq,
    F: JoltField,
{
    if blocks.len() != bundles.len() || blocks.len() != fold_inputs.len() {
        return Err(BlockTraceError::BlockFoldInputChainLengthMismatch {
            blocks: blocks.len(),
            bundles: bundles.len(),
            fold_inputs: fold_inputs.len(),
        });
    }

    verify_block_proof_bundle_chain_with_external_lookahead(
        bytecode_preprocessing,
        blocks,
        external_lookahead_cycle,
        bundles,
    )?;

    for (index, ((block, bundle), fold_input)) in
        blocks.iter().zip(bundles).zip(fold_inputs).enumerate()
    {
        let lookahead = block_lookahead_cycle(blocks, index, external_lookahead_cycle);
        verify_block_fold_input(bytecode_preprocessing, block, lookahead, bundle, fold_input)?;
    }

    validate_block_fold_input_chain(fold_inputs)
}

pub fn build_block_fold_accumulator<Digest, F>(
    fold_inputs: &[BlockFoldInput<Digest, F>],
) -> Result<BlockFoldAccumulator<Digest>, BlockTraceError>
where
    Digest: Clone + PartialEq + AsRef<[u8]>,
    F: JoltField,
{
    build_block_fold_accumulator_with_backend(fold_inputs, &MockFoldingBackend)
}

pub fn verify_block_fold_accumulator<Digest, F>(
    fold_inputs: &[BlockFoldInput<Digest, F>],
    accumulator: &BlockFoldAccumulator<Digest>,
) -> Result<(), BlockTraceError>
where
    Digest: Clone + PartialEq + AsRef<[u8]>,
    F: JoltField,
{
    verify_block_fold_accumulator_with_backend(fold_inputs, accumulator, &MockFoldingBackend)
}

pub fn build_block_fold_accumulator_with_backend<Digest, F, Backend>(
    fold_inputs: &[BlockFoldInput<Digest, F>],
    folding_backend: &Backend,
) -> Result<Backend::Accumulator, BlockTraceError>
where
    Digest: Clone + PartialEq + AsRef<[u8]>,
    F: JoltField,
    Backend: BlockFoldingBackend<Digest, F>,
{
    folding_backend.fold(fold_inputs)
}

pub fn verify_block_fold_accumulator_with_backend<Digest, F, Backend>(
    fold_inputs: &[BlockFoldInput<Digest, F>],
    accumulator: &Backend::Accumulator,
    folding_backend: &Backend,
) -> Result<(), BlockTraceError>
where
    Digest: Clone + PartialEq + AsRef<[u8]>,
    F: JoltField,
    Backend: BlockFoldingBackend<Digest, F>,
{
    folding_backend.verify(fold_inputs, accumulator)
}

pub fn build_verified_block_fold_accumulator<Digest, F>(
    bytecode_preprocessing: &BytecodePreprocessing,
    blocks: &[TraceBlock],
    bundles: &[BlockProofBundle<Digest, F>],
    fold_inputs: &[BlockFoldInput<Digest, F>],
) -> Result<BlockFoldAccumulator<Digest>, BlockTraceError>
where
    Digest: Clone + PartialEq + AsRef<[u8]>,
    F: JoltField,
{
    verify_block_fold_input_chain(bytecode_preprocessing, blocks, bundles, fold_inputs)?;
    build_block_fold_accumulator(fold_inputs)
}

pub fn extract_block_io_claims(block: &TraceBlock) -> Result<BlockIOClaims, BlockTraceError> {
    validate_trace_block_shape(block)?;

    let mut register_reads = Vec::new();
    let mut register_writes = Vec::new();
    let mut ram_accesses = Vec::new();
    let mut lookup_claims = Vec::with_capacity(block.cycles.len());

    for (local_cycle, cycle) in block.cycles.iter().enumerate() {
        let global_cycle = block.global_cycle_start + local_cycle;

        if let Some((register_index, value)) = cycle.rs1_read() {
            register_reads.push(RegisterReadClaim {
                local_cycle,
                global_cycle,
                register_index,
                value,
                kind: RegisterReadKind::Rs1,
            });
        }

        if let Some((register_index, value)) = cycle.rs2_read() {
            register_reads.push(RegisterReadClaim {
                local_cycle,
                global_cycle,
                register_index,
                value,
                kind: RegisterReadKind::Rs2,
            });
        }

        if let Some((register_index, pre_value, post_value)) = cycle.rd_write() {
            register_writes.push(RegisterWriteClaim {
                local_cycle,
                global_cycle,
                register_index,
                pre_value,
                post_value,
            });
        }

        match cycle.ram_access() {
            tracer::instruction::RAMAccess::Read(read) => {
                ram_accesses.push(RamAccessClaim::Read {
                    local_cycle,
                    global_cycle,
                    address: read.address,
                    value: read.value,
                });
            }
            tracer::instruction::RAMAccess::Write(write) => {
                ram_accesses.push(RamAccessClaim::Write {
                    local_cycle,
                    global_cycle,
                    address: write.address,
                    pre_value: write.pre_value,
                    post_value: write.post_value,
                });
            }
            tracer::instruction::RAMAccess::NoOp => {}
        }

        let (left_instruction_input, right_instruction_input) =
            LookupQuery::<XLEN>::to_instruction_inputs(cycle);
        let (left_lookup_operand, right_lookup_operand) =
            LookupQuery::<XLEN>::to_lookup_operands(cycle);
        lookup_claims.push(LookupClaim {
            local_cycle,
            global_cycle,
            left_instruction_input,
            right_instruction_input,
            left_lookup_operand,
            right_lookup_operand,
            lookup_index: LookupQuery::<XLEN>::to_lookup_index(cycle),
            lookup_output: LookupQuery::<XLEN>::to_lookup_output(cycle),
        });
    }

    let claims = BlockIOClaims {
        block_index: block.block_index,
        global_cycle_start: block.global_cycle_start,
        active_cycles: block.active_cycles,
        register_reads,
        register_writes,
        ram_accesses,
        lookup_claims,
    };

    verify_block_io_claims(block, &claims)?;
    Ok(claims)
}

pub fn verify_block_io_claims(
    block: &TraceBlock,
    claims: &BlockIOClaims,
) -> Result<(), BlockTraceError> {
    validate_trace_block_shape(block)?;
    claims.validate_shape(block)?;

    let expected_claims = extract_block_io_claims_unchecked(block);
    if claims != &expected_claims {
        return Err(BlockTraceError::BlockIOClaimMismatch {
            block_index: block.block_index,
        });
    }

    validate_register_flow(block)?;
    validate_ram_flow(block)
}

pub fn verify_block_io_claim_chain(
    blocks: &[TraceBlock],
    claims: &[BlockIOClaims],
) -> Result<(), BlockTraceError> {
    if blocks.len() != claims.len() {
        return Err(BlockTraceError::BlockIOClaimChainLengthMismatch {
            blocks: blocks.len(),
            claims: claims.len(),
        });
    }

    let public_inputs = blocks
        .iter()
        .map(|block| BlockPublicInput::from_trace_block(block, ()))
        .collect::<Vec<_>>();
    validate_block_chain(&public_inputs)?;

    for (block, claim) in blocks.iter().zip(claims) {
        verify_block_io_claims(block, claim)?;
    }

    Ok(())
}

pub fn build_block_register_claim(
    block: &TraceBlock,
    io_claims: &BlockIOClaims,
) -> Result<BlockRegisterClaim, BlockTraceError> {
    verify_block_io_claims(block, io_claims)?;
    Ok(build_block_register_claim_unchecked(block, io_claims))
}

pub fn verify_block_register_claim(
    block: &TraceBlock,
    io_claims: &BlockIOClaims,
    register_claim: &BlockRegisterClaim,
) -> Result<(), BlockTraceError> {
    verify_block_io_claims(block, io_claims)?;
    register_claim.validate_shape(block, io_claims)?;

    let expected = build_block_register_claim_unchecked(block, io_claims);
    if register_claim != &expected {
        return Err(BlockTraceError::BlockRegisterClaimMismatch {
            block_index: block.block_index,
        });
    }

    Ok(())
}

pub fn verify_block_register_claim_chain(
    blocks: &[TraceBlock],
    io_claims: &[BlockIOClaims],
    register_claims: &[BlockRegisterClaim],
) -> Result<(), BlockTraceError> {
    if blocks.len() != io_claims.len() || blocks.len() != register_claims.len() {
        return Err(BlockTraceError::BlockRegisterClaimChainLengthMismatch {
            blocks: blocks.len(),
            io_claims: io_claims.len(),
            register_claims: register_claims.len(),
        });
    }

    verify_block_io_claim_chain(blocks, io_claims)?;

    for ((block, io_claim), register_claim) in blocks.iter().zip(io_claims).zip(register_claims) {
        verify_block_register_claim(block, io_claim, register_claim)?;
    }

    for window in register_claims.windows(2) {
        if window[0].end_register_digest != window[1].start_register_digest {
            return Err(BlockTraceError::BlockRegisterClaimBoundaryMismatch {
                current_block: window[0].block_index,
                next_block: window[1].block_index,
            });
        }
    }

    Ok(())
}

pub fn build_block_ram_claim(
    block: &TraceBlock,
    io_claims: &BlockIOClaims,
) -> Result<BlockRamClaim, BlockTraceError> {
    verify_block_io_claims(block, io_claims)?;
    Ok(build_block_ram_claim_unchecked(block, io_claims))
}

pub fn verify_block_ram_claim(
    block: &TraceBlock,
    io_claims: &BlockIOClaims,
    ram_claim: &BlockRamClaim,
) -> Result<(), BlockTraceError> {
    verify_block_io_claims(block, io_claims)?;
    ram_claim.validate_shape(block, io_claims)?;

    let expected = build_block_ram_claim_unchecked(block, io_claims);
    if ram_claim != &expected {
        return Err(BlockTraceError::BlockRamClaimMismatch {
            block_index: block.block_index,
        });
    }

    Ok(())
}

pub fn verify_block_ram_claim_chain(
    blocks: &[TraceBlock],
    io_claims: &[BlockIOClaims],
    ram_claims: &[BlockRamClaim],
) -> Result<(), BlockTraceError> {
    if blocks.len() != io_claims.len() || blocks.len() != ram_claims.len() {
        return Err(BlockTraceError::BlockRamClaimChainLengthMismatch {
            blocks: blocks.len(),
            io_claims: io_claims.len(),
            ram_claims: ram_claims.len(),
        });
    }

    verify_block_io_claim_chain(blocks, io_claims)?;

    for ((block, io_claim), ram_claim) in blocks.iter().zip(io_claims).zip(ram_claims) {
        verify_block_ram_claim(block, io_claim, ram_claim)?;
    }

    validate_ram_claim_continuity(io_claims)
}

pub fn build_block_lookup_claim(
    block: &TraceBlock,
    io_claims: &BlockIOClaims,
) -> Result<BlockLookupClaim, BlockTraceError> {
    verify_block_io_claims(block, io_claims)?;
    Ok(build_block_lookup_claim_unchecked(block, io_claims))
}

pub fn verify_block_lookup_claim(
    block: &TraceBlock,
    io_claims: &BlockIOClaims,
    lookup_claim: &BlockLookupClaim,
) -> Result<(), BlockTraceError> {
    verify_block_io_claims(block, io_claims)?;
    lookup_claim.validate_shape(block, io_claims)?;

    let expected = build_block_lookup_claim_unchecked(block, io_claims);
    if lookup_claim != &expected {
        if lookup_claim.logup_proof != expected.logup_proof {
            return Err(BlockTraceError::BlockLookupLogUpProofMismatch {
                block_index: block.block_index,
            });
        }
        return Err(BlockTraceError::BlockLookupClaimMismatch {
            block_index: block.block_index,
        });
    }

    Ok(())
}

pub fn verify_block_lookup_claim_chain(
    blocks: &[TraceBlock],
    io_claims: &[BlockIOClaims],
    lookup_claims: &[BlockLookupClaim],
) -> Result<(), BlockTraceError> {
    if blocks.len() != io_claims.len() || blocks.len() != lookup_claims.len() {
        return Err(BlockTraceError::BlockLookupClaimChainLengthMismatch {
            blocks: blocks.len(),
            io_claims: io_claims.len(),
            lookup_claims: lookup_claims.len(),
        });
    }

    verify_block_io_claim_chain(blocks, io_claims)?;

    for ((block, io_claim), lookup_claim) in blocks.iter().zip(io_claims).zip(lookup_claims) {
        verify_block_lookup_claim(block, io_claim, lookup_claim)?;
    }

    Ok(())
}

fn build_foldable_block_state<Digest, F>(
    bundle: &BlockProofBundle<Digest, F>,
) -> FoldableBlockState<F>
where
    F: JoltField,
{
    let public_input = bundle.public_input();
    let cpu_proof = &bundle.cpu_proof.inner_proof;
    let register_claim = &bundle.register_claim;
    let ram_claim = &bundle.ram_claim;
    let lookup_claim = &bundle.lookup_claim;
    let logup_proof = &lookup_claim.logup_proof;

    let mut state = FoldableBlockState {
        block_index: public_input.block_index,
        global_cycle_start: public_input.global_cycle_start,
        global_cycle_end: public_input.global_cycle_end(),
        active_cycles: public_input.active_cycles,
        start_state_digest: digest_machine_boundary_state(&public_input.start_state),
        end_state_digest: digest_machine_boundary_state(&public_input.end_state),
        start_register_digest: register_claim.start_register_digest,
        end_register_digest: register_claim.end_register_digest,
        register_reads_digest: register_claim.reads_digest,
        register_writes_digest: register_claim.writes_digest,
        register_read_count: register_claim.read_count,
        register_write_count: register_claim.write_count,
        ram_accesses_digest: ram_claim.accesses_digest,
        ram_touched_addresses_digest: ram_claim.touched_addresses_digest,
        ram_access_count: ram_claim.access_count,
        ram_touched_address_count: ram_claim.touched_address_count,
        lookup_claims_digest: lookup_claim.claims_digest,
        lookup_entry_summaries_digest: lookup_claim.entry_summaries_digest,
        lookup_count: lookup_claim.lookup_count,
        lookup_distinct_entry_count: lookup_claim.distinct_lookup_entry_count,
        lookup_logup_proof_digest: logup_proof.proof_digest,
        lookup_logup_tuple_challenge: logup_proof.tuple_challenge,
        lookup_logup_denominator_challenge: logup_proof.denominator_challenge,
        lookup_logup_denominator_retry_count: logup_proof.denominator_retry_count,
        lookup_logup_query_sum: logup_proof.query_sum,
        lookup_logup_table_sum: logup_proof.table_sum,
        r1cs_rows_checked: cpu_proof.r1cs_rows_checked,
        r1cs_num_steps: cpu_proof.r1cs_num_steps,
        r1cs_vk_digest: cpu_proof.r1cs_vk_digest,
        used_lookahead_cycle: cpu_proof.used_lookahead_cycle,
        lookahead_cycle_digest: cpu_proof.lookahead_cycle_digest,
        state_digest: [0u8; 32],
    };
    state.state_digest = digest_foldable_block_state(&state);
    state
}

fn validate_block_fold_input_chain<Digest, F>(
    fold_inputs: &[BlockFoldInput<Digest, F>],
) -> Result<(), BlockTraceError> {
    for window in fold_inputs.windows(2) {
        window[0].state.validate_contiguous_with(&window[1].state)?;
    }

    Ok(())
}

fn build_block_register_claim_unchecked(
    block: &TraceBlock,
    io_claims: &BlockIOClaims,
) -> BlockRegisterClaim {
    BlockRegisterClaim {
        block_index: block.block_index,
        global_cycle_start: block.global_cycle_start,
        active_cycles: block.active_cycles,
        start_register_digest: digest_register_state(&block.start_state),
        end_register_digest: digest_register_state(&block.end_state),
        reads_digest: digest_register_reads(&io_claims.register_reads),
        writes_digest: digest_register_writes(&io_claims.register_writes),
        read_count: io_claims.register_reads.len(),
        write_count: io_claims.register_writes.len(),
    }
}

fn build_block_ram_claim_unchecked(block: &TraceBlock, io_claims: &BlockIOClaims) -> BlockRamClaim {
    let address_summaries = ram_address_summaries(&io_claims.ram_accesses);

    BlockRamClaim {
        block_index: block.block_index,
        global_cycle_start: block.global_cycle_start,
        active_cycles: block.active_cycles,
        access_count: io_claims.ram_accesses.len(),
        touched_address_count: address_summaries.len(),
        accesses_digest: digest_ram_accesses(&io_claims.ram_accesses),
        touched_addresses_digest: digest_ram_summaries(&address_summaries),
    }
}

fn validate_ram_claim_continuity(io_claims: &[BlockIOClaims]) -> Result<(), BlockTraceError> {
    let mut latest_values = HashMap::<u64, u64>::new();

    for claims in io_claims {
        for summary in ram_address_summaries(&claims.ram_accesses) {
            if let Some(expected) = latest_values.get(&summary.address).copied() {
                if summary.first_value != expected {
                    return Err(BlockTraceError::BlockRamClaimContinuityMismatch {
                        block_index: claims.block_index,
                        address: summary.address,
                        expected,
                        actual: summary.first_value,
                    });
                }
            }

            latest_values.insert(summary.address, summary.final_value);
        }
    }

    Ok(())
}

fn build_block_lookup_claim_unchecked(
    block: &TraceBlock,
    io_claims: &BlockIOClaims,
) -> BlockLookupClaim {
    let entry_summaries = lookup_entry_summaries(&io_claims.lookup_claims);
    let claims_digest = digest_lookup_claims(&io_claims.lookup_claims);
    let entry_summaries_digest = digest_lookup_entry_summaries(&entry_summaries);
    let logup_proof = build_block_logup_proof(
        block,
        &io_claims.lookup_claims,
        &entry_summaries,
        claims_digest,
        entry_summaries_digest,
    );

    BlockLookupClaim {
        block_index: block.block_index,
        global_cycle_start: block.global_cycle_start,
        active_cycles: block.active_cycles,
        lookup_count: io_claims.lookup_claims.len(),
        distinct_lookup_entry_count: entry_summaries.len(),
        claims_digest,
        entry_summaries_digest,
        logup_proof,
    }
}

fn digest_register_state(state: &MachineBoundaryState) -> [u8; 32] {
    let mut hasher = Sha3_256::new();
    hasher.update(b"JOLT_NOVA_REGISTER_STATE_V1");
    update_usize(&mut hasher, REGISTER_COUNT as usize);

    for (register_index, value) in state.registers.iter().enumerate() {
        hasher.update([register_index as u8]);
        hasher.update(value.to_le_bytes());
    }

    finalize_digest(hasher)
}

fn digest_machine_boundary_state(state: &MachineBoundaryState) -> [u8; 32] {
    let mut hasher = Sha3_256::new();
    hasher.update(b"JOLT_NOVA_MACHINE_BOUNDARY_STATE_V1");
    update_usize(&mut hasher, state.global_cycle);
    update_usize(&mut hasher, state.emulator_trace_len);
    hasher.update(state.pc.to_le_bytes());
    hasher.update([u8::from(state.terminated)]);
    update_usize(&mut hasher, REGISTER_COUNT as usize);

    for (register_index, value) in state.registers.iter().enumerate() {
        hasher.update([register_index as u8]);
        hasher.update(value.to_le_bytes());
    }

    finalize_digest(hasher)
}

fn digest_cpu_lookahead_cycle(
    block_index: usize,
    lookahead_cycle: Option<&Cycle>,
) -> Result<Option<[u8; 32]>, BlockTraceError> {
    let Some(lookahead_cycle) = lookahead_cycle else {
        return Ok(None);
    };
    let bytes = postcard::to_stdvec(lookahead_cycle)
        .map_err(|_| BlockTraceError::CpuLookaheadSerializationFailed { block_index })?;
    let mut hasher = Sha3_256::new();
    hasher.update(b"JOLT_NOVA_CPU_LOOKAHEAD_CYCLE_V1");
    update_usize(&mut hasher, bytes.len());
    hasher.update(bytes);
    Ok(Some(finalize_digest(hasher)))
}

fn digest_foldable_block_state<F>(state: &FoldableBlockState<F>) -> [u8; 32]
where
    F: JoltField,
{
    let mut hasher = Sha3_256::new();
    hasher.update(b"JOLT_NOVA_FOLDABLE_BLOCK_STATE_V3");
    update_usize(&mut hasher, state.block_index);
    update_usize(&mut hasher, state.global_cycle_start);
    update_usize(&mut hasher, state.global_cycle_end);
    update_usize(&mut hasher, state.active_cycles);
    hasher.update(state.start_state_digest);
    hasher.update(state.end_state_digest);
    hasher.update(state.start_register_digest);
    hasher.update(state.end_register_digest);
    hasher.update(state.register_reads_digest);
    hasher.update(state.register_writes_digest);
    update_usize(&mut hasher, state.register_read_count);
    update_usize(&mut hasher, state.register_write_count);
    hasher.update(state.ram_accesses_digest);
    hasher.update(state.ram_touched_addresses_digest);
    update_usize(&mut hasher, state.ram_access_count);
    update_usize(&mut hasher, state.ram_touched_address_count);
    hasher.update(state.lookup_claims_digest);
    hasher.update(state.lookup_entry_summaries_digest);
    update_usize(&mut hasher, state.lookup_count);
    update_usize(&mut hasher, state.lookup_distinct_entry_count);
    hasher.update(state.lookup_logup_proof_digest);
    update_field(&mut hasher, state.lookup_logup_tuple_challenge);
    update_field(&mut hasher, state.lookup_logup_denominator_challenge);
    update_usize(&mut hasher, state.lookup_logup_denominator_retry_count);
    update_field(&mut hasher, state.lookup_logup_query_sum);
    update_field(&mut hasher, state.lookup_logup_table_sum);
    update_usize(&mut hasher, state.r1cs_rows_checked);
    update_usize(&mut hasher, state.r1cs_num_steps);
    update_field(&mut hasher, state.r1cs_vk_digest);
    hasher.update([u8::from(state.used_lookahead_cycle)]);
    match state.lookahead_cycle_digest {
        Some(digest) => {
            hasher.update([1]);
            hasher.update(digest);
        }
        None => hasher.update([0]),
    }

    finalize_digest(hasher)
}

fn digest_empty_fold_accumulator() -> [u8; 32] {
    let mut hasher = Sha3_256::new();
    hasher.update(b"JOLT_NOVA_BLOCK_FOLD_ACCUMULATOR_EMPTY_V1");
    finalize_digest(hasher)
}

fn digest_final_folded_instance<Digest>(instance: &FinalFoldedInstance<Digest>) -> [u8; 32]
where
    Digest: AsRef<[u8]>,
{
    let mut hasher = Sha3_256::new();
    hasher.update(FINAL_FOLDED_INSTANCE_VERSION.as_bytes());
    update_nova_fold_config(&mut hasher, &instance.config);
    update_block_fold_metadata(&mut hasher, &instance.metadata);
    hasher.update(instance.recursive_snark_output_digest);
    for scalar_bytes in instance.recursive_z_state {
        hasher.update(scalar_bytes);
    }
    finalize_digest(hasher)
}

fn encode_spartan_final_public_input_bytes<Digest>(
    instance: &FinalFoldedInstance<Digest>,
) -> Vec<u8>
where
    Digest: AsRef<[u8]>,
{
    let mut output = Vec::new();
    append_bytes(
        &mut output,
        SPARTAN_FINAL_INSTANCE_ENCODING_VERSION.as_bytes(),
    );
    append_bytes(&mut output, b"public-inputs");
    append_bytes(&mut output, &instance.instance_digest);
    append_bytes(&mut output, &instance.metadata.accumulator_digest);
    append_bytes(&mut output, &instance.recursive_snark_output_digest);
    append_bytes(
        &mut output,
        &digest_nova_z_state(&instance.recursive_z_state),
    );
    append_usize(&mut output, instance.metadata.absorbed_blocks);
    append_optional_usize(&mut output, instance.metadata.first_block_index);
    append_optional_usize(&mut output, instance.metadata.last_block_index);
    append_optional_usize(&mut output, instance.metadata.global_cycle_start);
    append_optional_usize(&mut output, instance.metadata.global_cycle_end);
    append_usize(&mut output, instance.metadata.total_active_cycles);
    append_usize(&mut output, instance.metadata.total_register_reads);
    append_usize(&mut output, instance.metadata.total_register_writes);
    append_usize(&mut output, instance.metadata.total_ram_accesses);
    append_usize(&mut output, instance.metadata.total_lookup_claims);
    output
}

fn encode_spartan_final_witness_bytes<Digest>(instance: &FinalFoldedInstance<Digest>) -> Vec<u8>
where
    Digest: AsRef<[u8]>,
{
    let mut output = Vec::new();
    append_bytes(
        &mut output,
        SPARTAN_FINAL_INSTANCE_ENCODING_VERSION.as_bytes(),
    );
    append_bytes(&mut output, b"witness");
    append_nova_fold_config(&mut output, &instance.config);
    append_block_fold_metadata(&mut output, &instance.metadata);
    append_bytes(&mut output, &instance.recursive_snark_output_digest);
    append_usize(&mut output, NOVA_Z_ARITY);
    for (index, scalar_bytes) in instance.recursive_z_state.iter().enumerate() {
        append_usize(&mut output, index);
        append_bytes(&mut output, scalar_bytes);
    }
    append_bytes(&mut output, &instance.instance_digest);
    output
}

fn digest_nova_z_state(state: &NovaFoldZState) -> [u8; 32] {
    let mut hasher = Sha3_256::new();
    hasher.update(b"JOLT_NOVA_Z_STATE_V1");
    update_usize(&mut hasher, NOVA_Z_ARITY);
    for (index, scalar_bytes) in state.iter().enumerate() {
        update_usize(&mut hasher, index);
        hasher.update(scalar_bytes);
    }
    finalize_digest(hasher)
}

fn digest_spartan_final_encoding_component(label: &'static str, bytes: &[u8]) -> [u8; 32] {
    let mut hasher = Sha3_256::new();
    hasher.update(b"JOLT_NOVA_SPARTAN_FINAL_ENCODING_COMPONENT_V1");
    hasher.update(SPARTAN_FINAL_INSTANCE_ENCODING_VERSION.as_bytes());
    hasher.update(label.as_bytes());
    update_usize(&mut hasher, bytes.len());
    hasher.update(bytes);
    finalize_digest(hasher)
}

fn digest_spartan_final_instance_encoding(
    public_input_digest: [u8; 32],
    witness_digest: [u8; 32],
) -> [u8; 32] {
    let mut hasher = Sha3_256::new();
    hasher.update(b"JOLT_NOVA_SPARTAN_FINAL_INSTANCE_ENCODING_V1");
    hasher.update(SPARTAN_FINAL_INSTANCE_ENCODING_VERSION.as_bytes());
    hasher.update(public_input_digest);
    hasher.update(witness_digest);
    finalize_digest(hasher)
}

fn encode_final_folded_proof_envelope_size_bytes<Digest>(
    proof: &FinalFoldedProof<Digest>,
) -> Vec<u8> {
    let mut output = Vec::new();
    append_bytes(
        &mut output,
        b"JOLT_NOVA_FINAL_FOLDED_PROOF_SIZE_ENVELOPE_V1",
    );
    append_bytes(&mut output, proof.proof_system.as_bytes());
    append_bytes(&mut output, &proof.instance.instance_digest);
    append_optional_digest(&mut output, proof.spartan_encoding_digest);
    append_bytes(&mut output, &proof.proof_digest);
    append_usize(
        &mut output,
        proof
            .spartan_proof_bytes
            .as_ref()
            .map(Vec::len)
            .unwrap_or(0),
    );
    output
}

fn digest_final_folded_proof<Digest>(
    proof_system: &'static str,
    instance: &FinalFoldedInstance<Digest>,
    spartan_encoding_digest: Option<[u8; 32]>,
    proof_bytes: Option<&[u8]>,
) -> [u8; 32]
where
    Digest: AsRef<[u8]>,
{
    let mut hasher = Sha3_256::new();
    hasher.update(b"JOLT_NOVA_FINAL_FOLDED_PROOF_V1");
    hasher.update(proof_system.as_bytes());
    hasher.update(instance.instance_digest);
    update_optional_digest(&mut hasher, spartan_encoding_digest);
    match proof_bytes {
        Some(bytes) => {
            hasher.update([1]);
            update_usize(&mut hasher, bytes.len());
            hasher.update(bytes);
        }
        None => hasher.update([0]),
    }
    finalize_digest(hasher)
}

fn digest_fold_accumulator_step<Digest, F>(
    previous_digest: [u8; 32],
    fold_input: &BlockFoldInput<Digest, F>,
) -> [u8; 32]
where
    Digest: AsRef<[u8]>,
    F: JoltField,
{
    let mut hasher = Sha3_256::new();
    hasher.update(b"JOLT_NOVA_BLOCK_FOLD_ACCUMULATOR_STEP_V1");
    hasher.update(previous_digest);
    hasher.update(fold_input.program_digest.as_ref());
    update_usize(&mut hasher, fold_input.state.block_index);
    update_usize(&mut hasher, fold_input.state.global_cycle_start);
    update_usize(&mut hasher, fold_input.state.global_cycle_end);
    hasher.update(fold_input.state.start_state_digest);
    hasher.update(fold_input.state.end_state_digest);
    hasher.update(fold_input.state.start_register_digest);
    hasher.update(fold_input.state.end_register_digest);
    hasher.update(fold_input.state.state_digest);

    finalize_digest(hasher)
}

fn digest_register_reads(reads: &[RegisterReadClaim]) -> [u8; 32] {
    let mut hasher = Sha3_256::new();
    hasher.update(b"JOLT_NOVA_REGISTER_READS_V1");
    update_usize(&mut hasher, reads.len());

    for read in reads {
        update_usize(&mut hasher, read.local_cycle);
        update_usize(&mut hasher, read.global_cycle);
        hasher.update([read.register_index]);
        hasher.update(read.value.to_le_bytes());
        hasher.update([match read.kind {
            RegisterReadKind::Rs1 => 1,
            RegisterReadKind::Rs2 => 2,
        }]);
    }

    finalize_digest(hasher)
}

fn digest_register_writes(writes: &[RegisterWriteClaim]) -> [u8; 32] {
    let mut hasher = Sha3_256::new();
    hasher.update(b"JOLT_NOVA_REGISTER_WRITES_V1");
    update_usize(&mut hasher, writes.len());

    for write in writes {
        update_usize(&mut hasher, write.local_cycle);
        update_usize(&mut hasher, write.global_cycle);
        hasher.update([write.register_index]);
        hasher.update(write.pre_value.to_le_bytes());
        hasher.update(write.post_value.to_le_bytes());
    }

    finalize_digest(hasher)
}

fn digest_ram_accesses(accesses: &[RamAccessClaim]) -> [u8; 32] {
    let mut hasher = Sha3_256::new();
    hasher.update(b"JOLT_NOVA_RAM_ACCESSES_V1");
    update_usize(&mut hasher, accesses.len());

    for access in accesses {
        match access {
            RamAccessClaim::Read {
                local_cycle,
                global_cycle,
                address,
                value,
            } => {
                hasher.update([1]);
                update_usize(&mut hasher, *local_cycle);
                update_usize(&mut hasher, *global_cycle);
                hasher.update(address.to_le_bytes());
                hasher.update(value.to_le_bytes());
            }
            RamAccessClaim::Write {
                local_cycle,
                global_cycle,
                address,
                pre_value,
                post_value,
            } => {
                hasher.update([2]);
                update_usize(&mut hasher, *local_cycle);
                update_usize(&mut hasher, *global_cycle);
                hasher.update(address.to_le_bytes());
                hasher.update(pre_value.to_le_bytes());
                hasher.update(post_value.to_le_bytes());
            }
        }
    }

    finalize_digest(hasher)
}

fn digest_ram_summaries(summaries: &[RamAddressSummary]) -> [u8; 32] {
    let mut hasher = Sha3_256::new();
    hasher.update(b"JOLT_NOVA_RAM_ADDRESS_SUMMARIES_V1");
    update_usize(&mut hasher, summaries.len());

    for summary in summaries {
        hasher.update(summary.address.to_le_bytes());
        hasher.update(summary.first_value.to_le_bytes());
        hasher.update(summary.final_value.to_le_bytes());
        update_usize(&mut hasher, summary.read_count);
        update_usize(&mut hasher, summary.write_count);
        update_usize(&mut hasher, summary.first_global_cycle);
        update_usize(&mut hasher, summary.last_global_cycle);
    }

    finalize_digest(hasher)
}

fn ram_address_summaries(accesses: &[RamAccessClaim]) -> Vec<RamAddressSummary> {
    let mut summaries = HashMap::<u64, RamAddressSummary>::new();

    for access in accesses {
        let (address, first_value, final_value, read_delta, write_delta, global_cycle) =
            match access {
                RamAccessClaim::Read {
                    global_cycle,
                    address,
                    value,
                    ..
                } => (*address, *value, *value, 1, 0, *global_cycle),
                RamAccessClaim::Write {
                    global_cycle,
                    address,
                    pre_value,
                    post_value,
                    ..
                } => (*address, *pre_value, *post_value, 0, 1, *global_cycle),
            };

        if let Some(summary) = summaries.get_mut(&address) {
            summary.final_value = final_value;
            summary.read_count += read_delta;
            summary.write_count += write_delta;
            summary.last_global_cycle = global_cycle;
        } else {
            summaries.insert(
                address,
                RamAddressSummary {
                    address,
                    first_value,
                    final_value,
                    read_count: read_delta,
                    write_count: write_delta,
                    first_global_cycle: global_cycle,
                    last_global_cycle: global_cycle,
                },
            );
        }
    }

    let mut summaries = summaries.into_values().collect::<Vec<_>>();
    summaries.sort_by_key(|summary| summary.address);
    summaries
}

fn digest_lookup_claims(claims: &[LookupClaim]) -> [u8; 32] {
    let mut hasher = Sha3_256::new();
    hasher.update(b"JOLT_NOVA_LOOKUP_CLAIMS_V1");
    update_usize(&mut hasher, claims.len());

    for claim in claims {
        update_usize(&mut hasher, claim.local_cycle);
        update_usize(&mut hasher, claim.global_cycle);
        hasher.update(claim.left_instruction_input.to_le_bytes());
        hasher.update(claim.right_instruction_input.to_le_bytes());
        hasher.update(claim.left_lookup_operand.to_le_bytes());
        hasher.update(claim.right_lookup_operand.to_le_bytes());
        hasher.update(claim.lookup_index.to_le_bytes());
        hasher.update(claim.lookup_output.to_le_bytes());
    }

    finalize_digest(hasher)
}

fn digest_lookup_entry_summaries(summaries: &[LookupEntrySummary]) -> [u8; 32] {
    let mut hasher = Sha3_256::new();
    hasher.update(b"JOLT_NOVA_LOOKUP_ENTRY_SUMMARIES_V1");
    update_usize(&mut hasher, summaries.len());

    for summary in summaries {
        hasher.update(summary.lookup_index.to_le_bytes());
        hasher.update(summary.left_lookup_operand.to_le_bytes());
        hasher.update(summary.right_lookup_operand.to_le_bytes());
        hasher.update(summary.lookup_output.to_le_bytes());
        update_usize(&mut hasher, summary.count);
        update_usize(&mut hasher, summary.first_global_cycle);
        update_usize(&mut hasher, summary.last_global_cycle);
    }

    finalize_digest(hasher)
}

fn build_block_logup_proof(
    block: &TraceBlock,
    claims: &[LookupClaim],
    query_entry_summaries: &[LookupEntrySummary],
    claims_digest: [u8; 32],
    entry_summaries_digest: [u8; 32],
) -> BlockLogUpProof {
    let table_entry_summaries = lookup_entry_summaries_for_block(block);
    debug_assert_eq!(query_entry_summaries, table_entry_summaries);

    let mut tuple_challenge = derive_logup_challenge(
        b"tuple-compression",
        block,
        claims_digest,
        entry_summaries_digest,
    );
    if tuple_challenge.is_zero() {
        tuple_challenge = <ark_bn254::Fr as JoltField>::from_u64(1);
    }

    let query_values = claims
        .iter()
        .map(|claim| {
            compress_logup_entry(
                tuple_challenge,
                claim.lookup_index,
                claim.left_lookup_operand,
                claim.right_lookup_operand,
                claim.lookup_output,
            )
        })
        .collect::<Vec<_>>();
    let table_values = table_entry_summaries
        .iter()
        .map(|summary| {
            compress_logup_entry(
                tuple_challenge,
                summary.lookup_index,
                summary.left_lookup_operand,
                summary.right_lookup_operand,
                summary.lookup_output,
            )
        })
        .collect::<Vec<_>>();

    let base_denominator_challenge =
        derive_logup_challenge(b"denominator", block, claims_digest, entry_summaries_digest);
    let mut denominator_retry_count = 0usize;
    let denominator_challenge = loop {
        let candidate = base_denominator_challenge
            + <ark_bn254::Fr as JoltField>::from_u64(denominator_retry_count as u64);
        let has_zero_denominator = query_values
            .iter()
            .chain(table_values.iter())
            .any(|value| (candidate + value).is_zero());
        if !has_zero_denominator {
            break candidate;
        }
        denominator_retry_count += 1;
    };

    let query_denominators = query_values
        .iter()
        .map(|value| denominator_challenge + value)
        .collect::<Vec<_>>();
    let table_denominators = table_values
        .iter()
        .map(|value| denominator_challenge + value)
        .collect::<Vec<_>>();
    let query_weights = vec![1usize; query_denominators.len()];
    let table_weights = table_entry_summaries
        .iter()
        .map(|summary| summary.count)
        .collect::<Vec<_>>();
    let query_sum = logup_weighted_inverse_sum(&query_denominators, &query_weights);
    let table_sum = logup_weighted_inverse_sum(&table_denominators, &table_weights);

    let mut proof = BlockLogUpProof {
        tuple_challenge,
        denominator_challenge,
        denominator_retry_count,
        query_sum,
        table_sum,
        query_count: claims.len(),
        table_distinct_entry_count: table_entry_summaries.len(),
        proof_digest: [0u8; 32],
    };
    proof.proof_digest = digest_block_logup_proof(&proof);
    proof
}

fn derive_logup_challenge(
    label: &[u8],
    block: &TraceBlock,
    claims_digest: [u8; 32],
    entry_summaries_digest: [u8; 32],
) -> ark_bn254::Fr {
    let mut hasher = Sha3_256::new();
    hasher.update(b"JOLT_NOVA_BLOCK_LOGUP_CHALLENGE_V1");
    update_usize(&mut hasher, label.len());
    hasher.update(label);
    update_usize(&mut hasher, block.block_index);
    update_usize(&mut hasher, block.global_cycle_start);
    update_usize(&mut hasher, block.active_cycles);
    hasher.update(claims_digest);
    hasher.update(entry_summaries_digest);
    <ark_bn254::Fr as JoltField>::from_bytes(&finalize_digest(hasher))
}

fn compress_logup_entry(
    tuple_challenge: ark_bn254::Fr,
    lookup_index: u128,
    left_lookup_operand: u64,
    right_lookup_operand: u128,
    lookup_output: u64,
) -> ark_bn254::Fr {
    let alpha_squared = tuple_challenge.square();
    let alpha_cubed = alpha_squared * tuple_challenge;
    <ark_bn254::Fr as JoltField>::from_u64(left_lookup_operand)
        + tuple_challenge * <ark_bn254::Fr as JoltField>::from_u128(right_lookup_operand)
        + alpha_squared * <ark_bn254::Fr as JoltField>::from_u128(lookup_index)
        + alpha_cubed * <ark_bn254::Fr as JoltField>::from_u64(lookup_output)
}

fn logup_weighted_inverse_sum(denominators: &[ark_bn254::Fr], weights: &[usize]) -> ark_bn254::Fr {
    debug_assert_eq!(denominators.len(), weights.len());
    if denominators.is_empty() {
        return ark_bn254::Fr::zero();
    }

    let one = <ark_bn254::Fr as JoltField>::from_u64(1);
    let mut prefixes = Vec::with_capacity(denominators.len());
    let mut product = one;
    for denominator in denominators {
        prefixes.push(product);
        product *= denominator;
    }

    let mut inverse_suffix = product
        .inverse()
        .expect("LogUp denominator product must be non-zero");
    let mut sum = ark_bn254::Fr::zero();
    for index in (0..denominators.len()).rev() {
        let denominator_inverse = inverse_suffix * prefixes[index];
        inverse_suffix *= denominators[index];
        sum += denominator_inverse * <ark_bn254::Fr as JoltField>::from_u64(weights[index] as u64);
    }
    sum
}

fn digest_block_logup_proof(proof: &BlockLogUpProof) -> [u8; 32] {
    let mut hasher = Sha3_256::new();
    hasher.update(b"JOLT_NOVA_BLOCK_LOGUP_PROOF_V1");
    update_field(&mut hasher, proof.tuple_challenge);
    update_field(&mut hasher, proof.denominator_challenge);
    update_usize(&mut hasher, proof.denominator_retry_count);
    update_field(&mut hasher, proof.query_sum);
    update_field(&mut hasher, proof.table_sum);
    update_usize(&mut hasher, proof.query_count);
    update_usize(&mut hasher, proof.table_distinct_entry_count);
    finalize_digest(hasher)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct LookupEntryKey {
    lookup_index: u128,
    left_lookup_operand: u64,
    right_lookup_operand: u128,
    lookup_output: u64,
}

fn lookup_entry_summaries(claims: &[LookupClaim]) -> Vec<LookupEntrySummary> {
    let mut summaries = HashMap::<LookupEntryKey, LookupEntrySummary>::new();

    for claim in claims {
        let key = LookupEntryKey {
            lookup_index: claim.lookup_index,
            left_lookup_operand: claim.left_lookup_operand,
            right_lookup_operand: claim.right_lookup_operand,
            lookup_output: claim.lookup_output,
        };

        if let Some(summary) = summaries.get_mut(&key) {
            summary.count += 1;
            summary.last_global_cycle = claim.global_cycle;
        } else {
            summaries.insert(
                key,
                LookupEntrySummary {
                    lookup_index: claim.lookup_index,
                    left_lookup_operand: claim.left_lookup_operand,
                    right_lookup_operand: claim.right_lookup_operand,
                    lookup_output: claim.lookup_output,
                    count: 1,
                    first_global_cycle: claim.global_cycle,
                    last_global_cycle: claim.global_cycle,
                },
            );
        }
    }

    let mut summaries = summaries.into_values().collect::<Vec<_>>();
    summaries.sort_by_key(|summary| {
        (
            summary.lookup_index,
            summary.left_lookup_operand,
            summary.right_lookup_operand,
            summary.lookup_output,
        )
    });
    summaries
}

fn lookup_entry_summaries_for_block(block: &TraceBlock) -> Vec<LookupEntrySummary> {
    let claims = block
        .cycles
        .iter()
        .enumerate()
        .map(|(local_cycle, cycle)| {
            let (left_instruction_input, right_instruction_input) =
                LookupQuery::<XLEN>::to_instruction_inputs(cycle);
            let (left_lookup_operand, right_lookup_operand) =
                LookupQuery::<XLEN>::to_lookup_operands(cycle);
            LookupClaim {
                local_cycle,
                global_cycle: block.global_cycle_start + local_cycle,
                left_instruction_input,
                right_instruction_input,
                left_lookup_operand,
                right_lookup_operand,
                lookup_index: LookupQuery::<XLEN>::to_lookup_index(cycle),
                lookup_output: LookupQuery::<XLEN>::to_lookup_output(cycle),
            }
        })
        .collect::<Vec<_>>();
    lookup_entry_summaries(&claims)
}

fn append_usize(output: &mut Vec<u8>, value: usize) {
    output.extend_from_slice(&(value as u64).to_le_bytes());
}

fn append_bytes(output: &mut Vec<u8>, bytes: &[u8]) {
    append_usize(output, bytes.len());
    output.extend_from_slice(bytes);
}

fn append_optional_usize(output: &mut Vec<u8>, value: Option<usize>) {
    match value {
        Some(value) => {
            output.push(1);
            append_usize(output, value);
        }
        None => output.push(0),
    }
}

fn append_optional_digest(output: &mut Vec<u8>, value: Option<[u8; 32]>) {
    match value {
        Some(value) => {
            output.push(1);
            append_bytes(output, &value);
        }
        None => output.push(0),
    }
}

fn append_optional_bytes<Digest>(output: &mut Vec<u8>, value: Option<&Digest>)
where
    Digest: AsRef<[u8]>,
{
    match value {
        Some(value) => {
            output.push(1);
            append_bytes(output, value.as_ref());
        }
        None => output.push(0),
    }
}

fn append_nova_fold_config(output: &mut Vec<u8>, config: &NovaFoldConfig) {
    append_bytes(output, config.backend_name.as_bytes());
    append_bytes(output, config.relation_name.as_bytes());
    append_bytes(output, config.subclaim_backend_name.as_bytes());
    append_bytes(output, config.final_proof_backend_name.as_bytes());
    output.push(u8::from(config.use_zero_knowledge));
}

fn append_block_fold_metadata<Digest>(output: &mut Vec<u8>, metadata: &BlockFoldAccumulator<Digest>)
where
    Digest: AsRef<[u8]>,
{
    append_optional_bytes(output, metadata.program_digest.as_ref());
    append_usize(output, metadata.absorbed_blocks);
    append_optional_usize(output, metadata.first_block_index);
    append_optional_usize(output, metadata.last_block_index);
    append_optional_usize(output, metadata.global_cycle_start);
    append_optional_usize(output, metadata.global_cycle_end);
    append_optional_digest(output, metadata.latest_state_digest);
    append_optional_digest(output, metadata.latest_machine_state_digest);
    append_optional_digest(output, metadata.latest_register_digest);
    append_usize(output, metadata.total_active_cycles);
    append_usize(output, metadata.total_register_reads);
    append_usize(output, metadata.total_register_writes);
    append_usize(output, metadata.total_ram_accesses);
    append_usize(output, metadata.total_lookup_claims);
    append_bytes(output, &metadata.accumulator_digest);
}

fn update_usize(hasher: &mut Sha3_256, value: usize) {
    hasher.update((value as u64).to_le_bytes());
}

fn update_optional_usize(hasher: &mut Sha3_256, value: Option<usize>) {
    match value {
        Some(value) => {
            hasher.update([1]);
            update_usize(hasher, value);
        }
        None => hasher.update([0]),
    }
}

fn update_optional_digest(hasher: &mut Sha3_256, value: Option<[u8; 32]>) {
    match value {
        Some(value) => {
            hasher.update([1]);
            hasher.update(value);
        }
        None => hasher.update([0]),
    }
}

fn update_optional_bytes<Digest>(hasher: &mut Sha3_256, value: Option<&Digest>)
where
    Digest: AsRef<[u8]>,
{
    match value {
        Some(value) => {
            let bytes = value.as_ref();
            hasher.update([1]);
            update_usize(hasher, bytes.len());
            hasher.update(bytes);
        }
        None => hasher.update([0]),
    }
}

fn update_nova_fold_config(hasher: &mut Sha3_256, config: &NovaFoldConfig) {
    hasher.update(config.backend_name.as_bytes());
    hasher.update(config.relation_name.as_bytes());
    hasher.update(config.subclaim_backend_name.as_bytes());
    hasher.update(config.final_proof_backend_name.as_bytes());
    hasher.update([u8::from(config.use_zero_knowledge)]);
}

fn update_block_fold_metadata<Digest>(
    hasher: &mut Sha3_256,
    metadata: &BlockFoldAccumulator<Digest>,
) where
    Digest: AsRef<[u8]>,
{
    update_optional_bytes(hasher, metadata.program_digest.as_ref());
    update_usize(hasher, metadata.absorbed_blocks);
    update_optional_usize(hasher, metadata.first_block_index);
    update_optional_usize(hasher, metadata.last_block_index);
    update_optional_usize(hasher, metadata.global_cycle_start);
    update_optional_usize(hasher, metadata.global_cycle_end);
    update_optional_digest(hasher, metadata.latest_state_digest);
    update_optional_digest(hasher, metadata.latest_machine_state_digest);
    update_optional_digest(hasher, metadata.latest_register_digest);
    update_usize(hasher, metadata.total_active_cycles);
    update_usize(hasher, metadata.total_register_reads);
    update_usize(hasher, metadata.total_register_writes);
    update_usize(hasher, metadata.total_ram_accesses);
    update_usize(hasher, metadata.total_lookup_claims);
    hasher.update(metadata.accumulator_digest);
}

fn update_field<F>(hasher: &mut Sha3_256, value: F)
where
    F: JoltField,
{
    let mut bytes = Vec::new();
    value
        .serialize_compressed(&mut bytes)
        .expect("serializing a field element into Vec<u8> should not fail");
    update_usize(hasher, bytes.len());
    hasher.update(bytes);
}

#[cfg(feature = "nova")]
fn update_nova_scalar(hasher: &mut Sha3_256, value: NovaScalar) {
    let bytes = value.to_bytes();
    update_usize(hasher, bytes.len());
    hasher.update(bytes);
}

fn finalize_digest(hasher: Sha3_256) -> [u8; 32] {
    let digest = hasher.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&digest);
    out
}

fn extract_block_io_claims_unchecked(block: &TraceBlock) -> BlockIOClaims {
    let mut register_reads = Vec::new();
    let mut register_writes = Vec::new();
    let mut ram_accesses = Vec::new();
    let mut lookup_claims = Vec::with_capacity(block.cycles.len());

    for (local_cycle, cycle) in block.cycles.iter().enumerate() {
        let global_cycle = block.global_cycle_start + local_cycle;

        if let Some((register_index, value)) = cycle.rs1_read() {
            register_reads.push(RegisterReadClaim {
                local_cycle,
                global_cycle,
                register_index,
                value,
                kind: RegisterReadKind::Rs1,
            });
        }
        if let Some((register_index, value)) = cycle.rs2_read() {
            register_reads.push(RegisterReadClaim {
                local_cycle,
                global_cycle,
                register_index,
                value,
                kind: RegisterReadKind::Rs2,
            });
        }
        if let Some((register_index, pre_value, post_value)) = cycle.rd_write() {
            register_writes.push(RegisterWriteClaim {
                local_cycle,
                global_cycle,
                register_index,
                pre_value,
                post_value,
            });
        }

        match cycle.ram_access() {
            tracer::instruction::RAMAccess::Read(read) => {
                ram_accesses.push(RamAccessClaim::Read {
                    local_cycle,
                    global_cycle,
                    address: read.address,
                    value: read.value,
                });
            }
            tracer::instruction::RAMAccess::Write(write) => {
                ram_accesses.push(RamAccessClaim::Write {
                    local_cycle,
                    global_cycle,
                    address: write.address,
                    pre_value: write.pre_value,
                    post_value: write.post_value,
                });
            }
            tracer::instruction::RAMAccess::NoOp => {}
        }

        let (left_instruction_input, right_instruction_input) =
            LookupQuery::<XLEN>::to_instruction_inputs(cycle);
        let (left_lookup_operand, right_lookup_operand) =
            LookupQuery::<XLEN>::to_lookup_operands(cycle);
        lookup_claims.push(LookupClaim {
            local_cycle,
            global_cycle,
            left_instruction_input,
            right_instruction_input,
            left_lookup_operand,
            right_lookup_operand,
            lookup_index: LookupQuery::<XLEN>::to_lookup_index(cycle),
            lookup_output: LookupQuery::<XLEN>::to_lookup_output(cycle),
        });
    }

    BlockIOClaims {
        block_index: block.block_index,
        global_cycle_start: block.global_cycle_start,
        active_cycles: block.active_cycles,
        register_reads,
        register_writes,
        ram_accesses,
        lookup_claims,
    }
}

fn validate_register_flow(block: &TraceBlock) -> Result<(), BlockTraceError> {
    let mut registers = block.start_state.registers.map(|register| register as u64);

    for (local_cycle, cycle) in block.cycles.iter().enumerate() {
        if let Some((register_index, value)) = cycle.rs1_read() {
            let expected = registers[register_index as usize];
            if expected != value {
                return Err(BlockTraceError::RegisterReadMismatch {
                    block_index: block.block_index,
                    row_index: local_cycle,
                    register_index,
                    expected,
                    actual: value,
                });
            }
        }

        if let Some((register_index, value)) = cycle.rs2_read() {
            let expected = registers[register_index as usize];
            if expected != value {
                return Err(BlockTraceError::RegisterReadMismatch {
                    block_index: block.block_index,
                    row_index: local_cycle,
                    register_index,
                    expected,
                    actual: value,
                });
            }
        }

        if let Some((register_index, pre_value, post_value)) = cycle.rd_write() {
            let expected = registers[register_index as usize];
            if expected != pre_value {
                return Err(BlockTraceError::RegisterWritePreValueMismatch {
                    block_index: block.block_index,
                    row_index: local_cycle,
                    register_index,
                    expected,
                    actual: pre_value,
                });
            }

            registers[register_index as usize] = post_value;
        }
    }

    for register_index in 0..REGISTER_COUNT as usize {
        let expected = registers[register_index];
        let actual = block.end_state.registers[register_index] as u64;
        if expected != actual {
            return Err(BlockTraceError::RegisterBoundaryMismatch {
                block_index: block.block_index,
                register_index: register_index as u8,
                expected,
                actual,
            });
        }
    }

    Ok(())
}

fn validate_ram_flow(block: &TraceBlock) -> Result<(), BlockTraceError> {
    let mut memory = HashMap::<u64, u64>::new();

    for (local_cycle, cycle) in block.cycles.iter().enumerate() {
        match cycle.ram_access() {
            tracer::instruction::RAMAccess::Read(read) => {
                if let Some(expected) = memory.get(&read.address).copied() {
                    if expected != read.value {
                        return Err(BlockTraceError::RamReadMismatch {
                            block_index: block.block_index,
                            row_index: local_cycle,
                            address: read.address,
                            expected,
                            actual: read.value,
                        });
                    }
                } else {
                    memory.insert(read.address, read.value);
                }
            }
            tracer::instruction::RAMAccess::Write(write) => {
                if let Some(expected) = memory.get(&write.address).copied() {
                    if expected != write.pre_value {
                        return Err(BlockTraceError::RamWritePreValueMismatch {
                            block_index: block.block_index,
                            row_index: local_cycle,
                            address: write.address,
                            expected,
                            actual: write.pre_value,
                        });
                    }
                }

                memory.insert(write.address, write.post_value);
            }
            tracer::instruction::RAMAccess::NoOp => {}
        }
    }

    Ok(())
}

fn validate_trace_block_shape(block: &TraceBlock) -> Result<(), BlockTraceError> {
    if block.active_cycles != block.cycles.len() {
        return Err(BlockTraceError::TraceCycleCountMismatch {
            block_index: block.block_index,
            active_cycles: block.active_cycles,
            actual_cycles: block.cycles.len(),
        });
    }

    if !block.ended_at_tick_boundary {
        return Err(BlockTraceError::BlockDidNotEndAtTickBoundary {
            block_index: block.block_index,
        });
    }

    Ok(())
}

pub fn validate_cpu_r1cs_block<F>(
    bytecode_preprocessing: &BytecodePreprocessing,
    block: &TraceBlock,
    lookahead_cycle: Option<&Cycle>,
) -> Result<(), BlockTraceError>
where
    F: JoltField,
{
    for (row_index, cycle) in block.cycles.iter().enumerate() {
        let next_cycle = block.cycles.get(row_index + 1).or_else(|| lookahead_cycle);
        let row =
            R1CSCycleInputs::from_cycle_with_next::<F>(bytecode_preprocessing, cycle, next_cycle);
        validate_cpu_r1cs_row::<F>(block.block_index, row_index, &row)?;
    }

    Ok(())
}

fn validate_cpu_r1cs_row<F>(
    block_index: usize,
    row_index: usize,
    row: &R1CSCycleInputs,
) -> Result<(), BlockTraceError>
where
    F: JoltField,
{
    let eval = R1CSEval::<F>::from_cycle_inputs(row);
    let az_first = eval.eval_az_first_group();
    let bz_first = eval.eval_bz_first_group();

    ensure_cpu_constraint(
        block_index,
        row_index,
        "first",
        "RamAddrEqZeroIfNotLoadStore",
        az_first.not_load_store,
        bz_first.ram_addr == 0,
    )?;
    ensure_cpu_constraint(
        block_index,
        row_index,
        "first",
        "RamReadEqRamWriteIfLoad",
        az_first.load_a,
        bz_first.ram_read_minus_ram_write.to_i128() == 0,
    )?;
    ensure_cpu_constraint(
        block_index,
        row_index,
        "first",
        "RamReadEqRdWriteIfLoad",
        az_first.load_b,
        bz_first.ram_read_minus_rd_write.to_i128() == 0,
    )?;
    ensure_cpu_constraint(
        block_index,
        row_index,
        "first",
        "Rs2EqRamWriteIfStore",
        az_first.store,
        bz_first.rs2_minus_ram_write.to_i128() == 0,
    )?;
    ensure_cpu_constraint(
        block_index,
        row_index,
        "first",
        "LeftLookupZeroUnlessAddSubMul",
        az_first.add_sub_mul,
        bz_first.left_lookup == 0,
    )?;
    ensure_cpu_constraint(
        block_index,
        row_index,
        "first",
        "LeftLookupEqLeftInputOtherwise",
        az_first.not_add_sub_mul,
        bz_first.left_lookup_minus_left_input.to_i128() == 0,
    )?;
    ensure_cpu_constraint(
        block_index,
        row_index,
        "first",
        "AssertLookupOne",
        az_first.assert_flag,
        bz_first.lookup_output_minus_one.to_i128() == 0,
    )?;
    ensure_cpu_constraint(
        block_index,
        row_index,
        "first",
        "NextUnexpPCEqLookupIfShouldJump",
        az_first.should_jump,
        bz_first.next_unexp_pc_minus_lookup_output.to_i128() == 0,
    )?;
    ensure_cpu_constraint(
        block_index,
        row_index,
        "first",
        "NextPCEqPCPlusOneIfInline",
        az_first.virtual_instr_not_last,
        bz_first.next_pc_minus_pc_plus_one.to_i128() == 0,
    )?;
    ensure_cpu_constraint(
        block_index,
        row_index,
        "first",
        "MustStartSequenceFromBeginning",
        az_first.must_start_sequence,
        !bz_first.one_minus_do_not_update_unexpanded_pc,
    )?;

    let az_second = eval.eval_az_second_group();
    let bz_second = eval.eval_bz_second_group();
    ensure_cpu_constraint(
        block_index,
        row_index,
        "second",
        "RamAddrEqRs1PlusImmIfLoadStore",
        az_second.load_or_store,
        bz_second.ram_addr_minus_rs1_plus_imm == 0,
    )?;
    ensure_cpu_constraint(
        block_index,
        row_index,
        "second",
        "RightLookupEqAddResult",
        az_second.add,
        bz_second.right_lookup_minus_add_result.is_zero(),
    )?;
    ensure_cpu_constraint(
        block_index,
        row_index,
        "second",
        "RightLookupEqSubResult",
        az_second.sub,
        bz_second.right_lookup_minus_sub_result.is_zero(),
    )?;
    ensure_cpu_constraint(
        block_index,
        row_index,
        "second",
        "RightLookupEqProduct",
        az_second.mul,
        bz_second.right_lookup_minus_product.is_zero(),
    )?;
    ensure_cpu_constraint(
        block_index,
        row_index,
        "second",
        "RightLookupEqRightInputOtherwise",
        az_second.not_add_sub_mul_advice,
        bz_second.right_lookup_minus_right_input.is_zero(),
    )?;
    ensure_cpu_constraint(
        block_index,
        row_index,
        "second",
        "RdWriteEqLookupOutput",
        az_second.write_lookup_to_rd,
        bz_second.rd_write_minus_lookup_output.is_zero(),
    )?;
    ensure_cpu_constraint(
        block_index,
        row_index,
        "second",
        "RdWriteEqPCPlusConstIfJump",
        az_second.write_pc_to_rd,
        bz_second.rd_write_minus_pc_plus_const.is_zero(),
    )?;
    ensure_cpu_constraint(
        block_index,
        row_index,
        "second",
        "NextUnexpPCEqPCPlusImmIfBranch",
        az_second.should_branch,
        bz_second.next_unexp_pc_minus_pc_plus_imm == 0,
    )?;
    ensure_cpu_constraint(
        block_index,
        row_index,
        "second",
        "NextUnexpPCEqExpectedOtherwise",
        az_second.not_jump_or_branch,
        bz_second.next_unexp_pc_minus_expected.is_zero(),
    )
}

fn ensure_cpu_constraint(
    block_index: usize,
    row_index: usize,
    group: &'static str,
    constraint: &'static str,
    guard: bool,
    satisfied: bool,
) -> Result<(), BlockTraceError> {
    if guard && !satisfied {
        return Err(BlockTraceError::CpuR1CSConstraintViolation {
            block_index,
            row_index,
            group,
            constraint,
        });
    }

    Ok(())
}

fn r1cs_num_steps_for_block(active_cycles: usize) -> usize {
    active_cycles.next_power_of_two().max(1)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BlockPublicInputError {
    EmptyBlock {
        block_index: usize,
    },
    StartCycleMismatch {
        block_index: usize,
        expected: usize,
        actual: usize,
    },
    EndCycleMismatch {
        block_index: usize,
        expected: usize,
        actual: usize,
    },
    BlockIndexGap {
        current: usize,
        next: usize,
    },
    CycleGap {
        current_block: usize,
        next_block: usize,
        current_end: usize,
        next_start: usize,
    },
    BoundaryStateMismatch {
        current_block: usize,
        next_block: usize,
    },
}

impl fmt::Display for BlockPublicInputError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyBlock { block_index } => {
                write!(f, "block {block_index} has no active cycles")
            }
            Self::StartCycleMismatch {
                block_index,
                expected,
                actual,
            } => write!(
                f,
                "block {block_index} start cycle mismatch: expected {expected}, got {actual}"
            ),
            Self::EndCycleMismatch {
                block_index,
                expected,
                actual,
            } => write!(
                f,
                "block {block_index} end cycle mismatch: expected {expected}, got {actual}"
            ),
            Self::BlockIndexGap { current, next } => {
                write!(f, "block index gap: current {current}, next {next}")
            }
            Self::CycleGap {
                current_block,
                next_block,
                current_end,
                next_start,
            } => write!(
                f,
                "cycle gap between block {current_block} and {next_block}: current ends at {current_end}, next starts at {next_start}"
            ),
            Self::BoundaryStateMismatch {
                current_block,
                next_block,
            } => write!(
                f,
                "boundary state mismatch between block {current_block} and block {next_block}"
            ),
        }
    }
}

impl Error for BlockPublicInputError {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BlockTraceError {
    PublicInput(BlockPublicInputError),
    TraceCycleCountMismatch {
        block_index: usize,
        active_cycles: usize,
        actual_cycles: usize,
    },
    ProofCycleCountMismatch {
        block_index: usize,
        public_input_cycles: usize,
        proof_cycles: usize,
    },
    BlockDidNotEndAtTickBoundary {
        block_index: usize,
    },
    CpuR1CSRowsCheckedMismatch {
        block_index: usize,
        expected: usize,
        actual: usize,
    },
    CpuR1CSNumStepsMismatch {
        block_index: usize,
        expected: usize,
        actual: usize,
    },
    CpuR1CSShapeDigestMismatch {
        block_index: usize,
    },
    CpuPublicInputMismatch {
        block_index: usize,
    },
    CpuLookaheadMismatch {
        block_index: usize,
        proof_used_lookahead: bool,
        actual_used_lookahead: bool,
    },
    CpuLookaheadDigestMismatch {
        block_index: usize,
    },
    CpuLookaheadSerializationFailed {
        block_index: usize,
    },
    CpuR1CSConstraintViolation {
        block_index: usize,
        row_index: usize,
        group: &'static str,
        constraint: &'static str,
    },
    BlockIOClaimShapeMismatch {
        block_index: usize,
        reason: &'static str,
    },
    BlockIOClaimMismatch {
        block_index: usize,
    },
    BlockIOClaimChainLengthMismatch {
        blocks: usize,
        claims: usize,
    },
    RegisterReadMismatch {
        block_index: usize,
        row_index: usize,
        register_index: u8,
        expected: u64,
        actual: u64,
    },
    RegisterWritePreValueMismatch {
        block_index: usize,
        row_index: usize,
        register_index: u8,
        expected: u64,
        actual: u64,
    },
    RegisterBoundaryMismatch {
        block_index: usize,
        register_index: u8,
        expected: u64,
        actual: u64,
    },
    RamReadMismatch {
        block_index: usize,
        row_index: usize,
        address: u64,
        expected: u64,
        actual: u64,
    },
    RamWritePreValueMismatch {
        block_index: usize,
        row_index: usize,
        address: u64,
        expected: u64,
        actual: u64,
    },
    BlockRegisterClaimShapeMismatch {
        block_index: usize,
        reason: &'static str,
    },
    BlockRegisterClaimMismatch {
        block_index: usize,
    },
    BlockRegisterClaimChainLengthMismatch {
        blocks: usize,
        io_claims: usize,
        register_claims: usize,
    },
    BlockRegisterClaimBoundaryMismatch {
        current_block: usize,
        next_block: usize,
    },
    BlockRamClaimShapeMismatch {
        block_index: usize,
        reason: &'static str,
    },
    BlockRamClaimMismatch {
        block_index: usize,
    },
    BlockRamClaimChainLengthMismatch {
        blocks: usize,
        io_claims: usize,
        ram_claims: usize,
    },
    BlockRamClaimContinuityMismatch {
        block_index: usize,
        address: u64,
        expected: u64,
        actual: u64,
    },
    BlockLookupClaimShapeMismatch {
        block_index: usize,
        reason: &'static str,
    },
    BlockLookupClaimMismatch {
        block_index: usize,
    },
    BlockLookupLogUpProofMismatch {
        block_index: usize,
    },
    BlockLookupClaimChainLengthMismatch {
        blocks: usize,
        io_claims: usize,
        lookup_claims: usize,
    },
    BlockProofBundleChainLengthMismatch {
        blocks: usize,
        bundles: usize,
    },
    BlockFoldInputMismatch {
        block_index: usize,
    },
    BlockFoldInputChainLengthMismatch {
        blocks: usize,
        bundles: usize,
        fold_inputs: usize,
    },
    BlockFoldInputBoundaryMismatch {
        current_block: usize,
        next_block: usize,
        reason: &'static str,
    },
    BlockFoldAccumulatorAbsorbMismatch {
        block_index: usize,
        reason: &'static str,
    },
    BlockFoldAccumulatorProgramDigestMismatch {
        block_index: usize,
    },
    BlockFoldAccumulatorBoundaryMismatch {
        current_block: usize,
        next_block: usize,
        reason: &'static str,
    },
    BlockFoldAccumulatorMismatch {
        expected_blocks: usize,
        actual_blocks: usize,
    },
    NovaFoldingBackendUnavailable {
        block_index: usize,
        reason: &'static str,
    },
    NovaFoldingBackendError {
        block_index: usize,
        reason: &'static str,
    },
}

impl From<BlockPublicInputError> for BlockTraceError {
    fn from(error: BlockPublicInputError) -> Self {
        Self::PublicInput(error)
    }
}

impl fmt::Display for BlockTraceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PublicInput(error) => error.fmt(f),
            Self::TraceCycleCountMismatch {
                block_index,
                active_cycles,
                actual_cycles,
            } => write!(
                f,
                "trace block {block_index} cycle count mismatch: active_cycles={active_cycles}, actual_cycles={actual_cycles}"
            ),
            Self::ProofCycleCountMismatch {
                block_index,
                public_input_cycles,
                proof_cycles,
            } => write!(
                f,
                "block proof {block_index} cycle count mismatch: public_input={public_input_cycles}, proof={proof_cycles}"
            ),
            Self::BlockDidNotEndAtTickBoundary { block_index } => {
                write!(f, "trace block {block_index} did not end at a tick boundary")
            }
            Self::CpuR1CSRowsCheckedMismatch {
                block_index,
                expected,
                actual,
            } => write!(
                f,
                "CPU/R1CS proof for block {block_index} checked {actual} rows, expected {expected}"
            ),
            Self::CpuR1CSNumStepsMismatch {
                block_index,
                expected,
                actual,
            } => write!(
                f,
                "CPU/R1CS proof for block {block_index} uses {actual} padded steps, expected {expected}"
            ),
            Self::CpuR1CSShapeDigestMismatch { block_index } => write!(
                f,
                "CPU/R1CS proof for block {block_index} has an unexpected R1CS shape digest"
            ),
            Self::CpuPublicInputMismatch { block_index } => write!(
                f,
                "CPU/R1CS proof public input does not match trace block {block_index}"
            ),
            Self::CpuLookaheadMismatch {
                block_index,
                proof_used_lookahead,
                actual_used_lookahead,
            } => write!(
                f,
                "CPU/R1CS proof lookahead mismatch for block {block_index}: proof={proof_used_lookahead}, actual={actual_used_lookahead}"
            ),
            Self::CpuLookaheadDigestMismatch { block_index } => write!(
                f,
                "CPU/R1CS proof lookahead digest does not match the cycle supplied for block {block_index}"
            ),
            Self::CpuLookaheadSerializationFailed { block_index } => write!(
                f,
                "failed to serialize the CPU/R1CS lookahead cycle for block {block_index}"
            ),
            Self::CpuR1CSConstraintViolation {
                block_index,
                row_index,
                group,
                constraint,
            } => write!(
                f,
                "CPU/R1CS {group}-group constraint {constraint} failed at block {block_index}, row {row_index}"
            ),
            Self::BlockIOClaimShapeMismatch {
                block_index,
                reason,
            } => write!(
                f,
                "block IO claim shape mismatch for block {block_index}: {reason}"
            ),
            Self::BlockIOClaimMismatch { block_index } => {
                write!(f, "block IO claims do not match trace block {block_index}")
            }
            Self::BlockIOClaimChainLengthMismatch { blocks, claims } => write!(
                f,
                "block IO claim chain length mismatch: {blocks} blocks, {claims} claims"
            ),
            Self::RegisterReadMismatch {
                block_index,
                row_index,
                register_index,
                expected,
                actual,
            } => write!(
                f,
                "register read mismatch at block {block_index}, row {row_index}, x{register_index}: expected {expected}, got {actual}"
            ),
            Self::RegisterWritePreValueMismatch {
                block_index,
                row_index,
                register_index,
                expected,
                actual,
            } => write!(
                f,
                "register write pre-value mismatch at block {block_index}, row {row_index}, x{register_index}: expected {expected}, got {actual}"
            ),
            Self::RegisterBoundaryMismatch {
                block_index,
                register_index,
                expected,
                actual,
            } => write!(
                f,
                "register boundary mismatch at block {block_index}, x{register_index}: expected end value {expected}, got {actual}"
            ),
            Self::RamReadMismatch {
                block_index,
                row_index,
                address,
                expected,
                actual,
            } => write!(
                f,
                "RAM read mismatch at block {block_index}, row {row_index}, address {address}: expected {expected}, got {actual}"
            ),
            Self::RamWritePreValueMismatch {
                block_index,
                row_index,
                address,
                expected,
                actual,
            } => write!(
                f,
                "RAM write pre-value mismatch at block {block_index}, row {row_index}, address {address}: expected {expected}, got {actual}"
            ),
            Self::BlockRegisterClaimShapeMismatch {
                block_index,
                reason,
            } => write!(
                f,
                "block register claim shape mismatch for block {block_index}: {reason}"
            ),
            Self::BlockRegisterClaimMismatch { block_index } => write!(
                f,
                "block register accumulator claim does not match block {block_index}"
            ),
            Self::BlockRegisterClaimChainLengthMismatch {
                blocks,
                io_claims,
                register_claims,
            } => write!(
                f,
                "block register claim chain length mismatch: {blocks} blocks, {io_claims} IO claims, {register_claims} register claims"
            ),
            Self::BlockRegisterClaimBoundaryMismatch {
                current_block,
                next_block,
            } => write!(
                f,
                "register accumulator boundary mismatch between block {current_block} and block {next_block}"
            ),
            Self::BlockRamClaimShapeMismatch {
                block_index,
                reason,
            } => write!(
                f,
                "block RAM claim shape mismatch for block {block_index}: {reason}"
            ),
            Self::BlockRamClaimMismatch { block_index } => write!(
                f,
                "block RAM accumulator claim does not match block {block_index}"
            ),
            Self::BlockRamClaimChainLengthMismatch {
                blocks,
                io_claims,
                ram_claims,
            } => write!(
                f,
                "block RAM claim chain length mismatch: {blocks} blocks, {io_claims} IO claims, {ram_claims} RAM claims"
            ),
            Self::BlockRamClaimContinuityMismatch {
                block_index,
                address,
                expected,
                actual,
            } => write!(
                f,
                "RAM accumulator continuity mismatch at block {block_index}, address {address}: expected first value {expected}, got {actual}"
            ),
            Self::BlockLookupClaimShapeMismatch {
                block_index,
                reason,
            } => write!(
                f,
                "block lookup claim shape mismatch for block {block_index}: {reason}"
            ),
            Self::BlockLookupClaimMismatch { block_index } => write!(
                f,
                "block lookup accumulator claim does not match block {block_index}"
            ),
            Self::BlockLookupLogUpProofMismatch { block_index } => write!(
                f,
                "block LogUp proof does not match lookup claims for block {block_index}"
            ),
            Self::BlockLookupClaimChainLengthMismatch {
                blocks,
                io_claims,
                lookup_claims,
            } => write!(
                f,
                "block lookup claim chain length mismatch: {blocks} blocks, {io_claims} IO claims, {lookup_claims} lookup claims"
            ),
            Self::BlockProofBundleChainLengthMismatch { blocks, bundles } => write!(
                f,
                "block proof bundle chain length mismatch: {blocks} blocks, {bundles} bundles"
            ),
            Self::BlockFoldInputMismatch { block_index } => write!(
                f,
                "block fold input does not match block proof bundle {block_index}"
            ),
            Self::BlockFoldInputChainLengthMismatch {
                blocks,
                bundles,
                fold_inputs,
            } => write!(
                f,
                "block fold input chain length mismatch: {blocks} blocks, {bundles} bundles, {fold_inputs} fold inputs"
            ),
            Self::BlockFoldInputBoundaryMismatch {
                current_block,
                next_block,
                reason,
            } => write!(
                f,
                "block fold input boundary mismatch between block {current_block} and block {next_block}: {reason}"
            ),
            Self::BlockFoldAccumulatorAbsorbMismatch {
                block_index,
                reason,
            } => write!(
                f,
                "block fold accumulator cannot absorb block {block_index}: {reason}"
            ),
            Self::BlockFoldAccumulatorProgramDigestMismatch { block_index } => write!(
                f,
                "block fold accumulator program digest mismatch at block {block_index}"
            ),
            Self::BlockFoldAccumulatorBoundaryMismatch {
                current_block,
                next_block,
                reason,
            } => write!(
                f,
                "block fold accumulator boundary mismatch between block {current_block} and block {next_block}: {reason}"
            ),
            Self::BlockFoldAccumulatorMismatch {
                expected_blocks,
                actual_blocks,
            } => write!(
                f,
                "block fold accumulator mismatch: expected {expected_blocks} absorbed blocks, got {actual_blocks}"
            ),
            Self::NovaFoldingBackendUnavailable {
                block_index,
                reason,
            } => write!(
                f,
                "Nova folding backend is unavailable at block {block_index}: {reason}"
            ),
            Self::NovaFoldingBackendError {
                block_index,
                reason,
            } => write!(
                f,
                "Nova folding backend failed at block {block_index}: {reason}"
            ),
        }
    }
}

impl Error for BlockTraceError {}

pub fn validate_block_chain<Digest>(
    blocks: &[BlockPublicInput<Digest>],
) -> Result<(), BlockPublicInputError> {
    for block in blocks {
        block.validate_shape()?;
    }

    for window in blocks.windows(2) {
        window[0].validate_contiguous_with(&window[1])?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use common::constants::REGISTER_COUNT;
    use tracer::instruction::Cycle;

    fn boundary(global_cycle: usize, pc: u64) -> MachineBoundaryState {
        let mut registers = [0i64; REGISTER_COUNT as usize];
        registers[1] = pc as i64;

        MachineBoundaryState {
            global_cycle,
            emulator_trace_len: global_cycle,
            pc,
            registers,
            terminated: false,
        }
    }

    fn input(
        block_index: usize,
        start: MachineBoundaryState,
        end: MachineBoundaryState,
    ) -> BlockPublicInput {
        BlockPublicInput {
            program_digest: [7u8; 32],
            block_index,
            target_size: end.global_cycle - start.global_cycle,
            global_cycle_start: start.global_cycle,
            active_cycles: end.global_cycle - start.global_cycle,
            start_state: start,
            end_state: end,
        }
    }

    fn trace_block(
        block_index: usize,
        start: MachineBoundaryState,
        end: MachineBoundaryState,
    ) -> TraceBlock {
        let active_cycles = end.global_cycle - start.global_cycle;
        TraceBlock {
            block_index,
            global_cycle_start: start.global_cycle,
            active_cycles,
            target_size: active_cycles,
            start_state: start,
            end_state: end,
            cycles: vec![Cycle::NoOp; active_cycles],
            ended_at_tick_boundary: true,
        }
    }

    fn repeated_digest(byte: u8) -> [u8; 32] {
        [byte; 32]
    }

    fn temp_artifact_path(test_name: &str, file_name: &str) -> std::path::PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let mut path = std::env::temp_dir();
        path.push(format!(
            "jolt-nova-{test_name}-{}-{nonce}",
            std::process::id()
        ));
        path.push(file_name);
        path
    }

    fn sample_final_proof_size_baseline(
        configured_backend_name: &'static str,
        proof_system: &'static str,
        proof_payload_bytes_len: usize,
        proof_total_bytes_len: usize,
        digest_byte: u8,
    ) -> JoltNovaFinalProofSizeBaseline {
        JoltNovaFinalProofSizeBaseline {
            configured_backend_name,
            proof_system,
            absorbed_blocks: 2,
            total_active_cycles: 4,
            recursive_snark_bytes_len: Some(128),
            final_public_input_bytes_len: 64,
            final_witness_bytes_len: 96,
            proof_envelope_bytes_len: proof_total_bytes_len - proof_payload_bytes_len,
            proof_payload_bytes_len,
            proof_total_bytes_len,
            final_instance_digest: repeated_digest(digest_byte),
            spartan_encoding_digest: repeated_digest(digest_byte + 1),
            proof_digest: repeated_digest(digest_byte + 2),
        }
    }

    fn sample_final_proof_size_scaling_report() -> NovaBlockProofPipelineFinalProofSizeScalingReport
    {
        let placeholder = sample_final_proof_size_baseline(
            SPARTAN_PLACEHOLDER_PROOF_SYSTEM_NAME,
            SPARTAN_PLACEHOLDER_PROOF_SYSTEM_NAME,
            0,
            80,
            3,
        );
        let spartan = sample_final_proof_size_baseline(
            SPARTAN_FINAL_PROOF_SYSTEM_NAME,
            SPARTAN_FINAL_PROOF_SYSTEM_NAME,
            512,
            600,
            6,
        );

        NovaBlockProofPipelineFinalProofSizeScalingReport {
            rows: vec![NovaBlockProofPipelineFinalProofSizeScalingRow {
                block_count: 2,
                first_block_index: Some(0),
                last_block_index: Some(1),
                total_active_cycles: 4,
                recursive_snark_bytes_len: Some(128),
                final_proof_size_comparison: JoltNovaFinalProofSizeComparison {
                    folded_accumulator_digest: repeated_digest(1),
                    absorbed_blocks: 2,
                    total_active_cycles: 4,
                    recursive_snark_bytes_len: Some(128),
                    placeholder,
                    spartan,
                    spartan_payload_extra_bytes: 512,
                    spartan_total_extra_bytes: 520,
                },
            }],
        }
    }

    #[test]
    fn jolt_nova_report_output_format_uses_json_as_canonical_schema_format() {
        assert_eq!(JOLT_NOVA_REPORT_SCHEMA_VERSION, "jolt-nova-report-v1");
        assert_eq!(
            JOLT_NOVA_REPORT_CANONICAL_OUTPUT_FORMAT,
            JoltNovaReportOutputFormat::Json
        );
        assert!(JoltNovaReportOutputFormat::Json.is_canonical());
        assert!(!JoltNovaReportOutputFormat::Csv.is_canonical());
        assert_eq!(JoltNovaReportOutputFormat::Json.as_str(), "json");
        assert_eq!(JoltNovaReportOutputFormat::Csv.as_str(), "csv");
        assert_eq!(JoltNovaReportOutputFormat::Json.to_string(), "json");
        assert_eq!(
            JoltNovaReportOutputFormat::parse("json"),
            Some(JoltNovaReportOutputFormat::Json)
        );
        assert_eq!(
            JoltNovaReportOutputFormat::parse("csv"),
            Some(JoltNovaReportOutputFormat::Csv)
        );
        assert_eq!(JoltNovaReportOutputFormat::parse("debug"), None);
    }

    #[test]
    fn jolt_nova_final_proof_size_scaling_report_exports_stable_json() {
        let report = sample_final_proof_size_scaling_report();
        let json =
            export_nova_final_proof_size_scaling_report(&report, JoltNovaReportOutputFormat::Json)
                .unwrap();

        assert_eq!(
            json,
            export_nova_final_proof_size_scaling_report_json(&report)
        );
        assert!(json.starts_with(
            "{\"schema_version\":\"jolt-nova-report-v1\",\"format\":\"json\",\"report_kind\":\"final-proof-size-scaling\",\"row_count\":1,\"rows\":[{\"block_count\":2"
        ));
        assert!(json.contains("\"first_block_index\":0"));
        assert!(json.contains("\"last_block_index\":1"));
        assert!(json.contains("\"recursive_snark_bytes_len\":128"));
        assert!(json.contains(
            "\"folded_accumulator_digest\":\"0101010101010101010101010101010101010101010101010101010101010101\""
        ));
        assert!(json.contains("\"configured_backend_name\":\"spartan-placeholder\""));
        assert!(json.contains("\"configured_backend_name\":\"spartan-final-proof\""));
        assert!(json.contains(
            "\"final_instance_digest\":\"0303030303030303030303030303030303030303030303030303030303030303\""
        ));
        assert!(json.contains(
            "\"spartan_encoding_digest\":\"0707070707070707070707070707070707070707070707070707070707070707\""
        ));
        assert!(json.contains("\"spartan_payload_extra_bytes\":512"));
        assert!(json.contains("\"spartan_total_extra_bytes\":520"));
    }

    #[test]
    fn jolt_nova_final_proof_size_scaling_report_rejects_csv_until_stage8_csv_export() {
        let report = sample_final_proof_size_scaling_report();

        assert_eq!(
            export_nova_final_proof_size_scaling_report(&report, JoltNovaReportOutputFormat::Csv)
                .unwrap_err(),
            BlockTraceError::NovaFoldingBackendError {
                block_index: 0,
                reason: "CSV report export is not implemented yet",
            }
        );
    }

    #[test]
    fn jolt_nova_final_proof_size_benchmark_artifact_writes_serialized_report() {
        let report = sample_final_proof_size_scaling_report();
        let serialized_report = export_nova_final_proof_size_scaling_report_json(&report);
        let artifact = NovaBlockProofPipelineFinalProofSizeBenchmarkArtifact {
            output_format: JoltNovaReportOutputFormat::Json,
            report,
            serialized_report: serialized_report.clone(),
        };
        let path = temp_artifact_path("final-proof-size-benchmark-artifact-writes", "report.json");

        artifact.write_to_path(&path).unwrap();
        let written_report = std::fs::read_to_string(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        std::fs::remove_dir(path.parent().unwrap()).unwrap();

        assert_eq!(artifact.file_extension(), "json");
        assert_eq!(artifact.serialized_bytes(), serialized_report.as_bytes());
        assert_eq!(written_report, serialized_report);
    }

    #[test]
    fn validates_contiguous_block_chain() {
        let b0 = input(0, boundary(0, 100), boundary(4, 104));
        let b1 = input(1, b0.end_state.clone(), boundary(8, 108));

        validate_block_chain(&[b0, b1]).unwrap();
    }

    #[test]
    fn rejects_boundary_state_mismatch() {
        let b0 = input(0, boundary(0, 100), boundary(4, 104));
        let b1 = input(1, boundary(4, 999), boundary(8, 108));

        assert_eq!(
            validate_block_chain(&[b0, b1]).unwrap_err(),
            BlockPublicInputError::BoundaryStateMismatch {
                current_block: 0,
                next_block: 1,
            }
        );
    }

    #[test]
    fn rejects_end_cycle_mismatch() {
        let mut block = input(0, boundary(0, 100), boundary(4, 104));
        block.active_cycles = 3;

        assert_eq!(
            block.validate_shape().unwrap_err(),
            BlockPublicInputError::EndCycleMismatch {
                block_index: 0,
                expected: 3,
                actual: 4,
            }
        );
    }

    #[test]
    fn block_trace_prover_emits_and_verifies_placeholder_proof() {
        let block = trace_block(0, boundary(0, 100), boundary(4, 104));
        let prover = BlockTraceProver::new([9u8; 32]);

        let proof = prover.prove_block(&block).unwrap();

        assert_eq!(proof.public_input.block_index, 0);
        assert_eq!(proof.public_input.active_cycles, 4);
        assert_eq!(proof.inner_proof.cycle_count, 4);
        verify_placeholder_block_proof(&proof).unwrap();
    }

    #[test]
    fn block_trace_prover_validates_placeholder_proof_chain() {
        let block0 = trace_block(0, boundary(0, 100), boundary(4, 104));
        let block1 = trace_block(1, block0.end_state.clone(), boundary(8, 108));
        let prover = BlockTraceProver::new([9u8; 32]);

        let proofs = prover.prove_blocks([&block0, &block1]).unwrap();

        verify_placeholder_block_proof_chain(&proofs).unwrap();
    }

    #[test]
    fn block_trace_prover_rejects_trace_cycle_count_mismatch() {
        let mut block = trace_block(0, boundary(0, 100), boundary(4, 104));
        block.active_cycles = 5;
        let prover = BlockTraceProver::new([9u8; 32]);

        assert_eq!(
            prover.prove_block(&block).unwrap_err(),
            BlockTraceError::TraceCycleCountMismatch {
                block_index: 0,
                active_cycles: 5,
                actual_cycles: 4,
            }
        );
    }

    #[test]
    fn placeholder_verifier_rejects_proof_cycle_count_mismatch() {
        let block = trace_block(0, boundary(0, 100), boundary(4, 104));
        let prover = BlockTraceProver::new([9u8; 32]);
        let mut proof = prover.prove_block(&block).unwrap();
        proof.inner_proof.cycle_count = 3;

        assert_eq!(
            verify_placeholder_block_proof(&proof).unwrap_err(),
            BlockTraceError::ProofCycleCountMismatch {
                block_index: 0,
                public_input_cycles: 4,
                proof_cycles: 3,
            }
        );
    }

    #[test]
    fn block_trace_prover_rejects_non_tick_boundary_block() {
        let mut block = trace_block(0, boundary(0, 100), boundary(4, 104));
        block.ended_at_tick_boundary = false;
        let prover = BlockTraceProver::new([9u8; 32]);

        assert_eq!(
            prover.prove_block(&block).unwrap_err(),
            BlockTraceError::BlockDidNotEndAtTickBoundary { block_index: 0 }
        );
    }

    #[test]
    fn block_cpu_prover_emits_and_verifies_noop_r1cs_proof() {
        let bytecode = BytecodePreprocessing::default();
        let block = trace_block(0, boundary(0, 0), boundary(4, 0));
        let prover = BlockCpuProver::<_, ark_bn254::Fr>::new([9u8; 32]);

        let proof = prover.prove_block(&bytecode, &block, None).unwrap();

        assert_eq!(proof.public_input.block_index, 0);
        assert_eq!(proof.inner_proof.cycle_count, 4);
        assert_eq!(proof.inner_proof.r1cs_rows_checked, 4);
        assert_eq!(proof.inner_proof.r1cs_num_steps, 4);
        assert!(!proof.inner_proof.used_lookahead_cycle);
        verify_cpu_block_proof(&proof).unwrap();
        verify_cpu_block_witness(&bytecode, &block, None, &proof).unwrap();
    }

    #[test]
    fn block_cpu_prover_validates_chain_with_lookahead() {
        let bytecode = BytecodePreprocessing::default();
        let block0 = trace_block(0, boundary(0, 0), boundary(2, 0));
        let block1 = trace_block(1, block0.end_state.clone(), boundary(4, 0));
        let prover = BlockCpuProver::<_, ark_bn254::Fr>::new([9u8; 32]);

        let proofs = prover
            .prove_blocks(&bytecode, &[block0.clone(), block1.clone()])
            .unwrap();

        assert_eq!(proofs.len(), 2);
        assert!(proofs[0].inner_proof.used_lookahead_cycle);
        assert!(proofs[0].inner_proof.lookahead_cycle_digest.is_some());
        assert!(!proofs[1].inner_proof.used_lookahead_cycle);
        assert!(proofs[1].inner_proof.lookahead_cycle_digest.is_none());
        verify_cpu_block_witness(&bytecode, &block0, block1.cycles.first(), &proofs[0]).unwrap();
        verify_cpu_block_witness(&bytecode, &block1, None, &proofs[1]).unwrap();
    }

    #[test]
    fn block_cpu_prefix_proof_publicly_binds_external_lookahead() {
        let bytecode = BytecodePreprocessing::default();
        let block0 = trace_block(0, boundary(0, 0), boundary(2, 0));
        let block1 = trace_block(1, block0.end_state.clone(), boundary(4, 0));
        let external_lookahead = block1.cycles.first().unwrap();
        let prover = BlockCpuProver::<_, ark_bn254::Fr>::new([9u8; 32]);

        let proofs = prover
            .prove_blocks_with_external_lookahead(
                &bytecode,
                std::slice::from_ref(&block0),
                Some(external_lookahead),
            )
            .unwrap();

        assert!(proofs[0].inner_proof.used_lookahead_cycle);
        assert_eq!(
            proofs[0].inner_proof.lookahead_cycle_digest,
            digest_cpu_lookahead_cycle(block0.block_index, Some(external_lookahead)).unwrap()
        );
        verify_cpu_block_witness(&bytecode, &block0, Some(external_lookahead), &proofs[0]).unwrap();
        assert!(matches!(
            verify_cpu_block_witness(&bytecode, &block0, None, &proofs[0]),
            Err(BlockTraceError::CpuLookaheadMismatch { .. })
        ));

        let mut tampered = proofs[0].clone();
        tampered
            .inner_proof
            .lookahead_cycle_digest
            .as_mut()
            .unwrap()[0] ^= 1;
        assert_eq!(
            verify_cpu_block_witness(&bytecode, &block0, Some(external_lookahead), &tampered,)
                .unwrap_err(),
            BlockTraceError::CpuLookaheadDigestMismatch { block_index: 0 }
        );
    }

    #[test]
    fn block_cpu_prover_uses_noop_lookahead_for_terminal_block() {
        let bytecode = BytecodePreprocessing::default();
        let mut block = trace_block(0, boundary(0, 0), boundary(2, 0));
        block.end_state.terminated = true;
        let prover = BlockCpuProver::<_, ark_bn254::Fr>::new([9u8; 32]);

        let proofs = prover.prove_blocks(&bytecode, &[block.clone()]).unwrap();

        assert!(proofs[0].inner_proof.used_lookahead_cycle);
        verify_cpu_block_witness(
            &bytecode,
            &block,
            Some(&TERMINAL_LOOKAHEAD_CYCLE),
            &proofs[0],
        )
        .unwrap();
    }

    #[test]
    fn cpu_verifier_rejects_rows_checked_mismatch() {
        let bytecode = BytecodePreprocessing::default();
        let block = trace_block(0, boundary(0, 0), boundary(4, 0));
        let prover = BlockCpuProver::<_, ark_bn254::Fr>::new([9u8; 32]);
        let mut proof = prover.prove_block(&bytecode, &block, None).unwrap();
        proof.inner_proof.r1cs_rows_checked = 3;

        assert_eq!(
            verify_cpu_block_proof(&proof).unwrap_err(),
            BlockTraceError::CpuR1CSRowsCheckedMismatch {
                block_index: 0,
                expected: 4,
                actual: 3,
            }
        );
    }

    #[test]
    fn cpu_witness_verifier_rejects_lookahead_mismatch() {
        let bytecode = BytecodePreprocessing::default();
        let block = trace_block(0, boundary(0, 0), boundary(2, 0));
        let next = trace_block(1, block.end_state.clone(), boundary(4, 0));
        let prover = BlockCpuProver::<_, ark_bn254::Fr>::new([9u8; 32]);
        let proof = prover
            .prove_block(&bytecode, &block, next.cycles.first())
            .unwrap();

        assert_eq!(
            verify_cpu_block_witness(&bytecode, &block, None, &proof).unwrap_err(),
            BlockTraceError::CpuLookaheadMismatch {
                block_index: 0,
                proof_used_lookahead: true,
                actual_used_lookahead: false,
            }
        );
    }

    #[test]
    fn block_io_claims_extract_noop_block() {
        let block = trace_block(0, boundary(0, 7), boundary(3, 7));

        let claims = extract_block_io_claims(&block).unwrap();

        assert_eq!(claims.block_index, 0);
        assert_eq!(claims.active_cycles, 3);
        assert!(claims.register_reads.is_empty());
        assert!(claims.register_writes.is_empty());
        assert!(claims.ram_accesses.is_empty());
        assert_eq!(claims.lookup_claims.len(), 3);
        assert!(claims.lookup_claims.iter().all(|claim| {
            claim.left_instruction_input == 0
                && claim.right_instruction_input == 0
                && claim.left_lookup_operand == 0
                && claim.right_lookup_operand == 0
                && claim.lookup_index == 0
                && claim.lookup_output == 0
        }));
        verify_block_io_claims(&block, &claims).unwrap();
    }

    #[test]
    fn block_io_claim_chain_validates_contiguous_blocks() {
        let block0 = trace_block(0, boundary(0, 7), boundary(2, 7));
        let block1 = trace_block(1, block0.end_state.clone(), boundary(4, 7));
        let claims0 = extract_block_io_claims(&block0).unwrap();
        let claims1 = extract_block_io_claims(&block1).unwrap();

        verify_block_io_claim_chain(&[block0, block1], &[claims0, claims1]).unwrap();
    }

    #[test]
    fn block_io_claims_reject_register_boundary_mismatch() {
        let block = trace_block(0, boundary(0, 7), boundary(3, 8));

        assert_eq!(
            extract_block_io_claims(&block).unwrap_err(),
            BlockTraceError::RegisterBoundaryMismatch {
                block_index: 0,
                register_index: 1,
                expected: 7,
                actual: 8,
            }
        );
    }

    #[test]
    fn block_io_claim_verifier_rejects_tampered_lookup_claim() {
        let block = trace_block(0, boundary(0, 7), boundary(3, 7));
        let mut claims = extract_block_io_claims(&block).unwrap();
        claims.lookup_claims[0].lookup_output = 1;

        assert_eq!(
            verify_block_io_claims(&block, &claims).unwrap_err(),
            BlockTraceError::BlockIOClaimMismatch { block_index: 0 }
        );
    }

    #[test]
    fn block_io_claim_chain_rejects_length_mismatch() {
        let block = trace_block(0, boundary(0, 7), boundary(3, 7));

        assert_eq!(
            verify_block_io_claim_chain(&[block], &[]).unwrap_err(),
            BlockTraceError::BlockIOClaimChainLengthMismatch {
                blocks: 1,
                claims: 0,
            }
        );
    }

    #[test]
    fn block_register_claim_builds_noop_accumulator() {
        let block = trace_block(0, boundary(0, 7), boundary(3, 7));
        let io_claims = extract_block_io_claims(&block).unwrap();

        let register_claim = build_block_register_claim(&block, &io_claims).unwrap();

        assert_eq!(register_claim.block_index, 0);
        assert_eq!(register_claim.active_cycles, 3);
        assert_eq!(register_claim.read_count, 0);
        assert_eq!(register_claim.write_count, 0);
        assert_eq!(
            register_claim.start_register_digest,
            register_claim.end_register_digest
        );
        verify_block_register_claim(&block, &io_claims, &register_claim).unwrap();
    }

    #[test]
    fn block_register_claim_chain_validates_contiguous_blocks() {
        let block0 = trace_block(0, boundary(0, 7), boundary(2, 7));
        let block1 = trace_block(1, block0.end_state.clone(), boundary(4, 7));
        let io0 = extract_block_io_claims(&block0).unwrap();
        let io1 = extract_block_io_claims(&block1).unwrap();
        let reg0 = build_block_register_claim(&block0, &io0).unwrap();
        let reg1 = build_block_register_claim(&block1, &io1).unwrap();

        verify_block_register_claim_chain(&[block0, block1], &[io0, io1], &[reg0, reg1]).unwrap();
    }

    #[test]
    fn block_register_claim_verifier_rejects_tampered_digest() {
        let block = trace_block(0, boundary(0, 7), boundary(3, 7));
        let io_claims = extract_block_io_claims(&block).unwrap();
        let mut register_claim = build_block_register_claim(&block, &io_claims).unwrap();
        register_claim.reads_digest[0] ^= 1;

        assert_eq!(
            verify_block_register_claim(&block, &io_claims, &register_claim).unwrap_err(),
            BlockTraceError::BlockRegisterClaimMismatch { block_index: 0 }
        );
    }

    #[test]
    fn block_register_claim_verifier_rejects_count_mismatch() {
        let block = trace_block(0, boundary(0, 7), boundary(3, 7));
        let io_claims = extract_block_io_claims(&block).unwrap();
        let mut register_claim = build_block_register_claim(&block, &io_claims).unwrap();
        register_claim.read_count = 1;

        assert_eq!(
            verify_block_register_claim(&block, &io_claims, &register_claim).unwrap_err(),
            BlockTraceError::BlockRegisterClaimShapeMismatch {
                block_index: 0,
                reason: "register read count mismatch",
            }
        );
    }

    #[test]
    fn block_register_claim_chain_rejects_length_mismatch() {
        let block = trace_block(0, boundary(0, 7), boundary(3, 7));
        let io_claims = extract_block_io_claims(&block).unwrap();

        assert_eq!(
            verify_block_register_claim_chain(&[block], &[io_claims], &[]).unwrap_err(),
            BlockTraceError::BlockRegisterClaimChainLengthMismatch {
                blocks: 1,
                io_claims: 1,
                register_claims: 0,
            }
        );
    }

    #[test]
    fn block_ram_claim_builds_noop_accumulator() {
        let block = trace_block(0, boundary(0, 7), boundary(3, 7));
        let io_claims = extract_block_io_claims(&block).unwrap();

        let ram_claim = build_block_ram_claim(&block, &io_claims).unwrap();

        assert_eq!(ram_claim.block_index, 0);
        assert_eq!(ram_claim.active_cycles, 3);
        assert_eq!(ram_claim.access_count, 0);
        assert_eq!(ram_claim.touched_address_count, 0);
        verify_block_ram_claim(&block, &io_claims, &ram_claim).unwrap();
    }

    #[test]
    fn block_ram_claim_chain_validates_contiguous_blocks() {
        let block0 = trace_block(0, boundary(0, 7), boundary(2, 7));
        let block1 = trace_block(1, block0.end_state.clone(), boundary(4, 7));
        let io0 = extract_block_io_claims(&block0).unwrap();
        let io1 = extract_block_io_claims(&block1).unwrap();
        let ram0 = build_block_ram_claim(&block0, &io0).unwrap();
        let ram1 = build_block_ram_claim(&block1, &io1).unwrap();

        verify_block_ram_claim_chain(&[block0, block1], &[io0, io1], &[ram0, ram1]).unwrap();
    }

    #[test]
    fn block_ram_claim_verifier_rejects_tampered_digest() {
        let block = trace_block(0, boundary(0, 7), boundary(3, 7));
        let io_claims = extract_block_io_claims(&block).unwrap();
        let mut ram_claim = build_block_ram_claim(&block, &io_claims).unwrap();
        ram_claim.accesses_digest[0] ^= 1;

        assert_eq!(
            verify_block_ram_claim(&block, &io_claims, &ram_claim).unwrap_err(),
            BlockTraceError::BlockRamClaimMismatch { block_index: 0 }
        );
    }

    #[test]
    fn block_ram_claim_verifier_rejects_count_mismatch() {
        let block = trace_block(0, boundary(0, 7), boundary(3, 7));
        let io_claims = extract_block_io_claims(&block).unwrap();
        let mut ram_claim = build_block_ram_claim(&block, &io_claims).unwrap();
        ram_claim.access_count = 1;

        assert_eq!(
            verify_block_ram_claim(&block, &io_claims, &ram_claim).unwrap_err(),
            BlockTraceError::BlockRamClaimShapeMismatch {
                block_index: 0,
                reason: "RAM access count mismatch",
            }
        );
    }

    #[test]
    fn block_ram_claim_chain_rejects_length_mismatch() {
        let block = trace_block(0, boundary(0, 7), boundary(3, 7));
        let io_claims = extract_block_io_claims(&block).unwrap();

        assert_eq!(
            verify_block_ram_claim_chain(&[block], &[io_claims], &[]).unwrap_err(),
            BlockTraceError::BlockRamClaimChainLengthMismatch {
                blocks: 1,
                io_claims: 1,
                ram_claims: 0,
            }
        );
    }

    #[test]
    fn ram_address_summaries_track_first_and_final_values() {
        let accesses = vec![
            RamAccessClaim::Read {
                local_cycle: 0,
                global_cycle: 10,
                address: 5,
                value: 7,
            },
            RamAccessClaim::Write {
                local_cycle: 1,
                global_cycle: 11,
                address: 5,
                pre_value: 7,
                post_value: 9,
            },
            RamAccessClaim::Read {
                local_cycle: 2,
                global_cycle: 12,
                address: 8,
                value: 3,
            },
        ];

        let summaries = ram_address_summaries(&accesses);

        assert_eq!(
            summaries,
            vec![
                RamAddressSummary {
                    address: 5,
                    first_value: 7,
                    final_value: 9,
                    read_count: 1,
                    write_count: 1,
                    first_global_cycle: 10,
                    last_global_cycle: 11,
                },
                RamAddressSummary {
                    address: 8,
                    first_value: 3,
                    final_value: 3,
                    read_count: 1,
                    write_count: 0,
                    first_global_cycle: 12,
                    last_global_cycle: 12,
                },
            ]
        );
    }

    #[test]
    fn ram_claim_continuity_rejects_cross_block_value_gap() {
        let claims0 = BlockIOClaims {
            block_index: 0,
            global_cycle_start: 0,
            active_cycles: 1,
            register_reads: vec![],
            register_writes: vec![],
            ram_accesses: vec![RamAccessClaim::Write {
                local_cycle: 0,
                global_cycle: 0,
                address: 5,
                pre_value: 7,
                post_value: 9,
            }],
            lookup_claims: vec![],
        };
        let claims1 = BlockIOClaims {
            block_index: 1,
            global_cycle_start: 1,
            active_cycles: 1,
            register_reads: vec![],
            register_writes: vec![],
            ram_accesses: vec![RamAccessClaim::Read {
                local_cycle: 0,
                global_cycle: 1,
                address: 5,
                value: 8,
            }],
            lookup_claims: vec![],
        };

        assert_eq!(
            validate_ram_claim_continuity(&[claims0, claims1]).unwrap_err(),
            BlockTraceError::BlockRamClaimContinuityMismatch {
                block_index: 1,
                address: 5,
                expected: 9,
                actual: 8,
            }
        );
    }

    #[test]
    fn block_lookup_claim_builds_noop_accumulator() {
        let block = trace_block(0, boundary(0, 7), boundary(3, 7));
        let io_claims = extract_block_io_claims(&block).unwrap();

        let lookup_claim = build_block_lookup_claim(&block, &io_claims).unwrap();

        assert_eq!(lookup_claim.block_index, 0);
        assert_eq!(lookup_claim.active_cycles, 3);
        assert_eq!(lookup_claim.lookup_count, 3);
        assert_eq!(lookup_claim.distinct_lookup_entry_count, 1);
        assert_eq!(lookup_claim.logup_proof.query_count, 3);
        assert_eq!(lookup_claim.logup_proof.table_distinct_entry_count, 1);
        assert_eq!(
            lookup_claim.logup_proof.query_sum,
            lookup_claim.logup_proof.table_sum
        );
        assert_ne!(lookup_claim.logup_proof.proof_digest, [0u8; 32]);
        verify_block_lookup_claim(&block, &io_claims, &lookup_claim).unwrap();
    }

    #[test]
    fn block_lookup_claim_chain_validates_contiguous_blocks() {
        let block0 = trace_block(0, boundary(0, 7), boundary(2, 7));
        let block1 = trace_block(1, block0.end_state.clone(), boundary(4, 7));
        let io0 = extract_block_io_claims(&block0).unwrap();
        let io1 = extract_block_io_claims(&block1).unwrap();
        let lookup0 = build_block_lookup_claim(&block0, &io0).unwrap();
        let lookup1 = build_block_lookup_claim(&block1, &io1).unwrap();

        verify_block_lookup_claim_chain(&[block0, block1], &[io0, io1], &[lookup0, lookup1])
            .unwrap();
    }

    #[test]
    fn block_lookup_claim_verifier_rejects_tampered_digest() {
        let block = trace_block(0, boundary(0, 7), boundary(3, 7));
        let io_claims = extract_block_io_claims(&block).unwrap();
        let mut lookup_claim = build_block_lookup_claim(&block, &io_claims).unwrap();
        lookup_claim.claims_digest[0] ^= 1;

        assert_eq!(
            verify_block_lookup_claim(&block, &io_claims, &lookup_claim).unwrap_err(),
            BlockTraceError::BlockLookupClaimMismatch { block_index: 0 }
        );
    }

    #[test]
    fn block_lookup_claim_verifier_rejects_tampered_logup_sum() {
        let block = trace_block(0, boundary(0, 7), boundary(3, 7));
        let io_claims = extract_block_io_claims(&block).unwrap();
        let mut lookup_claim = build_block_lookup_claim(&block, &io_claims).unwrap();
        lookup_claim.logup_proof.query_sum += <ark_bn254::Fr as JoltField>::from_u64(1);

        assert_eq!(
            verify_block_lookup_claim(&block, &io_claims, &lookup_claim).unwrap_err(),
            BlockTraceError::BlockLookupLogUpProofMismatch { block_index: 0 }
        );
    }

    #[test]
    fn block_lookup_claim_verifier_rejects_count_mismatch() {
        let block = trace_block(0, boundary(0, 7), boundary(3, 7));
        let io_claims = extract_block_io_claims(&block).unwrap();
        let mut lookup_claim = build_block_lookup_claim(&block, &io_claims).unwrap();
        lookup_claim.lookup_count = 2;

        assert_eq!(
            verify_block_lookup_claim(&block, &io_claims, &lookup_claim).unwrap_err(),
            BlockTraceError::BlockLookupClaimShapeMismatch {
                block_index: 0,
                reason: "lookup count mismatch",
            }
        );
    }

    #[test]
    fn block_lookup_claim_chain_rejects_length_mismatch() {
        let block = trace_block(0, boundary(0, 7), boundary(3, 7));
        let io_claims = extract_block_io_claims(&block).unwrap();

        assert_eq!(
            verify_block_lookup_claim_chain(&[block], &[io_claims], &[]).unwrap_err(),
            BlockTraceError::BlockLookupClaimChainLengthMismatch {
                blocks: 1,
                io_claims: 1,
                lookup_claims: 0,
            }
        );
    }

    #[test]
    fn lookup_entry_summaries_track_distinct_table_entries() {
        let claims = vec![
            LookupClaim {
                local_cycle: 0,
                global_cycle: 10,
                left_instruction_input: 1,
                right_instruction_input: 2,
                left_lookup_operand: 3,
                right_lookup_operand: 4,
                lookup_index: 5,
                lookup_output: 6,
            },
            LookupClaim {
                local_cycle: 1,
                global_cycle: 11,
                left_instruction_input: 10,
                right_instruction_input: 20,
                left_lookup_operand: 3,
                right_lookup_operand: 4,
                lookup_index: 5,
                lookup_output: 6,
            },
            LookupClaim {
                local_cycle: 2,
                global_cycle: 12,
                left_instruction_input: 1,
                right_instruction_input: 2,
                left_lookup_operand: 7,
                right_lookup_operand: 8,
                lookup_index: 5,
                lookup_output: 9,
            },
        ];

        let summaries = lookup_entry_summaries(&claims);

        assert_eq!(
            summaries,
            vec![
                LookupEntrySummary {
                    lookup_index: 5,
                    left_lookup_operand: 3,
                    right_lookup_operand: 4,
                    lookup_output: 6,
                    count: 2,
                    first_global_cycle: 10,
                    last_global_cycle: 11,
                },
                LookupEntrySummary {
                    lookup_index: 5,
                    left_lookup_operand: 7,
                    right_lookup_operand: 8,
                    lookup_output: 9,
                    count: 1,
                    first_global_cycle: 12,
                    last_global_cycle: 12,
                },
            ]
        );
    }

    #[test]
    fn block_bundle_prover_emits_and_verifies_noop_bundle() {
        let bytecode = BytecodePreprocessing::default();
        let block = trace_block(0, boundary(0, 0), boundary(4, 0));
        let prover = BlockProofBundleProver::<_, ark_bn254::Fr>::new([9u8; 32]);

        let bundle = prover.prove_block(&bytecode, &block, None).unwrap();

        assert_eq!(bundle.public_input().block_index, 0);
        assert_eq!(bundle.public_input().active_cycles, 4);
        assert_eq!(bundle.cpu_proof.inner_proof.r1cs_rows_checked, 4);
        assert_eq!(bundle.io_claims.lookup_claims.len(), 4);
        assert_eq!(bundle.register_claim.read_count, 0);
        assert_eq!(bundle.ram_claim.access_count, 0);
        assert_eq!(bundle.lookup_claim.lookup_count, 4);
        verify_block_proof_bundle(&bytecode, &block, None, &bundle).unwrap();
    }

    #[test]
    fn block_bundle_prover_validates_chain_with_lookahead() {
        let bytecode = BytecodePreprocessing::default();
        let block0 = trace_block(0, boundary(0, 0), boundary(2, 0));
        let block1 = trace_block(1, block0.end_state.clone(), boundary(4, 0));
        let prover = BlockProofBundleProver::<_, ark_bn254::Fr>::new([9u8; 32]);

        let bundles = prover
            .prove_blocks(&bytecode, &[block0.clone(), block1.clone()])
            .unwrap();

        assert_eq!(bundles.len(), 2);
        assert!(bundles[0].cpu_proof.inner_proof.used_lookahead_cycle);
        assert!(!bundles[1].cpu_proof.inner_proof.used_lookahead_cycle);
        verify_block_proof_bundle_chain(&bytecode, &[block0, block1], &bundles).unwrap();
    }

    #[test]
    fn block_bundle_verifier_rejects_tampered_register_claim() {
        let bytecode = BytecodePreprocessing::default();
        let block = trace_block(0, boundary(0, 0), boundary(4, 0));
        let prover = BlockProofBundleProver::<_, ark_bn254::Fr>::new([9u8; 32]);
        let mut bundle = prover.prove_block(&bytecode, &block, None).unwrap();
        bundle.register_claim.reads_digest[0] ^= 1;

        assert_eq!(
            verify_block_proof_bundle(&bytecode, &block, None, &bundle).unwrap_err(),
            BlockTraceError::BlockRegisterClaimMismatch { block_index: 0 }
        );
    }

    #[test]
    fn block_bundle_chain_rejects_length_mismatch() {
        let bytecode = BytecodePreprocessing::default();
        let block = trace_block(0, boundary(0, 0), boundary(4, 0));

        assert_eq!(
            verify_block_proof_bundle_chain::<[u8; 32], ark_bn254::Fr>(&bytecode, &[block], &[])
                .unwrap_err(),
            BlockTraceError::BlockProofBundleChainLengthMismatch {
                blocks: 1,
                bundles: 0,
            }
        );
    }

    #[test]
    fn block_fold_input_builds_from_bundle() {
        let bytecode = BytecodePreprocessing::default();
        let block = trace_block(0, boundary(0, 0), boundary(4, 0));
        let prover = BlockProofBundleProver::<_, ark_bn254::Fr>::new([9u8; 32]);
        let bundle = prover.prove_block(&bytecode, &block, None).unwrap();

        let fold_input = build_block_fold_input(&bundle);

        assert_eq!(fold_input.program_digest, [9u8; 32]);
        assert_eq!(fold_input.state.block_index, 0);
        assert_eq!(fold_input.state.global_cycle_start, 0);
        assert_eq!(fold_input.state.global_cycle_end, 4);
        assert_eq!(fold_input.state.active_cycles, 4);
        assert_eq!(fold_input.state.r1cs_rows_checked, 4);
        assert_eq!(fold_input.state.r1cs_num_steps, 4);
        assert_eq!(fold_input.state.lookup_count, 4);
        assert_ne!(fold_input.state.state_digest, [0u8; 32]);
        verify_block_fold_input(&bytecode, &block, None, &bundle, &fold_input).unwrap();
    }

    #[test]
    fn block_fold_input_chain_validates_bundles() {
        let bytecode = BytecodePreprocessing::default();
        let block0 = trace_block(0, boundary(0, 0), boundary(2, 0));
        let block1 = trace_block(1, block0.end_state.clone(), boundary(4, 0));
        let prover = BlockProofBundleProver::<_, ark_bn254::Fr>::new([9u8; 32]);
        let bundles = prover
            .prove_blocks(&bytecode, &[block0.clone(), block1.clone()])
            .unwrap();

        let fold_inputs = build_block_fold_inputs(&bundles);

        assert_eq!(fold_inputs.len(), 2);
        assert_eq!(
            fold_inputs[0].state.end_state_digest,
            fold_inputs[1].state.start_state_digest
        );
        assert!(fold_inputs[0].state.used_lookahead_cycle);
        assert!(fold_inputs[0].state.lookahead_cycle_digest.is_some());
        assert!(!fold_inputs[1].state.used_lookahead_cycle);
        assert!(fold_inputs[1].state.lookahead_cycle_digest.is_none());
        verify_block_fold_input_chain(&bytecode, &[block0, block1], &bundles, &fold_inputs)
            .unwrap();
    }

    #[test]
    fn block_fold_input_verifier_rejects_tampered_state_digest() {
        let bytecode = BytecodePreprocessing::default();
        let block = trace_block(0, boundary(0, 0), boundary(4, 0));
        let prover = BlockProofBundleProver::<_, ark_bn254::Fr>::new([9u8; 32]);
        let bundle = prover.prove_block(&bytecode, &block, None).unwrap();
        let mut fold_input = build_block_fold_input(&bundle);
        fold_input.state.state_digest[0] ^= 1;

        assert_eq!(
            verify_block_fold_input(&bytecode, &block, None, &bundle, &fold_input).unwrap_err(),
            BlockTraceError::BlockFoldInputMismatch { block_index: 0 }
        );
    }

    #[test]
    fn block_fold_input_chain_rejects_length_mismatch() {
        let bytecode = BytecodePreprocessing::default();
        let block = trace_block(0, boundary(0, 0), boundary(4, 0));
        let prover = BlockProofBundleProver::<_, ark_bn254::Fr>::new([9u8; 32]);
        let bundle = prover.prove_block(&bytecode, &block, None).unwrap();

        assert_eq!(
            verify_block_fold_input_chain(&bytecode, &[block], &[bundle], &[]).unwrap_err(),
            BlockTraceError::BlockFoldInputChainLengthMismatch {
                blocks: 1,
                bundles: 1,
                fold_inputs: 0,
            }
        );
    }

    #[test]
    fn foldable_block_state_rejects_boundary_digest_gap() {
        let bytecode = BytecodePreprocessing::default();
        let block0 = trace_block(0, boundary(0, 0), boundary(2, 0));
        let block1 = trace_block(1, block0.end_state.clone(), boundary(4, 0));
        let prover = BlockProofBundleProver::<_, ark_bn254::Fr>::new([9u8; 32]);
        let bundles = prover
            .prove_blocks(&bytecode, &[block0.clone(), block1.clone()])
            .unwrap();
        let mut fold_inputs = build_block_fold_inputs(&bundles);
        fold_inputs[1].state.start_state_digest[0] ^= 1;

        assert_eq!(
            validate_block_fold_input_chain(&fold_inputs).unwrap_err(),
            BlockTraceError::BlockFoldInputBoundaryMismatch {
                current_block: 0,
                next_block: 1,
                reason: "machine boundary state digest mismatch",
            }
        );
    }

    #[test]
    fn block_fold_accumulator_absorbs_fold_inputs() {
        let bytecode = BytecodePreprocessing::default();
        let block0 = trace_block(0, boundary(0, 0), boundary(2, 0));
        let block1 = trace_block(1, block0.end_state.clone(), boundary(4, 0));
        let prover = BlockProofBundleProver::<_, ark_bn254::Fr>::new([9u8; 32]);
        let bundles = prover
            .prove_blocks(&bytecode, &[block0.clone(), block1.clone()])
            .unwrap();
        let fold_inputs = build_block_fold_inputs(&bundles);

        let accumulator = build_block_fold_accumulator(&fold_inputs).unwrap();

        assert_eq!(accumulator.program_digest, Some([9u8; 32]));
        assert_eq!(accumulator.absorbed_blocks, 2);
        assert_eq!(accumulator.first_block_index, Some(0));
        assert_eq!(accumulator.last_block_index, Some(1));
        assert_eq!(accumulator.global_cycle_start, Some(0));
        assert_eq!(accumulator.global_cycle_end, Some(4));
        assert_eq!(accumulator.total_active_cycles, 4);
        assert_eq!(accumulator.total_lookup_claims, 4);
        assert_eq!(
            accumulator.latest_state_digest,
            Some(fold_inputs[1].state.state_digest)
        );
        assert_ne!(
            accumulator.accumulator_digest,
            BlockFoldAccumulator::<[u8; 32]>::new().accumulator_digest
        );
        verify_block_fold_accumulator(&fold_inputs, &accumulator).unwrap();
    }

    #[test]
    fn verified_block_fold_accumulator_validates_pipeline_first() {
        let bytecode = BytecodePreprocessing::default();
        let block0 = trace_block(0, boundary(0, 0), boundary(2, 0));
        let block1 = trace_block(1, block0.end_state.clone(), boundary(4, 0));
        let prover = BlockProofBundleProver::<_, ark_bn254::Fr>::new([9u8; 32]);
        let bundles = prover
            .prove_blocks(&bytecode, &[block0.clone(), block1.clone()])
            .unwrap();
        let fold_inputs = build_block_fold_inputs(&bundles);

        let accumulator = build_verified_block_fold_accumulator(
            &bytecode,
            &[block0, block1],
            &bundles,
            &fold_inputs,
        )
        .unwrap();

        assert_eq!(accumulator.absorbed_blocks, 2);
        assert_eq!(accumulator.total_active_cycles, 4);
    }

    #[test]
    fn block_fold_accumulator_rejects_tampered_fold_state_digest() {
        let bytecode = BytecodePreprocessing::default();
        let block = trace_block(0, boundary(0, 0), boundary(4, 0));
        let prover = BlockProofBundleProver::<_, ark_bn254::Fr>::new([9u8; 32]);
        let bundle = prover.prove_block(&bytecode, &block, None).unwrap();
        let mut fold_input = build_block_fold_input(&bundle);
        fold_input.state.state_digest[0] ^= 1;

        assert_eq!(
            build_block_fold_accumulator(&[fold_input]).unwrap_err(),
            BlockTraceError::BlockFoldAccumulatorAbsorbMismatch {
                block_index: 0,
                reason: "foldable state digest mismatch",
            }
        );
    }

    #[test]
    fn block_fold_accumulator_rejects_program_digest_mismatch() {
        let bytecode = BytecodePreprocessing::default();
        let block0 = trace_block(0, boundary(0, 0), boundary(2, 0));
        let block1 = trace_block(1, block0.end_state.clone(), boundary(4, 0));
        let prover = BlockProofBundleProver::<_, ark_bn254::Fr>::new([9u8; 32]);
        let bundles = prover
            .prove_blocks(&bytecode, &[block0.clone(), block1.clone()])
            .unwrap();
        let mut fold_inputs = build_block_fold_inputs(&bundles);
        fold_inputs[1].program_digest = [8u8; 32];

        assert_eq!(
            build_block_fold_accumulator(&fold_inputs).unwrap_err(),
            BlockTraceError::BlockFoldAccumulatorProgramDigestMismatch { block_index: 1 }
        );
    }

    #[test]
    fn block_fold_accumulator_rejects_boundary_digest_gap() {
        let bytecode = BytecodePreprocessing::default();
        let block0 = trace_block(0, boundary(0, 0), boundary(2, 0));
        let block1 = trace_block(1, block0.end_state.clone(), boundary(4, 0));
        let prover = BlockProofBundleProver::<_, ark_bn254::Fr>::new([9u8; 32]);
        let bundles = prover
            .prove_blocks(&bytecode, &[block0.clone(), block1.clone()])
            .unwrap();
        let mut fold_inputs = build_block_fold_inputs(&bundles);
        fold_inputs[1].state.start_register_digest[0] ^= 1;
        fold_inputs[1].state.state_digest = digest_foldable_block_state(&fold_inputs[1].state);

        assert_eq!(
            build_block_fold_accumulator(&fold_inputs).unwrap_err(),
            BlockTraceError::BlockFoldAccumulatorBoundaryMismatch {
                current_block: 0,
                next_block: 1,
                reason: "register boundary digest mismatch",
            }
        );
    }

    #[test]
    fn block_fold_accumulator_verifier_rejects_tampered_accumulator() {
        let bytecode = BytecodePreprocessing::default();
        let block = trace_block(0, boundary(0, 0), boundary(4, 0));
        let prover = BlockProofBundleProver::<_, ark_bn254::Fr>::new([9u8; 32]);
        let bundle = prover.prove_block(&bytecode, &block, None).unwrap();
        let fold_inputs = vec![build_block_fold_input(&bundle)];
        let mut accumulator = build_block_fold_accumulator(&fold_inputs).unwrap();
        accumulator.accumulator_digest[0] ^= 1;

        assert_eq!(
            verify_block_fold_accumulator(&fold_inputs, &accumulator).unwrap_err(),
            BlockTraceError::BlockFoldAccumulatorMismatch {
                expected_blocks: 1,
                actual_blocks: 1,
            }
        );
    }

    #[test]
    fn mock_folding_backend_matches_legacy_accumulator() {
        let bytecode = BytecodePreprocessing::default();
        let block0 = trace_block(0, boundary(0, 0), boundary(2, 0));
        let block1 = trace_block(1, block0.end_state.clone(), boundary(4, 0));
        let prover = BlockProofBundleProver::<_, ark_bn254::Fr>::new([9u8; 32]);
        let bundles = prover
            .prove_blocks(&bytecode, &[block0.clone(), block1.clone()])
            .unwrap();
        let fold_inputs = build_block_fold_inputs(&bundles);
        let backend = MockFoldingBackend;

        let legacy_accumulator = build_block_fold_accumulator(&fold_inputs).unwrap();
        let backend_accumulator =
            build_block_fold_accumulator_with_backend(&fold_inputs, &backend).unwrap();

        assert_eq!(
            <MockFoldingBackend as BlockFoldingBackend<[u8; 32], ark_bn254::Fr>>::name(&backend),
            "mock-hash-chain"
        );
        assert_eq!(backend_accumulator, legacy_accumulator);
        verify_block_fold_accumulator_with_backend(&fold_inputs, &backend_accumulator, &backend)
            .unwrap();
    }

    #[test]
    fn block_proof_pipeline_accepts_explicit_mock_backend() {
        let bytecode = BytecodePreprocessing::default();
        let block0 = trace_block(0, boundary(0, 0), boundary(2, 0));
        let block1 = trace_block(1, block0.end_state.clone(), boundary(4, 0));
        let pipeline = BlockProofPipeline::<_, ark_bn254::Fr, MockFoldingBackend>::with_backend(
            [9u8; 32],
            MockFoldingBackend,
        );

        let output = pipeline
            .prove_blocks(&bytecode, &[block0.clone(), block1.clone()])
            .unwrap();

        assert_eq!(output.accumulator.absorbed_blocks, 2);
        assert!(output.final_proof.is_none());
        verify_block_proof_pipeline_with_backend(
            &bytecode,
            &[block0, block1],
            &output,
            &MockFoldingBackend,
        )
        .unwrap();
    }

    #[test]
    fn nova_folding_backend_exposes_placeholder_config() {
        let backend = NovaFoldingBackend::default();
        let accumulator =
            <NovaFoldingBackend as BlockFoldingBackend<[u8; 32], ark_bn254::Fr>>::new_accumulator(
                &backend,
            );
        let expected_backend_name = if cfg!(feature = "nova") {
            "nova-recursive-snark"
        } else {
            "nova-placeholder"
        };

        assert_eq!(
            <NovaFoldingBackend as BlockFoldingBackend<[u8; 32], ark_bn254::Fr>>::name(&backend),
            expected_backend_name
        );
        assert_eq!(
            <NovaFoldingBackend as BlockFoldingBackend<[u8; 32], ark_bn254::Fr>>::relation_name(
                &backend
            ),
            NOVA_BLOCK_FOLD_RELATION_NAME
        );
        assert_eq!(
            accumulator.config.relation_name,
            NOVA_BLOCK_FOLD_RELATION_NAME
        );
        assert_eq!(
            accumulator.config.subclaim_backend_name,
            NOVA_TRANSCRIPT_SUBCLAIM_BACKEND_NAME
        );
        assert_eq!(
            accumulator.config.final_proof_backend_name,
            SPARTAN_PLACEHOLDER_PROOF_SYSTEM_NAME
        );
        assert!(accumulator.config.use_zero_knowledge);
        assert_eq!(accumulator.metadata.absorbed_blocks, 0);
        assert_eq!(accumulator.metadata.program_digest, None);
        assert_eq!(accumulator.metadata.latest_state_digest, None);
        assert!(accumulator.recursive_snark_bytes.is_none());
        assert!(accumulator.recursive_snark_output_digest.is_none());
        assert!(accumulator.recursive_z_state.is_none());
    }

    #[test]
    fn nova_folding_backend_rejects_unsupported_relation_name() {
        let bytecode = BytecodePreprocessing::default();
        let block = trace_block(0, boundary(0, 0), boundary(4, 0));
        let prover = BlockProofBundleProver::<_, ark_bn254::Fr>::new([9u8; 32]);
        let bundle = prover.prove_block(&bytecode, &block, None).unwrap();
        let fold_input = build_block_fold_input(&bundle);
        let backend = NovaFoldingBackend::new(NovaFoldConfig {
            relation_name: "jolt-nova-block-fold-unsupported",
            ..NovaFoldConfig::default()
        });

        assert_eq!(
            build_block_fold_accumulator_with_backend(&[fold_input], &backend).unwrap_err(),
            BlockTraceError::NovaFoldingBackendError {
                block_index: 0,
                reason: "unsupported Nova fold relation",
            }
        );
    }

    #[test]
    fn nova_folding_backend_rejects_unsupported_subclaim_backend_name() {
        let bytecode = BytecodePreprocessing::default();
        let block = trace_block(0, boundary(0, 0), boundary(4, 0));
        let prover = BlockProofBundleProver::<_, ark_bn254::Fr>::new([9u8; 32]);
        let bundle = prover.prove_block(&bytecode, &block, None).unwrap();
        let fold_input = build_block_fold_input(&bundle);
        let backend = NovaFoldingBackend::new(NovaFoldConfig {
            subclaim_backend_name: "unsupported-subclaim-backend",
            ..NovaFoldConfig::default()
        });

        assert_eq!(
            build_block_fold_accumulator_with_backend(&[fold_input], &backend).unwrap_err(),
            BlockTraceError::NovaFoldingBackendError {
                block_index: 0,
                reason: "unsupported Nova subclaim folding backend",
            }
        );
    }

    #[test]
    fn final_folded_instance_rejects_empty_nova_accumulator() {
        let backend = NovaFoldingBackend::default();
        let accumulator =
            <NovaFoldingBackend as BlockFoldingBackend<[u8; 32], ark_bn254::Fr>>::new_accumulator(
                &backend,
            );

        assert_eq!(
            build_final_folded_instance(&accumulator).unwrap_err(),
            BlockTraceError::NovaFoldingBackendError {
                block_index: 0,
                reason: "final folded instance requires a non-empty Nova accumulator",
            }
        );
    }

    #[cfg(feature = "nova")]
    #[test]
    fn nova_transcript_challenges_are_domain_separated() {
        let semantic = nova_transcript_challenge_scalar(
            NOVA_TRANSCRIPT_DOMAIN_SEMANTIC,
            "lookup_claims_digest",
        );
        let lookup =
            nova_transcript_challenge_scalar(NOVA_TRANSCRIPT_DOMAIN_LOOKUP, "lookup_claims_digest");
        let different_label = nova_transcript_challenge_scalar(
            NOVA_TRANSCRIPT_DOMAIN_SEMANTIC,
            "lookup_entry_summaries_digest",
        );

        assert_ne!(semantic, NovaScalar::zero());
        assert_ne!(lookup, NovaScalar::zero());
        assert_ne!(different_label, NovaScalar::zero());
        assert_ne!(semantic, lookup);
        assert_ne!(semantic, different_label);
    }

    #[cfg(feature = "nova")]
    #[test]
    fn nova_hash_to_field_uses_full_digest_width() {
        let mut left = [0u8; 32];
        let mut right = [0u8; 32];
        left[0] = 7;
        right[31] = 7;

        let left_scalar = nova_hash_bytes_to_scalar("test", "digest", &left);
        let right_scalar = nova_hash_bytes_to_scalar("test", "digest", &right);

        assert_ne!(left_scalar, NovaScalar::zero());
        assert_ne!(right_scalar, NovaScalar::zero());
        assert_ne!(left_scalar, right_scalar);
    }

    #[cfg(feature = "nova")]
    #[test]
    fn nova_block_fold_statement_digest_tracks_fold_input_claims() {
        let bytecode = BytecodePreprocessing::default();
        let block = trace_block(0, boundary(0, 0), boundary(4, 0));
        let prover = BlockProofBundleProver::<_, ark_bn254::Fr>::new([9u8; 32]);
        let bundle = prover.prove_block(&bytecode, &block, None).unwrap();
        let mut fold_input = build_block_fold_input(&bundle);
        let statement = BlockFoldStatement::from_fold_input(&fold_input);
        let statement_digest = statement.digest();
        let statement_fingerprint = statement.statement_digest_scalar();
        let lookup_delta = statement.lookup_delta();

        fold_input.state.lookup_count += 1;
        fold_input.state.state_digest = digest_foldable_block_state(&fold_input.state);
        let tampered_statement = BlockFoldStatement::from_fold_input(&fold_input);

        assert_ne!(statement_digest, tampered_statement.digest());
        assert_ne!(
            statement_fingerprint,
            tampered_statement.statement_digest_scalar()
        );
        assert_ne!(lookup_delta, tampered_statement.lookup_delta());
        assert_ne!(
            statement.semantic_delta(),
            tampered_statement.semantic_delta()
        );
    }

    #[cfg(feature = "nova")]
    #[test]
    fn nova_register_fingerprint_tracks_register_claims() {
        let bytecode = BytecodePreprocessing::default();
        let block = trace_block(0, boundary(0, 0), boundary(4, 0));
        let prover = BlockProofBundleProver::<_, ark_bn254::Fr>::new([9u8; 32]);
        let bundle = prover.prove_block(&bytecode, &block, None).unwrap();
        let mut fold_input = build_block_fold_input(&bundle);
        let statement = BlockFoldStatement::from_fold_input(&fold_input);
        let register_fingerprint = statement.register_fingerprint();

        fold_input.state.register_read_count += 1;
        fold_input.state.state_digest = digest_foldable_block_state(&fold_input.state);
        let tampered_statement = BlockFoldStatement::from_fold_input(&fold_input);

        assert_ne!(
            register_fingerprint,
            tampered_statement.register_fingerprint()
        );
        assert_ne!(
            statement.register_delta(),
            tampered_statement.register_delta()
        );
    }

    #[cfg(feature = "nova")]
    #[test]
    fn nova_ram_fingerprint_tracks_ram_claims() {
        let bytecode = BytecodePreprocessing::default();
        let block = trace_block(0, boundary(0, 0), boundary(4, 0));
        let prover = BlockProofBundleProver::<_, ark_bn254::Fr>::new([9u8; 32]);
        let bundle = prover.prove_block(&bytecode, &block, None).unwrap();
        let mut fold_input = build_block_fold_input(&bundle);
        let statement = BlockFoldStatement::from_fold_input(&fold_input);
        let ram_fingerprint = statement.ram_fingerprint();

        fold_input.state.ram_access_count += 1;
        fold_input.state.state_digest = digest_foldable_block_state(&fold_input.state);
        let tampered_statement = BlockFoldStatement::from_fold_input(&fold_input);

        assert_ne!(ram_fingerprint, tampered_statement.ram_fingerprint());
        assert_ne!(statement.ram_delta(), tampered_statement.ram_delta());
    }

    #[cfg(feature = "nova")]
    #[test]
    fn nova_lookup_fingerprint_tracks_lookup_claims() {
        let bytecode = BytecodePreprocessing::default();
        let block = trace_block(0, boundary(0, 0), boundary(4, 0));
        let prover = BlockProofBundleProver::<_, ark_bn254::Fr>::new([9u8; 32]);
        let bundle = prover.prove_block(&bytecode, &block, None).unwrap();
        let mut fold_input = build_block_fold_input(&bundle);
        let statement = BlockFoldStatement::from_fold_input(&fold_input);
        let lookup_fingerprint = statement.lookup_fingerprint();

        fold_input.state.lookup_count += 1;
        fold_input.state.state_digest = digest_foldable_block_state(&fold_input.state);
        let tampered_statement = BlockFoldStatement::from_fold_input(&fold_input);

        assert_ne!(lookup_fingerprint, tampered_statement.lookup_fingerprint());
        assert_ne!(statement.lookup_delta(), tampered_statement.lookup_delta());
    }

    #[cfg(feature = "nova")]
    #[test]
    fn nova_cpu_fingerprint_tracks_cpu_claims() {
        let bytecode = BytecodePreprocessing::default();
        let block = trace_block(0, boundary(0, 0), boundary(4, 0));
        let prover = BlockProofBundleProver::<_, ark_bn254::Fr>::new([9u8; 32]);
        let bundle = prover.prove_block(&bytecode, &block, None).unwrap();
        let mut fold_input = build_block_fold_input(&bundle);
        let statement = BlockFoldStatement::from_fold_input(&fold_input);
        let cpu_fingerprint = statement.cpu_fingerprint();

        fold_input.state.r1cs_rows_checked += 1;
        fold_input.state.state_digest = digest_foldable_block_state(&fold_input.state);
        let tampered_statement = BlockFoldStatement::from_fold_input(&fold_input);

        assert_ne!(cpu_fingerprint, tampered_statement.cpu_fingerprint());
        assert_ne!(statement.cpu_delta(), tampered_statement.cpu_delta());
    }

    #[cfg(feature = "nova")]
    #[test]
    fn nova_subclaim_fingerprints_match_individual_deltas() {
        let bytecode = BytecodePreprocessing::default();
        let block = trace_block(0, boundary(0, 0), boundary(4, 0));
        let prover = BlockProofBundleProver::<_, ark_bn254::Fr>::new([9u8; 32]);
        let bundle = prover.prove_block(&bytecode, &block, None).unwrap();
        let fold_input = build_block_fold_input(&bundle);
        let statement = BlockFoldStatement::from_fold_input(&fold_input);
        let statement_subclaims = statement.subclaim_fingerprints();
        let witness = JoltNovaStepWitness::from_fold_input(&fold_input);

        assert_eq!(statement_subclaims.register, statement.register_delta());
        assert_eq!(statement_subclaims.ram, statement.ram_delta());
        assert_eq!(statement_subclaims.lookup, statement.lookup_delta());
        assert_eq!(statement_subclaims.cpu, statement.cpu_delta());
        assert_eq!(witness.subclaim_fingerprints(), statement_subclaims);
        assert_eq!(
            witness.semantic_delta(),
            statement.semantic_delta_with_subclaims(statement_subclaims)
        );
    }

    #[cfg(feature = "nova")]
    #[test]
    fn nova_witness_semantic_delta_binds_subclaim_fingerprints() {
        let bytecode = BytecodePreprocessing::default();
        let block = trace_block(0, boundary(0, 0), boundary(4, 0));
        let prover = BlockProofBundleProver::<_, ark_bn254::Fr>::new([9u8; 32]);
        let bundle = prover.prove_block(&bytecode, &block, None).unwrap();
        let fold_input = build_block_fold_input(&bundle);
        let mut witness = JoltNovaStepWitness::from_fold_input(&fold_input);
        let statement_semantic_delta = witness.statement().semantic_delta();
        let witness_semantic_delta = witness.semantic_delta();

        assert_eq!(witness_semantic_delta, statement_semantic_delta);

        witness.lookup_claim_fingerprint = witness.lookup_claim_fingerprint + NovaScalar::from(1);

        assert_ne!(witness.semantic_delta(), witness_semantic_delta);
        assert_eq!(
            witness.statement().semantic_delta(),
            statement_semantic_delta
        );
    }

    #[cfg(feature = "nova")]
    #[test]
    fn nova_transcript_subclaim_backend_matches_statement_fingerprints() {
        let bytecode = BytecodePreprocessing::default();
        let block = trace_block(0, boundary(0, 0), boundary(4, 0));
        let prover = BlockProofBundleProver::<_, ark_bn254::Fr>::new([9u8; 32]);
        let bundle = prover.prove_block(&bytecode, &block, None).unwrap();
        let fold_input = build_block_fold_input(&bundle);
        let statement = BlockFoldStatement::from_fold_input(&fold_input);
        let backend = TranscriptSubclaimFoldingBackend;
        let expected_subclaims = statement.subclaim_fingerprints();
        let witness =
            JoltNovaStepWitness::from_statement_with_subclaim_backend(statement.clone(), &backend);

        assert_eq!(backend.name(), "transcript-subclaim-fingerprints");
        assert_eq!(
            backend.subclaim_fingerprints(&statement),
            expected_subclaims
        );
        assert_eq!(witness.subclaim_fingerprints(), expected_subclaims);
        assert_eq!(
            witness.semantic_delta(),
            statement.semantic_delta_with_subclaims(expected_subclaims)
        );
    }

    #[cfg(feature = "nova")]
    #[test]
    fn nova_custom_subclaim_backend_feeds_witness_and_circuit_binding() {
        struct LookupOffsetSubclaimBackend;

        impl NovaSubclaimFoldingBackend for LookupOffsetSubclaimBackend {
            fn name(&self) -> &'static str {
                "lookup-offset-test-subclaim-backend"
            }

            fn subclaim_fingerprints(
                &self,
                statement: &BlockFoldStatement,
            ) -> BlockFoldSubclaimFingerprints {
                let mut subclaims =
                    TranscriptSubclaimFoldingBackend.subclaim_fingerprints(statement);
                subclaims.lookup = subclaims.lookup + NovaScalar::from(1);
                subclaims
            }
        }

        let bytecode = BytecodePreprocessing::default();
        let block = trace_block(0, boundary(0, 0), boundary(4, 0));
        let prover = BlockProofBundleProver::<_, ark_bn254::Fr>::new([9u8; 32]);
        let bundle = prover.prove_block(&bytecode, &block, None).unwrap();
        let fold_input = build_block_fold_input(&bundle);
        let statement = BlockFoldStatement::from_fold_input(&fold_input);
        let default_witness = JoltNovaStepWitness::from_fold_input(&fold_input);
        let custom_witness = JoltNovaStepWitness::from_statement_with_subclaim_backend(
            statement,
            &LookupOffsetSubclaimBackend,
        );

        assert_eq!(
            custom_witness.register_delta(),
            default_witness.register_delta()
        );
        assert_eq!(custom_witness.ram_delta(), default_witness.ram_delta());
        assert_eq!(custom_witness.cpu_delta(), default_witness.cpu_delta());
        assert_ne!(
            custom_witness.lookup_delta(),
            default_witness.lookup_delta()
        );
        assert_ne!(
            custom_witness.semantic_delta(),
            default_witness.semantic_delta()
        );

        let circuit = JoltNovaStepCircuit {
            witness: custom_witness,
        };
        let cs = synthesize_nova_step_circuit_for_test(&circuit);

        assert_eq!(
            cs.which_is_unsatisfied(),
            Some("lookup claim fingerprint binds lookup fields")
        );
    }

    #[cfg(feature = "nova")]
    #[test]
    fn nova_configured_transcript_subclaim_backend_matches_default_paths() {
        let bytecode = BytecodePreprocessing::default();
        let block = trace_block(0, boundary(0, 0), boundary(4, 0));
        let prover = BlockProofBundleProver::<_, ark_bn254::Fr>::new([9u8; 32]);
        let bundle = prover.prove_block(&bytecode, &block, None).unwrap();
        let fold_input = build_block_fold_input(&bundle);
        let config = NovaFoldConfig::default();
        let subclaim_backend =
            nova_subclaim_backend_from_config(&config, fold_input.state.block_index).unwrap();

        let default_witness = JoltNovaStepWitness::from_fold_input(&fold_input);
        let configured_witness = JoltNovaStepWitness::from_fold_input_with_subclaim_backend(
            &fold_input,
            &subclaim_backend,
        );
        let default_circuit = nova_step_circuit_for_fold_input(&fold_input);
        let configured_circuit =
            nova_step_circuit_for_fold_input_with_subclaim_backend(&fold_input, &subclaim_backend);
        let initial_z_state = nova_initial_z_state();

        assert_eq!(
            subclaim_backend.name(),
            NOVA_TRANSCRIPT_SUBCLAIM_BACKEND_NAME
        );
        assert_eq!(configured_witness, default_witness);
        assert_eq!(configured_circuit, default_circuit);
        assert_eq!(
            nova_next_z_state_with_subclaim_backend(
                initial_z_state,
                &fold_input,
                &subclaim_backend,
            ),
            nova_next_z_state(initial_z_state, &fold_input)
        );
        assert_eq!(
            nova_expected_z_state_with_subclaim_backend(&[fold_input.clone()], &subclaim_backend),
            nova_expected_z_state(&[fold_input])
        );
    }

    #[test]
    fn spartan_placeholder_final_proof_backend_exposes_adapter_name() {
        let backend = SpartanPlaceholderFinalProofBackend;

        assert_eq!(
            <SpartanPlaceholderFinalProofBackend as FinalFoldedProofBackend<[u8; 32]>>::name(
                &backend
            ),
            SPARTAN_PLACEHOLDER_PROOF_SYSTEM_NAME
        );
    }

    #[test]
    fn spartan_final_proof_backend_exposes_adapter_name() {
        let backend = SpartanFinalProofBackend;

        assert_eq!(
            <SpartanFinalProofBackend as FinalFoldedProofBackend<[u8; 32]>>::name(&backend),
            SPARTAN_FINAL_PROOF_SYSTEM_NAME
        );
    }

    #[cfg(feature = "nova")]
    #[test]
    fn nova_final_folded_proof_backend_matches_legacy_placeholder_functions() {
        let bytecode = BytecodePreprocessing::default();
        let block0 = trace_block(0, boundary(0, 0), boundary(2, 0));
        let block1 = trace_block(1, block0.end_state.clone(), boundary(4, 0));
        let prover = BlockProofBundleProver::<_, ark_bn254::Fr>::new([9u8; 32]);
        let bundles = prover
            .prove_blocks(&bytecode, &[block0.clone(), block1.clone()])
            .unwrap();
        let fold_inputs = build_block_fold_inputs(&bundles);
        let folding_backend = NovaFoldingBackend::default();
        let accumulator =
            build_block_fold_accumulator_with_backend(&fold_inputs, &folding_backend).unwrap();
        let instance = build_final_folded_instance(&accumulator).unwrap();
        let proof_backend = SpartanPlaceholderFinalProofBackend;

        let backend_proof =
            prove_final_folded_instance_with_backend(&proof_backend, instance.clone()).unwrap();
        let legacy_proof = prove_spartan_placeholder_final_instance(instance);

        assert_eq!(backend_proof, legacy_proof);
        verify_final_folded_proof_with_backend(&proof_backend, &accumulator, &backend_proof)
            .unwrap();
        verify_spartan_placeholder_final_proof(&accumulator, &legacy_proof).unwrap();
    }

    #[cfg(feature = "nova")]
    #[test]
    fn nova_configured_final_folded_proof_uses_placeholder_backend_by_default() {
        let bytecode = BytecodePreprocessing::default();
        let block = trace_block(0, boundary(0, 0), boundary(4, 0));
        let prover = BlockProofBundleProver::<_, ark_bn254::Fr>::new([9u8; 32]);
        let bundle = prover.prove_block(&bytecode, &block, None).unwrap();
        let fold_input = build_block_fold_input(&bundle);
        let folding_backend = NovaFoldingBackend::default();
        let accumulator =
            build_block_fold_accumulator_with_backend(&[fold_input], &folding_backend).unwrap();
        let instance = build_final_folded_instance(&accumulator).unwrap();

        let configured_proof = prove_configured_final_folded_instance(instance.clone()).unwrap();
        let placeholder_proof = prove_spartan_placeholder_final_instance(instance);

        assert_eq!(configured_proof, placeholder_proof);
        verify_configured_final_folded_proof(&accumulator, &configured_proof).unwrap();
    }

    #[cfg(feature = "nova")]
    #[test]
    fn nova_configured_final_folded_proof_rejects_backend_mismatch() {
        let bytecode = BytecodePreprocessing::default();
        let block = trace_block(0, boundary(0, 0), boundary(4, 0));
        let prover = BlockProofBundleProver::<_, ark_bn254::Fr>::new([9u8; 32]);
        let bundle = prover.prove_block(&bytecode, &block, None).unwrap();
        let fold_input = build_block_fold_input(&bundle);
        let folding_backend = NovaFoldingBackend::default();
        let accumulator =
            build_block_fold_accumulator_with_backend(&[fold_input], &folding_backend).unwrap();
        let instance = build_final_folded_instance(&accumulator).unwrap();
        let mut proof = prove_spartan_placeholder_final_instance(instance);
        proof.proof_system = SPARTAN_FINAL_PROOF_SYSTEM_NAME;

        assert_eq!(
            verify_configured_final_folded_proof(&accumulator, &proof).unwrap_err(),
            BlockTraceError::NovaFoldingBackendError {
                block_index: 0,
                reason: "final folded proof system does not match configured backend",
            }
        );
    }

    #[cfg(feature = "nova")]
    #[test]
    fn nova_configured_final_folded_proof_rejects_unsupported_backend() {
        let bytecode = BytecodePreprocessing::default();
        let block = trace_block(0, boundary(0, 0), boundary(4, 0));
        let prover = BlockProofBundleProver::<_, ark_bn254::Fr>::new([9u8; 32]);
        let bundle = prover.prove_block(&bytecode, &block, None).unwrap();
        let fold_input = build_block_fold_input(&bundle);
        let folding_backend = NovaFoldingBackend::new(NovaFoldConfig {
            final_proof_backend_name: "unsupported-final-proof-backend",
            ..NovaFoldConfig::default()
        });
        let accumulator =
            build_block_fold_accumulator_with_backend(&[fold_input], &folding_backend).unwrap();
        let instance = build_final_folded_instance(&accumulator).unwrap();

        assert_eq!(
            prove_configured_final_folded_instance(instance).unwrap_err(),
            BlockTraceError::NovaFoldingBackendError {
                block_index: 0,
                reason: "unsupported final folded proof backend",
            }
        );
    }

    #[cfg(feature = "nova")]
    #[test]
    fn nova_spartan_final_proof_backend_instance_prove_requires_accumulator() {
        let bytecode = BytecodePreprocessing::default();
        let block = trace_block(0, boundary(0, 0), boundary(4, 0));
        let prover = BlockProofBundleProver::<_, ark_bn254::Fr>::new([9u8; 32]);
        let bundle = prover.prove_block(&bytecode, &block, None).unwrap();
        let fold_input = build_block_fold_input(&bundle);
        let folding_backend = NovaFoldingBackend::new(NovaFoldConfig {
            final_proof_backend_name: SPARTAN_FINAL_PROOF_SYSTEM_NAME,
            ..NovaFoldConfig::default()
        });
        let accumulator =
            build_block_fold_accumulator_with_backend(&[fold_input], &folding_backend).unwrap();
        let instance = build_final_folded_instance(&accumulator).unwrap();

        assert_eq!(
            prove_configured_final_folded_instance(instance).unwrap_err(),
            BlockTraceError::NovaFoldingBackendUnavailable {
                block_index: 0,
                reason: "Spartan final proof proving requires a Nova accumulator",
            }
        );
    }

    #[cfg(feature = "nova")]
    #[test]
    fn nova_spartan_final_proof_backend_compresses_and_verifies_recursive_snark() {
        let bytecode = BytecodePreprocessing::default();
        let block = trace_block(0, boundary(0, 0), boundary(4, 0));
        let prover = BlockProofBundleProver::<_, ark_bn254::Fr>::new([9u8; 32]);
        let bundle = prover.prove_block(&bytecode, &block, None).unwrap();
        let fold_input = build_block_fold_input(&bundle);
        let folding_backend = NovaFoldingBackend::new(NovaFoldConfig {
            final_proof_backend_name: SPARTAN_FINAL_PROOF_SYSTEM_NAME,
            ..NovaFoldConfig::default()
        });
        let accumulator =
            build_block_fold_accumulator_with_backend(&[fold_input], &folding_backend).unwrap();

        let proof = prove_configured_final_folded_accumulator(&accumulator).unwrap();

        assert_eq!(proof.proof_system, SPARTAN_FINAL_PROOF_SYSTEM_NAME);
        assert!(proof.spartan_encoding_digest.is_some());
        assert!(proof.spartan_proof_bytes.as_ref().unwrap().len() > 0);
        verify_final_folded_proof_envelope(&accumulator, &proof).unwrap();
        verify_configured_final_folded_proof(&accumulator, &proof).unwrap();

        let baseline = summarize_jolt_nova_phase6_baseline(&accumulator, Some(&proof)).unwrap();
        assert_eq!(
            baseline.final_proof_system,
            Some(SPARTAN_FINAL_PROOF_SYSTEM_NAME)
        );
        assert_eq!(baseline.final_proof_digest, Some(proof.proof_digest));
        assert_eq!(
            baseline.final_proof_bytes_len,
            proof.spartan_proof_bytes.as_ref().map(Vec::len)
        );

        let final_proof_size_baseline =
            summarize_jolt_nova_final_proof_size_baseline(&accumulator, &proof).unwrap();
        let encoding = encode_final_folded_instance_for_spartan(&proof.instance).unwrap();

        assert_eq!(
            final_proof_size_baseline.configured_backend_name,
            SPARTAN_FINAL_PROOF_SYSTEM_NAME
        );
        assert_eq!(
            final_proof_size_baseline.proof_system,
            SPARTAN_FINAL_PROOF_SYSTEM_NAME
        );
        assert_eq!(
            final_proof_size_baseline.absorbed_blocks,
            accumulator.metadata.absorbed_blocks
        );
        assert_eq!(
            final_proof_size_baseline.total_active_cycles,
            accumulator.metadata.total_active_cycles
        );
        assert_eq!(
            final_proof_size_baseline.recursive_snark_bytes_len,
            accumulator.recursive_snark_bytes.as_ref().map(Vec::len)
        );
        assert_eq!(
            final_proof_size_baseline.final_public_input_bytes_len,
            encoding.public_input_bytes.len()
        );
        assert_eq!(
            final_proof_size_baseline.final_witness_bytes_len,
            encoding.witness_bytes.len()
        );
        assert_eq!(
            final_proof_size_baseline.proof_payload_bytes_len,
            proof.spartan_proof_bytes.as_ref().map(Vec::len).unwrap()
        );
        assert!(final_proof_size_baseline.proof_payload_bytes_len > 0);
        assert!(final_proof_size_baseline.proof_envelope_bytes_len > 0);
        assert_eq!(
            final_proof_size_baseline.proof_total_bytes_len,
            final_proof_size_baseline.proof_envelope_bytes_len
                + final_proof_size_baseline.proof_payload_bytes_len
        );
        assert!(
            final_proof_size_baseline.proof_total_bytes_len
                > final_proof_size_baseline.proof_envelope_bytes_len
        );
        assert_eq!(
            final_proof_size_baseline.final_instance_digest,
            proof.instance.instance_digest
        );
        assert_eq!(
            final_proof_size_baseline.spartan_encoding_digest,
            encoding.encoding_digest
        );
        assert_eq!(final_proof_size_baseline.proof_digest, proof.proof_digest);
    }

    #[cfg(feature = "nova")]
    #[test]
    fn nova_spartan_final_proof_assembly_binds_bytes_and_envelope() {
        let bytecode = BytecodePreprocessing::default();
        let block = trace_block(0, boundary(0, 0), boundary(4, 0));
        let prover = BlockProofBundleProver::<_, ark_bn254::Fr>::new([9u8; 32]);
        let bundle = prover.prove_block(&bytecode, &block, None).unwrap();
        let fold_input = build_block_fold_input(&bundle);
        let folding_backend = NovaFoldingBackend::new(NovaFoldConfig {
            final_proof_backend_name: SPARTAN_FINAL_PROOF_SYSTEM_NAME,
            ..NovaFoldConfig::default()
        });
        let accumulator =
            build_block_fold_accumulator_with_backend(&[fold_input], &folding_backend).unwrap();
        let instance = build_final_folded_instance(&accumulator).unwrap();
        let proof_bytes = vec![1, 2, 3, 5, 8, 13];
        let proof = assemble_spartan_final_proof(instance, proof_bytes.clone()).unwrap();

        assert_eq!(proof.proof_system, SPARTAN_FINAL_PROOF_SYSTEM_NAME);
        assert_eq!(proof.spartan_proof_bytes, Some(proof_bytes));
        assert_eq!(
            proof.proof_digest,
            digest_final_folded_proof(
                proof.proof_system,
                &proof.instance,
                proof.spartan_encoding_digest,
                proof.spartan_proof_bytes.as_deref()
            )
        );
        verify_final_folded_proof_envelope(&accumulator, &proof).unwrap();
        assert_eq!(
            verify_configured_final_folded_proof(&accumulator, &proof).unwrap_err(),
            BlockTraceError::NovaFoldingBackendError {
                block_index: 0,
                reason: "Spartan compressed SNARK deserialization failed",
            }
        );
    }

    #[cfg(feature = "nova")]
    #[test]
    fn nova_spartan_final_proof_assembly_rejects_empty_bytes() {
        let bytecode = BytecodePreprocessing::default();
        let block = trace_block(0, boundary(0, 0), boundary(4, 0));
        let prover = BlockProofBundleProver::<_, ark_bn254::Fr>::new([9u8; 32]);
        let bundle = prover.prove_block(&bytecode, &block, None).unwrap();
        let fold_input = build_block_fold_input(&bundle);
        let folding_backend = NovaFoldingBackend::new(NovaFoldConfig {
            final_proof_backend_name: SPARTAN_FINAL_PROOF_SYSTEM_NAME,
            ..NovaFoldConfig::default()
        });
        let accumulator =
            build_block_fold_accumulator_with_backend(&[fold_input], &folding_backend).unwrap();
        let instance = build_final_folded_instance(&accumulator).unwrap();

        assert_eq!(
            assemble_spartan_final_proof(instance, Vec::new()).unwrap_err(),
            BlockTraceError::NovaFoldingBackendError {
                block_index: 0,
                reason: "Spartan final proof bytes must not be empty",
            }
        );
    }

    #[cfg(feature = "nova")]
    #[test]
    fn nova_final_folded_proof_backend_rejects_unsupported_proof_system() {
        let bytecode = BytecodePreprocessing::default();
        let block = trace_block(0, boundary(0, 0), boundary(4, 0));
        let prover = BlockProofBundleProver::<_, ark_bn254::Fr>::new([9u8; 32]);
        let bundle = prover.prove_block(&bytecode, &block, None).unwrap();
        let fold_input = build_block_fold_input(&bundle);
        let folding_backend = NovaFoldingBackend::default();
        let accumulator =
            build_block_fold_accumulator_with_backend(&[fold_input], &folding_backend).unwrap();
        let instance = build_final_folded_instance(&accumulator).unwrap();
        let proof_backend = SpartanPlaceholderFinalProofBackend;
        let mut proof = prove_final_folded_instance_with_backend(&proof_backend, instance).unwrap();
        proof.proof_system = "unsupported-final-proof-system";

        assert_eq!(
            verify_final_folded_proof_with_backend(&proof_backend, &accumulator, &proof)
                .unwrap_err(),
            BlockTraceError::NovaFoldingBackendError {
                block_index: 0,
                reason: "unsupported final folded proof system",
            }
        );
    }

    #[cfg(feature = "nova")]
    #[test]
    fn nova_spartan_final_instance_encoding_maps_public_input_and_witness() {
        let bytecode = BytecodePreprocessing::default();
        let block0 = trace_block(0, boundary(0, 0), boundary(2, 0));
        let block1 = trace_block(1, block0.end_state.clone(), boundary(4, 0));
        let prover = BlockProofBundleProver::<_, ark_bn254::Fr>::new([9u8; 32]);
        let bundles = prover
            .prove_blocks(&bytecode, &[block0.clone(), block1.clone()])
            .unwrap();
        let fold_inputs = build_block_fold_inputs(&bundles);
        let backend = NovaFoldingBackend::default();
        let accumulator =
            build_block_fold_accumulator_with_backend(&fold_inputs, &backend).unwrap();
        let instance = build_final_folded_instance(&accumulator).unwrap();

        let encoding = encode_final_folded_instance_for_spartan(&instance).unwrap();

        assert_eq!(encoding.version, SPARTAN_FINAL_INSTANCE_ENCODING_VERSION);
        assert!(!encoding.public_input_bytes.is_empty());
        assert!(!encoding.witness_bytes.is_empty());
        assert_eq!(
            encoding.public_input_digest,
            digest_spartan_final_encoding_component("public-inputs", &encoding.public_input_bytes)
        );
        assert_eq!(
            encoding.witness_digest,
            digest_spartan_final_encoding_component("witness", &encoding.witness_bytes)
        );
        assert_eq!(
            encoding.encoding_digest,
            digest_spartan_final_instance_encoding(
                encoding.public_input_digest,
                encoding.witness_digest
            )
        );
        verify_spartan_final_instance_encoding(&instance, &encoding).unwrap();

        let mut tampered_instance = instance.clone();
        tampered_instance.recursive_z_state[NOVA_SEMANTIC_ACCUMULATOR_INDEX][0] ^= 1;
        tampered_instance.instance_digest = digest_final_folded_instance(&tampered_instance);
        let tampered_encoding =
            encode_final_folded_instance_for_spartan(&tampered_instance).unwrap();

        assert_ne!(encoding.encoding_digest, tampered_encoding.encoding_digest);
    }

    #[cfg(feature = "nova")]
    #[test]
    fn nova_spartan_final_instance_encoding_verifier_rejects_tampering() {
        let bytecode = BytecodePreprocessing::default();
        let block = trace_block(0, boundary(0, 0), boundary(4, 0));
        let prover = BlockProofBundleProver::<_, ark_bn254::Fr>::new([9u8; 32]);
        let bundle = prover.prove_block(&bytecode, &block, None).unwrap();
        let fold_input = build_block_fold_input(&bundle);
        let backend = NovaFoldingBackend::default();
        let accumulator =
            build_block_fold_accumulator_with_backend(&[fold_input], &backend).unwrap();
        let instance = build_final_folded_instance(&accumulator).unwrap();
        let mut encoding = encode_final_folded_instance_for_spartan(&instance).unwrap();
        encoding.public_input_bytes[0] ^= 1;

        assert_eq!(
            verify_spartan_final_instance_encoding(&instance, &encoding).unwrap_err(),
            BlockTraceError::NovaFoldingBackendError {
                block_index: 0,
                reason: "Spartan final instance encoding mismatch",
            }
        );
    }

    #[cfg(feature = "nova")]
    #[test]
    fn nova_final_folded_instance_extracts_and_verifies_spartan_placeholder() {
        let bytecode = BytecodePreprocessing::default();
        let block0 = trace_block(0, boundary(0, 0), boundary(2, 0));
        let block1 = trace_block(1, block0.end_state.clone(), boundary(4, 0));
        let prover = BlockProofBundleProver::<_, ark_bn254::Fr>::new([9u8; 32]);
        let bundles = prover
            .prove_blocks(&bytecode, &[block0.clone(), block1.clone()])
            .unwrap();
        let fold_inputs = build_block_fold_inputs(&bundles);
        let backend = NovaFoldingBackend::default();
        let accumulator =
            build_block_fold_accumulator_with_backend(&fold_inputs, &backend).unwrap();

        let instance = build_final_folded_instance(&accumulator).unwrap();

        assert_eq!(instance.config, accumulator.config);
        assert_eq!(instance.metadata, accumulator.metadata);
        assert_eq!(
            instance.recursive_snark_output_digest,
            accumulator.recursive_snark_output_digest.unwrap()
        );
        assert_eq!(
            instance.recursive_z_state,
            accumulator.recursive_z_state.unwrap()
        );
        assert_eq!(
            instance.instance_digest,
            digest_final_folded_instance(&instance)
        );
        verify_final_folded_instance(&accumulator, &instance).unwrap();

        let proof = prove_spartan_placeholder_final_instance(instance.clone());
        assert_eq!(proof.proof_system, SPARTAN_PLACEHOLDER_PROOF_SYSTEM_NAME);
        assert_eq!(proof.instance, instance);
        assert!(proof.spartan_proof_bytes.is_none());
        let encoding = encode_final_folded_instance_for_spartan(&proof.instance).unwrap();
        assert_eq!(
            proof.spartan_encoding_digest,
            Some(encoding.encoding_digest)
        );
        assert_eq!(
            proof.proof_digest,
            digest_final_folded_proof(
                proof.proof_system,
                &proof.instance,
                proof.spartan_encoding_digest,
                None
            )
        );
        verify_spartan_placeholder_final_proof(&accumulator, &proof).unwrap();
    }

    #[cfg(feature = "nova")]
    #[test]
    fn nova_final_folded_instance_verifier_rejects_tampering() {
        let bytecode = BytecodePreprocessing::default();
        let block0 = trace_block(0, boundary(0, 0), boundary(2, 0));
        let block1 = trace_block(1, block0.end_state.clone(), boundary(4, 0));
        let prover = BlockProofBundleProver::<_, ark_bn254::Fr>::new([9u8; 32]);
        let bundles = prover
            .prove_blocks(&bytecode, &[block0.clone(), block1.clone()])
            .unwrap();
        let fold_inputs = build_block_fold_inputs(&bundles);
        let backend = NovaFoldingBackend::default();
        let accumulator =
            build_block_fold_accumulator_with_backend(&fold_inputs, &backend).unwrap();
        let mut instance = build_final_folded_instance(&accumulator).unwrap();
        instance.recursive_z_state[NOVA_SEMANTIC_ACCUMULATOR_INDEX][0] ^= 1;

        assert_eq!(
            verify_final_folded_instance(&accumulator, &instance).unwrap_err(),
            BlockTraceError::NovaFoldingBackendError {
                block_index: 1,
                reason: "final folded instance mismatch",
            }
        );
    }

    #[cfg(feature = "nova")]
    #[test]
    fn nova_spartan_placeholder_proof_verifier_rejects_tampering() {
        let bytecode = BytecodePreprocessing::default();
        let block = trace_block(0, boundary(0, 0), boundary(4, 0));
        let prover = BlockProofBundleProver::<_, ark_bn254::Fr>::new([9u8; 32]);
        let bundle = prover.prove_block(&bytecode, &block, None).unwrap();
        let fold_input = build_block_fold_input(&bundle);
        let backend = NovaFoldingBackend::default();
        let accumulator =
            build_block_fold_accumulator_with_backend(&[fold_input], &backend).unwrap();
        let instance = build_final_folded_instance(&accumulator).unwrap();
        let mut proof = prove_spartan_placeholder_final_instance(instance);
        proof.proof_digest[0] ^= 1;

        assert_eq!(
            verify_spartan_placeholder_final_proof(&accumulator, &proof).unwrap_err(),
            BlockTraceError::NovaFoldingBackendError {
                block_index: 0,
                reason: "final folded proof digest mismatch",
            }
        );
    }

    #[cfg(feature = "nova")]
    #[test]
    fn nova_spartan_placeholder_proof_verifier_rejects_encoding_digest_tampering() {
        let bytecode = BytecodePreprocessing::default();
        let block = trace_block(0, boundary(0, 0), boundary(4, 0));
        let prover = BlockProofBundleProver::<_, ark_bn254::Fr>::new([9u8; 32]);
        let bundle = prover.prove_block(&bytecode, &block, None).unwrap();
        let fold_input = build_block_fold_input(&bundle);
        let backend = NovaFoldingBackend::default();
        let accumulator =
            build_block_fold_accumulator_with_backend(&[fold_input], &backend).unwrap();
        let instance = build_final_folded_instance(&accumulator).unwrap();
        let mut proof = prove_spartan_placeholder_final_instance(instance);
        proof.spartan_encoding_digest.as_mut().unwrap()[0] ^= 1;

        assert_eq!(
            verify_spartan_placeholder_final_proof(&accumulator, &proof).unwrap_err(),
            BlockTraceError::NovaFoldingBackendError {
                block_index: 0,
                reason: "Spartan final instance encoding digest mismatch",
            }
        );
    }

    #[cfg(feature = "nova")]
    #[test]
    fn nova_phase6_baseline_summarizes_verified_placeholder_envelope() {
        let bytecode = BytecodePreprocessing::default();
        let block0 = trace_block(0, boundary(0, 0), boundary(2, 0));
        let block1 = trace_block(1, block0.end_state.clone(), boundary(4, 0));
        let prover = BlockProofBundleProver::<_, ark_bn254::Fr>::new([9u8; 32]);
        let bundles = prover
            .prove_blocks(&bytecode, &[block0.clone(), block1.clone()])
            .unwrap();
        let fold_inputs = build_block_fold_inputs(&bundles);
        let backend = NovaFoldingBackend::default();
        let accumulator =
            build_block_fold_accumulator_with_backend(&fold_inputs, &backend).unwrap();
        let instance = build_final_folded_instance(&accumulator).unwrap();
        let encoding = encode_final_folded_instance_for_spartan(&instance).unwrap();
        let proof = prove_spartan_placeholder_final_instance(instance.clone());

        let baseline = summarize_jolt_nova_phase6_baseline(&accumulator, Some(&proof)).unwrap();

        assert_eq!(baseline.absorbed_blocks, 2);
        assert_eq!(
            baseline.total_active_cycles,
            accumulator.metadata.total_active_cycles
        );
        assert_eq!(
            baseline.total_register_reads,
            accumulator.metadata.total_register_reads
        );
        assert_eq!(
            baseline.total_register_writes,
            accumulator.metadata.total_register_writes
        );
        assert_eq!(
            baseline.total_ram_accesses,
            accumulator.metadata.total_ram_accesses
        );
        assert_eq!(
            baseline.total_lookup_claims,
            accumulator.metadata.total_lookup_claims
        );
        assert_eq!(
            baseline.recursive_snark_bytes_len,
            accumulator.recursive_snark_bytes.as_ref().map(Vec::len)
        );
        assert_eq!(baseline.recursive_z_state_words, NOVA_Z_ARITY);
        assert_eq!(
            baseline.final_public_input_bytes_len,
            encoding.public_input_bytes.len()
        );
        assert_eq!(
            baseline.final_witness_bytes_len,
            encoding.witness_bytes.len()
        );
        assert_eq!(baseline.final_instance_digest, instance.instance_digest);
        assert_eq!(baseline.spartan_encoding_digest, encoding.encoding_digest);
        assert_eq!(
            baseline.final_proof_system,
            Some(SPARTAN_PLACEHOLDER_PROOF_SYSTEM_NAME)
        );
        assert_eq!(baseline.final_proof_digest, Some(proof.proof_digest));
        assert_eq!(baseline.final_proof_bytes_len, None);

        let final_proof_size_baseline =
            summarize_jolt_nova_final_proof_size_baseline(&accumulator, &proof).unwrap();
        assert_eq!(
            final_proof_size_baseline.configured_backend_name,
            SPARTAN_PLACEHOLDER_PROOF_SYSTEM_NAME
        );
        assert_eq!(
            final_proof_size_baseline.proof_system,
            SPARTAN_PLACEHOLDER_PROOF_SYSTEM_NAME
        );
        assert_eq!(final_proof_size_baseline.absorbed_blocks, 2);
        assert_eq!(
            final_proof_size_baseline.total_active_cycles,
            accumulator.metadata.total_active_cycles
        );
        assert_eq!(
            final_proof_size_baseline.recursive_snark_bytes_len,
            accumulator.recursive_snark_bytes.as_ref().map(Vec::len)
        );
        assert_eq!(
            final_proof_size_baseline.final_public_input_bytes_len,
            encoding.public_input_bytes.len()
        );
        assert_eq!(
            final_proof_size_baseline.final_witness_bytes_len,
            encoding.witness_bytes.len()
        );
        assert_eq!(final_proof_size_baseline.proof_payload_bytes_len, 0);
        assert!(final_proof_size_baseline.proof_envelope_bytes_len > 0);
        assert_eq!(
            final_proof_size_baseline.proof_total_bytes_len,
            final_proof_size_baseline.proof_envelope_bytes_len
        );
        assert_eq!(
            final_proof_size_baseline.final_instance_digest,
            instance.instance_digest
        );
        assert_eq!(
            final_proof_size_baseline.spartan_encoding_digest,
            encoding.encoding_digest
        );
        assert_eq!(final_proof_size_baseline.proof_digest, proof.proof_digest);
    }

    #[cfg(feature = "nova")]
    #[test]
    fn nova_final_proof_size_comparison_reports_placeholder_vs_spartan_for_multiblock_accumulator()
    {
        let bytecode = BytecodePreprocessing::default();
        let block0 = trace_block(0, boundary(0, 0), boundary(2, 0));
        let block1 = trace_block(1, block0.end_state.clone(), boundary(4, 0));
        let block2 = trace_block(2, block1.end_state.clone(), boundary(6, 0));
        let prover = BlockProofBundleProver::<_, ark_bn254::Fr>::new([9u8; 32]);
        let bundles = prover
            .prove_blocks(&bytecode, &[block0.clone(), block1.clone(), block2.clone()])
            .unwrap();
        let fold_inputs = build_block_fold_inputs(&bundles);
        let backend = NovaFoldingBackend::default();
        let accumulator =
            build_block_fold_accumulator_with_backend(&fold_inputs, &backend).unwrap();

        let comparison = summarize_jolt_nova_final_proof_size_comparison(&accumulator).unwrap();

        assert_eq!(
            comparison.folded_accumulator_digest,
            accumulator.metadata.accumulator_digest
        );
        assert_eq!(comparison.absorbed_blocks, 3);
        assert_eq!(
            comparison.total_active_cycles,
            accumulator.metadata.total_active_cycles
        );
        assert_eq!(
            comparison.recursive_snark_bytes_len,
            accumulator.recursive_snark_bytes.as_ref().map(Vec::len)
        );
        assert_eq!(
            comparison.placeholder.configured_backend_name,
            SPARTAN_PLACEHOLDER_PROOF_SYSTEM_NAME
        );
        assert_eq!(
            comparison.placeholder.proof_system,
            SPARTAN_PLACEHOLDER_PROOF_SYSTEM_NAME
        );
        assert_eq!(
            comparison.spartan.configured_backend_name,
            SPARTAN_FINAL_PROOF_SYSTEM_NAME
        );
        assert_eq!(
            comparison.spartan.proof_system,
            SPARTAN_FINAL_PROOF_SYSTEM_NAME
        );
        assert_eq!(
            comparison.placeholder.absorbed_blocks,
            comparison.spartan.absorbed_blocks
        );
        assert_eq!(
            comparison.placeholder.total_active_cycles,
            comparison.spartan.total_active_cycles
        );
        assert_eq!(comparison.placeholder.proof_payload_bytes_len, 0);
        assert!(comparison.placeholder.proof_envelope_bytes_len > 0);
        assert!(comparison.spartan.proof_payload_bytes_len > 0);
        assert_eq!(
            comparison.spartan_payload_extra_bytes,
            comparison.spartan.proof_payload_bytes_len
        );
        assert_eq!(
            comparison.spartan_total_extra_bytes,
            comparison.spartan.proof_total_bytes_len as i128
                - comparison.placeholder.proof_total_bytes_len as i128
        );
        assert!(
            comparison.spartan.proof_total_bytes_len > comparison.placeholder.proof_total_bytes_len
        );
        verify_block_fold_accumulator_with_backend(&fold_inputs, &accumulator, &backend).unwrap();
    }

    #[cfg(feature = "nova")]
    fn synthesize_nova_step_circuit_for_test(
        circuit: &JoltNovaStepCircuit,
    ) -> nova_snark::frontend::test_cs::TestConstraintSystem<NovaScalar> {
        use nova_snark::frontend::{
            num::AllocatedNum, test_cs::TestConstraintSystem, ConstraintSystem,
        };
        use nova_snark::traits::circuit::StepCircuit;

        let mut cs = TestConstraintSystem::<NovaScalar>::new();
        let z_values = nova_initial_input();
        let z = z_values
            .iter()
            .enumerate()
            .map(|(index, value)| {
                AllocatedNum::alloc(cs.namespace(|| format!("z_{index}")), || Ok(*value)).unwrap()
            })
            .collect::<Vec<_>>();

        let output = circuit.synthesize(&mut cs, &z).unwrap();
        assert_eq!(output.len(), NOVA_Z_ARITY);
        cs
    }

    #[cfg(feature = "nova")]
    #[test]
    fn nova_configured_logup_backend_satisfies_internal_balance_relation() {
        let bytecode = BytecodePreprocessing::default();
        let block = trace_block(0, boundary(0, 0), boundary(4, 0));
        let prover = BlockProofBundleProver::<_, ark_bn254::Fr>::new([9u8; 32]);
        let bundle = prover.prove_block(&bytecode, &block, None).unwrap();
        let fold_input = build_block_fold_input(&bundle);
        let statement = BlockFoldStatement::from_fold_input(&fold_input);
        let config = NovaFoldConfig {
            subclaim_backend_name: NOVA_LOGUP_SUBCLAIM_BACKEND_NAME,
            ..NovaFoldConfig::default()
        };
        let backend =
            nova_subclaim_backend_from_config(&config, fold_input.state.block_index).unwrap();
        let witness =
            JoltNovaStepWitness::from_fold_input_with_subclaim_backend(&fold_input, &backend);
        let circuit = nova_step_circuit_for_fold_input_with_subclaim_backend(&fold_input, &backend);
        let cs = synthesize_nova_step_circuit_for_test(&circuit);

        assert_eq!(backend.name(), NOVA_LOGUP_SUBCLAIM_BACKEND_NAME);
        assert_eq!(backend.lookup_backend_selector(), NovaScalar::from(1));
        assert_eq!(witness.lookup_delta(), statement.lookup_logup_fingerprint());
        assert_eq!(
            witness.lookup_logup_query_sum,
            witness.lookup_logup_table_sum
        );
        assert!(
            cs.is_satisfied(),
            "unsatisfied constraint: {:?}",
            cs.which_is_unsatisfied()
        );
    }

    #[cfg(feature = "nova")]
    #[test]
    fn nova_logup_backend_rejects_unbalanced_fractional_sum() {
        let bytecode = BytecodePreprocessing::default();
        let block = trace_block(0, boundary(0, 0), boundary(4, 0));
        let prover = BlockProofBundleProver::<_, ark_bn254::Fr>::new([9u8; 32]);
        let bundle = prover.prove_block(&bytecode, &block, None).unwrap();
        let fold_input = build_block_fold_input(&bundle);
        let mut statement = BlockFoldStatement::from_fold_input(&fold_input);
        statement.lookup_logup_query_sum += NovaScalar::from(1);
        let witness = JoltNovaStepWitness::from_statement_with_subclaim_backend(
            statement,
            &LogUpSubclaimFoldingBackend,
        );
        let circuit = JoltNovaStepCircuit { witness };
        let cs = synthesize_nova_step_circuit_for_test(&circuit);

        assert_eq!(
            cs.which_is_unsatisfied(),
            Some("LogUp query and table sums balance")
        );
    }

    #[cfg(feature = "nova")]
    #[test]
    fn nova_lookup_backend_selector_rejects_non_boolean_value() {
        let bytecode = BytecodePreprocessing::default();
        let block = trace_block(0, boundary(0, 0), boundary(4, 0));
        let prover = BlockProofBundleProver::<_, ark_bn254::Fr>::new([9u8; 32]);
        let bundle = prover.prove_block(&bytecode, &block, None).unwrap();
        let fold_input = build_block_fold_input(&bundle);
        let mut witness = JoltNovaStepWitness::from_fold_input_with_subclaim_backend(
            &fold_input,
            &LogUpSubclaimFoldingBackend,
        );
        witness.lookup_backend_selector = NovaScalar::from(2);
        let circuit = JoltNovaStepCircuit { witness };
        let cs = synthesize_nova_step_circuit_for_test(&circuit);

        assert_eq!(
            cs.which_is_unsatisfied(),
            Some("lookup backend selector is boolean")
        );
    }

    #[cfg(feature = "nova")]
    #[test]
    fn nova_step_circuit_satisfies_statement_digest_binding() {
        let bytecode = BytecodePreprocessing::default();
        let block = trace_block(0, boundary(0, 0), boundary(4, 0));
        let prover = BlockProofBundleProver::<_, ark_bn254::Fr>::new([9u8; 32]);
        let bundle = prover.prove_block(&bytecode, &block, None).unwrap();
        let fold_input = build_block_fold_input(&bundle);
        let circuit = nova_step_circuit_for_fold_input(&fold_input);
        let cs = synthesize_nova_step_circuit_for_test(&circuit);

        assert!(
            cs.is_satisfied(),
            "unsatisfied constraint: {:?}",
            cs.which_is_unsatisfied()
        );
    }

    #[cfg(feature = "nova")]
    #[test]
    fn nova_step_circuit_satisfies_register_fingerprint_binding() {
        let bytecode = BytecodePreprocessing::default();
        let block = trace_block(0, boundary(0, 0), boundary(4, 0));
        let prover = BlockProofBundleProver::<_, ark_bn254::Fr>::new([9u8; 32]);
        let bundle = prover.prove_block(&bytecode, &block, None).unwrap();
        let fold_input = build_block_fold_input(&bundle);
        let circuit = nova_step_circuit_for_fold_input(&fold_input);
        let cs = synthesize_nova_step_circuit_for_test(&circuit);

        assert!(
            cs.is_satisfied(),
            "unsatisfied constraint: {:?}",
            cs.which_is_unsatisfied()
        );
    }

    #[cfg(feature = "nova")]
    #[test]
    fn nova_step_circuit_satisfies_ram_fingerprint_binding() {
        let bytecode = BytecodePreprocessing::default();
        let block = trace_block(0, boundary(0, 0), boundary(4, 0));
        let prover = BlockProofBundleProver::<_, ark_bn254::Fr>::new([9u8; 32]);
        let bundle = prover.prove_block(&bytecode, &block, None).unwrap();
        let fold_input = build_block_fold_input(&bundle);
        let circuit = nova_step_circuit_for_fold_input(&fold_input);
        let cs = synthesize_nova_step_circuit_for_test(&circuit);

        assert!(
            cs.is_satisfied(),
            "unsatisfied constraint: {:?}",
            cs.which_is_unsatisfied()
        );
    }

    #[cfg(feature = "nova")]
    #[test]
    fn nova_step_circuit_satisfies_lookup_fingerprint_binding() {
        let bytecode = BytecodePreprocessing::default();
        let block = trace_block(0, boundary(0, 0), boundary(4, 0));
        let prover = BlockProofBundleProver::<_, ark_bn254::Fr>::new([9u8; 32]);
        let bundle = prover.prove_block(&bytecode, &block, None).unwrap();
        let fold_input = build_block_fold_input(&bundle);
        let circuit = nova_step_circuit_for_fold_input(&fold_input);
        let cs = synthesize_nova_step_circuit_for_test(&circuit);

        assert!(
            cs.is_satisfied(),
            "unsatisfied constraint: {:?}",
            cs.which_is_unsatisfied()
        );
    }

    #[cfg(feature = "nova")]
    #[test]
    fn nova_step_circuit_satisfies_cpu_fingerprint_binding() {
        let bytecode = BytecodePreprocessing::default();
        let block = trace_block(0, boundary(0, 0), boundary(4, 0));
        let prover = BlockProofBundleProver::<_, ark_bn254::Fr>::new([9u8; 32]);
        let bundle = prover.prove_block(&bytecode, &block, None).unwrap();
        let fold_input = build_block_fold_input(&bundle);
        let circuit = nova_step_circuit_for_fold_input(&fold_input);
        let cs = synthesize_nova_step_circuit_for_test(&circuit);

        assert!(
            cs.is_satisfied(),
            "unsatisfied constraint: {:?}",
            cs.which_is_unsatisfied()
        );
    }

    #[cfg(feature = "nova")]
    #[test]
    fn nova_step_circuit_rejects_tampered_statement_digest_witness() {
        let bytecode = BytecodePreprocessing::default();
        let block = trace_block(0, boundary(0, 0), boundary(4, 0));
        let prover = BlockProofBundleProver::<_, ark_bn254::Fr>::new([9u8; 32]);
        let bundle = prover.prove_block(&bytecode, &block, None).unwrap();
        let fold_input = build_block_fold_input(&bundle);
        let mut circuit = nova_step_circuit_for_fold_input(&fold_input);
        circuit.witness.statement_digest = circuit.witness.statement_digest + NovaScalar::from(1);

        let cs = synthesize_nova_step_circuit_for_test(&circuit);

        assert_eq!(
            cs.which_is_unsatisfied(),
            Some("statement digest binds statement fields")
        );
    }

    #[cfg(feature = "nova")]
    #[test]
    fn nova_step_circuit_rejects_tampered_register_fingerprint_witness() {
        let bytecode = BytecodePreprocessing::default();
        let block = trace_block(0, boundary(0, 0), boundary(4, 0));
        let prover = BlockProofBundleProver::<_, ark_bn254::Fr>::new([9u8; 32]);
        let bundle = prover.prove_block(&bytecode, &block, None).unwrap();
        let fold_input = build_block_fold_input(&bundle);
        let mut circuit = nova_step_circuit_for_fold_input(&fold_input);
        circuit.witness.register_claim_fingerprint =
            circuit.witness.register_claim_fingerprint + NovaScalar::from(1);

        let cs = synthesize_nova_step_circuit_for_test(&circuit);

        assert_eq!(
            cs.which_is_unsatisfied(),
            Some("register claim fingerprint binds register fields")
        );
    }

    #[cfg(feature = "nova")]
    #[test]
    fn nova_step_circuit_rejects_tampered_ram_fingerprint_witness() {
        let bytecode = BytecodePreprocessing::default();
        let block = trace_block(0, boundary(0, 0), boundary(4, 0));
        let prover = BlockProofBundleProver::<_, ark_bn254::Fr>::new([9u8; 32]);
        let bundle = prover.prove_block(&bytecode, &block, None).unwrap();
        let fold_input = build_block_fold_input(&bundle);
        let mut circuit = nova_step_circuit_for_fold_input(&fold_input);
        circuit.witness.ram_claim_fingerprint =
            circuit.witness.ram_claim_fingerprint + NovaScalar::from(1);

        let cs = synthesize_nova_step_circuit_for_test(&circuit);

        assert_eq!(
            cs.which_is_unsatisfied(),
            Some("ram claim fingerprint binds RAM fields")
        );
    }

    #[cfg(feature = "nova")]
    #[test]
    fn nova_step_circuit_rejects_tampered_lookup_fingerprint_witness() {
        let bytecode = BytecodePreprocessing::default();
        let block = trace_block(0, boundary(0, 0), boundary(4, 0));
        let prover = BlockProofBundleProver::<_, ark_bn254::Fr>::new([9u8; 32]);
        let bundle = prover.prove_block(&bytecode, &block, None).unwrap();
        let fold_input = build_block_fold_input(&bundle);
        let mut circuit = nova_step_circuit_for_fold_input(&fold_input);
        circuit.witness.lookup_claim_fingerprint =
            circuit.witness.lookup_claim_fingerprint + NovaScalar::from(1);

        let cs = synthesize_nova_step_circuit_for_test(&circuit);

        assert_eq!(
            cs.which_is_unsatisfied(),
            Some("lookup claim fingerprint binds lookup fields")
        );
    }

    #[cfg(feature = "nova")]
    #[test]
    fn nova_step_circuit_rejects_tampered_cpu_fingerprint_witness() {
        let bytecode = BytecodePreprocessing::default();
        let block = trace_block(0, boundary(0, 0), boundary(4, 0));
        let prover = BlockProofBundleProver::<_, ark_bn254::Fr>::new([9u8; 32]);
        let bundle = prover.prove_block(&bytecode, &block, None).unwrap();
        let fold_input = build_block_fold_input(&bundle);
        let mut circuit = nova_step_circuit_for_fold_input(&fold_input);
        circuit.witness.cpu_claim_fingerprint =
            circuit.witness.cpu_claim_fingerprint + NovaScalar::from(1);

        let cs = synthesize_nova_step_circuit_for_test(&circuit);

        assert_eq!(
            cs.which_is_unsatisfied(),
            Some("CPU R1CS claim fingerprint binds CPU fields")
        );
    }

    #[cfg(feature = "nova")]
    #[test]
    fn nova_public_params_are_cached() {
        let first = nova_public_params(0).unwrap();
        let second = nova_public_params(1).unwrap();

        assert!(std::ptr::eq(first, second));
    }

    #[cfg(feature = "nova")]
    #[test]
    fn nova_folding_backend_verifies_empty_accumulator() {
        let backend = NovaFoldingBackend::default();
        let accumulator =
            <NovaFoldingBackend as BlockFoldingBackend<[u8; 32], ark_bn254::Fr>>::new_accumulator(
                &backend,
            );
        let fold_inputs = Vec::<BlockFoldInput<[u8; 32], ark_bn254::Fr>>::new();

        verify_block_fold_accumulator_with_backend(&fold_inputs, &accumulator, &backend).unwrap();
    }

    #[cfg(feature = "nova")]
    #[test]
    fn nova_folding_backend_rejects_empty_accumulator_with_recursive_state() {
        let backend = NovaFoldingBackend::default();
        let mut accumulator = <NovaFoldingBackend as BlockFoldingBackend<
            [u8; 32],
            ark_bn254::Fr,
        >>::new_accumulator(&backend);
        accumulator.recursive_z_state = Some(nova_initial_z_state_storage());
        let fold_inputs = Vec::<BlockFoldInput<[u8; 32], ark_bn254::Fr>>::new();

        assert!(matches!(
            verify_block_fold_accumulator_with_backend(&fold_inputs, &accumulator, &backend)
                .unwrap_err(),
            BlockTraceError::NovaFoldingBackendError {
                block_index: 0,
                reason: "empty Nova accumulator must not contain recursive state",
            }
        ));
    }

    #[cfg(feature = "nova")]
    #[test]
    fn nova_step_delta_vector_tracks_structured_accumulators() {
        let bytecode = BytecodePreprocessing::default();
        let block = trace_block(0, boundary(0, 0), boundary(4, 0));
        let prover = BlockProofBundleProver::<_, ark_bn254::Fr>::new([9u8; 32]);
        let bundle = prover.prove_block(&bytecode, &block, None).unwrap();
        let fold_input = build_block_fold_input(&bundle);
        let witness = JoltNovaStepWitness::from_fold_input(&fold_input);
        let delta = nova_step_delta_vector(&witness);

        assert_eq!(
            delta[NOVA_SEMANTIC_ACCUMULATOR_INDEX],
            witness.semantic_delta()
        );
        assert_eq!(delta[NOVA_NEXT_BLOCK_INDEX_INDEX], NovaScalar::from(1));
        assert_eq!(delta[NOVA_TOTAL_ACTIVE_CYCLES_INDEX], witness.active_cycles);
        assert_eq!(
            delta[NOVA_REGISTER_ACCUMULATOR_INDEX],
            witness.register_delta()
        );
        assert_eq!(delta[NOVA_RAM_ACCUMULATOR_INDEX], witness.ram_delta());
        assert_eq!(delta[NOVA_LOOKUP_ACCUMULATOR_INDEX], witness.lookup_delta());
        assert_eq!(delta[NOVA_CPU_ACCUMULATOR_INDEX], witness.cpu_delta());
    }

    #[cfg(not(feature = "nova"))]
    #[test]
    fn nova_folding_backend_rejects_without_nova_feature() {
        let bytecode = BytecodePreprocessing::default();
        let block = trace_block(0, boundary(0, 0), boundary(4, 0));
        let prover = BlockProofBundleProver::<_, ark_bn254::Fr>::new([9u8; 32]);
        let bundle = prover.prove_block(&bytecode, &block, None).unwrap();
        let fold_input = build_block_fold_input(&bundle);
        let backend = NovaFoldingBackend::default();

        assert_eq!(
            build_block_fold_accumulator_with_backend(&[fold_input], &backend).unwrap_err(),
            BlockTraceError::NovaFoldingBackendUnavailable {
                block_index: 0,
                reason: "compile jolt-core with the `nova` feature to enable Nova folding",
            }
        );
    }

    #[cfg(not(feature = "nova"))]
    #[test]
    fn block_proof_pipeline_with_nova_backend_fails_without_nova_feature() {
        let bytecode = BytecodePreprocessing::default();
        let block = trace_block(0, boundary(0, 0), boundary(4, 0));
        let pipeline = BlockProofPipeline::<_, ark_bn254::Fr, NovaFoldingBackend>::with_backend(
            [9u8; 32],
            NovaFoldingBackend::default(),
        );

        assert_eq!(
            pipeline.prove_blocks(&bytecode, &[block]).unwrap_err(),
            BlockTraceError::NovaFoldingBackendUnavailable {
                block_index: 0,
                reason: "compile jolt-core with the `nova` feature to enable Nova folding",
            }
        );
    }

    #[cfg(feature = "nova")]
    #[test]
    fn nova_folding_backend_absorbs_and_verifies_recursive_snark_steps() {
        let bytecode = BytecodePreprocessing::default();
        let block0 = trace_block(0, boundary(0, 0), boundary(2, 0));
        let block1 = trace_block(1, block0.end_state.clone(), boundary(4, 0));
        let prover = BlockProofBundleProver::<_, ark_bn254::Fr>::new([9u8; 32]);
        let bundles = prover
            .prove_blocks(&bytecode, &[block0.clone(), block1.clone()])
            .unwrap();
        let fold_inputs = build_block_fold_inputs(&bundles);
        let backend = NovaFoldingBackend::default();

        let accumulator =
            build_block_fold_accumulator_with_backend(&fold_inputs, &backend).unwrap();

        assert_eq!(accumulator.metadata.absorbed_blocks, 2);
        assert_eq!(accumulator.metadata.program_digest, Some([9u8; 32]));
        assert_eq!(
            accumulator.metadata.latest_state_digest,
            Some(fold_inputs[1].state.state_digest)
        );
        assert!(accumulator
            .recursive_snark_bytes
            .as_ref()
            .is_some_and(|bytes| !bytes.is_empty()));
        assert!(accumulator.recursive_snark_output_digest.is_some());
        let expected_z_state = nova_expected_z_state(&fold_inputs);
        assert_eq!(
            accumulator.recursive_z_state,
            Some(nova_z_state_to_storage(expected_z_state))
        );
        assert_eq!(
            expected_z_state[NOVA_NEXT_BLOCK_INDEX_INDEX],
            NovaScalar::from(2)
        );
        assert_eq!(
            expected_z_state[NOVA_TOTAL_ACTIVE_CYCLES_INDEX],
            NovaScalar::from(4)
        );
        assert_ne!(
            expected_z_state[NOVA_REGISTER_ACCUMULATOR_INDEX],
            NovaScalar::zero()
        );
        assert_ne!(
            expected_z_state[NOVA_RAM_ACCUMULATOR_INDEX],
            NovaScalar::zero()
        );
        assert_ne!(
            expected_z_state[NOVA_LOOKUP_ACCUMULATOR_INDEX],
            NovaScalar::zero()
        );
        assert_ne!(
            expected_z_state[NOVA_CPU_ACCUMULATOR_INDEX],
            NovaScalar::zero()
        );

        let recursive_snark = postcard::from_bytes::<NovaRecursiveSnark>(
            accumulator.recursive_snark_bytes.as_deref().unwrap(),
        )
        .unwrap();
        let output = recursive_snark
            .verify(
                nova_public_params(0).unwrap(),
                accumulator.metadata.absorbed_blocks,
                &nova_initial_input(),
            )
            .unwrap();
        assert_eq!(output.as_slice(), expected_z_state.as_slice());

        verify_block_fold_accumulator_with_backend(&fold_inputs, &accumulator, &backend).unwrap();
    }

    #[cfg(feature = "nova")]
    #[test]
    fn nova_recursive_lifecycle_verifies_each_absorbed_step() {
        let bytecode = BytecodePreprocessing::default();
        let block0 = trace_block(0, boundary(0, 0), boundary(2, 0));
        let block1 = trace_block(1, block0.end_state.clone(), boundary(4, 0));
        let prover = BlockProofBundleProver::<_, ark_bn254::Fr>::new([9u8; 32]);
        let bundles = prover.prove_blocks(&bytecode, &[block0, block1]).unwrap();
        let fold_inputs = build_block_fold_inputs(&bundles);
        let backend = NovaFoldingBackend::default();
        let mut accumulator = <NovaFoldingBackend as BlockFoldingBackend<
            [u8; 32],
            ark_bn254::Fr,
        >>::new_accumulator(&backend);

        <NovaFoldingBackend as BlockFoldingBackend<[u8; 32], ark_bn254::Fr>>::absorb(
            &backend,
            &mut accumulator,
            &fold_inputs[0],
        )
        .unwrap();
        let first_z_state = nova_expected_z_state(&fold_inputs[..1]);
        assert_eq!(accumulator.metadata.absorbed_blocks, 1);
        assert_eq!(
            accumulator.recursive_z_state,
            Some(nova_z_state_to_storage(first_z_state))
        );
        verify_block_fold_accumulator_with_backend(&fold_inputs[..1], &accumulator, &backend)
            .unwrap();

        <NovaFoldingBackend as BlockFoldingBackend<[u8; 32], ark_bn254::Fr>>::absorb(
            &backend,
            &mut accumulator,
            &fold_inputs[1],
        )
        .unwrap();
        let second_z_state = nova_expected_z_state(&fold_inputs);
        assert_eq!(accumulator.metadata.absorbed_blocks, 2);
        assert_ne!(first_z_state, second_z_state);
        assert_eq!(
            accumulator.recursive_z_state,
            Some(nova_z_state_to_storage(second_z_state))
        );
        verify_block_fold_accumulator_with_backend(&fold_inputs, &accumulator, &backend).unwrap();
    }

    #[cfg(feature = "nova")]
    #[test]
    fn nova_folding_backend_verifier_rejects_tampered_recursive_snark() {
        let bytecode = BytecodePreprocessing::default();
        let block = trace_block(0, boundary(0, 0), boundary(4, 0));
        let prover = BlockProofBundleProver::<_, ark_bn254::Fr>::new([9u8; 32]);
        let bundle = prover.prove_block(&bytecode, &block, None).unwrap();
        let fold_inputs = vec![build_block_fold_input(&bundle)];
        let backend = NovaFoldingBackend::default();
        let mut accumulator =
            build_block_fold_accumulator_with_backend(&fold_inputs, &backend).unwrap();
        accumulator.recursive_snark_bytes = Some(Vec::new());

        assert!(matches!(
            verify_block_fold_accumulator_with_backend(&fold_inputs, &accumulator, &backend)
                .unwrap_err(),
            BlockTraceError::NovaFoldingBackendError {
                block_index: 0,
                reason: "Nova recursive SNARK deserialization failed",
            }
        ));
    }

    #[cfg(feature = "nova")]
    #[test]
    fn nova_folding_backend_verifier_rejects_tampered_output_digest() {
        let bytecode = BytecodePreprocessing::default();
        let block = trace_block(0, boundary(0, 0), boundary(4, 0));
        let prover = BlockProofBundleProver::<_, ark_bn254::Fr>::new([9u8; 32]);
        let bundle = prover.prove_block(&bytecode, &block, None).unwrap();
        let fold_inputs = vec![build_block_fold_input(&bundle)];
        let backend = NovaFoldingBackend::default();
        let mut accumulator =
            build_block_fold_accumulator_with_backend(&fold_inputs, &backend).unwrap();
        accumulator.recursive_snark_output_digest.as_mut().unwrap()[0] ^= 1;

        assert!(matches!(
            verify_block_fold_accumulator_with_backend(&fold_inputs, &accumulator, &backend)
                .unwrap_err(),
            BlockTraceError::NovaFoldingBackendError {
                block_index: 0,
                reason: "Nova recursive SNARK output digest mismatch",
            }
        ));
    }

    #[cfg(feature = "nova")]
    #[test]
    fn nova_folding_backend_verifier_rejects_tampered_z_state() {
        let bytecode = BytecodePreprocessing::default();
        let block = trace_block(0, boundary(0, 0), boundary(4, 0));
        let prover = BlockProofBundleProver::<_, ark_bn254::Fr>::new([9u8; 32]);
        let bundle = prover.prove_block(&bytecode, &block, None).unwrap();
        let fold_inputs = vec![build_block_fold_input(&bundle)];
        let backend = NovaFoldingBackend::default();
        let mut accumulator =
            build_block_fold_accumulator_with_backend(&fold_inputs, &backend).unwrap();
        accumulator.recursive_z_state.as_mut().unwrap()[NOVA_SEMANTIC_ACCUMULATOR_INDEX][0] ^= 1;

        assert!(matches!(
            verify_block_fold_accumulator_with_backend(&fold_inputs, &accumulator, &backend)
                .unwrap_err(),
            BlockTraceError::NovaFoldingBackendError {
                block_index: 0,
                reason: "Nova recursive z-state mismatch",
            }
        ));
    }

    #[cfg(feature = "nova")]
    #[test]
    fn nova_folding_backend_verifier_rejects_tampered_register_accumulator() {
        let bytecode = BytecodePreprocessing::default();
        let block = trace_block(0, boundary(0, 0), boundary(4, 0));
        let prover = BlockProofBundleProver::<_, ark_bn254::Fr>::new([9u8; 32]);
        let bundle = prover.prove_block(&bytecode, &block, None).unwrap();
        let fold_inputs = vec![build_block_fold_input(&bundle)];
        let backend = NovaFoldingBackend::default();
        let mut accumulator =
            build_block_fold_accumulator_with_backend(&fold_inputs, &backend).unwrap();
        accumulator.recursive_z_state.as_mut().unwrap()[NOVA_REGISTER_ACCUMULATOR_INDEX][0] ^= 1;

        assert!(matches!(
            verify_block_fold_accumulator_with_backend(&fold_inputs, &accumulator, &backend)
                .unwrap_err(),
            BlockTraceError::NovaFoldingBackendError {
                block_index: 0,
                reason: "Nova recursive z-state mismatch",
            }
        ));
    }

    #[cfg(feature = "nova")]
    #[test]
    fn nova_folding_backend_verifier_rejects_tampered_ram_accumulator() {
        let bytecode = BytecodePreprocessing::default();
        let block = trace_block(0, boundary(0, 0), boundary(4, 0));
        let prover = BlockProofBundleProver::<_, ark_bn254::Fr>::new([9u8; 32]);
        let bundle = prover.prove_block(&bytecode, &block, None).unwrap();
        let fold_inputs = vec![build_block_fold_input(&bundle)];
        let backend = NovaFoldingBackend::default();
        let mut accumulator =
            build_block_fold_accumulator_with_backend(&fold_inputs, &backend).unwrap();
        accumulator.recursive_z_state.as_mut().unwrap()[NOVA_RAM_ACCUMULATOR_INDEX][0] ^= 1;

        assert!(matches!(
            verify_block_fold_accumulator_with_backend(&fold_inputs, &accumulator, &backend)
                .unwrap_err(),
            BlockTraceError::NovaFoldingBackendError {
                block_index: 0,
                reason: "Nova recursive z-state mismatch",
            }
        ));
    }

    #[cfg(feature = "nova")]
    #[test]
    fn nova_folding_backend_verifier_rejects_tampered_lookup_accumulator() {
        let bytecode = BytecodePreprocessing::default();
        let block = trace_block(0, boundary(0, 0), boundary(4, 0));
        let prover = BlockProofBundleProver::<_, ark_bn254::Fr>::new([9u8; 32]);
        let bundle = prover.prove_block(&bytecode, &block, None).unwrap();
        let fold_inputs = vec![build_block_fold_input(&bundle)];
        let backend = NovaFoldingBackend::default();
        let mut accumulator =
            build_block_fold_accumulator_with_backend(&fold_inputs, &backend).unwrap();
        accumulator.recursive_z_state.as_mut().unwrap()[NOVA_LOOKUP_ACCUMULATOR_INDEX][0] ^= 1;

        assert!(matches!(
            verify_block_fold_accumulator_with_backend(&fold_inputs, &accumulator, &backend)
                .unwrap_err(),
            BlockTraceError::NovaFoldingBackendError {
                block_index: 0,
                reason: "Nova recursive z-state mismatch",
            }
        ));
    }

    #[cfg(feature = "nova")]
    #[test]
    fn nova_folding_backend_verifier_rejects_tampered_cpu_accumulator() {
        let bytecode = BytecodePreprocessing::default();
        let block = trace_block(0, boundary(0, 0), boundary(4, 0));
        let prover = BlockProofBundleProver::<_, ark_bn254::Fr>::new([9u8; 32]);
        let bundle = prover.prove_block(&bytecode, &block, None).unwrap();
        let fold_inputs = vec![build_block_fold_input(&bundle)];
        let backend = NovaFoldingBackend::default();
        let mut accumulator =
            build_block_fold_accumulator_with_backend(&fold_inputs, &backend).unwrap();
        accumulator.recursive_z_state.as_mut().unwrap()[NOVA_CPU_ACCUMULATOR_INDEX][0] ^= 1;

        assert!(matches!(
            verify_block_fold_accumulator_with_backend(&fold_inputs, &accumulator, &backend)
                .unwrap_err(),
            BlockTraceError::NovaFoldingBackendError {
                block_index: 0,
                reason: "Nova recursive z-state mismatch",
            }
        ));
    }

    #[cfg(feature = "nova")]
    #[test]
    fn nova_folding_backend_verifier_rejects_tampered_metadata() {
        let bytecode = BytecodePreprocessing::default();
        let block = trace_block(0, boundary(0, 0), boundary(4, 0));
        let prover = BlockProofBundleProver::<_, ark_bn254::Fr>::new([9u8; 32]);
        let bundle = prover.prove_block(&bytecode, &block, None).unwrap();
        let fold_inputs = vec![build_block_fold_input(&bundle)];
        let backend = NovaFoldingBackend::default();
        let mut accumulator =
            build_block_fold_accumulator_with_backend(&fold_inputs, &backend).unwrap();
        accumulator.metadata.accumulator_digest[0] ^= 1;

        assert_eq!(
            verify_block_fold_accumulator_with_backend(&fold_inputs, &accumulator, &backend)
                .unwrap_err(),
            BlockTraceError::BlockFoldAccumulatorMismatch {
                expected_blocks: 1,
                actual_blocks: 1,
            }
        );
    }

    #[cfg(feature = "nova")]
    #[test]
    fn block_proof_pipeline_with_nova_backend_proves_and_verifies_blocks() {
        let bytecode = BytecodePreprocessing::default();
        let block0 = trace_block(0, boundary(0, 0), boundary(2, 0));
        let block1 = trace_block(1, block0.end_state.clone(), boundary(4, 0));
        let pipeline = BlockProofPipeline::<_, ark_bn254::Fr, NovaFoldingBackend>::with_backend(
            [9u8; 32],
            NovaFoldingBackend::default(),
        );

        let output = pipeline
            .prove_blocks(&bytecode, &[block0.clone(), block1.clone()])
            .unwrap();

        assert_eq!(output.bundles.len(), 2);
        assert_eq!(output.fold_inputs.len(), 2);
        assert_eq!(output.accumulator.metadata.absorbed_blocks, 2);
        assert!(output.accumulator.recursive_snark_bytes.is_some());
        assert!(output.final_proof.is_none());
        verify_block_proof_pipeline_with_backend(
            &bytecode,
            &[block0, block1],
            &output,
            &NovaFoldingBackend::default(),
        )
        .unwrap();
    }

    #[cfg(feature = "nova")]
    #[test]
    fn block_proof_pipeline_with_logup_backend_proves_and_verifies_blocks() {
        let bytecode = BytecodePreprocessing::default();
        let block0 = trace_block(0, boundary(0, 0), boundary(2, 0));
        let block1 = trace_block(1, block0.end_state.clone(), boundary(4, 0));
        let backend = NovaFoldingBackend::new(NovaFoldConfig {
            subclaim_backend_name: NOVA_LOGUP_SUBCLAIM_BACKEND_NAME,
            ..NovaFoldConfig::default()
        });
        let pipeline = BlockProofPipeline::<_, ark_bn254::Fr, NovaFoldingBackend>::with_backend(
            [9u8; 32],
            backend.clone(),
        );

        let output = pipeline
            .prove_blocks(&bytecode, &[block0.clone(), block1.clone()])
            .unwrap();

        assert_eq!(
            output.accumulator.config.subclaim_backend_name,
            NOVA_LOGUP_SUBCLAIM_BACKEND_NAME
        );
        assert!(output
            .bundles
            .iter()
            .all(|bundle| bundle.lookup_claim.logup_proof.query_sum
                == bundle.lookup_claim.logup_proof.table_sum));
        verify_block_proof_pipeline_with_backend(&bytecode, &[block0, block1], &output, &backend)
            .unwrap();
    }

    #[test]
    fn block_proof_pipeline_proves_and_verifies_blocks() {
        let bytecode = BytecodePreprocessing::default();
        let block0 = trace_block(0, boundary(0, 0), boundary(2, 0));
        let block1 = trace_block(1, block0.end_state.clone(), boundary(4, 0));
        let pipeline = BlockProofPipeline::<_, ark_bn254::Fr>::new([9u8; 32]);

        let output = pipeline
            .prove_blocks(&bytecode, &[block0.clone(), block1.clone()])
            .unwrap();

        assert_eq!(output.bundles.len(), 2);
        assert_eq!(output.fold_inputs.len(), 2);
        assert_eq!(output.accumulator.absorbed_blocks, 2);
        assert_eq!(output.accumulator.total_active_cycles, 4);
        assert_eq!(output.accumulator.total_lookup_claims, 4);
        assert!(output.final_proof.is_none());
        verify_block_proof_pipeline(&bytecode, &[block0, block1], &output).unwrap();
    }

    #[test]
    fn block_proof_pipeline_proves_and_verifies_partial_prefix_with_external_lookahead() {
        let bytecode = BytecodePreprocessing::default();
        let block0 = trace_block(0, boundary(0, 0), boundary(2, 0));
        let block1 = trace_block(1, block0.end_state.clone(), boundary(4, 0));
        let external_lookahead = block1.cycles.first().unwrap();
        let pipeline = BlockProofPipeline::<_, ark_bn254::Fr>::new([9u8; 32]);

        let output = pipeline
            .prove_blocks_with_external_lookahead(
                &bytecode,
                std::slice::from_ref(&block0),
                Some(external_lookahead),
            )
            .unwrap();

        assert_eq!(output.bundles.len(), 1);
        assert!(output.bundles[0]
            .cpu_proof
            .inner_proof
            .lookahead_cycle_digest
            .is_some());
        assert_eq!(
            output.fold_inputs[0].state.lookahead_cycle_digest,
            output.bundles[0]
                .cpu_proof
                .inner_proof
                .lookahead_cycle_digest
        );
        verify_block_proof_pipeline_with_external_lookahead(
            &bytecode,
            std::slice::from_ref(&block0),
            Some(external_lookahead),
            &output,
        )
        .unwrap();
        assert!(verify_block_proof_pipeline(&bytecode, &[block0], &output).is_err());
    }

    #[cfg(feature = "nova")]
    #[test]
    fn nova_block_proof_pipeline_with_placeholder_final_proof_proves_and_verifies() {
        let bytecode = BytecodePreprocessing::default();
        let block0 = trace_block(0, boundary(0, 0), boundary(2, 0));
        let block1 = trace_block(1, block0.end_state.clone(), boundary(4, 0));
        let backend = NovaFoldingBackend::default();
        let pipeline = BlockProofPipeline::<_, ark_bn254::Fr, NovaFoldingBackend>::with_backend(
            [9u8; 32],
            backend.clone(),
        );

        let output = pipeline
            .prove_blocks_with_final_proof(&bytecode, &[block0.clone(), block1.clone()])
            .unwrap();
        let final_proof = output.final_proof.as_ref().unwrap();

        assert_eq!(output.accumulator.metadata.absorbed_blocks, 2);
        assert_eq!(
            final_proof.proof_system,
            SPARTAN_PLACEHOLDER_PROOF_SYSTEM_NAME
        );
        assert!(final_proof.spartan_proof_bytes.is_none());
        verify_nova_block_proof_pipeline_with_final_proof(
            &bytecode,
            &[block0, block1],
            &output,
            &backend,
        )
        .unwrap();
    }

    #[cfg(feature = "nova")]
    #[test]
    fn nova_final_proof_for_partial_prefix_binds_external_lookahead() {
        let bytecode = BytecodePreprocessing::default();
        let block0 = trace_block(0, boundary(0, 0), boundary(2, 0));
        let block1 = trace_block(1, block0.end_state.clone(), boundary(4, 0));
        let external_lookahead = block1.cycles.first().unwrap();
        let backend = NovaFoldingBackend::default();
        let pipeline = BlockProofPipeline::<_, ark_bn254::Fr, NovaFoldingBackend>::with_backend(
            [9u8; 32],
            backend.clone(),
        );

        let output = pipeline
            .prove_blocks_with_final_proof_and_external_lookahead(
                &bytecode,
                std::slice::from_ref(&block0),
                Some(external_lookahead),
            )
            .unwrap();

        assert!(output.final_proof.is_some());
        assert!(output.fold_inputs[0].state.lookahead_cycle_digest.is_some());
        verify_nova_block_proof_pipeline_with_final_proof_and_external_lookahead(
            &bytecode,
            std::slice::from_ref(&block0),
            Some(external_lookahead),
            &output,
            &backend,
        )
        .unwrap();
        assert!(verify_nova_block_proof_pipeline_with_final_proof(
            &bytecode,
            &[block0],
            &output,
            &backend,
        )
        .is_err());
    }

    #[cfg(feature = "nova")]
    #[test]
    fn nova_block_proof_pipeline_final_proof_size_report_proves_and_reports() {
        let bytecode = BytecodePreprocessing::default();
        let block0 = trace_block(0, boundary(0, 0), boundary(2, 0));
        let block1 = trace_block(1, block0.end_state.clone(), boundary(4, 0));
        let backend = NovaFoldingBackend::default();
        let pipeline = BlockProofPipeline::<_, ark_bn254::Fr, NovaFoldingBackend>::with_backend(
            [9u8; 32],
            backend.clone(),
        );

        let report = pipeline
            .prove_blocks_with_final_proof_size_report(&bytecode, &[block0.clone(), block1.clone()])
            .unwrap();
        let output = &report.output;
        let comparison = &report.final_proof_size_comparison;

        assert_eq!(output.bundles.len(), 2);
        assert_eq!(output.fold_inputs.len(), 2);
        assert_eq!(output.accumulator.metadata.absorbed_blocks, 2);
        assert!(output.accumulator.recursive_snark_bytes.is_some());
        assert!(output.final_proof.is_none());
        assert_eq!(
            comparison.folded_accumulator_digest,
            output.accumulator.metadata.accumulator_digest
        );
        assert_eq!(comparison.absorbed_blocks, 2);
        assert_eq!(
            comparison.total_active_cycles,
            output.accumulator.metadata.total_active_cycles
        );
        assert_eq!(
            comparison.placeholder.configured_backend_name,
            SPARTAN_PLACEHOLDER_PROOF_SYSTEM_NAME
        );
        assert_eq!(
            comparison.spartan.configured_backend_name,
            SPARTAN_FINAL_PROOF_SYSTEM_NAME
        );
        assert_eq!(comparison.placeholder.proof_payload_bytes_len, 0);
        assert!(comparison.spartan.proof_payload_bytes_len > 0);
        assert_eq!(
            comparison.spartan_payload_extra_bytes,
            comparison.spartan.proof_payload_bytes_len
        );
        assert!(
            comparison.spartan.proof_total_bytes_len > comparison.placeholder.proof_total_bytes_len
        );
        verify_block_proof_pipeline_with_backend(&bytecode, &[block0, block1], output, &backend)
            .unwrap();
    }

    #[cfg(feature = "nova")]
    #[test]
    fn nova_block_proof_pipeline_final_proof_size_scaling_report_tracks_prefixes() {
        let bytecode = BytecodePreprocessing::default();
        let block0 = trace_block(0, boundary(0, 0), boundary(2, 0));
        let block1 = trace_block(1, block0.end_state.clone(), boundary(4, 0));
        let backend = NovaFoldingBackend::default();
        let pipeline = BlockProofPipeline::<_, ark_bn254::Fr, NovaFoldingBackend>::with_backend(
            [9u8; 32], backend,
        );

        let report = pipeline
            .prove_block_prefixes_with_final_proof_size_scaling_report(
                &bytecode,
                &[block0, block1],
                &[1, 2],
            )
            .unwrap();

        assert_eq!(report.rows.len(), 2);
        assert_eq!(report.rows[0].block_count, 1);
        assert_eq!(report.rows[0].first_block_index, Some(0));
        assert_eq!(report.rows[0].last_block_index, Some(0));
        assert_eq!(report.rows[0].total_active_cycles, 2);
        assert_eq!(
            report.rows[0].final_proof_size_comparison.absorbed_blocks,
            1
        );
        assert_eq!(report.rows[1].block_count, 2);
        assert_eq!(report.rows[1].first_block_index, Some(0));
        assert_eq!(report.rows[1].last_block_index, Some(1));
        assert_eq!(report.rows[1].total_active_cycles, 4);
        assert_eq!(
            report.rows[1].final_proof_size_comparison.absorbed_blocks,
            2
        );
        assert_eq!(
            report.rows[0].recursive_snark_bytes_len,
            report.rows[0]
                .final_proof_size_comparison
                .recursive_snark_bytes_len
        );
        assert_eq!(
            report.rows[1].recursive_snark_bytes_len,
            report.rows[1]
                .final_proof_size_comparison
                .recursive_snark_bytes_len
        );
        assert!(report.rows[1].total_active_cycles > report.rows[0].total_active_cycles);
        for row in &report.rows {
            assert_eq!(
                row.final_proof_size_comparison
                    .placeholder
                    .proof_payload_bytes_len,
                0
            );
            assert!(
                row.final_proof_size_comparison
                    .spartan
                    .proof_payload_bytes_len
                    > 0
            );
            assert!(
                row.final_proof_size_comparison
                    .spartan
                    .proof_total_bytes_len
                    > row
                        .final_proof_size_comparison
                        .placeholder
                        .proof_total_bytes_len
            );
        }
    }

    #[cfg(feature = "nova")]
    #[test]
    fn nova_block_proof_pipeline_final_proof_size_benchmark_artifact_exports_json() {
        let bytecode = BytecodePreprocessing::default();
        let block0 = trace_block(0, boundary(0, 0), boundary(2, 0));
        let block1 = trace_block(1, block0.end_state.clone(), boundary(4, 0));
        let backend = NovaFoldingBackend::default();
        let pipeline = BlockProofPipeline::<_, ark_bn254::Fr, NovaFoldingBackend>::with_backend(
            [9u8; 32], backend,
        );

        let artifact = pipeline
            .prove_block_prefixes_with_final_proof_size_benchmark_artifact(
                &bytecode,
                &[block0, block1],
                &[1, 2],
                JoltNovaReportOutputFormat::Json,
            )
            .unwrap();

        assert_eq!(artifact.output_format, JoltNovaReportOutputFormat::Json);
        assert_eq!(artifact.report.rows.len(), 2);
        assert_eq!(artifact.report.rows[0].block_count, 1);
        assert_eq!(artifact.report.rows[1].block_count, 2);
        assert_eq!(
            artifact.serialized_report,
            export_nova_final_proof_size_scaling_report_json(&artifact.report)
        );
        assert!(artifact
            .serialized_report
            .contains("\"schema_version\":\"jolt-nova-report-v1\""));
        assert!(artifact
            .serialized_report
            .contains("\"report_kind\":\"final-proof-size-scaling\""));
        assert!(artifact.serialized_report.contains("\"row_count\":2"));
        assert!(artifact.serialized_report.contains("\"block_count\":1"));
        assert!(artifact.serialized_report.contains("\"block_count\":2"));
        assert!(artifact
            .serialized_report
            .contains("\"spartan-final-proof\""));
    }

    #[cfg(feature = "nova")]
    #[test]
    fn nova_block_proof_pipeline_final_proof_size_benchmark_artifact_writes_json_file() {
        let bytecode = BytecodePreprocessing::default();
        let block0 = trace_block(0, boundary(0, 0), boundary(2, 0));
        let backend = NovaFoldingBackend::default();
        let pipeline = BlockProofPipeline::<_, ark_bn254::Fr, NovaFoldingBackend>::with_backend(
            [9u8; 32], backend,
        );
        let path = temp_artifact_path(
            "nova-final-proof-size-benchmark-artifact-writes",
            "report.json",
        );

        let artifact = pipeline
            .prove_block_prefixes_and_write_final_proof_size_benchmark_artifact(
                &bytecode,
                &[block0],
                &[1],
                JoltNovaReportOutputFormat::Json,
                &path,
            )
            .unwrap();
        let written_report = std::fs::read_to_string(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        std::fs::remove_dir(path.parent().unwrap()).unwrap();

        assert_eq!(artifact.output_format, JoltNovaReportOutputFormat::Json);
        assert_eq!(artifact.report.rows.len(), 1);
        assert_eq!(artifact.report.rows[0].block_count, 1);
        assert_eq!(written_report, artifact.serialized_report);
        assert!(written_report.contains("\"schema_version\":\"jolt-nova-report-v1\""));
        assert!(written_report.contains("\"report_kind\":\"final-proof-size-scaling\""));
        assert!(written_report.contains("\"row_count\":1"));
    }

    #[cfg(feature = "nova")]
    #[test]
    fn nova_block_proof_pipeline_final_proof_size_scaling_report_rejects_invalid_counts() {
        let bytecode = BytecodePreprocessing::default();
        let block0 = trace_block(0, boundary(0, 0), boundary(2, 0));
        let block1 = trace_block(1, block0.end_state.clone(), boundary(4, 0));
        let backend = NovaFoldingBackend::default();
        let pipeline = BlockProofPipeline::<_, ark_bn254::Fr, NovaFoldingBackend>::with_backend(
            [9u8; 32], backend,
        );
        let blocks = [block0, block1];

        assert_eq!(
            pipeline
                .prove_block_prefixes_with_final_proof_size_scaling_report(&bytecode, &blocks, &[])
                .unwrap_err(),
            BlockTraceError::NovaFoldingBackendError {
                block_index: 0,
                reason: "final proof size scaling requires at least one block count",
            }
        );
        assert_eq!(
            pipeline
                .prove_block_prefixes_with_final_proof_size_scaling_report(&bytecode, &blocks, &[0])
                .unwrap_err(),
            BlockTraceError::NovaFoldingBackendError {
                block_index: 0,
                reason: "final proof size scaling block count is out of range",
            }
        );
        assert_eq!(
            pipeline
                .prove_block_prefixes_with_final_proof_size_scaling_report(
                    &bytecode,
                    &blocks,
                    &[2, 1]
                )
                .unwrap_err(),
            BlockTraceError::NovaFoldingBackendError {
                block_index: 0,
                reason: "final proof size scaling block counts must be strictly increasing",
            }
        );
        assert_eq!(
            pipeline
                .prove_block_prefixes_with_final_proof_size_scaling_report(&bytecode, &blocks, &[3])
                .unwrap_err(),
            BlockTraceError::NovaFoldingBackendError {
                block_index: 2,
                reason: "final proof size scaling block count is out of range",
            }
        );
    }

    #[cfg(feature = "nova")]
    #[test]
    fn nova_stage7_final_proof_reporting_lifecycle_closes_pipeline_surface() {
        let bytecode = BytecodePreprocessing::default();
        let block0 = trace_block(0, boundary(0, 0), boundary(2, 0));
        let block1 = trace_block(1, block0.end_state.clone(), boundary(4, 0));
        let blocks = [block0, block1];
        let backend = NovaFoldingBackend::default();
        let pipeline = BlockProofPipeline::<_, ark_bn254::Fr, NovaFoldingBackend>::with_backend(
            [9u8; 32],
            backend.clone(),
        );

        let final_proof_output = pipeline
            .prove_blocks_with_final_proof(&bytecode, &blocks)
            .unwrap();
        let final_proof = final_proof_output.final_proof.as_ref().unwrap();

        assert_eq!(
            final_proof.proof_system,
            SPARTAN_PLACEHOLDER_PROOF_SYSTEM_NAME
        );
        assert!(final_proof.spartan_proof_bytes.is_none());
        verify_nova_block_proof_pipeline_with_final_proof(
            &bytecode,
            &blocks,
            &final_proof_output,
            &backend,
        )
        .unwrap();

        let scaling_report = pipeline
            .prove_block_prefixes_with_final_proof_size_scaling_report(&bytecode, &blocks, &[2])
            .unwrap();
        let row = &scaling_report.rows[0];
        let comparison = &row.final_proof_size_comparison;

        assert_eq!(scaling_report.rows.len(), 1);
        assert_eq!(row.block_count, 2);
        assert_eq!(
            comparison.folded_accumulator_digest,
            final_proof_output.accumulator.metadata.accumulator_digest
        );
        assert_eq!(
            comparison.absorbed_blocks,
            final_proof_output.accumulator.metadata.absorbed_blocks
        );
        assert_eq!(
            comparison.placeholder.proof_system,
            SPARTAN_PLACEHOLDER_PROOF_SYSTEM_NAME
        );
        assert_eq!(
            comparison.spartan.proof_system,
            SPARTAN_FINAL_PROOF_SYSTEM_NAME
        );
        assert_eq!(comparison.placeholder.proof_payload_bytes_len, 0);
        assert!(comparison.spartan.proof_payload_bytes_len > 0);
        assert!(
            comparison.spartan.proof_total_bytes_len > comparison.placeholder.proof_total_bytes_len
        );
    }

    #[cfg(feature = "nova")]
    #[test]
    fn nova_block_proof_pipeline_generic_verifier_rejects_embedded_final_proof() {
        let bytecode = BytecodePreprocessing::default();
        let block = trace_block(0, boundary(0, 0), boundary(4, 0));
        let backend = NovaFoldingBackend::default();
        let pipeline = BlockProofPipeline::<_, ark_bn254::Fr, NovaFoldingBackend>::with_backend(
            [9u8; 32],
            backend.clone(),
        );
        let output = pipeline
            .prove_blocks_with_final_proof(&bytecode, &[block.clone()])
            .unwrap();

        assert!(output.final_proof.is_some());
        assert_eq!(
            verify_block_proof_pipeline_with_backend(&bytecode, &[block], &output, &backend)
                .unwrap_err(),
            BlockTraceError::NovaFoldingBackendError {
                block_index: 0,
                reason: "pipeline output carries a final folded proof; use the Nova final-proof verifier",
            }
        );
    }

    #[cfg(feature = "nova")]
    #[test]
    fn nova_block_proof_pipeline_with_spartan_final_proof_proves_and_verifies() {
        let bytecode = BytecodePreprocessing::default();
        let block = trace_block(0, boundary(0, 0), boundary(4, 0));
        let backend = NovaFoldingBackend::new(NovaFoldConfig {
            final_proof_backend_name: SPARTAN_FINAL_PROOF_SYSTEM_NAME,
            ..NovaFoldConfig::default()
        });
        let pipeline = BlockProofPipeline::<_, ark_bn254::Fr, NovaFoldingBackend>::with_backend(
            [9u8; 32],
            backend.clone(),
        );

        let output = pipeline
            .prove_blocks_with_final_proof(&bytecode, &[block.clone()])
            .unwrap();
        let final_proof = output.final_proof.as_ref().unwrap();

        assert_eq!(output.accumulator.metadata.absorbed_blocks, 1);
        assert_eq!(final_proof.proof_system, SPARTAN_FINAL_PROOF_SYSTEM_NAME);
        assert!(final_proof.spartan_proof_bytes.as_ref().unwrap().len() > 0);
        verify_nova_block_proof_pipeline_with_final_proof(&bytecode, &[block], &output, &backend)
            .unwrap();
    }

    #[cfg(feature = "nova")]
    #[test]
    fn nova_block_proof_pipeline_final_proof_verifier_rejects_tampering() {
        let bytecode = BytecodePreprocessing::default();
        let block = trace_block(0, boundary(0, 0), boundary(4, 0));
        let backend = NovaFoldingBackend::default();
        let pipeline = BlockProofPipeline::<_, ark_bn254::Fr, NovaFoldingBackend>::with_backend(
            [9u8; 32],
            backend.clone(),
        );
        let mut output = pipeline
            .prove_blocks_with_final_proof(&bytecode, &[block.clone()])
            .unwrap();
        output.final_proof.as_mut().unwrap().proof_digest[0] ^= 1;

        assert_eq!(
            verify_nova_block_proof_pipeline_with_final_proof(
                &bytecode,
                &[block],
                &output,
                &backend,
            )
            .unwrap_err(),
            BlockTraceError::NovaFoldingBackendError {
                block_index: 0,
                reason: "final folded proof digest mismatch",
            }
        );
    }

    #[test]
    fn block_proof_pipeline_verifier_rejects_tampered_accumulator() {
        let bytecode = BytecodePreprocessing::default();
        let block = trace_block(0, boundary(0, 0), boundary(4, 0));
        let pipeline = BlockProofPipeline::<_, ark_bn254::Fr>::new([9u8; 32]);
        let mut output = pipeline.prove_blocks(&bytecode, &[block.clone()]).unwrap();
        output.accumulator.accumulator_digest[0] ^= 1;

        assert_eq!(
            verify_block_proof_pipeline(&bytecode, &[block], &output).unwrap_err(),
            BlockTraceError::BlockFoldAccumulatorMismatch {
                expected_blocks: 1,
                actual_blocks: 1,
            }
        );
    }

    #[test]
    fn block_proof_pipeline_verifier_rejects_tampered_bundle() {
        let bytecode = BytecodePreprocessing::default();
        let block = trace_block(0, boundary(0, 0), boundary(4, 0));
        let pipeline = BlockProofPipeline::<_, ark_bn254::Fr>::new([9u8; 32]);
        let mut output = pipeline.prove_blocks(&bytecode, &[block.clone()]).unwrap();
        output.bundles[0].lookup_claim.claims_digest[0] ^= 1;

        assert_eq!(
            verify_block_proof_pipeline(&bytecode, &[block], &output).unwrap_err(),
            BlockTraceError::BlockLookupClaimMismatch { block_index: 0 }
        );
    }

    #[test]
    fn block_proof_pipeline_verifier_rejects_tampered_fold_input() {
        let bytecode = BytecodePreprocessing::default();
        let block = trace_block(0, boundary(0, 0), boundary(4, 0));
        let pipeline = BlockProofPipeline::<_, ark_bn254::Fr>::new([9u8; 32]);
        let mut output = pipeline.prove_blocks(&bytecode, &[block.clone()]).unwrap();
        output.fold_inputs[0].state.state_digest[0] ^= 1;

        assert_eq!(
            verify_block_proof_pipeline(&bytecode, &[block], &output).unwrap_err(),
            BlockTraceError::BlockFoldInputMismatch { block_index: 0 }
        );
    }

    #[cfg(feature = "nova")]
    #[test]
    fn nova_snark_trivial_recursive_demo_runs() {
        use nova_snark::{
            nova::{PublicParams, RecursiveSNARK},
            provider::{PallasEngine, VestaEngine},
            traits::{circuit::TrivialCircuit, snark::default_ck_hint, Engine},
        };

        type E1 = PallasEngine;
        type E2 = VestaEngine;
        type C = TrivialCircuit<<E1 as Engine>::Scalar>;

        let circuit = C::default();
        let pp = PublicParams::<E1, E2, C>::setup(
            &circuit,
            &*default_ck_hint::<E1>(),
            &*default_ck_hint::<E2>(),
        )
        .unwrap();

        let z0 = [<E1 as Engine>::Scalar::default()];
        let mut recursive_snark = RecursiveSNARK::<E1, E2, C>::new(&pp, &circuit, &z0).unwrap();

        recursive_snark.prove_step(&pp, &circuit).unwrap();
        recursive_snark.verify(&pp, 1, &z0).unwrap();
    }
}
