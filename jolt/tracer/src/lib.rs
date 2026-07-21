#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate alloc;
extern crate core;

use itertools::Itertools;
use std::vec;
use tracing::{error, info};

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, vec::Vec};

use common::{self, constants::REGISTER_COUNT, jolt_device::MemoryConfig};
use emulator::{cpu, default_terminal::DefaultTerminal};
use instruction::{Cycle, Instruction};
use jolt_riscv::RV64IMAC_JOLT;

pub mod emulator;
pub mod execution_backend;
pub mod instruction;
mod jolt_cycle_adapter;
pub mod trace_row;
pub mod utils;

pub use common::jolt_device::JoltDevice;
pub use cpu::{advice_tape_read, advice_tape_remaining, advice_tape_write, AdviceTape};
pub use execution_backend::TracerBackend;
pub use instruction::inline::{
    list_registered_inlines, InlineRegistration, TracerInlineExpansionProvider,
};
pub use jolt_riscv::InlineExtension;
pub use trace_row::{build_trace_rows, cycle_to_trace_row, CycleConversionError};

use crate::emulator::{
    memory::{Memory, MemoryData},
    Emulator,
};

/// Executes a RISC-V program and generates its execution trace along with emulator state checkpoints.
///
/// # Details
/// The function performs these steps:
/// 1. Sets up an emulator with the provided program and configuration
/// 2. Runs the program to completion while:
///    - Collecting execution traces of each instruction
///    - Optionally saving periodic checkpoints of the emulator state
///
/// # Arguments
///
/// * `elf_contents`
/// * `inputs`
/// * `memory_config`
/// * `checkpoint_interval` - Number of Cycle at which to save emulator checkpoints
///                          If None, no checkpoints will be saved
///
/// # Returns
///
/// Returns a tuple containing:
/// * `Vec<Cycle>` - Complete execution trace
/// * `JoltDevice`
/// * `Option<Vec<LazyTraceIterator>>` - If checkpoint_interval is not None, contains emulator
///                                      checkpoints every n Cycle. Otherwise None.
///
/// # Example Usage
///
/// let (execution_trace, checkpoints) = trace(elf_contents, inputs, memory_config, Some(5));
///
/// let full_execution_trace = checkpoints.as_ref().unwrap()[0].clone().collect::Vec<Cycle>();
/// assert!(execution_trace == full_execution_trace);
///
/// let trace_from_checkpoint_1 = checkpoints.as_ref().unwrap()[1].clone().collect::Vec<Cycle>();
/// assert!(trace_from_checkpoint_1 == execution_trace[n..])
///
/// let trace_from_checkpoint_2 = checkpoints.as_ref().unwrap()[2].clone().collect::Vec<Cycle>();
/// assert!(trace_from_checkpoint_2 == execution_trace[2*n..])
///
#[tracing::instrument(skip_all)]
pub fn trace(
    elf_contents: &[u8],
    elf_path: Option<&std::path::PathBuf>,
    inputs: &[u8],
    untrusted_advice: &[u8],
    trusted_advice: &[u8],
    memory_config: &MemoryConfig,
    advice_tape: Option<cpu::AdviceTape>,
) -> (
    LazyTraceIterator,
    Vec<Cycle>,
    Memory,
    JoltDevice,
    cpu::AdviceTape,
) {
    let mut lazy_trace_iter = trace_lazy(
        elf_contents,
        elf_path,
        inputs,
        untrusted_advice,
        trusted_advice,
        memory_config,
        advice_tape,
    );
    let lazy_trace_iter_ = lazy_trace_iter.clone();
    let trace: Vec<Cycle> = lazy_trace_iter.by_ref().collect();

    // Extract the populated advice tape before moving lazy_tracer
    let advice_tape_result = lazy_trace_iter.lazy_tracer.take_advice_tape();

    let final_memory_state =
        std::mem::take(&mut lazy_trace_iter.lazy_tracer.final_memory_state).unwrap();
    let jolt_device = lazy_trace_iter.lazy_tracer.get_jolt_device();
    (
        lazy_trace_iter_, // Return the clone since lazy_tracer was moved
        trace,
        final_memory_state,
        jolt_device,
        advice_tape_result, // Return the populated advice tape
    )
}

use crate::utils::trace_writer::{TraceBatchCollector, TraceWriter, TraceWriterConfig};

pub fn trace_to_file(
    elf_contents: &[u8],
    elf_path: Option<&std::path::PathBuf>,
    inputs: &[u8],
    untrusted_advice: &[u8],
    trusted_advice: &[u8],
    memory_config: &MemoryConfig,
    out_path: &std::path::PathBuf,
) -> (Memory, JoltDevice) {
    let config = TraceWriterConfig::default();

    let writer =
        TraceWriter::<Cycle>::new(out_path, config).expect("Failed to create trace writer");
    let mut collector = TraceBatchCollector::new(writer);
    let mut lazy = trace_lazy(
        elf_contents,
        elf_path,
        inputs,
        untrusted_advice,
        trusted_advice,
        memory_config,
        None,
    );

    for cycle in &mut lazy {
        collector.push(cycle);
    }

    let total = collector
        .finalize()
        .expect("Failed to finalize trace writer");

    info!("trace length: {total} cycles");

    let final_mem = lazy.lazy_tracer.final_memory_state.take().unwrap();
    (final_mem, lazy.lazy_tracer.get_jolt_device())
}

#[tracing::instrument(skip_all)]
pub fn trace_lazy(
    elf_contents: &[u8],
    elf_path: Option<&std::path::PathBuf>,
    inputs: &[u8],
    untrusted_advice: &[u8],
    trusted_advice: &[u8],
    memory_config: &MemoryConfig,
    advice_tape: Option<cpu::AdviceTape>,
) -> LazyTraceIterator {
    LazyTraceIterator::new(CheckpointingTracer::new(setup_emulator_with_backtraces(
        elf_contents,
        elf_path,
        inputs,
        untrusted_advice,
        trusted_advice,
        memory_config,
        advice_tape,
    )))
}

#[tracing::instrument(skip_all)]
pub fn trace_blocks(
    elf_contents: &[u8],
    elf_path: Option<&std::path::PathBuf>,
    inputs: &[u8],
    untrusted_advice: &[u8],
    trusted_advice: &[u8],
    memory_config: &MemoryConfig,
    advice_tape: Option<cpu::AdviceTape>,
    target_size: usize,
) -> TraceBlockIterator<CheckpointingTracer> {
    trace_lazy(
        elf_contents,
        elf_path,
        inputs,
        untrusted_advice,
        trusted_advice,
        memory_config,
        advice_tape,
    )
    .iter_blocks(target_size)
}

#[tracing::instrument(skip_all)]
pub fn trace_checkpoints(
    elf_contents: &[u8],
    inputs: &[u8],
    untrusted_advice: &[u8],
    trusted_advice: &[u8],
    memory_config: &MemoryConfig,
    checkpoint_interval: usize,
) -> (Vec<Checkpoint>, JoltDevice) {
    let mut emulator_trace_iter =
        GeneralizedLazyTraceIter::new(CheckpointingTracer::new(setup_emulator(
            elf_contents,
            inputs,
            untrusted_advice,
            trusted_advice,
            memory_config,
        )));
    emulator_trace_iter.lazy_tracer.start_saving_checkpoints();
    let mut checkpoints = Vec::new();

    loop {
        emulator_trace_iter = emulator_trace_iter.dropping(checkpoint_interval);
        let chkpt = emulator_trace_iter.lazy_tracer.save_checkpoint();
        checkpoints.push(chkpt);
        if emulator_trace_iter.lazy_tracer.has_terminated() {
            break;
        }
    }
    (
        checkpoints,
        emulator_trace_iter.lazy_tracer.get_jolt_device(),
    )
}

fn step_emulator(emulator: &mut Emulator, prev_pc: &mut u64, trace: Option<&mut Vec<Cycle>>) {
    let pc = emulator.get_cpu().read_pc();
    // This is a trick to see if the program has terminated by throwing itself
    // into an infinite loop. It seems to be a good heuristic for now but we
    // should eventually migrate to an explicit shutdown signal.
    if *prev_pc == pc {
        return;
    }
    emulator.tick(trace);
    *prev_pc = pc;
}

#[tracing::instrument(skip_all)]
fn setup_emulator(
    elf_contents: &[u8],
    inputs: &[u8],
    untrusted_advice: &[u8],
    trusted_advice: &[u8],
    memory_config: &MemoryConfig,
) -> Emulator {
    setup_emulator_with_backtraces(
        elf_contents,
        None,
        inputs,
        untrusted_advice,
        trusted_advice,
        memory_config,
        None, // No advice tape by default
    )
}

#[tracing::instrument(skip_all)]
/// Sets up an emulator instance with access to the elf-path for symbol loading and de-mangling.
fn setup_emulator_with_backtraces(
    elf_contents: &[u8],
    elf_path: Option<&std::path::PathBuf>,
    inputs: &[u8],
    untrusted_advice: &[u8],
    trusted_advice: &[u8],
    memory_config: &MemoryConfig,
    advice_tape: Option<cpu::AdviceTape>,
) -> Emulator {
    let term = DefaultTerminal::default();
    let mut emulator = Emulator::new(Box::new(term));
    // Set the advice tape if provided
    if let Some(tape) = advice_tape {
        emulator.set_advice_tape(tape);
    }

    assert!(
        trusted_advice.len() as u64 <= memory_config.max_trusted_advice_size,
        "Trusted advice too long: got {} bytes, max is {} bytes (set by MemoryConfig.max_trusted_advice_size).",
        trusted_advice.len(),
        memory_config.max_trusted_advice_size,
    );
    assert!(
        untrusted_advice.len() as u64 <= memory_config.max_untrusted_advice_size,
        "Untrusted advice too long: got {} bytes, max is {} bytes (set by MemoryConfig.max_untrusted_advice_size).",
        untrusted_advice.len(),
        memory_config.max_untrusted_advice_size,
    );
    assert!(
        inputs.len() as u64 <= memory_config.max_input_size,
        "Input too long: got {} bytes, max is {} bytes (set by MemoryConfig.max_input_size).",
        inputs.len(),
        memory_config.max_input_size,
    );

    let mut jolt_device = JoltDevice::new(memory_config);
    jolt_device.inputs = inputs.to_vec();
    jolt_device.trusted_advice = trusted_advice.to_vec();
    jolt_device.untrusted_advice = untrusted_advice.to_vec();
    emulator.get_mut_cpu().get_mut_mmu().jolt_device = Some(jolt_device);
    if let Some(elf_path) = elf_path {
        emulator.set_elf_path(elf_path);
    }
    emulator.setup_program(elf_contents);
    emulator
}

/// A type that can be used to lazily generate a trace, one [`Cycle`] at a time.
pub trait LazyTracer {
    /// Check if the program execution has terminated.
    fn has_terminated(&self) -> bool;

    /// Check if the program execution has panicked.
    fn has_panicked(&self) -> bool;

    /// Snapshot machine state at a trace-block boundary.
    ///
    /// This should only be called when [`LazyTracer::at_tick_boundary`] is true. The
    /// `global_cycle` argument is the global trace-row offset maintained by the block iterator.
    fn boundary_state(&self, global_cycle: usize) -> MachineBoundaryState;

    /// Returns whether the next execution of [`LazyTracer::lazy_step_cycle`] will emulate a new
    /// instruction or return the next [`Cycle`] in the last executed instruction.
    fn at_tick_boundary(&self) -> bool;

    /// Print a backtrace, assuming the program has panicked.
    fn print_panic_log(&self);

    /// Get the next [`Cycle`] in the program execution. If the program is at a tick boundary, this
    /// emulates the next instruction. Otherwise, it returns the next cycle within the last
    /// executed instruction.
    fn lazy_step_cycle(&mut self) -> Option<Cycle>;

    /// Take the [`JoltDevice`] from this tracer, consuming the tracer.
    fn get_jolt_device(self) -> JoltDevice;
}

/// An iterator that lazily generates execution traces from a RISC-V emulator checkpoint.
///
/// This iterator produces instruction traces one at a time, executing the emulator
/// as needed rather than generating the entire trace upfront. It buffers traces
/// in `current_traces` since some instructions generate multiple trace entries.
/// When the `current_traces` buffer is exhausted, it executes another emulator tick
/// to generate more.
#[derive(Clone, Debug)]
pub struct GeneralizedLazyTraceIter<T> {
    pub lazy_tracer: T,
}

pub type LazyTraceIterator = GeneralizedLazyTraceIter<CheckpointingTracer>;

unsafe impl<T: Send> Send for GeneralizedLazyTraceIter<T> {}

#[derive(Clone, Debug, PartialEq)]
pub struct MachineBoundaryState {
    pub global_cycle: usize,
    pub emulator_trace_len: usize,
    pub pc: u64,
    pub registers: [i64; REGISTER_COUNT as usize],
    pub terminated: bool,
}

impl MachineBoundaryState {
    fn from_emulator(emulator: &Emulator, global_cycle: usize, terminated: bool) -> Self {
        let cpu = emulator.get_cpu();
        Self {
            global_cycle,
            emulator_trace_len: cpu.trace_len,
            pc: cpu.read_pc(),
            registers: cpu.x,
            terminated,
        }
    }
}

/// A contiguous chunk of execution trace cycles cut only at tracer tick boundaries.
///
/// `target_size` is intentionally a soft target: if a single emulated instruction expands into
/// more cycles than the requested size, the block may grow beyond `target_size` so that virtual
/// and inline instruction sequences are not split across block boundaries.
#[derive(Clone, Debug, PartialEq)]
pub struct TraceBlock {
    pub block_index: usize,
    pub global_cycle_start: usize,
    pub active_cycles: usize,
    pub target_size: usize,
    pub start_state: MachineBoundaryState,
    pub end_state: MachineBoundaryState,
    pub cycles: Vec<Cycle>,
    pub ended_at_tick_boundary: bool,
}

/// Iterator adapter that groups a lazy trace into [`TraceBlock`]s.
pub struct TraceBlockIterator<T> {
    trace_iter: GeneralizedLazyTraceIter<T>,
    target_size: usize,
    block_index: usize,
    global_cycle_start: usize,
}

impl<T: LazyTracer> Iterator for GeneralizedLazyTraceIter<T> {
    type Item = Cycle;

    /// Advances the iterator and returns the next trace entry.
    ///
    /// # Returns
    ///
    /// * `Some(Cycle)` - The next instruction trace in the execution sequence
    /// * `None` - If program execution has completed.
    ///
    /// # Details
    ///
    /// The function follows this sequence:
    /// 1. Returns any remaining traces from the previous emulator tick
    /// 2. If buffer `current_traces` is empty, and the number of ticks
    ///    is not reached, executes another emulator tick``
    /// 3. Checks for program termination using the heuristic of PC not changing
    /// 4. Buffers new traces in FIFO order
    /// 5. Returns the next trace or None if execution is complete
    fn next(&mut self) -> Option<Self::Item> {
        if self.lazy_tracer.has_terminated() {
            return None;
        }

        let res = self.lazy_tracer.lazy_step_cycle();

        if res.is_none() && self.lazy_tracer.has_panicked() {
            self.lazy_tracer.print_panic_log();
        }

        res
    }
}

impl<T> GeneralizedLazyTraceIter<T> {
    pub fn new(lazy_tracer: T) -> Self {
        Self { lazy_tracer }
    }
}

impl<T: LazyTracer> GeneralizedLazyTraceIter<T> {
    pub fn iter_blocks(self, target_size: usize) -> TraceBlockIterator<T> {
        TraceBlockIterator::new(self, target_size)
    }
}

impl<T: LazyTracer> TraceBlockIterator<T> {
    pub fn new(trace_iter: GeneralizedLazyTraceIter<T>, target_size: usize) -> Self {
        assert!(target_size != 0, "trace block target size must be non-zero");
        Self {
            trace_iter,
            target_size,
            block_index: 0,
            global_cycle_start: 0,
        }
    }

    pub fn into_inner(self) -> GeneralizedLazyTraceIter<T> {
        self.trace_iter
    }
}

impl<T: LazyTracer> Iterator for TraceBlockIterator<T> {
    type Item = TraceBlock;

    fn next(&mut self) -> Option<Self::Item> {
        if self.trace_iter.lazy_tracer.has_terminated() {
            return None;
        }

        assert!(
            self.trace_iter.lazy_tracer.at_tick_boundary(),
            "trace blocks must start at a tick boundary"
        );

        let start_state = self
            .trace_iter
            .lazy_tracer
            .boundary_state(self.global_cycle_start);
        let mut cycles = Vec::with_capacity(self.target_size);

        while cycles.len() < self.target_size || !self.trace_iter.lazy_tracer.at_tick_boundary() {
            match self.trace_iter.next() {
                Some(cycle) => cycles.push(cycle),
                None => break,
            }
        }

        if cycles.is_empty() {
            return None;
        }

        let global_cycle_end = self.global_cycle_start + cycles.len();
        let end_state = self.trace_iter.lazy_tracer.boundary_state(global_cycle_end);

        let block = TraceBlock {
            block_index: self.block_index,
            global_cycle_start: self.global_cycle_start,
            active_cycles: cycles.len(),
            target_size: self.target_size,
            start_state,
            end_state,
            ended_at_tick_boundary: self.trace_iter.lazy_tracer.at_tick_boundary()
                || self.trace_iter.lazy_tracer.has_terminated(),
            cycles,
        };

        self.block_index += 1;
        self.global_cycle_start += block.active_cycles;

        Some(block)
    }
}

#[derive(Clone, Debug)]
pub struct Checkpoint {
    emulator_state: Emulator,
    prev_pc: u64,
    current_traces: Vec<Cycle>,
    /// The remaining number of cycles that can be replayed for this checkpoint
    trace_steps_remaining: usize,
    /// The total number of cycles executed so far, including the ones prior to this checkpoint
    cycle_count: usize,
}

// SAFETY: Checkpoint contains only owned data and can be safely sent between threads
unsafe impl Send for Checkpoint {}

impl Checkpoint {
    pub(crate) fn new_with_empty_memory(
        emulator_state: &Emulator,
        prev_pc: u64,
        current_traces: &[Cycle],
        cycle_count: usize,
    ) -> Self {
        Self {
            emulator_state: emulator_state.save_state_with_empty_memory(),
            prev_pc,
            current_traces: current_traces.to_vec(),
            trace_steps_remaining: 0,
            cycle_count,
        }
    }

    pub(crate) fn set_memory_state(&mut self, data: MemoryData, cycles_remaining: usize) {
        self.trace_steps_remaining = cycles_remaining;
        self.emulator_state
            .get_mut_cpu()
            .get_mut_mmu()
            .memory
            .memory
            .data = data;
    }
}

impl LazyTracer for Checkpoint {
    fn has_terminated(&self) -> bool {
        self.trace_steps_remaining == 0
    }

    fn has_panicked(&self) -> bool {
        self.emulator_state
            .get_cpu()
            .mmu
            .jolt_device
            .as_ref()
            .unwrap()
            .panic
    }

    fn boundary_state(&self, global_cycle: usize) -> MachineBoundaryState {
        MachineBoundaryState::from_emulator(
            &self.emulator_state,
            global_cycle,
            self.has_terminated(),
        )
    }

    fn at_tick_boundary(&self) -> bool {
        self.current_traces.is_empty()
    }

    fn print_panic_log(&self) {
        error!(
            "Guest program terminated due to panic after {} cycles.",
            self.emulator_state.get_cpu().trace_len
        );
        utils::panic::display_panic_backtrace(&self.emulator_state);
    }

    fn lazy_step_cycle(&mut self) -> Option<Cycle> {
        if !self.current_traces.is_empty() {
            self.trace_steps_remaining -= 1;
            return self.current_traces.pop();
        }

        self.cycle_count += 1;

        step_emulator(
            &mut self.emulator_state,
            &mut self.prev_pc,
            Some(&mut self.current_traces),
        );
        if self.current_traces.is_empty() {
            None
        } else {
            self.trace_steps_remaining -= 1;
            self.current_traces.reverse();
            self.current_traces.pop()
        }
    }

    fn get_jolt_device(mut self) -> JoltDevice {
        self.emulator_state
            .get_mut_cpu()
            .get_mut_mmu()
            .jolt_device
            .take()
            .unwrap()
    }
}

/// A tracer that uses a `Vec<u64>` memory backend but additionally stores the initial value of
/// each memory access to a [`Checkpoint`], which can be saved and replayed from.
#[derive(Clone, Debug)]
pub struct CheckpointingTracer {
    emulator_state: Emulator,
    prev_pc: u64,
    current_traces: Vec<Cycle>,
    trace_steps_since_last_checkpoint: usize,
    cycle_count: usize,
    finished: bool,
    saved_processor_state: Option<Checkpoint>,
    pub(crate) final_memory_state: Option<Memory>,
}

impl CheckpointingTracer {
    pub fn new(emulator_state: Emulator) -> Self {
        Self {
            emulator_state,
            prev_pc: 0,
            current_traces: vec![],
            trace_steps_since_last_checkpoint: 0,
            cycle_count: 0,
            finished: false,
            saved_processor_state: None,
            final_memory_state: None,
        }
    }

    pub fn new_for_test() -> Self {
        let minimal_elf = vec![
            0x7f, 0x45, 0x4c, 0x46, 0x02, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x02, 0x00, 0xf3, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0x00, 0x38, 0x00,
            0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        ];

        use crate::MemoryConfig;
        let memory_config = MemoryConfig {
            program_size: Some(1024),
            ..Default::default()
        };

        let emulator_state = setup_emulator(&minimal_elf, b"[]", &[], &[], &memory_config);

        Self::new(emulator_state)
    }

    /// Start recording memory accesses so that checkpoints can be saved using
    /// [`CheckpointingTracer::save_checkpoint`].
    pub fn start_saving_checkpoints(&mut self) {
        self.saved_processor_state = Some(Checkpoint::new_with_empty_memory(
            &self.emulator_state,
            self.prev_pc,
            &self.current_traces,
            self.cycle_count,
        ));
        self.emulator_state
            .get_mut_cpu()
            .get_mut_mmu()
            .memory
            .memory
            .data
            .start_saving_checkpoints();
    }

    /// Save the recorded memory traces to a new [`Checkpoint`] and reset the hashmap to which
    /// they're recorded. The chunk of the trace that has been executed since the last call to
    /// [`CheckpointingTracer::save_checkpoint`] or
    /// [`CheckpointingTracer::start_saving_checkpoints`] can be replayed from the resulting
    /// [`Checkpoint`].
    pub fn save_checkpoint(&mut self) -> Checkpoint {
        assert!(self
            .emulator_state
            .get_cpu()
            .mmu
            .memory
            .memory
            .data
            .is_saving_checkpoints());

        // Save the processor state at the start of the current chunk
        let mut new_processor_state = Checkpoint::new_with_empty_memory(
            &self.emulator_state,
            self.prev_pc,
            &self.current_traces,
            self.cycle_count,
        );
        core::mem::swap(
            self.saved_processor_state.as_mut().unwrap(),
            &mut new_processor_state,
        );

        // Store the hashmap of memory assignments since the last chunk
        let data = self
            .emulator_state
            .get_mut_cpu()
            .get_mut_mmu()
            .memory
            .memory
            .data
            .save_checkpoint();
        new_processor_state.set_memory_state(data, self.trace_steps_since_last_checkpoint);
        self.trace_steps_since_last_checkpoint = 0;

        new_processor_state
    }

    /// Take ownership of the advice tape from the emulator, replacing it with an empty one
    pub fn take_advice_tape(&mut self) -> cpu::AdviceTape {
        self.emulator_state.take_advice_tape()
    }
}

impl LazyTracer for CheckpointingTracer {
    fn has_terminated(&self) -> bool {
        self.finished
    }

    fn has_panicked(&self) -> bool {
        self.emulator_state
            .get_cpu()
            .mmu
            .jolt_device
            .as_ref()
            .unwrap()
            .panic
    }

    fn boundary_state(&self, global_cycle: usize) -> MachineBoundaryState {
        MachineBoundaryState::from_emulator(
            &self.emulator_state,
            global_cycle,
            self.has_terminated(),
        )
    }

    fn at_tick_boundary(&self) -> bool {
        self.current_traces.is_empty()
    }

    fn print_panic_log(&self) {
        error!(
            "Guest program terminated due to panic after {} cycles.",
            self.emulator_state.get_cpu().trace_len
        );
        utils::panic::display_panic_backtrace(&self.emulator_state);
    }

    fn lazy_step_cycle(&mut self) -> Option<Cycle> {
        if !self.current_traces.is_empty() {
            self.trace_steps_since_last_checkpoint += 1;
            return self.current_traces.pop();
        }

        self.cycle_count += 1;

        step_emulator(
            &mut self.emulator_state,
            &mut self.prev_pc,
            Some(&mut self.current_traces),
        );
        if self.current_traces.is_empty() {
            self.finished = true;
            let emulator = &mut self.emulator_state;
            let cpu = emulator.get_mut_cpu();
            self.final_memory_state = Some(cpu.mmu.memory.memory.take_memory());
            None
        } else {
            self.trace_steps_since_last_checkpoint += 1;
            self.current_traces.reverse();
            self.current_traces.pop()
        }
    }

    fn get_jolt_device(mut self) -> JoltDevice {
        self.emulator_state
            .get_mut_cpu()
            .get_mut_mmu()
            .jolt_device
            .take()
            .expect("JoltDevice was not initialized")
    }
}

#[tracing::instrument(skip_all)]
pub fn decode(elf: &[u8]) -> (Vec<Instruction>, Vec<(u64, u8)>, u64, u64) {
    let obj = object::File::parse(elf).unwrap();
    if matches!(&obj, object::File::Elf32(_)) {
        panic!("tracer only supports RV64 ELF inputs");
    }

    let image = jolt_program::image::decode_elf(elf, RV64IMAC_JOLT)
        .expect("jolt-program ELF64 decoding failed");
    let instructions = image
        .instructions
        .into_iter()
        .map(|instruction| {
            Instruction::try_from_source_instruction(instruction)
                .expect("jolt-program image decoder produced an unknown tracer row")
        })
        .collect();
    (
        instructions,
        image.memory_init,
        image.program_end,
        image.entry_address,
    )
}

pub struct IterChunks<I: Iterator> {
    chunk_size: usize,
    iter: I,
}

pub trait ChunksIterator: Iterator + Sized {
    fn iter_chunks(self, size: usize) -> IterChunks<Self> {
        assert!(size != 0, "chunk size must be non-zero");
        IterChunks {
            chunk_size: size,
            iter: self,
        }
    }
}

impl<I: Iterator + Sized> ChunksIterator for I {}

impl<I: Iterator<Item: Clone>> Iterator for IterChunks<I> {
    type Item = Vec<I::Item>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut chunk = Vec::with_capacity(self.chunk_size);
        chunk.extend(self.iter.by_ref().take(self.chunk_size));
        if chunk.is_empty() {
            return None;
        }
        Some(chunk)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use common::jolt_device::MemoryConfig;

    #[derive(Clone, Debug)]
    struct ScriptedTracer {
        sequence_lengths: Vec<usize>,
        sequence_index: usize,
        offset_in_sequence: usize,
        emitted_cycles: usize,
    }

    impl ScriptedTracer {
        fn new(sequence_lengths: Vec<usize>) -> Self {
            assert!(
                sequence_lengths.iter().all(|len| *len != 0),
                "scripted trace sequence lengths must be non-zero"
            );
            Self {
                sequence_lengths,
                sequence_index: 0,
                offset_in_sequence: 0,
                emitted_cycles: 0,
            }
        }
    }

    impl LazyTracer for ScriptedTracer {
        fn has_terminated(&self) -> bool {
            self.sequence_index >= self.sequence_lengths.len()
        }

        fn has_panicked(&self) -> bool {
            false
        }

        fn boundary_state(&self, global_cycle: usize) -> MachineBoundaryState {
            let mut registers = [0i64; REGISTER_COUNT as usize];
            registers[1] = self.sequence_index as i64;

            MachineBoundaryState {
                global_cycle,
                emulator_trace_len: self.emitted_cycles,
                pc: self.sequence_index as u64,
                registers,
                terminated: self.has_terminated(),
            }
        }

        fn at_tick_boundary(&self) -> bool {
            self.offset_in_sequence == 0
        }

        fn print_panic_log(&self) {}

        fn lazy_step_cycle(&mut self) -> Option<Cycle> {
            if self.has_terminated() {
                return None;
            }

            let sequence_len = self.sequence_lengths[self.sequence_index];
            self.offset_in_sequence += 1;
            self.emitted_cycles += 1;
            if self.offset_in_sequence == sequence_len {
                self.sequence_index += 1;
                self.offset_in_sequence = 0;
            }

            Some(Cycle::NoOp)
        }

        fn get_jolt_device(self) -> JoltDevice {
            unreachable!("scripted tracer tests do not consume a JoltDevice")
        }
    }

    #[test]
    fn trace_blocks_respect_tick_boundaries_and_offsets() {
        let blocks = GeneralizedLazyTraceIter::new(ScriptedTracer::new(vec![2, 1, 3, 1]))
            .iter_blocks(3)
            .collect::<Vec<_>>();

        assert_eq!(blocks.len(), 3);
        assert_eq!(
            blocks
                .iter()
                .map(|block| block.active_cycles)
                .collect::<Vec<_>>(),
            vec![3, 3, 1]
        );
        assert_eq!(
            blocks
                .iter()
                .map(|block| block.global_cycle_start)
                .collect::<Vec<_>>(),
            vec![0, 3, 6]
        );
        assert_eq!(
            blocks
                .iter()
                .map(|block| block.block_index)
                .collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        assert_eq!(blocks[0].start_state.global_cycle, 0);
        assert_eq!(blocks[0].end_state.global_cycle, 3);
        assert_eq!(blocks[1].start_state, blocks[0].end_state);
        assert_eq!(blocks[2].start_state, blocks[1].end_state);
        assert_eq!(blocks[2].end_state.global_cycle, 7);
        assert_eq!(blocks[2].end_state.emulator_trace_len, 7);
        assert!(blocks[2].end_state.terminated);
        assert!(blocks.iter().all(|block| block.ended_at_tick_boundary));
    }

    #[test]
    fn trace_blocks_do_not_split_long_instruction_sequence() {
        let blocks = GeneralizedLazyTraceIter::new(ScriptedTracer::new(vec![5, 1]))
            .iter_blocks(3)
            .collect::<Vec<_>>();

        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].active_cycles, 5);
        assert_eq!(blocks[1].active_cycles, 1);
        assert_eq!(blocks[0].start_state.global_cycle, 0);
        assert_eq!(blocks[0].end_state.global_cycle, 5);
        assert_eq!(blocks[1].start_state, blocks[0].end_state);
        assert_eq!(blocks[1].end_state.global_cycle, 6);
        assert!(blocks.iter().all(|block| block.ended_at_tick_boundary));
    }

    #[test]
    #[should_panic(expected = "trace block target size must be non-zero")]
    fn trace_blocks_reject_zero_target_size() {
        let _ = GeneralizedLazyTraceIter::new(ScriptedTracer::new(vec![1])).iter_blocks(0);
    }

    fn minimal_elf() -> Vec<u8> {
        vec![
            0x7f, 0x45, 0x4c, 0x46, 0x02, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x02, 0x00, 0xf3, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0x00,
            0x38, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        ]
    }

    fn minimal_elf32() -> Vec<u8> {
        let mut elf = vec![0; 52];
        elf[0..4].copy_from_slice(b"\x7fELF");
        elf[4] = 1; // ELFCLASS32
        elf[5] = 1; // little endian
        elf[6] = 1; // current ELF version
        elf[16..18].copy_from_slice(&2u16.to_le_bytes()); // executable
        elf[18..20].copy_from_slice(&243u16.to_le_bytes()); // RISC-V
        elf[20..24].copy_from_slice(&1u32.to_le_bytes());
        elf[40..42].copy_from_slice(&52u16.to_le_bytes());
        elf[42..44].copy_from_slice(&32u16.to_le_bytes());
        elf[46..48].copy_from_slice(&40u16.to_le_bytes());
        elf
    }

    #[test]
    #[should_panic(expected = "tracer only supports RV64 ELF inputs")]
    fn decode_rejects_elf32() {
        decode(&minimal_elf32());
    }

    #[test]
    #[should_panic(expected = "Trusted advice too long")]
    fn panics_when_trusted_advice_exceeds_max() {
        let elf = minimal_elf();
        let memory_config = MemoryConfig {
            program_size: Some(1024),
            max_trusted_advice_size: 2048,
            ..Default::default()
        };
        let _ = setup_emulator(&elf, b"[]", &[], &[0u8; 4096], &memory_config);
    }

    #[test]
    #[should_panic(expected = "Input too long")]
    fn panics_when_input_exceeds_max() {
        let elf = minimal_elf();
        let memory_config = MemoryConfig {
            program_size: Some(1024),
            max_input_size: 64,
            ..Default::default()
        };
        let _ = setup_emulator(&elf, &[0u8; 128], &[], &[], &memory_config);
    }

    #[test]
    #[should_panic(expected = "Untrusted advice too long")]
    fn panics_when_untrusted_advice_exceeds_max() {
        let elf = minimal_elf();
        let memory_config = MemoryConfig {
            program_size: Some(1024),
            max_untrusted_advice_size: 128,
            ..Default::default()
        };
        let _ = setup_emulator(&elf, b"[]", &[0u8; 256], &[], &memory_config);
    }

    /// Build the muldiv guest and return the ELF bytes.
    /// Mirrors the pattern used by `host::Program::build()` in jolt-core.
    fn build_muldiv_guest() -> Vec<u8> {
        let guest = "muldiv-guest";
        let func = "muldiv";
        let target_dir = format!("/tmp/jolt-guest-targets/{guest}-{func}");

        let output = std::process::Command::new("jolt")
            .args([
                "build",
                "-p",
                guest,
                "--stack-size",
                &common::constants::DEFAULT_STACK_SIZE.to_string(),
                "--heap-size",
                "32768",
                "--",
                "--release",
                "--target-dir",
                &target_dir,
                "--features",
                "guest",
            ])
            .env("JOLT_FUNC_NAME", func)
            .output()
            .expect("failed to run jolt CLI — install with: cargo install --path .");

        if !output.status.success() {
            panic!(
                "failed to build muldiv guest:\n{}",
                String::from_utf8_lossy(&output.stderr)
            );
        }

        let elf_path = format!("{target_dir}/riscv64imac-unknown-none-elf/release/{guest}");
        std::fs::read(&elf_path).unwrap_or_else(|e| panic!("failed to read ELF at {elf_path}: {e}"))
    }

    const INPUTS: [u8; 6] = [0xbd, 0xaa, 0xde, 0x5, 0x11, 0x5c];

    #[test]
    /// Test that the trace function produces the expected number of cycles for a given ELF input.
    /// Test the checkpointing functionality by verifying the number of checkpoints created and
    /// if the traces from checkpoints match the overall execution trace.
    /// The test is based on the muldiv benchmark.
    fn test_checkpoints() {
        let elf = build_muldiv_guest();
        let n = 50;
        let memory_config = MemoryConfig {
            program_size: Some(elf.len() as u64),
            ..Default::default()
        };
        let (_, execution_trace, _, _, _) =
            trace(&elf, None, &INPUTS, &[], &[], &memory_config, None);
        let (checkpoints, _) = trace_checkpoints(&elf, &INPUTS, &[], &[], &memory_config, n);
        assert!(
            !execution_trace.is_empty(),
            "execution trace should not be empty"
        );
        assert_eq!(checkpoints.len(), execution_trace.len().div_ceil(n));

        let trace_chunk = execution_trace
            .chunks(n)
            .map(|x| x.to_vec())
            .collect::<Vec<_>>();
        for (i, checkpoint) in checkpoints.into_iter().enumerate() {
            let ti: Vec<Cycle> = GeneralizedLazyTraceIter::new(checkpoint).collect();
            assert_eq!(trace_chunk[i], ti);
        }
    }

    #[test]
    fn test_trace_length() {
        let elf = build_muldiv_guest();
        let memory_config = MemoryConfig {
            program_size: Some(elf.len() as u64),
            ..Default::default()
        };

        let (_, execution_trace, _, _, _) =
            trace(&elf, None, &INPUTS, &[], &[], &memory_config, None);
        let mut emulator: Emulator = setup_emulator(&elf, &INPUTS, &[], &[], &memory_config);
        let mut prev_pc: u64 = 0;
        let mut trace = vec![];
        let mut prev_trace_len = 0;
        loop {
            step_emulator(&mut emulator, &mut prev_pc, Some(&mut trace));
            if trace.len() - prev_trace_len == 0 {
                break;
            }
            prev_trace_len = trace.len();
        }
        assert_eq!(execution_trace, trace);
    }
}
