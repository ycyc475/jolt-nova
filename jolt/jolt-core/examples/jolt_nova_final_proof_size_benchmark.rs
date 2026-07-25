use std::{
    error::Error,
    fmt, io,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
    thread,
    time::{Duration, Instant},
};

use clap::{Parser, ValueEnum};
use common::{
    constants::{RAM_START_ADDRESS, REGISTER_COUNT},
    jolt_device::MemoryConfig,
};
use jolt_core::{
    ark_bn254,
    zkvm::{
        block::{
            validate_block_chain, BlockProofPipeline, BlockPublicInput, JoltNovaReportOutputFormat,
            NovaBlockProofPipelineFinalProofSizeBenchmarkArtifact, NovaFoldingBackend,
            JOLT_NOVA_FINAL_PROOF_SIZE_SCALING_REPORT_KIND, JOLT_NOVA_REPORT_SCHEMA_VERSION,
        },
        bytecode::BytecodePreprocessing,
    },
};
use jolt_riscv::RV64IMAC_JOLT;
use serde::{Deserialize, Serialize};
use sha3::{Digest as ShaDigest, Sha3_256};
use tracer::{
    instruction::Cycle, LazyTracer, MachineBoundaryState, TraceBlock, TracerInlineExpansionProvider,
};

const DEFAULT_OUTPUT_PATH: &str = "benchmark-runs/jolt-nova/final-proof-size-scaling.json";
const RUNNER_NAME: &str = "jolt_nova_final_proof_size_benchmark";
const MANIFEST_SCHEMA_VERSION: &str = "jolt-nova-benchmark-runner-v3";
const PERFORMANCE_SCHEMA_VERSION: &str = "jolt-nova-performance-baseline-v1";
const TRACE_FILE_SCHEMA_VERSION: &str = "jolt-nova-trace-blocks-v1";
const TRACE_BUNDLE_SCHEMA_VERSION: &str = "jolt-nova-trace-bundle-v1";
const MAX_MEMORY_SAMPLE_INTERVAL_MS: u64 = 1_000;

/// Jolt-Nova runner for synthetic, serialized, and real RV64 ELF trace blocks.
///
/// ELF mode executes the program with Jolt's tracer, exports a bytecode-bound
/// trace bundle, reads that bundle back, and runs the Nova folding/final-proof
/// reporting path. Synthetic mode remains available for fast smoke tests.
#[derive(Debug, Clone, Parser)]
struct Args {
    /// Source used to obtain trace blocks.
    #[arg(long, value_enum, default_value_t = TraceSource::Synthetic)]
    trace_source: TraceSource,

    /// Synthetic trace profile to generate.
    #[arg(long, value_enum, default_value_t = TraceProfile::SyntheticNoop)]
    trace_profile: TraceProfile,

    /// Input path for a versioned Jolt-Nova trace-block JSON document.
    #[arg(long)]
    trace_input: Option<PathBuf>,

    /// RV64 ELF program to execute when `--trace-source elf` is selected.
    #[arg(long)]
    elf_input: Option<PathBuf>,

    /// Versioned trace bundle written by the ELF exporter.
    #[arg(long)]
    trace_output: Option<PathBuf>,

    /// Raw guest input bytes supplied to the ELF program.
    #[arg(long)]
    guest_input: Option<PathBuf>,

    /// Raw untrusted-advice bytes supplied to the ELF program.
    #[arg(long)]
    untrusted_advice: Option<PathBuf>,

    /// Raw trusted-advice bytes supplied to the ELF program.
    #[arg(long)]
    trusted_advice: Option<PathBuf>,

    /// Soft target number of trace cycles per ELF trace block.
    #[arg(long, default_value_t = 1024)]
    trace_block_size: usize,

    /// Number of synthetic trace blocks to generate.
    #[arg(long, default_value_t = 2)]
    blocks: usize,

    /// Active cycles per synthetic block.
    #[arg(long, default_value_t = 2)]
    cycles_per_block: usize,

    /// Comma-separated block prefix lengths to report, e.g. `1,2,4,8`.
    ///
    /// If omitted, every prefix from `1..=blocks` is reported.
    #[arg(long, value_delimiter = ',')]
    block_counts: Vec<usize>,

    /// Output JSON artifact path.
    #[arg(long, default_value = DEFAULT_OUTPUT_PATH)]
    output: PathBuf,

    /// Optional manifest path recording the runner inputs and report summary.
    ///
    /// If omitted, the runner writes next to the report using
    /// `<report-stem>.manifest.json`.
    #[arg(long)]
    manifest_output: Option<PathBuf>,

    /// Optional performance artifact path containing timings, throughput, and
    /// best-effort peak physical memory.
    ///
    /// If omitted, the runner writes next to the report using
    /// `<report-stem>.performance.json`.
    #[arg(long)]
    performance_output: Option<PathBuf>,

    /// Interval used by the best-effort peak-memory sampler.
    #[arg(long, default_value_t = 10)]
    memory_sample_interval_ms: u64,

    /// Repeated byte used to form the synthetic program digest.
    #[arg(long, default_value_t = 9)]
    program_digest_byte: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum TraceSource {
    /// Generate trace blocks locally from synthetic runner options.
    Synthetic,
    /// Load trace blocks from a versioned JSON document.
    TraceFile,
    /// Execute an RV64 ELF, export a trace bundle, then read it back for proving.
    Elf,
}

impl TraceSource {
    fn as_str(self) -> &'static str {
        match self {
            Self::Synthetic => "synthetic",
            Self::TraceFile => "trace-file",
            Self::Elf => "elf",
        }
    }
}

impl fmt::Display for TraceSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum TraceProfile {
    /// Contiguous no-op blocks with stable machine boundary state.
    SyntheticNoop,
}

impl TraceProfile {
    fn as_str(self) -> &'static str {
        match self {
            Self::SyntheticNoop => "synthetic-noop",
        }
    }
}

impl fmt::Display for TraceProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug)]
struct LoadedTraceBlocks {
    blocks: Vec<TraceBlock>,
    bytecode: BytecodePreprocessing,
    program_digest: Option<[u8; 32]>,
    file_metadata: Option<TraceFileMetadata>,
}

#[derive(Debug)]
struct TraceFileMetadata {
    schema_version: String,
    sha3_256: [u8; 32],
    elf_sha3_256: Option<[u8; 32]>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct PerformanceBaselineArtifact {
    schema_version: String,
    runner: String,
    trace_source: String,
    build_profile: String,
    target_os: String,
    target_arch: String,
    source_block_count: usize,
    reported_block_counts: Vec<usize>,
    largest_reported_block_count: usize,
    largest_reported_active_cycles: usize,
    processed_block_count: usize,
    processed_active_cycles: usize,
    timings_ms: PerformanceTimings,
    throughput: PerformanceThroughput,
    memory: PerformanceMemory,
    report_path: String,
    manifest_path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct PerformanceTimings {
    trace_load: u64,
    nova_fold_and_spartan_report: u64,
    manifest_write: u64,
    measured_total: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct PerformanceThroughput {
    proving_processed_active_cycles_per_second: Option<f64>,
    end_to_end_processed_active_cycles_per_second: Option<f64>,
    end_to_end_processed_blocks_per_second: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct PerformanceMemory {
    sample_interval_ms: u64,
    initial_physical_bytes: Option<u64>,
    peak_physical_bytes: Option<u64>,
    peak_delta_bytes: Option<u64>,
}

#[derive(Debug)]
struct RunMeasurements {
    trace_load: Duration,
    prove_and_report: Duration,
    manifest_write: Duration,
    measured_total: Duration,
    memory: PerformanceMemory,
}

#[derive(Debug)]
struct PeakMemorySampler {
    stop: Arc<AtomicBool>,
    peak_physical_bytes: Arc<AtomicU64>,
    initial_physical_bytes: Option<u64>,
    handle: Option<thread::JoinHandle<()>>,
    sample_interval_ms: u64,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct TraceFileDocument {
    schema_version: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    program: Option<SerializedProgramBinding>,
    blocks: Vec<SerializedTraceBlock>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SerializedProgramBinding {
    elf_sha3_256: String,
    program_digest_sha3_256: String,
    bytecode: BytecodePreprocessing,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SerializedTraceBlock {
    block_index: usize,
    global_cycle_start: usize,
    active_cycles: usize,
    target_size: usize,
    start_state: SerializedMachineBoundaryState,
    end_state: SerializedMachineBoundaryState,
    cycles: Vec<Cycle>,
    ended_at_tick_boundary: bool,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SerializedMachineBoundaryState {
    global_cycle: usize,
    emulator_trace_len: usize,
    pc: u64,
    registers: Vec<i64>,
    terminated: bool,
}

impl From<&MachineBoundaryState> for SerializedMachineBoundaryState {
    fn from(state: &MachineBoundaryState) -> Self {
        Self {
            global_cycle: state.global_cycle,
            emulator_trace_len: state.emulator_trace_len,
            pc: state.pc,
            registers: state.registers.to_vec(),
            terminated: state.terminated,
        }
    }
}

impl From<&TraceBlock> for SerializedTraceBlock {
    fn from(block: &TraceBlock) -> Self {
        Self {
            block_index: block.block_index,
            global_cycle_start: block.global_cycle_start,
            active_cycles: block.active_cycles,
            target_size: block.target_size,
            start_state: (&block.start_state).into(),
            end_state: (&block.end_state).into(),
            cycles: block.cycles.clone(),
            ended_at_tick_boundary: block.ended_at_tick_boundary,
        }
    }
}

impl TryFrom<SerializedMachineBoundaryState> for MachineBoundaryState {
    type Error = String;

    fn try_from(state: SerializedMachineBoundaryState) -> Result<Self, Self::Error> {
        let actual_register_count = state.registers.len();
        let registers = state.registers.try_into().map_err(|_| {
            format!(
                "register count mismatch: expected {}, decoded {actual_register_count}",
                REGISTER_COUNT
            )
        })?;
        Ok(Self {
            global_cycle: state.global_cycle,
            emulator_trace_len: state.emulator_trace_len,
            pc: state.pc,
            registers,
            terminated: state.terminated,
        })
    }
}

impl TryFrom<SerializedTraceBlock> for TraceBlock {
    type Error = String;

    fn try_from(block: SerializedTraceBlock) -> Result<Self, Self::Error> {
        let block_index = block.block_index;
        let start_state = block
            .start_state
            .try_into()
            .map_err(|error| format!("block {block_index} start_state {error}"))?;
        let end_state = block
            .end_state
            .try_into()
            .map_err(|error| format!("block {block_index} end_state {error}"))?;
        Ok(Self {
            block_index: block.block_index,
            global_cycle_start: block.global_cycle_start,
            active_cycles: block.active_cycles,
            target_size: block.target_size,
            start_state,
            end_state,
            cycles: block.cycles,
            ended_at_tick_boundary: block.ended_at_tick_boundary,
        })
    }
}

impl PeakMemorySampler {
    fn start(sample_interval_ms: u64) -> Self {
        let initial_physical_bytes = current_physical_memory_bytes();
        let peak_physical_bytes =
            Arc::new(AtomicU64::new(initial_physical_bytes.unwrap_or_default()));
        let stop = Arc::new(AtomicBool::new(false));
        let sampler_peak = Arc::clone(&peak_physical_bytes);
        let sampler_stop = Arc::clone(&stop);
        let sample_interval = Duration::from_millis(sample_interval_ms);
        let handle = thread::spawn(move || {
            while !sampler_stop.load(Ordering::Relaxed) {
                if let Some(physical_bytes) = current_physical_memory_bytes() {
                    sampler_peak.fetch_max(physical_bytes, Ordering::Relaxed);
                }
                thread::sleep(sample_interval);
            }
        });

        Self {
            stop,
            peak_physical_bytes,
            initial_physical_bytes,
            handle: Some(handle),
            sample_interval_ms,
        }
    }

    fn sample_now(&self) {
        if let Some(physical_bytes) = current_physical_memory_bytes() {
            self.peak_physical_bytes
                .fetch_max(physical_bytes, Ordering::Relaxed);
        }
    }

    fn finish(mut self) -> PerformanceMemory {
        self.sample_now();
        self.stop_and_join();
        let observed_peak = self.peak_physical_bytes.load(Ordering::Relaxed);
        let peak_physical_bytes =
            (self.initial_physical_bytes.is_some() || observed_peak > 0).then_some(observed_peak);
        PerformanceMemory {
            sample_interval_ms: self.sample_interval_ms,
            initial_physical_bytes: self.initial_physical_bytes,
            peak_physical_bytes,
            peak_delta_bytes: peak_physical_bytes
                .zip(self.initial_physical_bytes)
                .map(|(peak, initial)| peak.saturating_sub(initial)),
        }
    }

    fn stop_and_join(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for PeakMemorySampler {
    fn drop(&mut self) {
        self.stop_and_join();
    }
}

fn current_physical_memory_bytes() -> Option<u64> {
    memory_stats::memory_stats().map(|stats| stats.physical_mem as u64)
}

fn main() {
    if let Err(error) = run(Args::parse()) {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run(args: Args) -> Result<(), Box<dyn Error>> {
    validate_runner_args(&args).map_err(invalid_input)?;
    let measured_total_started = Instant::now();
    let memory_sampler = PeakMemorySampler::start(args.memory_sample_interval_ms);

    let trace_load_started = Instant::now();
    let loaded_trace = load_trace_blocks(&args).map_err(invalid_input)?;
    let trace_load = trace_load_started.elapsed();
    let blocks = &loaded_trace.blocks;
    let block_counts =
        normalized_block_counts(&args, blocks.len(), loaded_trace.program_digest.is_some())
            .map_err(invalid_input)?;
    let manifest_output = normalized_manifest_output(&args.output, args.manifest_output.as_ref());
    let performance_output =
        normalized_performance_output(&args.output, args.performance_output.as_ref());
    let program_digest = loaded_trace
        .program_digest
        .unwrap_or([args.program_digest_byte; 32]);
    let pipeline = BlockProofPipeline::<_, ark_bn254::Fr, NovaFoldingBackend>::with_backend(
        program_digest,
        NovaFoldingBackend::default(),
    );

    let prove_and_report_started = Instant::now();
    let artifact = pipeline.prove_block_prefixes_and_write_final_proof_size_benchmark_artifact(
        &loaded_trace.bytecode,
        blocks,
        &block_counts,
        JoltNovaReportOutputFormat::Json,
        &args.output,
    )?;
    let prove_and_report = prove_and_report_started.elapsed();

    let manifest_write_started = Instant::now();
    write_manifest(
        &args,
        &block_counts,
        blocks.len(),
        loaded_trace.file_metadata.as_ref(),
        &artifact,
        &manifest_output,
    )?;
    let manifest_write = manifest_write_started.elapsed();
    let memory = memory_sampler.finish();
    let measurements = RunMeasurements {
        trace_load,
        prove_and_report,
        manifest_write,
        measured_total: measured_total_started.elapsed(),
        memory,
    };
    let performance = build_performance_artifact(
        &args,
        &block_counts,
        blocks,
        &manifest_output,
        &measurements,
    );
    write_performance_artifact(&performance_output, &performance)?;

    println!("wrote {}", manifest_path_string(&args.output));
    println!("manifest {}", manifest_path_string(&manifest_output));
    println!("performance {}", manifest_path_string(&performance_output));
    println!("trace_source {}", args.trace_source);
    println!(
        "trace_profile {}",
        selected_trace_profile(&args).unwrap_or("none")
    );
    if let Some(metadata) = &loaded_trace.file_metadata {
        println!("trace_file_schema {}", metadata.schema_version);
        println!("trace_input_sha3_256 {}", hex_digest(&metadata.sha3_256));
        if let Some(elf_sha3_256) = metadata.elf_sha3_256 {
            println!("elf_sha3_256 {}", hex_digest(&elf_sha3_256));
        }
    }
    println!("program_digest_sha3_256 {}", hex_digest(&program_digest));
    println!("format {}", artifact.output_format);
    println!("rows {}", artifact.report.rows.len());
    println!("bytes {}", artifact.serialized_bytes().len());
    println!(
        "measured_total_ms {}",
        performance.timings_ms.measured_total
    );
    println!(
        "peak_physical_bytes {:?}",
        performance.memory.peak_physical_bytes
    );
    for row in &artifact.report.rows {
        println!(
            "block_count={} active_cycles={} recursive_snark_bytes={:?} spartan_total_bytes={}",
            row.block_count,
            row.total_active_cycles,
            row.recursive_snark_bytes_len,
            row.final_proof_size_comparison
                .spartan
                .proof_total_bytes_len
        );
    }

    Ok(())
}

fn invalid_input(error: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, error)
}

fn validate_runner_args(args: &Args) -> Result<(), String> {
    if !(1..=MAX_MEMORY_SAMPLE_INTERVAL_MS).contains(&args.memory_sample_interval_ms) {
        return Err(format!(
            "memory-sample-interval-ms must be in 1..={MAX_MEMORY_SAMPLE_INTERVAL_MS}"
        ));
    }

    let manifest_output = normalized_manifest_output(&args.output, args.manifest_output.as_ref());
    let performance_output =
        normalized_performance_output(&args.output, args.performance_output.as_ref());
    let mut outputs = vec![
        ("report output", args.output.as_path()),
        ("manifest output", manifest_output.as_path()),
        ("performance output", performance_output.as_path()),
    ];
    if let Some(trace_output) = args.trace_output.as_deref() {
        outputs.push(("trace output", trace_output));
    }

    for index in 0..outputs.len() {
        let (left_label, left_path) = outputs[index];
        for &(right_label, right_path) in &outputs[index + 1..] {
            if left_path == right_path {
                return Err(format!(
                    "{left_label} and {right_label} must use distinct paths: {}",
                    manifest_path_string(left_path)
                ));
            }
        }
    }

    let inputs = [
        ("trace input", args.trace_input.as_deref()),
        ("ELF input", args.elf_input.as_deref()),
        ("guest input", args.guest_input.as_deref()),
        ("untrusted advice", args.untrusted_advice.as_deref()),
        ("trusted advice", args.trusted_advice.as_deref()),
    ];
    for (output_label, output_path) in outputs {
        for &(input_label, input_path) in &inputs {
            if input_path == Some(output_path) {
                return Err(format!(
                    "{output_label} must not overwrite {input_label}: {}",
                    manifest_path_string(output_path)
                ));
            }
        }
    }

    Ok(())
}

fn normalized_manifest_output(report_output: &Path, manifest_output: Option<&PathBuf>) -> PathBuf {
    manifest_output
        .cloned()
        .unwrap_or_else(|| default_manifest_output_path(report_output))
}

fn normalized_performance_output(
    report_output: &Path,
    performance_output: Option<&PathBuf>,
) -> PathBuf {
    performance_output
        .cloned()
        .unwrap_or_else(|| default_performance_output_path(report_output))
}

fn default_manifest_output_path(report_output: &Path) -> PathBuf {
    let mut manifest_output = report_output.to_path_buf();
    let stem = report_output
        .file_stem()
        .and_then(|stem| stem.to_str())
        .filter(|stem| !stem.is_empty())
        .unwrap_or("jolt-nova-final-proof-size-scaling");
    manifest_output.set_file_name(format!("{stem}.manifest.json"));
    manifest_output
}

fn default_performance_output_path(report_output: &Path) -> PathBuf {
    let mut performance_output = report_output.to_path_buf();
    let stem = report_output
        .file_stem()
        .and_then(|stem| stem.to_str())
        .filter(|stem| !stem.is_empty())
        .unwrap_or("jolt-nova-final-proof-size-scaling");
    performance_output.set_file_name(format!("{stem}.performance.json"));
    performance_output
}

fn normalized_block_counts(
    args: &Args,
    loaded_blocks: usize,
    has_program_binding: bool,
) -> Result<Vec<usize>, String> {
    if loaded_blocks == 0 {
        return Err("loaded trace must contain at least one block".to_string());
    }

    let block_counts = if args.block_counts.is_empty() && has_program_binding {
        vec![loaded_blocks]
    } else if args.block_counts.is_empty() {
        (1..=loaded_blocks).collect::<Vec<_>>()
    } else {
        args.block_counts.clone()
    };

    validate_block_counts(loaded_blocks, &block_counts)?;
    if has_program_binding && block_counts.iter().any(|&count| count != loaded_blocks) {
        return Err(
            "program-bound trace bundles currently support only the complete block sequence; \
             partial prefixes require an external lookahead binding"
                .to_string(),
        );
    }
    Ok(block_counts)
}

fn validate_block_counts(blocks: usize, block_counts: &[usize]) -> Result<(), String> {
    if block_counts.is_empty() {
        return Err("block-counts must contain at least one prefix length".to_string());
    }

    let mut previous = 0;
    for &block_count in block_counts {
        if block_count == 0 || block_count > blocks {
            return Err(format!(
                "block count {block_count} is out of range; expected 1..={blocks}"
            ));
        }
        if block_count <= previous {
            return Err("block-counts must be strictly increasing".to_string());
        }
        previous = block_count;
    }

    Ok(())
}

fn load_trace_blocks(args: &Args) -> Result<LoadedTraceBlocks, String> {
    match args.trace_source {
        TraceSource::Synthetic => {
            reject_non_synthetic_inputs(args)?;
            Ok(LoadedTraceBlocks {
                blocks: build_trace_blocks(args.trace_profile, args.blocks, args.cycles_per_block)?,
                bytecode: BytecodePreprocessing::default(),
                program_digest: None,
                file_metadata: None,
            })
        }
        TraceSource::TraceFile => {
            reject_elf_inputs(args)?;
            let trace_input = args.trace_input.as_ref().ok_or_else(|| {
                "trace-source trace-file requires --trace-input <path>".to_string()
            })?;
            load_trace_file(trace_input)
        }
        TraceSource::Elf => export_elf_trace_bundle(args),
    }
}

fn reject_non_synthetic_inputs(args: &Args) -> Result<(), String> {
    if let Some(trace_input) = &args.trace_input {
        return Err(format!(
            "trace-input {} is only valid with --trace-source trace-file",
            manifest_path_string(trace_input)
        ));
    }
    reject_elf_inputs(args)
}

fn reject_elf_inputs(args: &Args) -> Result<(), String> {
    for (name, value) in [
        ("elf-input", args.elf_input.as_ref()),
        ("trace-output", args.trace_output.as_ref()),
        ("guest-input", args.guest_input.as_ref()),
        ("untrusted-advice", args.untrusted_advice.as_ref()),
        ("trusted-advice", args.trusted_advice.as_ref()),
    ] {
        if let Some(path) = value {
            return Err(format!(
                "{name} {} is only valid with --trace-source elf",
                manifest_path_string(path)
            ));
        }
    }
    Ok(())
}

fn export_elf_trace_bundle(args: &Args) -> Result<LoadedTraceBlocks, String> {
    if args.trace_input.is_some() {
        return Err("trace-input is not valid with --trace-source elf".to_string());
    }
    if args.trace_block_size == 0 {
        return Err("trace-block-size must be greater than zero".to_string());
    }
    let elf_input = args
        .elf_input
        .as_ref()
        .ok_or_else(|| "trace-source elf requires --elf-input <path>".to_string())?;
    let trace_output = args
        .trace_output
        .as_ref()
        .ok_or_else(|| "trace-source elf requires --trace-output <path>".to_string())?;
    let elf_bytes = read_input_bytes("ELF input", elf_input)?;
    let guest_input = read_optional_input_bytes("guest input", args.guest_input.as_ref())?;
    let untrusted_advice =
        read_optional_input_bytes("untrusted advice", args.untrusted_advice.as_ref())?;
    let trusted_advice = read_optional_input_bytes("trusted advice", args.trusted_advice.as_ref())?;

    let mut inline_provider = TracerInlineExpansionProvider::new();
    let program = jolt_program::build_jolt_program_with_inline_provider(
        &elf_bytes,
        &mut inline_provider,
        RV64IMAC_JOLT,
    )
    .map_err(|error| format!("failed to decode and expand ELF program: {error}"))?;
    let bytecode = BytecodePreprocessing::preprocess(
        program.expanded_bytecode,
        program.entry_address,
        program.profile,
    )
    .map_err(|error| format!("failed to preprocess ELF bytecode: {error}"))?;
    let program_size = program
        .program_end
        .checked_sub(RAM_START_ADDRESS)
        .ok_or_else(|| "ELF program end is below the Jolt RAM start address".to_string())?;
    let memory_config = MemoryConfig {
        program_size: Some(program_size),
        ..Default::default()
    };

    let mut block_iterator = tracer::trace_blocks(
        &elf_bytes,
        Some(elf_input),
        &guest_input,
        &untrusted_advice,
        &trusted_advice,
        &memory_config,
        None,
        args.trace_block_size,
    );
    let mut blocks = Vec::new();
    for block in block_iterator.by_ref() {
        blocks.push(block);
    }
    let tracer = block_iterator.into_inner().lazy_tracer;
    if tracer.has_panicked() {
        return Err("ELF guest panicked while generating trace blocks".to_string());
    }
    if !tracer.has_terminated() {
        return Err("ELF tracer did not terminate after generating trace blocks".to_string());
    }
    if let Some(last_block) = blocks.last_mut() {
        last_block.end_state =
            tracer.boundary_state(last_block.global_cycle_start + last_block.active_cycles);
    }
    validate_loaded_trace_blocks(&blocks)?;

    let elf_sha3_256 = Sha3_256::digest(&elf_bytes).into();
    let program_digest = bytecode_digest(&bytecode)?;
    write_trace_bundle(
        trace_output,
        &blocks,
        bytecode,
        elf_sha3_256,
        program_digest,
    )?;

    let loaded = load_trace_file(trace_output)?;
    if loaded.program_digest != Some(program_digest) {
        return Err("exported trace bundle program digest changed during readback".to_string());
    }
    Ok(loaded)
}

fn read_input_bytes(label: &str, path: &Path) -> Result<Vec<u8>, String> {
    std::fs::read(path).map_err(|error| {
        format!(
            "failed to read {label} {}: {error}",
            manifest_path_string(path)
        )
    })
}

fn read_optional_input_bytes(label: &str, path: Option<&PathBuf>) -> Result<Vec<u8>, String> {
    path.map_or_else(|| Ok(Vec::new()), |path| read_input_bytes(label, path))
}

fn write_trace_bundle(
    trace_output: &Path,
    blocks: &[TraceBlock],
    bytecode: BytecodePreprocessing,
    elf_sha3_256: [u8; 32],
    program_digest: [u8; 32],
) -> Result<(), String> {
    if let Some(parent) = trace_output.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).map_err(|error| {
                format!(
                    "failed to create trace output directory {}: {error}",
                    manifest_path_string(parent)
                )
            })?;
        }
    }
    let document = TraceFileDocument {
        schema_version: TRACE_BUNDLE_SCHEMA_VERSION.to_string(),
        program: Some(SerializedProgramBinding {
            elf_sha3_256: hex_digest(&elf_sha3_256),
            program_digest_sha3_256: hex_digest(&program_digest),
            bytecode,
        }),
        blocks: blocks.iter().map(SerializedTraceBlock::from).collect(),
    };
    let json = serde_json::to_vec_pretty(&document)
        .map_err(|error| format!("failed to serialize trace bundle: {error}"))?;
    std::fs::write(trace_output, json).map_err(|error| {
        format!(
            "failed to write trace bundle {}: {error}",
            manifest_path_string(trace_output)
        )
    })
}

fn bytecode_digest(bytecode: &BytecodePreprocessing) -> Result<[u8; 32], String> {
    let bytes = serde_json::to_vec(bytecode)
        .map_err(|error| format!("failed to serialize bytecode for digest: {error}"))?;
    Ok(Sha3_256::digest(bytes).into())
}

fn load_trace_file(trace_input: &Path) -> Result<LoadedTraceBlocks, String> {
    let input_bytes = std::fs::read(trace_input).map_err(|error| {
        format!(
            "failed to read trace input {}: {error}",
            manifest_path_string(trace_input)
        )
    })?;
    let document = serde_json::from_slice::<TraceFileDocument>(&input_bytes).map_err(|error| {
        format!(
            "failed to parse trace input {} as JSON: {error}",
            manifest_path_string(trace_input)
        )
    })?;
    if document.schema_version != TRACE_FILE_SCHEMA_VERSION
        && document.schema_version != TRACE_BUNDLE_SCHEMA_VERSION
    {
        return Err(format!(
            "unsupported trace schema version {:?}; expected {:?} or {:?}",
            document.schema_version, TRACE_FILE_SCHEMA_VERSION, TRACE_BUNDLE_SCHEMA_VERSION
        ));
    }

    let (bytecode, program_digest, elf_sha3_256) =
        match (document.schema_version.as_str(), document.program) {
            (TRACE_FILE_SCHEMA_VERSION, None) => (BytecodePreprocessing::default(), None, None),
            (TRACE_FILE_SCHEMA_VERSION, Some(_)) => {
                return Err(
                    "legacy trace-block files must not contain a program binding".to_string(),
                )
            }
            (TRACE_BUNDLE_SCHEMA_VERSION, Some(program)) => {
                let declared_program_digest =
                    parse_hex_digest("program_digest_sha3_256", &program.program_digest_sha3_256)?;
                let actual_program_digest = bytecode_digest(&program.bytecode)?;
                if declared_program_digest != actual_program_digest {
                    return Err(format!(
                        "trace bundle bytecode digest mismatch: declared {}, computed {}",
                        hex_digest(&declared_program_digest),
                        hex_digest(&actual_program_digest)
                    ));
                }
                let elf_sha3_256 = parse_hex_digest("elf_sha3_256", &program.elf_sha3_256)?;
                (
                    program.bytecode,
                    Some(declared_program_digest),
                    Some(elf_sha3_256),
                )
            }
            (TRACE_BUNDLE_SCHEMA_VERSION, None) => {
                return Err("trace bundle is missing its program binding".to_string())
            }
            _ => unreachable!("trace schema was validated above"),
        };
    let blocks = document
        .blocks
        .into_iter()
        .map(TraceBlock::try_from)
        .collect::<Result<Vec<_>, _>>()?;
    validate_loaded_trace_blocks(&blocks)?;

    Ok(LoadedTraceBlocks {
        blocks,
        bytecode,
        program_digest,
        file_metadata: Some(TraceFileMetadata {
            schema_version: document.schema_version,
            sha3_256: Sha3_256::digest(&input_bytes).into(),
            elf_sha3_256,
        }),
    })
}

fn parse_hex_digest(label: &str, value: &str) -> Result<[u8; 32], String> {
    if value.len() != 64 {
        return Err(format!(
            "{label} must contain exactly 64 hexadecimal digits"
        ));
    }
    let mut digest = [0u8; 32];
    for (index, byte) in digest.iter_mut().enumerate() {
        let offset = index * 2;
        *byte = u8::from_str_radix(&value[offset..offset + 2], 16)
            .map_err(|_| format!("{label} contains a non-hexadecimal digit"))?;
    }
    Ok(digest)
}

fn validate_loaded_trace_blocks(blocks: &[TraceBlock]) -> Result<(), String> {
    let first = blocks
        .first()
        .ok_or_else(|| "trace file must contain at least one block".to_string())?;
    if first.block_index != 0 {
        return Err(format!(
            "trace file must start at block index 0; got {}",
            first.block_index
        ));
    }
    if first.global_cycle_start != 0 {
        return Err(format!(
            "trace file must start at global cycle 0; got {}",
            first.global_cycle_start
        ));
    }

    for block in blocks {
        if block.active_cycles != block.cycles.len() {
            return Err(format!(
                "block {} active cycle count mismatch: declared {}, decoded {}",
                block.block_index,
                block.active_cycles,
                block.cycles.len()
            ));
        }
        if !block.ended_at_tick_boundary {
            return Err(format!(
                "block {} did not end at a tracer tick boundary",
                block.block_index
            ));
        }
    }

    let public_inputs = blocks
        .iter()
        .map(|block| BlockPublicInput::from_trace_block(block, [0u8; 32]))
        .collect::<Vec<_>>();
    validate_block_chain(&public_inputs)
        .map_err(|error| format!("trace block chain validation failed: {error}"))
}

fn build_trace_blocks(
    trace_profile: TraceProfile,
    blocks: usize,
    cycles_per_block: usize,
) -> Result<Vec<TraceBlock>, String> {
    match trace_profile {
        TraceProfile::SyntheticNoop => build_noop_trace_blocks(blocks, cycles_per_block),
    }
}

fn build_noop_trace_blocks(
    blocks: usize,
    cycles_per_block: usize,
) -> Result<Vec<TraceBlock>, String> {
    if blocks == 0 {
        return Err("blocks must be greater than zero".to_string());
    }
    if cycles_per_block == 0 {
        return Err("cycles-per-block must be greater than zero".to_string());
    }

    let mut trace_blocks = Vec::with_capacity(blocks);
    let mut start = boundary(0, 0);
    for block_index in 0..blocks {
        let end_global_cycle = start.global_cycle + cycles_per_block;
        let end = boundary(end_global_cycle, start.pc);
        trace_blocks.push(TraceBlock {
            block_index,
            global_cycle_start: start.global_cycle,
            active_cycles: cycles_per_block,
            target_size: cycles_per_block,
            start_state: start.clone(),
            end_state: end.clone(),
            cycles: vec![Cycle::NoOp; cycles_per_block],
            ended_at_tick_boundary: true,
        });
        start = end;
    }

    Ok(trace_blocks)
}

fn build_performance_artifact(
    args: &Args,
    block_counts: &[usize],
    blocks: &[TraceBlock],
    manifest_output: &Path,
    measurements: &RunMeasurements,
) -> PerformanceBaselineArtifact {
    let cumulative_active_cycles = blocks
        .iter()
        .scan(0usize, |total, block| {
            *total = total.saturating_add(block.active_cycles);
            Some(*total)
        })
        .collect::<Vec<_>>();
    let largest_reported_block_count = block_counts.last().copied().unwrap_or_default();
    let largest_reported_active_cycles = largest_reported_block_count
        .checked_sub(1)
        .and_then(|index| cumulative_active_cycles.get(index).copied())
        .unwrap_or_default();
    let processed_block_count = block_counts
        .iter()
        .copied()
        .fold(0usize, usize::saturating_add);
    let processed_active_cycles = block_counts.iter().copied().fold(0usize, |total, count| {
        total.saturating_add(
            count
                .checked_sub(1)
                .and_then(|index| cumulative_active_cycles.get(index).copied())
                .unwrap_or_default(),
        )
    });
    PerformanceBaselineArtifact {
        schema_version: PERFORMANCE_SCHEMA_VERSION.to_string(),
        runner: RUNNER_NAME.to_string(),
        trace_source: args.trace_source.as_str().to_string(),
        build_profile: if cfg!(debug_assertions) {
            "debug".to_string()
        } else {
            "release".to_string()
        },
        target_os: std::env::consts::OS.to_string(),
        target_arch: std::env::consts::ARCH.to_string(),
        source_block_count: blocks.len(),
        reported_block_counts: block_counts.to_vec(),
        largest_reported_block_count,
        largest_reported_active_cycles,
        processed_block_count,
        processed_active_cycles,
        timings_ms: PerformanceTimings {
            trace_load: duration_millis(measurements.trace_load),
            nova_fold_and_spartan_report: duration_millis(measurements.prove_and_report),
            manifest_write: duration_millis(measurements.manifest_write),
            measured_total: duration_millis(measurements.measured_total),
        },
        throughput: PerformanceThroughput {
            proving_processed_active_cycles_per_second: rate_per_second(
                processed_active_cycles,
                measurements.prove_and_report,
            ),
            end_to_end_processed_active_cycles_per_second: rate_per_second(
                processed_active_cycles,
                measurements.measured_total,
            ),
            end_to_end_processed_blocks_per_second: rate_per_second(
                processed_block_count,
                measurements.measured_total,
            ),
        },
        memory: measurements.memory.clone(),
        report_path: manifest_path_string(&args.output),
        manifest_path: manifest_path_string(manifest_output),
    }
}

fn write_performance_artifact(
    performance_output: &Path,
    artifact: &PerformanceBaselineArtifact,
) -> io::Result<()> {
    if let Some(parent) = performance_output.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let mut json = serde_json::to_vec_pretty(artifact).map_err(io::Error::other)?;
    json.push(b'\n');
    std::fs::write(performance_output, json)
}

fn duration_millis(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

fn rate_per_second(count: usize, duration: Duration) -> Option<f64> {
    let seconds = duration.as_secs_f64();
    (seconds > 0.0).then_some(count as f64 / seconds)
}

fn write_manifest(
    args: &Args,
    block_counts: &[usize],
    source_block_count: usize,
    trace_file_metadata: Option<&TraceFileMetadata>,
    artifact: &NovaBlockProofPipelineFinalProofSizeBenchmarkArtifact,
    manifest_output: &Path,
) -> io::Result<()> {
    if let Some(parent) = manifest_output.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }

    std::fs::write(
        manifest_output,
        build_manifest_json(
            args,
            block_counts,
            source_block_count,
            trace_file_metadata,
            artifact,
            manifest_output,
        ),
    )
}

fn build_manifest_json(
    args: &Args,
    block_counts: &[usize],
    source_block_count: usize,
    trace_file_metadata: Option<&TraceFileMetadata>,
    artifact: &NovaBlockProofPipelineFinalProofSizeBenchmarkArtifact,
    manifest_output: &Path,
) -> String {
    let last_row = artifact.report.rows.last();
    let last_total_active_cycles = last_row.map(|row| row.total_active_cycles).unwrap_or(0);
    let last_spartan_total_bytes = last_row
        .map(|row| {
            row.final_proof_size_comparison
                .spartan
                .proof_total_bytes_len
        })
        .unwrap_or(0);
    let trace_input_sha3_256 = trace_file_metadata.map(|metadata| hex_digest(&metadata.sha3_256));

    let mut json = String::new();
    json.push('{');
    append_json_string_field(&mut json, "schema_version", MANIFEST_SCHEMA_VERSION);
    json.push(',');
    append_json_string_field(&mut json, "runner", RUNNER_NAME);
    json.push(',');
    append_json_string_field(&mut json, "trace_source", args.trace_source.as_str());
    json.push(',');
    append_json_optional_string_field(&mut json, "trace_profile", selected_trace_profile(args));
    json.push(',');
    append_json_optional_path_field(&mut json, "trace_input_path", selected_trace_path(args));
    json.push(',');
    append_json_optional_path_field(&mut json, "elf_input_path", args.elf_input.as_ref());
    json.push(',');
    append_json_optional_path_field(&mut json, "trace_output_path", args.trace_output.as_ref());
    json.push(',');
    append_json_optional_string_field(
        &mut json,
        "trace_file_schema_version",
        trace_file_metadata.map(|metadata| metadata.schema_version.as_str()),
    );
    json.push(',');
    append_json_optional_string_field(
        &mut json,
        "trace_input_sha3_256",
        trace_input_sha3_256.as_deref(),
    );
    json.push(',');
    append_json_string_field(
        &mut json,
        "report_schema_version",
        JOLT_NOVA_REPORT_SCHEMA_VERSION,
    );
    json.push(',');
    append_json_string_field(
        &mut json,
        "report_kind",
        JOLT_NOVA_FINAL_PROOF_SIZE_SCALING_REPORT_KIND,
    );
    json.push(',');
    append_json_string_field(
        &mut json,
        "report_output_format",
        artifact.output_format.as_str(),
    );
    json.push(',');
    append_json_string_field(
        &mut json,
        "report_path",
        &manifest_path_string(&args.output),
    );
    json.push(',');
    append_json_string_field(
        &mut json,
        "manifest_path",
        &manifest_path_string(manifest_output),
    );
    json.push(',');
    append_json_string_field(
        &mut json,
        "performance_schema_version",
        PERFORMANCE_SCHEMA_VERSION,
    );
    json.push(',');
    append_json_string_field(
        &mut json,
        "performance_path",
        &manifest_path_string(&normalized_performance_output(
            &args.output,
            args.performance_output.as_ref(),
        )),
    );
    json.push(',');
    append_json_u64_field(
        &mut json,
        "memory_sample_interval_ms",
        args.memory_sample_interval_ms,
    );
    json.push(',');
    append_json_usize_field(&mut json, "source_block_count", source_block_count);
    json.push(',');
    append_json_optional_usize_field(
        &mut json,
        "blocks",
        (args.trace_source == TraceSource::Synthetic).then_some(args.blocks),
    );
    json.push(',');
    append_json_optional_usize_field(
        &mut json,
        "cycles_per_block",
        (args.trace_source == TraceSource::Synthetic).then_some(args.cycles_per_block),
    );
    json.push(',');
    append_json_optional_usize_field(
        &mut json,
        "trace_block_size",
        (args.trace_source == TraceSource::Elf).then_some(args.trace_block_size),
    );
    json.push(',');
    append_json_usize_array_field(&mut json, "block_counts", block_counts);
    json.push(',');
    append_json_optional_usize_field(
        &mut json,
        "program_digest_byte",
        (trace_file_metadata
            .and_then(|metadata| metadata.elf_sha3_256)
            .is_none())
        .then_some(args.program_digest_byte as usize),
    );
    json.push(',');
    append_json_usize_field(&mut json, "row_count", artifact.report.rows.len());
    json.push(',');
    append_json_usize_field(&mut json, "report_bytes", artifact.serialized_bytes().len());
    json.push(',');
    append_json_usize_field(
        &mut json,
        "last_total_active_cycles",
        last_total_active_cycles,
    );
    json.push(',');
    append_json_usize_field(
        &mut json,
        "last_spartan_total_bytes",
        last_spartan_total_bytes,
    );
    json.push('}');
    json
}

fn manifest_path_string(path: &Path) -> String {
    path.display().to_string().replace('\\', "/")
}

fn selected_trace_profile(args: &Args) -> Option<&'static str> {
    (args.trace_source == TraceSource::Synthetic).then(|| args.trace_profile.as_str())
}

fn selected_trace_path(args: &Args) -> Option<&PathBuf> {
    match args.trace_source {
        TraceSource::Synthetic => None,
        TraceSource::TraceFile => args.trace_input.as_ref(),
        TraceSource::Elf => args.trace_output.as_ref(),
    }
}

fn hex_digest(digest: &[u8; 32]) -> String {
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn append_json_string_field(json: &mut String, name: &str, value: &str) {
    append_json_string(json, name);
    json.push(':');
    append_json_string(json, value);
}

fn append_json_optional_string_field(json: &mut String, name: &str, value: Option<&str>) {
    append_json_string(json, name);
    json.push(':');
    match value {
        Some(value) => append_json_string(json, value),
        None => json.push_str("null"),
    }
}

fn append_json_usize_field(json: &mut String, name: &str, value: usize) {
    append_json_string(json, name);
    json.push(':');
    json.push_str(&value.to_string());
}

fn append_json_u64_field(json: &mut String, name: &str, value: u64) {
    append_json_string(json, name);
    json.push(':');
    json.push_str(&value.to_string());
}

fn append_json_optional_usize_field(json: &mut String, name: &str, value: Option<usize>) {
    append_json_string(json, name);
    json.push(':');
    match value {
        Some(value) => json.push_str(&value.to_string()),
        None => json.push_str("null"),
    }
}

fn append_json_usize_array_field(json: &mut String, name: &str, values: &[usize]) {
    append_json_string(json, name);
    json.push(':');
    json.push('[');
    for (index, value) in values.iter().enumerate() {
        if index > 0 {
            json.push(',');
        }
        json.push_str(&value.to_string());
    }
    json.push(']');
}

fn append_json_optional_path_field(json: &mut String, name: &str, value: Option<&PathBuf>) {
    append_json_string(json, name);
    json.push(':');
    match value {
        Some(path) => append_json_string(json, &manifest_path_string(path)),
        None => json.push_str("null"),
    }
}

fn append_json_string(json: &mut String, value: &str) {
    json.push('"');
    for ch in value.chars() {
        match ch {
            '"' => json.push_str("\\\""),
            '\\' => json.push_str("\\\\"),
            '\n' => json.push_str("\\n"),
            '\r' => json.push_str("\\r"),
            '\t' => json.push_str("\\t"),
            ch if ch.is_control() => {
                json.push_str("\\u");
                json.push_str(&format!("{:04x}", ch as u32));
            }
            ch => json.push(ch),
        }
    }
    json.push('"');
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_core::zkvm::block::{
        JoltNovaFinalProofSizeBaseline, JoltNovaFinalProofSizeComparison,
        NovaBlockProofPipelineFinalProofSizeScalingReport,
        NovaBlockProofPipelineFinalProofSizeScalingRow, SPARTAN_FINAL_PROOF_SYSTEM_NAME,
        SPARTAN_PLACEHOLDER_PROOF_SYSTEM_NAME,
    };
    use jolt_core::zkvm::r1cs::inputs::R1CSCycleInputs;

    #[test]
    fn default_block_counts_cover_every_prefix() {
        let args = Args {
            trace_source: TraceSource::Synthetic,
            trace_profile: TraceProfile::SyntheticNoop,
            trace_input: None,
            elf_input: None,
            trace_output: None,
            guest_input: None,
            untrusted_advice: None,
            trusted_advice: None,
            trace_block_size: 1024,
            blocks: 3,
            cycles_per_block: 2,
            block_counts: Vec::new(),
            output: PathBuf::from(DEFAULT_OUTPUT_PATH),
            manifest_output: None,
            performance_output: None,
            memory_sample_interval_ms: 10,
            program_digest_byte: 9,
        };

        assert_eq!(
            normalized_block_counts(&args, 3, false).unwrap(),
            vec![1, 2, 3]
        );
        assert_eq!(normalized_block_counts(&args, 3, true).unwrap(), vec![3]);

        let mut partial_real_trace = args;
        partial_real_trace.block_counts = vec![1, 3];
        assert!(normalized_block_counts(&partial_real_trace, 3, true)
            .unwrap_err()
            .contains("external lookahead binding"));
    }

    #[test]
    fn validates_explicit_block_counts() {
        assert_eq!(validate_block_counts(4, &[1, 2, 4]), Ok(()));
        assert!(validate_block_counts(4, &[])
            .unwrap_err()
            .contains("at least one"));
        assert!(validate_block_counts(4, &[0])
            .unwrap_err()
            .contains("out of range"));
        assert!(validate_block_counts(4, &[5])
            .unwrap_err()
            .contains("out of range"));
        assert!(validate_block_counts(4, &[2, 2])
            .unwrap_err()
            .contains("strictly increasing"));
        assert!(validate_block_counts(4, &[3, 2])
            .unwrap_err()
            .contains("strictly increasing"));
    }

    #[test]
    fn default_manifest_output_sits_next_to_report() {
        assert_eq!(
            default_manifest_output_path(Path::new("benchmark-runs/jolt-nova/report.json")),
            PathBuf::from("benchmark-runs/jolt-nova/report.manifest.json")
        );
        assert_eq!(
            default_manifest_output_path(Path::new("report")),
            PathBuf::from("report.manifest.json")
        );
    }

    #[test]
    fn default_performance_output_sits_next_to_report() {
        assert_eq!(
            default_performance_output_path(Path::new("benchmark-runs/jolt-nova/report.json")),
            PathBuf::from("benchmark-runs/jolt-nova/report.performance.json")
        );
        assert_eq!(
            default_performance_output_path(Path::new("report")),
            PathBuf::from("report.performance.json")
        );
    }

    #[test]
    fn runner_args_reject_bad_memory_interval_and_path_collisions() {
        let args = Args {
            trace_source: TraceSource::Synthetic,
            trace_profile: TraceProfile::SyntheticNoop,
            trace_input: None,
            elf_input: None,
            trace_output: None,
            guest_input: None,
            untrusted_advice: None,
            trusted_advice: None,
            trace_block_size: 1024,
            blocks: 2,
            cycles_per_block: 2,
            block_counts: vec![1, 2],
            output: PathBuf::from("benchmark-runs/jolt-nova/report.json"),
            manifest_output: None,
            performance_output: None,
            memory_sample_interval_ms: 10,
            program_digest_byte: 9,
        };
        assert_eq!(validate_runner_args(&args), Ok(()));

        let mut zero_interval = args.clone();
        zero_interval.memory_sample_interval_ms = 0;
        assert!(validate_runner_args(&zero_interval)
            .unwrap_err()
            .contains("must be in 1..=1000"));

        let mut excessive_interval = args.clone();
        excessive_interval.memory_sample_interval_ms = MAX_MEMORY_SAMPLE_INTERVAL_MS + 1;
        assert!(validate_runner_args(&excessive_interval)
            .unwrap_err()
            .contains("must be in 1..=1000"));

        let mut colliding_output = args.clone();
        colliding_output.manifest_output = Some(colliding_output.output.clone());
        assert!(validate_runner_args(&colliding_output)
            .unwrap_err()
            .contains("must use distinct paths"));

        let mut overwriting_input = args;
        overwriting_input.trace_source = TraceSource::TraceFile;
        overwriting_input.trace_input = Some(overwriting_input.output.clone());
        assert!(validate_runner_args(&overwriting_input)
            .unwrap_err()
            .contains("must not overwrite trace input"));
    }

    #[test]
    fn noop_trace_blocks_are_contiguous() {
        let args = Args {
            trace_source: TraceSource::Synthetic,
            trace_profile: TraceProfile::SyntheticNoop,
            trace_input: None,
            elf_input: None,
            trace_output: None,
            guest_input: None,
            untrusted_advice: None,
            trusted_advice: None,
            trace_block_size: 1024,
            blocks: 3,
            cycles_per_block: 2,
            block_counts: Vec::new(),
            output: PathBuf::from(DEFAULT_OUTPUT_PATH),
            manifest_output: None,
            performance_output: None,
            memory_sample_interval_ms: 10,
            program_digest_byte: 9,
        };
        let loaded_trace = load_trace_blocks(&args).unwrap();
        let blocks = loaded_trace.blocks;

        assert_eq!(blocks.len(), 3);
        assert!(loaded_trace.file_metadata.is_none());
        assert_eq!(blocks[0].block_index, 0);
        assert_eq!(blocks[0].global_cycle_start, 0);
        assert_eq!(blocks[0].active_cycles, 2);
        assert_eq!(blocks[0].start_state.pc, blocks[0].end_state.pc);
        assert_eq!(
            blocks[0].start_state.registers,
            blocks[0].end_state.registers
        );
        assert_eq!(blocks[0].end_state, blocks[1].start_state);
        assert_eq!(blocks[1].end_state, blocks[2].start_state);
        assert_eq!(blocks[2].end_state.global_cycle, 6);
        assert!(blocks.iter().all(|block| block.ended_at_tick_boundary));
        assert!(blocks
            .iter()
            .all(|block| block.cycles == vec![Cycle::NoOp; block.active_cycles]));
    }

    #[test]
    fn trace_source_validates_required_inputs() {
        let synthetic_with_input = Args {
            trace_source: TraceSource::Synthetic,
            trace_profile: TraceProfile::SyntheticNoop,
            trace_input: Some(PathBuf::from("traces/demo.json")),
            elf_input: None,
            trace_output: None,
            guest_input: None,
            untrusted_advice: None,
            trusted_advice: None,
            trace_block_size: 1024,
            blocks: 1,
            cycles_per_block: 2,
            block_counts: Vec::new(),
            output: PathBuf::from(DEFAULT_OUTPUT_PATH),
            manifest_output: None,
            performance_output: None,
            memory_sample_interval_ms: 10,
            program_digest_byte: 9,
        };
        assert!(load_trace_blocks(&synthetic_with_input)
            .unwrap_err()
            .contains("only valid with --trace-source trace-file"));

        let trace_file_without_input = Args {
            trace_source: TraceSource::TraceFile,
            trace_profile: TraceProfile::SyntheticNoop,
            trace_input: None,
            elf_input: None,
            trace_output: None,
            guest_input: None,
            untrusted_advice: None,
            trusted_advice: None,
            trace_block_size: 1024,
            blocks: 1,
            cycles_per_block: 2,
            block_counts: Vec::new(),
            output: PathBuf::from(DEFAULT_OUTPUT_PATH),
            manifest_output: None,
            performance_output: None,
            memory_sample_interval_ms: 10,
            program_digest_byte: 9,
        };
        assert!(load_trace_blocks(&trace_file_without_input)
            .unwrap_err()
            .contains("requires --trace-input"));
    }

    #[test]
    fn trace_file_loader_reads_versioned_blocks_and_digest() {
        let trace_input = trace_fixture_path();
        let args = Args {
            trace_source: TraceSource::TraceFile,
            trace_profile: TraceProfile::SyntheticNoop,
            trace_input: Some(trace_input.clone()),
            elf_input: None,
            trace_output: None,
            guest_input: None,
            untrusted_advice: None,
            trusted_advice: None,
            trace_block_size: 1024,
            blocks: 1,
            cycles_per_block: 2,
            block_counts: Vec::new(),
            output: PathBuf::from(DEFAULT_OUTPUT_PATH),
            manifest_output: None,
            performance_output: None,
            memory_sample_interval_ms: 10,
            program_digest_byte: 9,
        };
        let loaded_trace = load_trace_blocks(&args).unwrap();
        let metadata = loaded_trace.file_metadata.unwrap();
        let input_bytes = std::fs::read(trace_input).unwrap();

        assert_eq!(loaded_trace.blocks.len(), 2);
        assert_eq!(loaded_trace.blocks[0].block_index, 0);
        assert_eq!(loaded_trace.blocks[1].block_index, 1);
        assert_eq!(
            loaded_trace.blocks[0].end_state,
            loaded_trace.blocks[1].start_state
        );
        assert_eq!(
            loaded_trace.blocks[1].cycles,
            vec![Cycle::NoOp, Cycle::NoOp]
        );
        assert_eq!(metadata.schema_version, TRACE_FILE_SCHEMA_VERSION);
        assert_eq!(
            metadata.sha3_256,
            <[u8; 32]>::from(Sha3_256::digest(input_bytes))
        );
        assert_eq!(
            normalized_block_counts(&args, 2, false).unwrap(),
            vec![1, 2]
        );
    }

    #[test]
    fn elf_exporter_traces_binds_and_reads_back_real_program() {
        let elf_path = temp_manifest_path("elf-export", "tiny-rv64.elf");
        let trace_path = elf_path.with_file_name("tiny-rv64.trace.json");
        std::fs::create_dir_all(elf_path.parent().unwrap()).unwrap();
        let elf = tiny_rv64_elf();
        std::fs::write(&elf_path, &elf).unwrap();
        let args = Args {
            trace_source: TraceSource::Elf,
            trace_profile: TraceProfile::SyntheticNoop,
            trace_input: None,
            elf_input: Some(elf_path.clone()),
            trace_output: Some(trace_path.clone()),
            guest_input: None,
            untrusted_advice: None,
            trusted_advice: None,
            trace_block_size: 1,
            blocks: 1,
            cycles_per_block: 1,
            block_counts: Vec::new(),
            output: PathBuf::from(DEFAULT_OUTPUT_PATH),
            manifest_output: None,
            performance_output: None,
            memory_sample_interval_ms: 10,
            program_digest_byte: 9,
        };

        let loaded = load_trace_blocks(&args).unwrap();
        let metadata = loaded.file_metadata.as_ref().unwrap();
        let expected_elf_digest = <[u8; 32]>::from(Sha3_256::digest(&elf));

        assert_eq!(loaded.blocks.len(), 2);
        assert_eq!(loaded.blocks[0].start_state.pc, RAM_START_ADDRESS);
        assert_eq!(loaded.blocks[1].start_state.pc, RAM_START_ADDRESS + 4);
        assert_eq!(loaded.blocks[0].end_state, loaded.blocks[1].start_state);
        assert!(loaded.blocks[1].end_state.terminated);
        assert!(loaded.bytecode.code_size >= 2);
        let first_row = R1CSCycleInputs::from_cycle_with_next::<ark_bn254::Fr>(
            &loaded.bytecode,
            &loaded.blocks[0].cycles[0],
            Some(&loaded.blocks[1].cycles[0]),
        );
        assert_eq!(first_row.unexpanded_pc, RAM_START_ADDRESS);
        assert_eq!(first_row.next_unexpanded_pc, RAM_START_ADDRESS + 4);
        assert!(!first_row.should_branch);
        assert!(!first_row.flags[jolt_riscv::CircuitFlags::Jump as usize]);
        assert!(!first_row.flags[jolt_riscv::CircuitFlags::DoNotUpdateUnexpandedPC as usize]);
        assert!(!first_row.flags[jolt_riscv::CircuitFlags::IsCompressed as usize]);
        assert_eq!(
            loaded.program_digest,
            Some(bytecode_digest(&loaded.bytecode).unwrap())
        );
        assert_eq!(metadata.schema_version, TRACE_BUNDLE_SCHEMA_VERSION);
        assert_eq!(metadata.elf_sha3_256, Some(expected_elf_digest));
        assert!(trace_path.is_file());

        let declared_digest = hex_digest(&loaded.program_digest.unwrap());
        let tampered_path = trace_path.with_file_name("tiny-rv64.tampered.trace.json");
        let tampered = std::fs::read_to_string(&trace_path)
            .unwrap()
            .replace(&declared_digest, &"00".repeat(32));
        std::fs::write(&tampered_path, tampered).unwrap();
        assert!(load_trace_file(&tampered_path)
            .unwrap_err()
            .contains("bytecode digest mismatch"));

        std::fs::remove_file(tampered_path).unwrap();
        std::fs::remove_file(trace_path).unwrap();
        std::fs::remove_file(elf_path).unwrap();
        std::fs::remove_dir(args.elf_input.unwrap().parent().unwrap()).unwrap();
    }

    #[test]
    fn elf_runner_reaches_nova_folding_and_spartan_report() {
        let elf_path = temp_manifest_path("elf-e2e", "tiny-rv64.elf");
        let directory = elf_path.parent().unwrap().to_path_buf();
        let trace_path = directory.join("tiny-rv64.trace.json");
        let report_path = directory.join("report.json");
        let manifest_path = directory.join("report.manifest.json");
        let performance_path = directory.join("report.performance.json");
        std::fs::create_dir_all(&directory).unwrap();
        std::fs::write(&elf_path, tiny_rv64_elf()).unwrap();
        let args = Args {
            trace_source: TraceSource::Elf,
            trace_profile: TraceProfile::SyntheticNoop,
            trace_input: None,
            elf_input: Some(elf_path.clone()),
            trace_output: Some(trace_path.clone()),
            guest_input: None,
            untrusted_advice: None,
            trusted_advice: None,
            trace_block_size: 1,
            blocks: 1,
            cycles_per_block: 1,
            block_counts: vec![2],
            output: report_path.clone(),
            manifest_output: Some(manifest_path.clone()),
            performance_output: None,
            memory_sample_interval_ms: 1,
            program_digest_byte: 9,
        };

        run(args).unwrap();

        let report = std::fs::read_to_string(&report_path).unwrap();
        let manifest = std::fs::read_to_string(&manifest_path).unwrap();
        let performance_json = std::fs::read_to_string(&performance_path).unwrap();
        let performance: PerformanceBaselineArtifact =
            serde_json::from_str(&performance_json).unwrap();
        assert!(report.contains("\"block_count\":2"));
        assert!(report.contains("\"recursive_snark_bytes_len\":"));
        assert!(manifest.contains("\"trace_source\":\"elf\""));
        assert!(manifest.contains("\"trace_block_size\":1"));
        assert!(manifest.contains(&format!(
            "\"trace_file_schema_version\":\"{TRACE_BUNDLE_SCHEMA_VERSION}\""
        )));
        assert_eq!(performance.schema_version, PERFORMANCE_SCHEMA_VERSION);
        assert_eq!(performance.trace_source, "elf");
        assert_eq!(performance.source_block_count, 2);
        assert_eq!(performance.reported_block_counts, vec![2]);
        assert_eq!(performance.largest_reported_block_count, 2);
        assert_eq!(performance.largest_reported_active_cycles, 2);
        assert_eq!(performance.processed_block_count, 2);
        assert_eq!(performance.processed_active_cycles, 2);
        assert!(performance
            .throughput
            .proving_processed_active_cycles_per_second
            .is_some());
        assert_eq!(performance.memory.sample_interval_ms, 1);

        for path in [
            performance_path,
            manifest_path,
            report_path,
            trace_path,
            elf_path,
        ] {
            std::fs::remove_file(path).unwrap();
        }
        std::fs::remove_dir(directory).unwrap();
    }

    #[test]
    fn trace_file_loader_rejects_invalid_json_and_schema() {
        let invalid_json_path = write_temp_trace_file("invalid-json", "{not-json");
        assert!(load_trace_file(&invalid_json_path)
            .unwrap_err()
            .contains("failed to parse trace input"));
        remove_temp_trace_file(&invalid_json_path);

        let unsupported_schema = std::fs::read_to_string(trace_fixture_path())
            .unwrap()
            .replace(TRACE_FILE_SCHEMA_VERSION, "jolt-nova-trace-blocks-v2");
        let unsupported_schema_path =
            write_temp_trace_file("unsupported-schema", &unsupported_schema);
        assert!(load_trace_file(&unsupported_schema_path)
            .unwrap_err()
            .contains("unsupported trace schema version"));
        remove_temp_trace_file(&unsupported_schema_path);
    }

    #[test]
    fn trace_file_loader_rejects_invalid_block_chain() {
        let wrong_register_count = SerializedMachineBoundaryState {
            global_cycle: 0,
            emulator_trace_len: 0,
            pc: 0,
            registers: vec![0; REGISTER_COUNT as usize - 1],
            terminated: false,
        };
        assert!(MachineBoundaryState::try_from(wrong_register_count)
            .unwrap_err()
            .contains("register count mismatch"));

        let mut cycle_mismatch = build_noop_trace_blocks(2, 2).unwrap();
        cycle_mismatch[0].active_cycles = 3;
        assert!(validate_loaded_trace_blocks(&cycle_mismatch)
            .unwrap_err()
            .contains("active cycle count mismatch"));

        let mut non_tick_boundary = build_noop_trace_blocks(2, 2).unwrap();
        non_tick_boundary[0].ended_at_tick_boundary = false;
        assert!(validate_loaded_trace_blocks(&non_tick_boundary)
            .unwrap_err()
            .contains("tracer tick boundary"));

        let mut boundary_mismatch = build_noop_trace_blocks(2, 2).unwrap();
        boundary_mismatch[1].start_state.pc = 4;
        assert!(validate_loaded_trace_blocks(&boundary_mismatch)
            .unwrap_err()
            .contains("boundary state mismatch"));
    }

    #[test]
    fn manifest_json_records_runner_inputs_and_outputs() {
        let args = Args {
            trace_source: TraceSource::Synthetic,
            trace_profile: TraceProfile::SyntheticNoop,
            trace_input: None,
            elf_input: None,
            trace_output: None,
            guest_input: None,
            untrusted_advice: None,
            trusted_advice: None,
            trace_block_size: 1024,
            blocks: 2,
            cycles_per_block: 3,
            block_counts: vec![1, 2],
            output: PathBuf::from("benchmark-runs/jolt-nova/report.json"),
            manifest_output: Some(PathBuf::from(
                "benchmark-runs/jolt-nova/report.manifest.json",
            )),
            performance_output: None,
            memory_sample_interval_ms: 10,
            program_digest_byte: 11,
        };
        let block_counts = normalized_block_counts(&args, 2, false).unwrap();
        let artifact = sample_benchmark_artifact();
        let manifest_path = normalized_manifest_output(&args.output, args.manifest_output.as_ref());
        let json = build_manifest_json(&args, &block_counts, 2, None, &artifact, &manifest_path);

        assert!(json.contains("\"schema_version\":\"jolt-nova-benchmark-runner-v3\""));
        assert!(json.contains("\"runner\":\"jolt_nova_final_proof_size_benchmark\""));
        assert!(json.contains("\"trace_source\":\"synthetic\""));
        assert!(json.contains("\"trace_profile\":\"synthetic-noop\""));
        assert!(json.contains("\"trace_input_path\":null"));
        assert!(json.contains("\"trace_file_schema_version\":null"));
        assert!(json.contains("\"trace_input_sha3_256\":null"));
        assert!(json.contains("\"report_schema_version\":\"jolt-nova-report-v1\""));
        assert!(json.contains("\"report_kind\":\"final-proof-size-scaling\""));
        assert!(json.contains("\"report_output_format\":\"json\""));
        assert!(json.contains("\"report_path\":\"benchmark-runs/jolt-nova/report.json\""));
        assert!(
            json.contains("\"manifest_path\":\"benchmark-runs/jolt-nova/report.manifest.json\"")
        );
        assert!(
            json.contains("\"performance_schema_version\":\"jolt-nova-performance-baseline-v1\"")
        );
        assert!(json
            .contains("\"performance_path\":\"benchmark-runs/jolt-nova/report.performance.json\""));
        assert!(json.contains("\"memory_sample_interval_ms\":10"));
        assert!(json.contains("\"source_block_count\":2"));
        assert!(json.contains("\"blocks\":2"));
        assert!(json.contains("\"cycles_per_block\":3"));
        assert!(json.contains("\"block_counts\":[1,2]"));
        assert!(json.contains("\"program_digest_byte\":11"));
        assert!(json.contains("\"row_count\":1"));
        assert!(json.contains("\"report_bytes\":12"));
        assert!(json.contains("\"last_total_active_cycles\":6"));
        assert!(json.contains("\"last_spartan_total_bytes\":610"));
    }

    #[test]
    fn manifest_json_records_trace_file_binding() {
        let args = Args {
            trace_source: TraceSource::TraceFile,
            trace_profile: TraceProfile::SyntheticNoop,
            trace_input: Some(PathBuf::from(
                "jolt-core/examples/fixtures/jolt_nova_trace_blocks_v1.json",
            )),
            elf_input: None,
            trace_output: None,
            guest_input: None,
            untrusted_advice: None,
            trusted_advice: None,
            trace_block_size: 1024,
            blocks: 2,
            cycles_per_block: 2,
            block_counts: vec![1, 2],
            output: PathBuf::from("benchmark-runs/jolt-nova/report.json"),
            manifest_output: None,
            performance_output: None,
            memory_sample_interval_ms: 10,
            program_digest_byte: 9,
        };
        let artifact = sample_benchmark_artifact();
        let manifest_path = normalized_manifest_output(&args.output, args.manifest_output.as_ref());
        let metadata = TraceFileMetadata {
            schema_version: TRACE_FILE_SCHEMA_VERSION.to_string(),
            sha3_256: [0xabu8; 32],
            elf_sha3_256: None,
        };
        let json = build_manifest_json(
            &args,
            &[1, 2],
            2,
            Some(&metadata),
            &artifact,
            &manifest_path,
        );

        assert!(json.contains("\"trace_source\":\"trace-file\""));
        assert!(json.contains("\"trace_profile\":null"));
        assert!(json.contains(
            "\"trace_input_path\":\"jolt-core/examples/fixtures/jolt_nova_trace_blocks_v1.json\""
        ));
        assert!(json.contains("\"trace_file_schema_version\":\"jolt-nova-trace-blocks-v1\""));
        assert!(json.contains(&format!("\"trace_input_sha3_256\":\"{}\"", "ab".repeat(32))));
        assert!(json.contains("\"source_block_count\":2"));
        assert!(json.contains("\"blocks\":null"));
        assert!(json.contains("\"cycles_per_block\":null"));
    }

    #[test]
    fn write_manifest_creates_parent_directory() {
        let args = Args {
            trace_source: TraceSource::Synthetic,
            trace_profile: TraceProfile::SyntheticNoop,
            trace_input: None,
            elf_input: None,
            trace_output: None,
            guest_input: None,
            untrusted_advice: None,
            trusted_advice: None,
            trace_block_size: 1024,
            blocks: 1,
            cycles_per_block: 2,
            block_counts: vec![1],
            output: PathBuf::from("report.json"),
            manifest_output: None,
            performance_output: None,
            memory_sample_interval_ms: 10,
            program_digest_byte: 9,
        };
        let artifact = sample_benchmark_artifact();
        let manifest_path = temp_manifest_path("manifest-write", "report.manifest.json");

        write_manifest(&args, &[1], 1, None, &artifact, &manifest_path).unwrap();
        let manifest = std::fs::read_to_string(&manifest_path).unwrap();
        std::fs::remove_file(&manifest_path).unwrap();
        std::fs::remove_dir(manifest_path.parent().unwrap()).unwrap();

        assert!(manifest.contains("\"schema_version\":\"jolt-nova-benchmark-runner-v3\""));
        assert!(manifest.contains("\"block_counts\":[1]"));
    }

    #[test]
    fn performance_artifact_records_timings_throughput_and_memory() {
        let args = Args {
            trace_source: TraceSource::Synthetic,
            trace_profile: TraceProfile::SyntheticNoop,
            trace_input: None,
            elf_input: None,
            trace_output: None,
            guest_input: None,
            untrusted_advice: None,
            trusted_advice: None,
            trace_block_size: 1024,
            blocks: 2,
            cycles_per_block: 3,
            block_counts: vec![1, 2],
            output: PathBuf::from("benchmark-runs/jolt-nova/report.json"),
            manifest_output: None,
            performance_output: None,
            memory_sample_interval_ms: 5,
            program_digest_byte: 9,
        };
        let blocks = build_noop_trace_blocks(2, 3).unwrap();
        let measurements = RunMeasurements {
            trace_load: Duration::from_millis(100),
            prove_and_report: Duration::from_millis(2_000),
            manifest_write: Duration::from_millis(20),
            measured_total: Duration::from_millis(2_120),
            memory: PerformanceMemory {
                sample_interval_ms: 5,
                initial_physical_bytes: Some(100),
                peak_physical_bytes: Some(160),
                peak_delta_bytes: Some(60),
            },
        };

        let artifact = build_performance_artifact(
            &args,
            &[1, 2],
            &blocks,
            Path::new("benchmark-runs/jolt-nova/report.manifest.json"),
            &measurements,
        );

        assert_eq!(artifact.schema_version, PERFORMANCE_SCHEMA_VERSION);
        assert_eq!(artifact.source_block_count, 2);
        assert_eq!(artifact.largest_reported_block_count, 2);
        assert_eq!(artifact.largest_reported_active_cycles, 6);
        assert_eq!(artifact.processed_block_count, 3);
        assert_eq!(artifact.processed_active_cycles, 9);
        assert_eq!(artifact.timings_ms.trace_load, 100);
        assert_eq!(artifact.timings_ms.nova_fold_and_spartan_report, 2_000);
        assert_eq!(
            artifact
                .throughput
                .proving_processed_active_cycles_per_second,
            Some(4.5)
        );
        assert_eq!(artifact.memory.peak_delta_bytes, Some(60));
        assert_eq!(
            artifact.manifest_path,
            "benchmark-runs/jolt-nova/report.manifest.json"
        );
    }

    fn sample_benchmark_artifact() -> NovaBlockProofPipelineFinalProofSizeBenchmarkArtifact {
        let placeholder = sample_baseline(
            SPARTAN_PLACEHOLDER_PROOF_SYSTEM_NAME,
            SPARTAN_PLACEHOLDER_PROOF_SYSTEM_NAME,
            0,
            90,
            3,
        );
        let spartan = sample_baseline(
            SPARTAN_FINAL_PROOF_SYSTEM_NAME,
            SPARTAN_FINAL_PROOF_SYSTEM_NAME,
            512,
            610,
            6,
        );

        NovaBlockProofPipelineFinalProofSizeBenchmarkArtifact {
            output_format: JoltNovaReportOutputFormat::Json,
            serialized_report: "{\"rows\":[]}\n".to_string(),
            report: NovaBlockProofPipelineFinalProofSizeScalingReport {
                rows: vec![NovaBlockProofPipelineFinalProofSizeScalingRow {
                    block_count: 2,
                    first_block_index: Some(0),
                    last_block_index: Some(1),
                    total_active_cycles: 6,
                    recursive_snark_bytes_len: Some(128),
                    final_proof_size_comparison: JoltNovaFinalProofSizeComparison {
                        folded_accumulator_digest: repeated_digest(1),
                        absorbed_blocks: 2,
                        total_active_cycles: 6,
                        recursive_snark_bytes_len: Some(128),
                        placeholder,
                        spartan,
                        spartan_payload_extra_bytes: 512,
                        spartan_total_extra_bytes: 520,
                    },
                }],
            },
        }
    }

    fn sample_baseline(
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
            total_active_cycles: 6,
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

    fn repeated_digest(byte: u8) -> [u8; 32] {
        [byte; 32]
    }

    fn tiny_rv64_elf() -> Vec<u8> {
        const TEXT_OFFSET: usize = 64;
        const STRING_TABLE_OFFSET: usize = 72;
        const SECTION_HEADERS_OFFSET: usize = 96;
        const SECTION_HEADER_SIZE: usize = 64;
        const SECTION_COUNT: usize = 3;
        const TEXT_ADDRESS: u64 = RAM_START_ADDRESS;
        const STRING_TABLE: &[u8] = b"\0.text\0.shstrtab\0";

        let mut elf = vec![0u8; SECTION_HEADERS_OFFSET + SECTION_COUNT * SECTION_HEADER_SIZE];
        elf[0..16].copy_from_slice(&[0x7f, b'E', b'L', b'F', 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        write_u16(&mut elf, 16, 2);
        write_u16(&mut elf, 18, 243);
        write_u32(&mut elf, 20, 1);
        write_u64(&mut elf, 24, TEXT_ADDRESS);
        write_u64(&mut elf, 40, SECTION_HEADERS_OFFSET as u64);
        write_u16(&mut elf, 52, 64);
        write_u16(&mut elf, 58, SECTION_HEADER_SIZE as u16);
        write_u16(&mut elf, 60, SECTION_COUNT as u16);
        write_u16(&mut elf, 62, 2);

        write_u32(&mut elf, TEXT_OFFSET, 0x0010_0093); // addi x1, x0, 1
        write_u32(&mut elf, TEXT_OFFSET + 4, 0x0000_006f); // jal x0, 0
        elf[STRING_TABLE_OFFSET..STRING_TABLE_OFFSET + STRING_TABLE.len()]
            .copy_from_slice(STRING_TABLE);

        let text_header = SECTION_HEADERS_OFFSET + SECTION_HEADER_SIZE;
        write_u32(&mut elf, text_header, 1);
        write_u32(&mut elf, text_header + 4, 1);
        write_u64(&mut elf, text_header + 8, 0x6);
        write_u64(&mut elf, text_header + 16, TEXT_ADDRESS);
        write_u64(&mut elf, text_header + 24, TEXT_OFFSET as u64);
        write_u64(&mut elf, text_header + 32, 8);
        write_u64(&mut elf, text_header + 48, 4);

        let string_header = SECTION_HEADERS_OFFSET + 2 * SECTION_HEADER_SIZE;
        write_u32(&mut elf, string_header, 7);
        write_u32(&mut elf, string_header + 4, 3);
        write_u64(&mut elf, string_header + 24, STRING_TABLE_OFFSET as u64);
        write_u64(&mut elf, string_header + 32, STRING_TABLE.len() as u64);
        write_u64(&mut elf, string_header + 48, 1);
        elf
    }

    fn write_u16(bytes: &mut [u8], offset: usize, value: u16) {
        bytes[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
    }

    fn write_u32(bytes: &mut [u8], offset: usize, value: u32) {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }

    fn write_u64(bytes: &mut [u8], offset: usize, value: u64) {
        bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
    }

    fn trace_fixture_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("examples/fixtures/jolt_nova_trace_blocks_v1.json")
    }

    fn write_temp_trace_file(test_name: &str, contents: &str) -> PathBuf {
        let path = temp_manifest_path(test_name, "trace.json");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, contents).unwrap();
        path
    }

    fn remove_temp_trace_file(path: &Path) {
        std::fs::remove_file(path).unwrap();
        std::fs::remove_dir(path.parent().unwrap()).unwrap();
    }

    fn temp_manifest_path(test_name: &str, file_name: &str) -> PathBuf {
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
}
