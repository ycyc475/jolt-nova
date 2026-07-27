use std::{
    error::Error,
    fmt,
    fs::File,
    io::{self, BufReader, BufWriter, Read, Write},
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
            NovaBlockProofPipelineFinalProofSizeBenchmarkArtifact, NovaFoldConfig,
            NovaFoldingBackend, JOLT_NOVA_FINAL_PROOF_SIZE_SCALING_REPORT_KIND,
            JOLT_NOVA_REPORT_SCHEMA_VERSION, NOVA_LOGUP_SUBCLAIM_BACKEND_NAME,
            NOVA_TRANSCRIPT_SUBCLAIM_BACKEND_NAME,
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
const MANIFEST_SCHEMA_VERSION: &str = "jolt-nova-benchmark-runner-v5";
const PERFORMANCE_SCHEMA_VERSION: &str = "jolt-nova-performance-baseline-v2";
const MULTI_RUN_SCHEMA_VERSION: &str = "jolt-nova-multi-run-baseline-v2";
const LOOKUP_BACKEND_COMPARISON_SCHEMA_VERSION: &str = "jolt-nova-lookup-backend-comparison-v1";
const PRODUCTION_FIXTURE_SCHEMA_VERSION: &str = "jolt-nova-production-fixtures-v1";
const TRACE_FILE_SCHEMA_VERSION: &str = "jolt-nova-trace-blocks-v1";
const TRACE_BUNDLE_SCHEMA_VERSION: &str = "jolt-nova-trace-bundle-v1";
const BINARY_TRACE_SCHEMA_VERSION: &str = "jolt-nova-binary-trace-v1";
const BINARY_TRACE_MAGIC: &[u8; 8] = b"JNVTRC1\0";
const MAX_BINARY_TRACE_HEADER_BYTES: usize = 64 * 1024 * 1024;
const MAX_BINARY_TRACE_BLOCK_BYTES: usize = 256 * 1024 * 1024;
const MAX_BINARY_TRACE_BLOCKS: usize = 1 << 24;
const MAX_MEMORY_SAMPLE_INTERVAL_MS: u64 = 1_000;
const MAX_BENCHMARK_RUNS: usize = 100;

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

    /// Number of measured end-to-end runs included in the aggregate baseline.
    #[arg(long, default_value_t = 1)]
    measurement_runs: usize,

    /// Number of unmeasured warm-up runs performed before measurement.
    #[arg(long, default_value_t = 0)]
    warmup_runs: usize,

    /// Optional aggregate multi-run baseline artifact path.
    ///
    /// With multiple measured runs, the default is `<report-stem>.aggregate.json`.
    #[arg(long)]
    aggregate_output: Option<PathBuf>,

    /// Optional prior aggregate artifact used for an automated regression check.
    #[arg(long)]
    baseline_input: Option<PathBuf>,

    /// Maximum allowed regression for time, throughput, and peak-memory metrics.
    #[arg(long, default_value_t = 10.0)]
    max_regression_percent: f64,

    /// Built-in production-size RV64 guest used with `--trace-source fixture`.
    #[arg(long, value_enum, default_value_t = ProductionFixture::CpuLookup64k)]
    fixture: ProductionFixture,

    /// Lookup subclaim relation folded by Nova.
    #[arg(long, value_enum, default_value_t = LookupBackend::Transcript)]
    lookup_backend: LookupBackend,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum TraceSource {
    /// Generate trace blocks locally from synthetic runner options.
    Synthetic,
    /// Load trace blocks from a versioned JSON document.
    TraceFile,
    /// Execute an RV64 ELF, export a trace bundle, then read it back for proving.
    Elf,
    /// Execute a built-in, versioned production-size RV64 guest fixture.
    Fixture,
}

impl TraceSource {
    fn as_str(self) -> &'static str {
        match self {
            Self::Synthetic => "synthetic",
            Self::TraceFile => "trace-file",
            Self::Elf => "elf",
            Self::Fixture => "fixture",
        }
    }
}

impl fmt::Display for TraceSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum LookupBackend {
    /// Existing digest/count transcript fingerprint baseline.
    Transcript,
    /// LogUp fractional-sum relation with in-circuit balance enforcement.
    LogUp,
}

impl LookupBackend {
    fn as_str(self) -> &'static str {
        match self {
            Self::Transcript => "transcript",
            Self::LogUp => "logup",
        }
    }

    fn nova_subclaim_backend_name(self) -> &'static str {
        match self {
            Self::Transcript => NOVA_TRANSCRIPT_SUBCLAIM_BACKEND_NAME,
            Self::LogUp => NOVA_LOGUP_SUBCLAIM_BACKEND_NAME,
        }
    }
}

impl fmt::Display for LookupBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum ProductionFixture {
    /// Register, arithmetic, branch, and instruction-lookup workload (~64K cycles).
    CpuLookup64k,
    /// Stack load/store, register, branch, and instruction-lookup workload (~80K cycles).
    RamLookup80k,
}

impl ProductionFixture {
    const ITERATIONS: usize = 16_384;

    fn as_str(self) -> &'static str {
        match self {
            Self::CpuLookup64k => "cpu-lookup-64k",
            Self::RamLookup80k => "ram-lookup-80k",
        }
    }

    fn elf_bytes(self) -> Vec<u8> {
        let instructions = match self {
            Self::CpuLookup64k => vec![
                encode_lui(1, 4),
                encode_addi(1, 1, 0),
                encode_addi(2, 0, 0),
                0x0011_0133, // add x2, x2, x1
                encode_addi(1, 1, -1),
                encode_bne(1, 0, -8),
                0x0000_006f, // canonical Jolt termination
            ],
            Self::RamLookup80k => vec![
                encode_lui(1, 4),
                encode_addi(1, 1, 0),
                0x0000_1117, // auipc x2, 1; deterministic writable address
                encode_addi(3, 0, 0),
                0xfe31_3c23, // sd x3, -8(x2)
                0xff81_3203, // ld x4, -8(x2)
                encode_addi(3, 3, 1),
                encode_addi(1, 1, -1),
                encode_bne(1, 0, -16),
                0x0000_006f, // canonical Jolt termination
            ],
        };
        rv64_text_elf(&instructions)
    }

    fn expected_active_cycles(self) -> usize {
        match self {
            Self::CpuLookup64k => 3 + 3 * Self::ITERATIONS + 1,
            Self::RamLookup80k => 4 + 5 * Self::ITERATIONS + 1,
        }
    }
}

impl fmt::Display for ProductionFixture {
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
    storage_format: TraceStorageFormat,
    sha3_256: [u8; 32],
    elf_sha3_256: Option<[u8; 32]>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TraceStorageFormat {
    Json,
    Binary,
}

impl TraceStorageFormat {
    fn as_str(self) -> &'static str {
        match self {
            Self::Json => "json",
            Self::Binary => "binary",
        }
    }
}

impl fmt::Display for TraceStorageFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct PerformanceBaselineArtifact {
    schema_version: String,
    runner: String,
    trace_source: String,
    lookup_backend: String,
    build_profile: String,
    target_os: String,
    target_arch: String,
    workload_sha3_256: String,
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct MultiRunBaselineArtifact {
    schema_version: String,
    runner: String,
    trace_source: String,
    lookup_backend: String,
    fixture: Option<String>,
    fixture_schema_version: Option<String>,
    build_profile: String,
    target_os: String,
    target_arch: String,
    workload_sha3_256: String,
    measurement_runs: usize,
    warmup_runs: usize,
    source_block_count: usize,
    reported_block_counts: Vec<usize>,
    stats: MultiRunStatistics,
    samples: Vec<PerformanceBaselineArtifact>,
    comparison: Option<BaselineComparison>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct MultiRunStatistics {
    trace_load_ms: SummaryStatistics,
    nova_fold_and_spartan_report_ms: SummaryStatistics,
    measured_total_ms: SummaryStatistics,
    proving_processed_active_cycles_per_second: Option<SummaryStatistics>,
    end_to_end_processed_active_cycles_per_second: Option<SummaryStatistics>,
    peak_delta_bytes: Option<SummaryStatistics>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct SummaryStatistics {
    count: usize,
    min: f64,
    median: f64,
    mean: f64,
    max: f64,
    standard_deviation: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct BaselineComparison {
    baseline_path: String,
    max_regression_percent: f64,
    passed: bool,
    metrics: Vec<MetricComparison>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct MetricComparison {
    metric: String,
    direction: MetricDirection,
    baseline_mean: f64,
    current_mean: f64,
    regression_percent: f64,
    passed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct LookupBackendComparisonArtifact {
    schema_version: String,
    runner: String,
    trace_source: String,
    build_profile: String,
    target_os: String,
    target_arch: String,
    workload_sha3_256: String,
    left_lookup_backend: String,
    right_lookup_backend: String,
    measurement_runs: usize,
    warmup_runs: usize,
    source_block_count: usize,
    reported_block_counts: Vec<usize>,
    processed_block_count: usize,
    processed_active_cycles: usize,
    audit: LookupBackendComparisonAudit,
    metrics: Vec<LookupBackendMetricComparison>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct LookupBackendComparisonAudit {
    distinct_lookup_backends: bool,
    matching_workload_identity: bool,
    matching_block_shape: bool,
    matching_execution_volume: bool,
    passed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct LookupBackendMetricComparison {
    metric: String,
    direction: MetricDirection,
    left_mean: f64,
    right_mean: f64,
    right_vs_left_percent: Option<f64>,
    better_lookup_backend: Option<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
enum MetricDirection {
    LowerIsBetter,
    HigherIsBetter,
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
struct BinaryTraceHeader {
    container_schema_version: String,
    trace_schema_version: String,
    program: Option<SerializedProgramBinding>,
    block_count: u64,
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
    for warmup_index in 0..args.warmup_runs {
        println!("warmup_run {}/{}", warmup_index + 1, args.warmup_runs);
        run_once(&args)?;
    }

    let mut samples = Vec::with_capacity(args.measurement_runs);
    for run_index in 0..args.measurement_runs {
        println!(
            "measurement_run {}/{}",
            run_index + 1,
            args.measurement_runs
        );
        samples.push(run_once(&args)?);
    }

    let should_write_aggregate = args.measurement_runs > 1
        || args.aggregate_output.is_some()
        || args.baseline_input.is_some();
    if should_write_aggregate {
        let aggregate_output =
            normalized_aggregate_output(&args.output, args.aggregate_output.as_ref());
        let baseline = args
            .baseline_input
            .as_deref()
            .map(read_multi_run_baseline)
            .transpose()?;
        let mut aggregate = build_multi_run_baseline(&args, samples)?;
        if let Some((baseline_path, baseline)) =
            args.baseline_input.as_deref().zip(baseline.as_ref())
        {
            aggregate.comparison = Some(compare_multi_run_baselines(
                &aggregate,
                baseline,
                baseline_path,
                args.max_regression_percent,
            )?);
        }
        write_multi_run_baseline(&aggregate_output, &aggregate)?;
        println!("aggregate {}", manifest_path_string(&aggregate_output));
        if let Some(comparison) = &aggregate.comparison {
            println!("baseline_comparison_passed {}", comparison.passed);
            if !comparison.passed {
                let failures = comparison
                    .metrics
                    .iter()
                    .filter(|metric| !metric.passed)
                    .map(|metric| format!("{}={:.2}%", metric.metric, metric.regression_percent))
                    .collect::<Vec<_>>()
                    .join(", ");
                return Err(invalid_input(format!(
                    "multi-run baseline regression exceeded {:.2}%: {failures}",
                    args.max_regression_percent
                ))
                .into());
            }
        }
    }

    Ok(())
}

fn run_once(args: &Args) -> Result<PerformanceBaselineArtifact, Box<dyn Error>> {
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
    let workload_digest = loaded_trace
        .file_metadata
        .as_ref()
        .map(|metadata| metadata.sha3_256)
        .unwrap_or_else(|| synthetic_workload_digest(args, program_digest));
    let folding_backend = NovaFoldingBackend::new(NovaFoldConfig {
        subclaim_backend_name: args.lookup_backend.nova_subclaim_backend_name(),
        ..NovaFoldConfig::default()
    });
    let pipeline = BlockProofPipeline::<_, ark_bn254::Fr, NovaFoldingBackend>::with_backend(
        program_digest,
        folding_backend,
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
        workload_digest,
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
    println!("fixture {}", selected_fixture(args).unwrap_or("none"));
    println!("lookup_backend {}", args.lookup_backend);
    if let Some(metadata) = &loaded_trace.file_metadata {
        println!("trace_file_schema {}", metadata.schema_version);
        println!("trace_storage_format {}", metadata.storage_format);
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

    Ok(performance)
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
    if !(1..=MAX_BENCHMARK_RUNS).contains(&args.measurement_runs) {
        return Err(format!(
            "measurement-runs must be in 1..={MAX_BENCHMARK_RUNS}"
        ));
    }
    if args.warmup_runs > MAX_BENCHMARK_RUNS {
        return Err(format!("warmup-runs must be in 0..={MAX_BENCHMARK_RUNS}"));
    }
    if !args.max_regression_percent.is_finite() || args.max_regression_percent < 0.0 {
        return Err("max-regression-percent must be a finite non-negative number".to_string());
    }

    let manifest_output = normalized_manifest_output(&args.output, args.manifest_output.as_ref());
    let performance_output =
        normalized_performance_output(&args.output, args.performance_output.as_ref());
    let aggregate_output = (args.measurement_runs > 1
        || args.aggregate_output.is_some()
        || args.baseline_input.is_some())
    .then(|| normalized_aggregate_output(&args.output, args.aggregate_output.as_ref()));
    let mut outputs = vec![
        ("report output", args.output.as_path()),
        ("manifest output", manifest_output.as_path()),
        ("performance output", performance_output.as_path()),
    ];
    if let Some(trace_output) = args.trace_output.as_deref() {
        outputs.push(("trace output", trace_output));
    }
    if let Some(aggregate_output) = aggregate_output.as_deref() {
        outputs.push(("aggregate output", aggregate_output));
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
        ("baseline input", args.baseline_input.as_deref()),
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

fn normalized_aggregate_output(
    report_output: &Path,
    aggregate_output: Option<&PathBuf>,
) -> PathBuf {
    aggregate_output
        .cloned()
        .unwrap_or_else(|| default_aggregate_output_path(report_output))
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

fn default_aggregate_output_path(report_output: &Path) -> PathBuf {
    let mut aggregate_output = report_output.to_path_buf();
    let stem = report_output
        .file_stem()
        .and_then(|stem| stem.to_str())
        .filter(|stem| !stem.is_empty())
        .unwrap_or("jolt-nova-final-proof-size-scaling");
    aggregate_output.set_file_name(format!("{stem}.aggregate.json"));
    aggregate_output
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
        TraceSource::Fixture => export_fixture_trace_bundle(args),
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
    export_trace_bundle(
        args,
        &elf_bytes,
        Some(elf_input),
        trace_output,
        &guest_input,
        &untrusted_advice,
        &trusted_advice,
    )
}

fn export_fixture_trace_bundle(args: &Args) -> Result<LoadedTraceBlocks, String> {
    for (name, value) in [
        ("trace-input", args.trace_input.as_ref()),
        ("elf-input", args.elf_input.as_ref()),
        ("guest-input", args.guest_input.as_ref()),
        ("untrusted-advice", args.untrusted_advice.as_ref()),
        ("trusted-advice", args.trusted_advice.as_ref()),
    ] {
        if let Some(path) = value {
            return Err(format!(
                "{name} {} is not valid with --trace-source fixture",
                manifest_path_string(path)
            ));
        }
    }
    if args.trace_block_size == 0 {
        return Err("trace-block-size must be greater than zero".to_string());
    }
    let trace_output = args
        .trace_output
        .as_ref()
        .ok_or_else(|| "trace-source fixture requires --trace-output <path>".to_string())?;
    let loaded = export_trace_bundle(
        args,
        &args.fixture.elf_bytes(),
        None,
        trace_output,
        &[],
        &[],
        &[],
    )?;
    let active_cycles = loaded
        .blocks
        .iter()
        .map(|block| block.active_cycles)
        .sum::<usize>();
    if active_cycles != args.fixture.expected_active_cycles() {
        return Err(format!(
            "fixture {} cycle count mismatch: expected {}, traced {active_cycles}",
            args.fixture,
            args.fixture.expected_active_cycles()
        ));
    }
    Ok(loaded)
}

fn export_trace_bundle(
    args: &Args,
    elf_bytes: &[u8],
    elf_path: Option<&PathBuf>,
    trace_output: &Path,
    guest_input: &[u8],
    untrusted_advice: &[u8],
    trusted_advice: &[u8],
) -> Result<LoadedTraceBlocks, String> {
    let mut inline_provider = TracerInlineExpansionProvider::new();
    let program = jolt_program::build_jolt_program_with_inline_provider(
        elf_bytes,
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
        elf_bytes,
        elf_path,
        guest_input,
        untrusted_advice,
        trusted_advice,
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

    let elf_sha3_256 = Sha3_256::digest(elf_bytes).into();
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
    let program = SerializedProgramBinding {
        elf_sha3_256: hex_digest(&elf_sha3_256),
        program_digest_sha3_256: hex_digest(&program_digest),
        bytecode,
    };
    match trace_output_format(trace_output) {
        TraceStorageFormat::Json => write_json_trace_bundle(trace_output, blocks, program),
        TraceStorageFormat::Binary => write_binary_trace_bundle(trace_output, blocks, program),
    }
}

fn trace_output_format(path: &Path) -> TraceStorageFormat {
    match path.extension().and_then(|extension| extension.to_str()) {
        Some(extension)
            if extension.eq_ignore_ascii_case("bin")
                || extension.eq_ignore_ascii_case("jnvtrace") =>
        {
            TraceStorageFormat::Binary
        }
        _ => TraceStorageFormat::Json,
    }
}

fn write_json_trace_bundle(
    trace_output: &Path,
    blocks: &[TraceBlock],
    program: SerializedProgramBinding,
) -> Result<(), String> {
    let document = TraceFileDocument {
        schema_version: TRACE_BUNDLE_SCHEMA_VERSION.to_string(),
        program: Some(program),
        blocks: blocks.iter().map(SerializedTraceBlock::from).collect(),
    };
    let json = serde_json::to_vec_pretty(&document)
        .map_err(|error| format!("failed to serialize trace bundle as JSON: {error}"))?;
    std::fs::write(trace_output, json).map_err(|error| {
        format!(
            "failed to write JSON trace bundle {}: {error}",
            manifest_path_string(trace_output)
        )
    })
}

fn write_binary_trace_bundle(
    trace_output: &Path,
    blocks: &[TraceBlock],
    program: SerializedProgramBinding,
) -> Result<(), String> {
    let block_count = u64::try_from(blocks.len())
        .map_err(|_| "binary trace block count does not fit in u64".to_string())?;
    let header = BinaryTraceHeader {
        container_schema_version: BINARY_TRACE_SCHEMA_VERSION.to_string(),
        trace_schema_version: TRACE_BUNDLE_SCHEMA_VERSION.to_string(),
        program: Some(program),
        block_count,
    };
    let header_bytes = postcard::to_stdvec(&header)
        .map_err(|error| format!("failed to serialize binary trace header: {error}"))?;
    if header_bytes.len() > MAX_BINARY_TRACE_HEADER_BYTES {
        return Err(format!(
            "binary trace header is too large: {} bytes exceeds {}",
            header_bytes.len(),
            MAX_BINARY_TRACE_HEADER_BYTES
        ));
    }
    let header_len = u32::try_from(header_bytes.len())
        .map_err(|_| "binary trace header length does not fit in u32".to_string())?;
    let file = File::create(trace_output).map_err(|error| {
        format!(
            "failed to create binary trace bundle {}: {error}",
            manifest_path_string(trace_output)
        )
    })?;
    let mut writer = BufWriter::new(file);
    writer
        .write_all(BINARY_TRACE_MAGIC)
        .and_then(|_| writer.write_all(&header_len.to_le_bytes()))
        .and_then(|_| writer.write_all(&Sha3_256::digest(&header_bytes)))
        .and_then(|_| writer.write_all(&header_bytes))
        .map_err(|error| format!("failed to write binary trace header: {error}"))?;

    for block in blocks {
        let block_bytes =
            postcard::to_stdvec(&SerializedTraceBlock::from(block)).map_err(|error| {
                format!(
                    "failed to serialize trace block {}: {error}",
                    block.block_index
                )
            })?;
        if block_bytes.len() > MAX_BINARY_TRACE_BLOCK_BYTES {
            return Err(format!(
                "serialized trace block {} is too large: {} bytes exceeds {}",
                block.block_index,
                block_bytes.len(),
                MAX_BINARY_TRACE_BLOCK_BYTES
            ));
        }
        let block_len = u64::try_from(block_bytes.len()).map_err(|_| {
            format!(
                "trace block {} length does not fit in u64",
                block.block_index
            )
        })?;
        writer
            .write_all(&block_len.to_le_bytes())
            .and_then(|_| writer.write_all(&Sha3_256::digest(&block_bytes)))
            .and_then(|_| writer.write_all(&block_bytes))
            .map_err(|error| {
                format!(
                    "failed to write binary trace block {}: {error}",
                    block.block_index
                )
            })?;
    }
    writer.flush().map_err(|error| {
        format!(
            "failed to flush binary trace bundle {}: {error}",
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
    if trace_file_has_binary_magic(trace_input)? {
        load_binary_trace_file(trace_input)
    } else {
        load_json_trace_file(trace_input)
    }
}

fn trace_file_has_binary_magic(trace_input: &Path) -> Result<bool, String> {
    let mut file = File::open(trace_input).map_err(|error| {
        format!(
            "failed to open trace input {}: {error}",
            manifest_path_string(trace_input)
        )
    })?;
    let mut prefix = [0u8; BINARY_TRACE_MAGIC.len()];
    let read = file.read(&mut prefix).map_err(|error| {
        format!(
            "failed to inspect trace input {}: {error}",
            manifest_path_string(trace_input)
        )
    })?;
    Ok(read == BINARY_TRACE_MAGIC.len() && &prefix == BINARY_TRACE_MAGIC)
}

fn load_json_trace_file(trace_input: &Path) -> Result<LoadedTraceBlocks, String> {
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
        decode_program_binding(&document.schema_version, document.program)?;
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
            storage_format: TraceStorageFormat::Json,
            sha3_256: Sha3_256::digest(&input_bytes).into(),
            elf_sha3_256,
        }),
    })
}

fn load_binary_trace_file(trace_input: &Path) -> Result<LoadedTraceBlocks, String> {
    let file = File::open(trace_input).map_err(|error| {
        format!(
            "failed to open binary trace input {}: {error}",
            manifest_path_string(trace_input)
        )
    })?;
    let mut reader = BufReader::new(file);
    let magic = read_exact_array::<8>(&mut reader, "binary trace magic")?;
    if &magic != BINARY_TRACE_MAGIC {
        return Err("binary trace magic mismatch".to_string());
    }
    let header_len = u32::from_le_bytes(read_exact_array::<4>(
        &mut reader,
        "binary trace header length",
    )?) as usize;
    if header_len > MAX_BINARY_TRACE_HEADER_BYTES {
        return Err(format!(
            "binary trace header length {header_len} exceeds {MAX_BINARY_TRACE_HEADER_BYTES}"
        ));
    }
    let declared_header_digest = read_exact_array::<32>(&mut reader, "binary trace header digest")?;
    let header_bytes = read_exact_vec(&mut reader, header_len, "binary trace header")?;
    let actual_header_digest: [u8; 32] = Sha3_256::digest(&header_bytes).into();
    if declared_header_digest != actual_header_digest {
        return Err("binary trace header digest mismatch".to_string());
    }
    let header = postcard::from_bytes::<BinaryTraceHeader>(&header_bytes)
        .map_err(|error| format!("failed to decode binary trace header: {error}"))?;
    if header.container_schema_version != BINARY_TRACE_SCHEMA_VERSION {
        return Err(format!(
            "unsupported binary trace container schema {:?}; expected {:?}",
            header.container_schema_version, BINARY_TRACE_SCHEMA_VERSION
        ));
    }
    if header.trace_schema_version != TRACE_FILE_SCHEMA_VERSION
        && header.trace_schema_version != TRACE_BUNDLE_SCHEMA_VERSION
    {
        return Err(format!(
            "unsupported trace schema version {:?}; expected {:?} or {:?}",
            header.trace_schema_version, TRACE_FILE_SCHEMA_VERSION, TRACE_BUNDLE_SCHEMA_VERSION
        ));
    }
    let block_count = usize::try_from(header.block_count)
        .map_err(|_| "binary trace block count does not fit in usize".to_string())?;
    if block_count > MAX_BINARY_TRACE_BLOCKS {
        return Err(format!(
            "binary trace block count {block_count} exceeds {MAX_BINARY_TRACE_BLOCKS}"
        ));
    }

    let mut blocks = Vec::with_capacity(block_count);
    for record_index in 0..block_count {
        let block_len = u64::from_le_bytes(read_exact_array::<8>(
            &mut reader,
            &format!("binary trace block {record_index} length"),
        )?);
        let block_len = usize::try_from(block_len).map_err(|_| {
            format!("binary trace block {record_index} length does not fit in usize")
        })?;
        if block_len > MAX_BINARY_TRACE_BLOCK_BYTES {
            return Err(format!(
                "binary trace block {record_index} length {block_len} exceeds {MAX_BINARY_TRACE_BLOCK_BYTES}"
            ));
        }
        let declared_block_digest = read_exact_array::<32>(
            &mut reader,
            &format!("binary trace block {record_index} digest"),
        )?;
        let block_bytes = read_exact_vec(
            &mut reader,
            block_len,
            &format!("binary trace block {record_index} payload"),
        )?;
        let actual_block_digest: [u8; 32] = Sha3_256::digest(&block_bytes).into();
        if declared_block_digest != actual_block_digest {
            return Err(format!("binary trace block {record_index} digest mismatch"));
        }
        let serialized =
            postcard::from_bytes::<SerializedTraceBlock>(&block_bytes).map_err(|error| {
                format!("failed to decode binary trace block {record_index}: {error}")
            })?;
        blocks.push(TraceBlock::try_from(serialized)?);
    }
    let mut trailing = [0u8; 1];
    if reader
        .read(&mut trailing)
        .map_err(|error| format!("failed to check binary trace trailing bytes: {error}"))?
        != 0
    {
        return Err("binary trace contains trailing bytes".to_string());
    }

    let (bytecode, program_digest, elf_sha3_256) =
        decode_program_binding(&header.trace_schema_version, header.program)?;
    validate_loaded_trace_blocks(&blocks)?;
    Ok(LoadedTraceBlocks {
        blocks,
        bytecode,
        program_digest,
        file_metadata: Some(TraceFileMetadata {
            schema_version: header.trace_schema_version,
            storage_format: TraceStorageFormat::Binary,
            sha3_256: sha3_file(trace_input)?,
            elf_sha3_256,
        }),
    })
}

fn read_exact_array<const N: usize>(
    reader: &mut impl Read,
    label: &str,
) -> Result<[u8; N], String> {
    let mut bytes = [0u8; N];
    reader
        .read_exact(&mut bytes)
        .map_err(|error| format!("failed to read {label}: {error}"))?;
    Ok(bytes)
}

fn read_exact_vec(reader: &mut impl Read, len: usize, label: &str) -> Result<Vec<u8>, String> {
    let mut bytes = vec![0u8; len];
    reader
        .read_exact(&mut bytes)
        .map_err(|error| format!("failed to read {label}: {error}"))?;
    Ok(bytes)
}

fn sha3_file(path: &Path) -> Result<[u8; 32], String> {
    let file = File::open(path).map_err(|error| {
        format!(
            "failed to open trace input {} for digest: {error}",
            manifest_path_string(path)
        )
    })?;
    let mut reader = BufReader::new(file);
    let mut hasher = Sha3_256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = reader
            .read(&mut buffer)
            .map_err(|error| format!("failed to hash trace input: {error}"))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hasher.finalize().into())
}

fn decode_program_binding(
    schema_version: &str,
    program: Option<SerializedProgramBinding>,
) -> Result<(BytecodePreprocessing, Option<[u8; 32]>, Option<[u8; 32]>), String> {
    match (schema_version, program) {
        (TRACE_FILE_SCHEMA_VERSION, None) => Ok((BytecodePreprocessing::default(), None, None)),
        (TRACE_FILE_SCHEMA_VERSION, Some(_)) => {
            Err("legacy trace-block files must not contain a program binding".to_string())
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
            Ok((
                program.bytecode,
                Some(declared_program_digest),
                Some(elf_sha3_256),
            ))
        }
        (TRACE_BUNDLE_SCHEMA_VERSION, None) => {
            Err("trace bundle is missing its program binding".to_string())
        }
        _ => Err(format!(
            "unsupported trace schema version {schema_version:?}"
        )),
    }
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
    workload_digest: [u8; 32],
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
        lookup_backend: args.lookup_backend.as_str().to_string(),
        build_profile: if cfg!(debug_assertions) {
            "debug".to_string()
        } else {
            "release".to_string()
        },
        target_os: std::env::consts::OS.to_string(),
        target_arch: std::env::consts::ARCH.to_string(),
        workload_sha3_256: hex_digest(&workload_digest),
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

fn build_multi_run_baseline(
    args: &Args,
    samples: Vec<PerformanceBaselineArtifact>,
) -> Result<MultiRunBaselineArtifact, io::Error> {
    let first = samples.first().ok_or_else(|| {
        invalid_input("multi-run baseline requires at least one sample".to_string())
    })?;
    for (index, sample) in samples.iter().enumerate().skip(1) {
        if sample.trace_source != first.trace_source
            || sample.lookup_backend != first.lookup_backend
            || sample.build_profile != first.build_profile
            || sample.target_os != first.target_os
            || sample.target_arch != first.target_arch
            || sample.workload_sha3_256 != first.workload_sha3_256
            || sample.source_block_count != first.source_block_count
            || sample.reported_block_counts != first.reported_block_counts
            || sample.processed_block_count != first.processed_block_count
            || sample.processed_active_cycles != first.processed_active_cycles
        {
            return Err(invalid_input(format!(
                "measurement sample {} is incompatible with the first run",
                index + 1
            )));
        }
    }

    let stats = MultiRunStatistics {
        trace_load_ms: summarize_values(
            samples
                .iter()
                .map(|sample| sample.timings_ms.trace_load as f64)
                .collect(),
        )?,
        nova_fold_and_spartan_report_ms: summarize_values(
            samples
                .iter()
                .map(|sample| sample.timings_ms.nova_fold_and_spartan_report as f64)
                .collect(),
        )?,
        measured_total_ms: summarize_values(
            samples
                .iter()
                .map(|sample| sample.timings_ms.measured_total as f64)
                .collect(),
        )?,
        proving_processed_active_cycles_per_second: summarize_optional_values(
            samples
                .iter()
                .map(|sample| sample.throughput.proving_processed_active_cycles_per_second)
                .collect(),
        )?,
        end_to_end_processed_active_cycles_per_second: summarize_optional_values(
            samples
                .iter()
                .map(|sample| {
                    sample
                        .throughput
                        .end_to_end_processed_active_cycles_per_second
                })
                .collect(),
        )?,
        peak_delta_bytes: summarize_optional_values(
            samples
                .iter()
                .map(|sample| sample.memory.peak_delta_bytes.map(|value| value as f64))
                .collect(),
        )?,
    };

    Ok(MultiRunBaselineArtifact {
        schema_version: MULTI_RUN_SCHEMA_VERSION.to_string(),
        runner: RUNNER_NAME.to_string(),
        trace_source: first.trace_source.clone(),
        lookup_backend: first.lookup_backend.clone(),
        fixture: (args.trace_source == TraceSource::Fixture)
            .then(|| args.fixture.as_str().to_string()),
        fixture_schema_version: (args.trace_source == TraceSource::Fixture)
            .then(|| PRODUCTION_FIXTURE_SCHEMA_VERSION.to_string()),
        build_profile: first.build_profile.clone(),
        target_os: first.target_os.clone(),
        target_arch: first.target_arch.clone(),
        workload_sha3_256: first.workload_sha3_256.clone(),
        measurement_runs: samples.len(),
        warmup_runs: args.warmup_runs,
        source_block_count: first.source_block_count,
        reported_block_counts: first.reported_block_counts.clone(),
        stats,
        samples,
        comparison: None,
    })
}

fn summarize_optional_values(
    values: Vec<Option<f64>>,
) -> Result<Option<SummaryStatistics>, io::Error> {
    values
        .into_iter()
        .collect::<Option<Vec<_>>>()
        .map(summarize_values)
        .transpose()
}

fn summarize_values(mut values: Vec<f64>) -> Result<SummaryStatistics, io::Error> {
    if values.is_empty() {
        return Err(invalid_input(
            "summary statistics require at least one value".to_string(),
        ));
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(invalid_input(
            "summary statistics require finite values".to_string(),
        ));
    }
    values.sort_by(f64::total_cmp);
    let count = values.len();
    let min = values[0];
    let max = values[count - 1];
    let median = if count % 2 == 0 {
        (values[count / 2 - 1] + values[count / 2]) / 2.0
    } else {
        values[count / 2]
    };
    let mean = values.iter().sum::<f64>() / count as f64;
    let variance = values
        .iter()
        .map(|value| {
            let delta = value - mean;
            delta * delta
        })
        .sum::<f64>()
        / count as f64;

    Ok(SummaryStatistics {
        count,
        min,
        median,
        mean,
        max,
        standard_deviation: variance.sqrt(),
    })
}

fn compare_multi_run_baselines(
    current: &MultiRunBaselineArtifact,
    baseline: &MultiRunBaselineArtifact,
    baseline_path: &Path,
    max_regression_percent: f64,
) -> Result<BaselineComparison, io::Error> {
    validate_comparable_baselines(current, baseline)?;
    let mut metrics = vec![
        compare_metric(
            "nova_fold_and_spartan_report_ms",
            MetricDirection::LowerIsBetter,
            baseline.stats.nova_fold_and_spartan_report_ms.mean,
            current.stats.nova_fold_and_spartan_report_ms.mean,
            max_regression_percent,
        ),
        compare_metric(
            "measured_total_ms",
            MetricDirection::LowerIsBetter,
            baseline.stats.measured_total_ms.mean,
            current.stats.measured_total_ms.mean,
            max_regression_percent,
        ),
    ];
    if let (Some(baseline_throughput), Some(current_throughput)) = (
        &baseline.stats.proving_processed_active_cycles_per_second,
        &current.stats.proving_processed_active_cycles_per_second,
    ) {
        metrics.push(compare_metric(
            "proving_processed_active_cycles_per_second",
            MetricDirection::HigherIsBetter,
            baseline_throughput.mean,
            current_throughput.mean,
            max_regression_percent,
        ));
    }
    if let (Some(baseline_memory), Some(current_memory)) = (
        &baseline.stats.peak_delta_bytes,
        &current.stats.peak_delta_bytes,
    ) {
        metrics.push(compare_metric(
            "peak_delta_bytes",
            MetricDirection::LowerIsBetter,
            baseline_memory.mean,
            current_memory.mean,
            max_regression_percent,
        ));
    }
    let passed = metrics.iter().all(|metric| metric.passed);
    Ok(BaselineComparison {
        baseline_path: manifest_path_string(baseline_path),
        max_regression_percent,
        passed,
        metrics,
    })
}

fn compare_lookup_backend_baselines(
    left: &MultiRunBaselineArtifact,
    right: &MultiRunBaselineArtifact,
) -> Result<LookupBackendComparisonArtifact, io::Error> {
    validate_lookup_backend_comparison_inputs(left, right)?;
    let left_sample = left.samples.first().ok_or_else(|| {
        invalid_input("lookup backend comparison requires at least one left sample".to_string())
    })?;

    Ok(LookupBackendComparisonArtifact {
        schema_version: LOOKUP_BACKEND_COMPARISON_SCHEMA_VERSION.to_string(),
        runner: left.runner.clone(),
        trace_source: left.trace_source.clone(),
        build_profile: left.build_profile.clone(),
        target_os: left.target_os.clone(),
        target_arch: left.target_arch.clone(),
        workload_sha3_256: left.workload_sha3_256.clone(),
        left_lookup_backend: left.lookup_backend.clone(),
        right_lookup_backend: right.lookup_backend.clone(),
        measurement_runs: left.measurement_runs,
        warmup_runs: left.warmup_runs,
        source_block_count: left.source_block_count,
        reported_block_counts: left.reported_block_counts.clone(),
        processed_block_count: left_sample.processed_block_count,
        processed_active_cycles: left_sample.processed_active_cycles,
        audit: LookupBackendComparisonAudit {
            distinct_lookup_backends: left.lookup_backend != right.lookup_backend,
            matching_workload_identity: true,
            matching_block_shape: true,
            matching_execution_volume: true,
            passed: true,
        },
        metrics: vec![
            compare_lookup_backend_metric(
                "nova_fold_and_spartan_report_ms",
                MetricDirection::LowerIsBetter,
                Some(left.stats.nova_fold_and_spartan_report_ms.mean),
                &left.lookup_backend,
                Some(right.stats.nova_fold_and_spartan_report_ms.mean),
                &right.lookup_backend,
            ),
            compare_lookup_backend_metric(
                "measured_total_ms",
                MetricDirection::LowerIsBetter,
                Some(left.stats.measured_total_ms.mean),
                &left.lookup_backend,
                Some(right.stats.measured_total_ms.mean),
                &right.lookup_backend,
            ),
            compare_lookup_backend_metric(
                "proving_processed_active_cycles_per_second",
                MetricDirection::HigherIsBetter,
                left.stats
                    .proving_processed_active_cycles_per_second
                    .as_ref()
                    .map(|summary| summary.mean),
                &left.lookup_backend,
                right
                    .stats
                    .proving_processed_active_cycles_per_second
                    .as_ref()
                    .map(|summary| summary.mean),
                &right.lookup_backend,
            ),
            compare_lookup_backend_metric(
                "peak_delta_bytes",
                MetricDirection::LowerIsBetter,
                left.stats
                    .peak_delta_bytes
                    .as_ref()
                    .map(|summary| summary.mean),
                &left.lookup_backend,
                right
                    .stats
                    .peak_delta_bytes
                    .as_ref()
                    .map(|summary| summary.mean),
                &right.lookup_backend,
            ),
        ],
    })
}

fn validate_comparable_baselines(
    current: &MultiRunBaselineArtifact,
    baseline: &MultiRunBaselineArtifact,
) -> Result<(), io::Error> {
    if baseline.schema_version != MULTI_RUN_SCHEMA_VERSION {
        return Err(invalid_input(format!(
            "unsupported baseline schema {}; expected {MULTI_RUN_SCHEMA_VERSION}",
            baseline.schema_version
        )));
    }
    for (field, current_value, baseline_value) in [
        ("runner", current.runner.as_str(), baseline.runner.as_str()),
        (
            "trace_source",
            current.trace_source.as_str(),
            baseline.trace_source.as_str(),
        ),
        (
            "lookup_backend",
            current.lookup_backend.as_str(),
            baseline.lookup_backend.as_str(),
        ),
        (
            "build_profile",
            current.build_profile.as_str(),
            baseline.build_profile.as_str(),
        ),
        (
            "target_os",
            current.target_os.as_str(),
            baseline.target_os.as_str(),
        ),
        (
            "target_arch",
            current.target_arch.as_str(),
            baseline.target_arch.as_str(),
        ),
    ] {
        if current_value != baseline_value {
            return Err(invalid_input(format!(
                "baseline {field} mismatch: current {current_value}, baseline {baseline_value}"
            )));
        }
    }
    if current.fixture != baseline.fixture
        || current.fixture_schema_version != baseline.fixture_schema_version
        || current.workload_sha3_256 != baseline.workload_sha3_256
        || current.source_block_count != baseline.source_block_count
        || current.reported_block_counts != baseline.reported_block_counts
    {
        return Err(invalid_input(
            "baseline workload identity does not match the current run".to_string(),
        ));
    }
    Ok(())
}

fn validate_lookup_backend_comparison_inputs(
    left: &MultiRunBaselineArtifact,
    right: &MultiRunBaselineArtifact,
) -> Result<(), io::Error> {
    if left.schema_version != MULTI_RUN_SCHEMA_VERSION {
        return Err(invalid_input(format!(
            "unsupported left baseline schema {}; expected {MULTI_RUN_SCHEMA_VERSION}",
            left.schema_version
        )));
    }
    if right.schema_version != MULTI_RUN_SCHEMA_VERSION {
        return Err(invalid_input(format!(
            "unsupported right baseline schema {}; expected {MULTI_RUN_SCHEMA_VERSION}",
            right.schema_version
        )));
    }
    if left.lookup_backend == right.lookup_backend {
        return Err(invalid_input(
            "lookup backend comparison requires distinct lookup_backend values".to_string(),
        ));
    }
    for (field, left_value, right_value) in [
        ("runner", left.runner.as_str(), right.runner.as_str()),
        (
            "trace_source",
            left.trace_source.as_str(),
            right.trace_source.as_str(),
        ),
        (
            "build_profile",
            left.build_profile.as_str(),
            right.build_profile.as_str(),
        ),
        (
            "target_os",
            left.target_os.as_str(),
            right.target_os.as_str(),
        ),
        (
            "target_arch",
            left.target_arch.as_str(),
            right.target_arch.as_str(),
        ),
        (
            "workload_sha3_256",
            left.workload_sha3_256.as_str(),
            right.workload_sha3_256.as_str(),
        ),
        (
            "fixture",
            left.fixture.as_deref().unwrap_or("none"),
            right.fixture.as_deref().unwrap_or("none"),
        ),
        (
            "fixture_schema_version",
            left.fixture_schema_version.as_deref().unwrap_or("none"),
            right.fixture_schema_version.as_deref().unwrap_or("none"),
        ),
    ] {
        if left_value != right_value {
            return Err(invalid_input(format!(
                "lookup backend comparison field {field} does not match: left={left_value}, right={right_value}"
            )));
        }
    }
    if left.measurement_runs != right.measurement_runs {
        return Err(invalid_input(format!(
            "lookup backend comparison field measurement_runs does not match: left={}, right={}",
            left.measurement_runs, right.measurement_runs
        )));
    }
    if left.warmup_runs != right.warmup_runs {
        return Err(invalid_input(format!(
            "lookup backend comparison field warmup_runs does not match: left={}, right={}",
            left.warmup_runs, right.warmup_runs
        )));
    }
    if left.source_block_count != right.source_block_count {
        return Err(invalid_input(format!(
            "lookup backend comparison field source_block_count does not match: left={}, right={}",
            left.source_block_count, right.source_block_count
        )));
    }
    if left.reported_block_counts != right.reported_block_counts {
        return Err(invalid_input(
            "lookup backend comparison field reported_block_counts does not match".to_string(),
        ));
    }
    let left_sample = left.samples.first().ok_or_else(|| {
        invalid_input("lookup backend comparison requires at least one left sample".to_string())
    })?;
    let right_sample = right.samples.first().ok_or_else(|| {
        invalid_input("lookup backend comparison requires at least one right sample".to_string())
    })?;
    if left_sample.processed_block_count != right_sample.processed_block_count {
        return Err(invalid_input(format!(
            "lookup backend comparison field processed_block_count does not match: left={}, right={}",
            left_sample.processed_block_count, right_sample.processed_block_count
        )));
    }
    if left_sample.processed_active_cycles != right_sample.processed_active_cycles {
        return Err(invalid_input(format!(
            "lookup backend comparison field processed_active_cycles does not match: left={}, right={}",
            left_sample.processed_active_cycles, right_sample.processed_active_cycles
        )));
    }
    Ok(())
}

fn compare_lookup_backend_metric(
    metric: &str,
    direction: MetricDirection,
    left_mean: Option<f64>,
    left_backend: &str,
    right_mean: Option<f64>,
    right_backend: &str,
) -> LookupBackendMetricComparison {
    let left_mean = left_mean.unwrap_or(0.0);
    let right_mean = right_mean.unwrap_or(0.0);
    let right_vs_left_percent = if left_mean == 0.0 {
        None
    } else {
        Some(((right_mean - left_mean) / left_mean) * 100.0)
    };
    let better_lookup_backend = match direction {
        MetricDirection::LowerIsBetter if left_mean < right_mean => Some(left_backend.to_string()),
        MetricDirection::LowerIsBetter if right_mean < left_mean => Some(right_backend.to_string()),
        MetricDirection::HigherIsBetter if left_mean > right_mean => Some(left_backend.to_string()),
        MetricDirection::HigherIsBetter if right_mean > left_mean => {
            Some(right_backend.to_string())
        }
        _ => None,
    };

    LookupBackendMetricComparison {
        metric: metric.to_string(),
        direction,
        left_mean,
        right_mean,
        right_vs_left_percent,
        better_lookup_backend,
    }
}

fn compare_metric(
    metric: &str,
    direction: MetricDirection,
    baseline_mean: f64,
    current_mean: f64,
    max_regression_percent: f64,
) -> MetricComparison {
    let regression_percent = match direction {
        MetricDirection::LowerIsBetter => relative_increase_percent(baseline_mean, current_mean),
        MetricDirection::HigherIsBetter => relative_decrease_percent(baseline_mean, current_mean),
    };
    MetricComparison {
        metric: metric.to_string(),
        direction,
        baseline_mean,
        current_mean,
        regression_percent,
        passed: regression_percent <= max_regression_percent,
    }
}

fn relative_increase_percent(baseline: f64, current: f64) -> f64 {
    if baseline == 0.0 {
        if current == 0.0 {
            0.0
        } else {
            f64::MAX
        }
    } else {
        ((current - baseline) / baseline * 100.0).max(0.0)
    }
}

fn relative_decrease_percent(baseline: f64, current: f64) -> f64 {
    if baseline == 0.0 {
        0.0
    } else {
        ((baseline - current) / baseline * 100.0).max(0.0)
    }
}

fn read_multi_run_baseline(path: &Path) -> Result<MultiRunBaselineArtifact, io::Error> {
    let bytes = std::fs::read(path)?;
    serde_json::from_slice(&bytes).map_err(|error| {
        invalid_input(format!(
            "failed to decode multi-run baseline {}: {error}",
            manifest_path_string(path)
        ))
    })
}

fn write_multi_run_baseline(
    path: &Path,
    artifact: &MultiRunBaselineArtifact,
) -> Result<(), io::Error> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let mut json = serde_json::to_vec_pretty(artifact).map_err(io::Error::other)?;
    json.push(b'\n');
    std::fs::write(path, json)
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
    append_json_string_field(&mut json, "lookup_backend", args.lookup_backend.as_str());
    json.push(',');
    append_json_optional_string_field(&mut json, "trace_profile", selected_trace_profile(args));
    json.push(',');
    append_json_optional_string_field(&mut json, "fixture", selected_fixture(args));
    json.push(',');
    append_json_optional_string_field(
        &mut json,
        "fixture_schema_version",
        selected_fixture(args).map(|_| PRODUCTION_FIXTURE_SCHEMA_VERSION),
    );
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
        "trace_storage_format",
        trace_file_metadata.map(|metadata| metadata.storage_format.as_str()),
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
        matches!(args.trace_source, TraceSource::Elf | TraceSource::Fixture)
            .then_some(args.trace_block_size),
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

fn selected_fixture(args: &Args) -> Option<&'static str> {
    (args.trace_source == TraceSource::Fixture).then(|| args.fixture.as_str())
}

fn selected_trace_path(args: &Args) -> Option<&PathBuf> {
    match args.trace_source {
        TraceSource::Synthetic => None,
        TraceSource::TraceFile => args.trace_input.as_ref(),
        TraceSource::Elf => args.trace_output.as_ref(),
        TraceSource::Fixture => args.trace_output.as_ref(),
    }
}

fn hex_digest(digest: &[u8; 32]) -> String {
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn synthetic_workload_digest(args: &Args, program_digest: [u8; 32]) -> [u8; 32] {
    let mut hasher = Sha3_256::new();
    hasher.update(b"jolt-nova-synthetic-workload-v1");
    hasher.update(program_digest);
    hasher.update(args.trace_profile.as_str().as_bytes());
    hasher.update((args.blocks as u64).to_le_bytes());
    hasher.update((args.cycles_per_block as u64).to_le_bytes());
    hasher.finalize().into()
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

fn encode_lui(rd: u32, immediate_upper_20: u32) -> u32 {
    ((immediate_upper_20 & 0x000f_ffff) << 12) | ((rd & 0x1f) << 7) | 0x37
}

fn encode_addi(rd: u32, rs1: u32, immediate: i32) -> u32 {
    debug_assert!((-2048..=2047).contains(&immediate));
    (((immediate as u32) & 0x0fff) << 20) | ((rs1 & 0x1f) << 15) | ((rd & 0x1f) << 7) | 0x13
}

fn encode_bne(rs1: u32, rs2: u32, offset: i32) -> u32 {
    debug_assert!(offset % 2 == 0 && (-4096..=4094).contains(&offset));
    let immediate = (offset as u32) & 0x1fff;
    ((immediate >> 12) & 0x1) << 31
        | ((immediate >> 5) & 0x3f) << 25
        | ((rs2 & 0x1f) << 20)
        | ((rs1 & 0x1f) << 15)
        | (0x1 << 12)
        | ((immediate >> 1) & 0xf) << 8
        | ((immediate >> 11) & 0x1) << 7
        | 0x63
}

fn rv64_text_elf(instructions: &[u32]) -> Vec<u8> {
    const ELF_HEADER_SIZE: usize = 64;
    const SECTION_HEADER_SIZE: usize = 64;
    const SECTION_COUNT: usize = 3;
    const STRING_TABLE: &[u8] = b"\0.text\0.shstrtab\0";

    let text_offset = ELF_HEADER_SIZE;
    let text_size = instructions.len() * std::mem::size_of::<u32>();
    let string_table_offset = (text_offset + text_size + 7) & !7;
    let section_headers_offset = (string_table_offset + STRING_TABLE.len() + 7) & !7;
    let mut elf = vec![0u8; section_headers_offset + SECTION_COUNT * SECTION_HEADER_SIZE];

    elf[0..16].copy_from_slice(&[0x7f, b'E', b'L', b'F', 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    write_le_u16(&mut elf, 16, 2);
    write_le_u16(&mut elf, 18, 243);
    write_le_u32(&mut elf, 20, 1);
    write_le_u64(&mut elf, 24, RAM_START_ADDRESS);
    write_le_u64(&mut elf, 40, section_headers_offset as u64);
    write_le_u16(&mut elf, 52, ELF_HEADER_SIZE as u16);
    write_le_u16(&mut elf, 58, SECTION_HEADER_SIZE as u16);
    write_le_u16(&mut elf, 60, SECTION_COUNT as u16);
    write_le_u16(&mut elf, 62, 2);

    for (index, instruction) in instructions.iter().enumerate() {
        write_le_u32(
            &mut elf,
            text_offset + index * std::mem::size_of::<u32>(),
            *instruction,
        );
    }
    elf[string_table_offset..string_table_offset + STRING_TABLE.len()]
        .copy_from_slice(STRING_TABLE);

    let text_header = section_headers_offset + SECTION_HEADER_SIZE;
    write_le_u32(&mut elf, text_header, 1);
    write_le_u32(&mut elf, text_header + 4, 1);
    write_le_u64(&mut elf, text_header + 8, 0x6);
    write_le_u64(&mut elf, text_header + 16, RAM_START_ADDRESS);
    write_le_u64(&mut elf, text_header + 24, text_offset as u64);
    write_le_u64(&mut elf, text_header + 32, text_size as u64);
    write_le_u64(&mut elf, text_header + 48, 4);

    let string_header = section_headers_offset + 2 * SECTION_HEADER_SIZE;
    write_le_u32(&mut elf, string_header, 7);
    write_le_u32(&mut elf, string_header + 4, 3);
    write_le_u64(&mut elf, string_header + 24, string_table_offset as u64);
    write_le_u64(&mut elf, string_header + 32, STRING_TABLE.len() as u64);
    write_le_u64(&mut elf, string_header + 48, 1);
    elf
}

fn write_le_u16(bytes: &mut [u8], offset: usize, value: u16) {
    bytes[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}

fn write_le_u32(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn write_le_u64(bytes: &mut [u8], offset: usize, value: u64) {
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
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
            measurement_runs: 1,
            warmup_runs: 0,
            aggregate_output: None,
            baseline_input: None,
            max_regression_percent: 10.0,
            fixture: ProductionFixture::CpuLookup64k,
            lookup_backend: LookupBackend::Transcript,
        };

        assert_eq!(
            normalized_block_counts(&args, 3, false).unwrap(),
            vec![1, 2, 3]
        );
        assert_eq!(normalized_block_counts(&args, 3, true).unwrap(), vec![3]);

        let mut partial_real_trace = args;
        partial_real_trace.block_counts = vec![1, 3];
        assert_eq!(
            normalized_block_counts(&partial_real_trace, 3, true).unwrap(),
            vec![1, 3]
        );
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
    fn default_aggregate_output_sits_next_to_report() {
        assert_eq!(
            default_aggregate_output_path(Path::new("benchmark-runs/jolt-nova/report.json")),
            PathBuf::from("benchmark-runs/jolt-nova/report.aggregate.json")
        );
        assert_eq!(
            default_aggregate_output_path(Path::new("report")),
            PathBuf::from("report.aggregate.json")
        );
    }

    #[test]
    fn production_guest_fixtures_are_versioned_decodable_rv64_programs() {
        let cpu_elf = ProductionFixture::CpuLookup64k.elf_bytes();
        let ram_elf = ProductionFixture::RamLookup80k.elf_bytes();
        assert_ne!(Sha3_256::digest(&cpu_elf), Sha3_256::digest(&ram_elf));
        assert_eq!(
            ProductionFixture::CpuLookup64k.expected_active_cycles(),
            49_156
        );
        assert_eq!(
            ProductionFixture::RamLookup80k.expected_active_cycles(),
            81_925
        );
        assert_eq!(encode_bne(1, 0, -8), 0xfe00_9ce3);
        assert_eq!(encode_bne(1, 0, -16), 0xfe00_98e3);

        for (fixture, minimum_rows) in [
            (ProductionFixture::CpuLookup64k, 7),
            (ProductionFixture::RamLookup80k, 10),
        ] {
            let mut inline_provider = TracerInlineExpansionProvider::new();
            let program = jolt_program::build_jolt_program_with_inline_provider(
                &fixture.elf_bytes(),
                &mut inline_provider,
                RV64IMAC_JOLT,
            )
            .unwrap();
            assert_eq!(program.entry_address, RAM_START_ADDRESS);
            assert!(program.expanded_bytecode.len() >= minimum_rows);
        }
        assert_eq!(
            PRODUCTION_FIXTURE_SCHEMA_VERSION,
            "jolt-nova-production-fixtures-v1"
        );
    }

    #[test]
    fn production_guest_fixtures_execute_to_the_declared_cycle_count() {
        for fixture in [
            ProductionFixture::CpuLookup64k,
            ProductionFixture::RamLookup80k,
        ] {
            let elf = fixture.elf_bytes();
            let mut inline_provider = TracerInlineExpansionProvider::new();
            let program = jolt_program::build_jolt_program_with_inline_provider(
                &elf,
                &mut inline_provider,
                RV64IMAC_JOLT,
            )
            .unwrap();
            let memory_config = MemoryConfig {
                program_size: Some(program.program_end - RAM_START_ADDRESS),
                ..Default::default()
            };
            let mut iterator =
                tracer::trace_blocks(&elf, None, &[], &[], &[], &memory_config, None, 1024);
            let mut active_cycles = 0;
            for block in iterator.by_ref() {
                active_cycles += block.active_cycles;
            }
            let tracer = iterator.into_inner().lazy_tracer;
            assert!(!tracer.has_panicked(), "{fixture} panicked");
            assert!(tracer.has_terminated(), "{fixture} did not terminate");
            assert_eq!(active_cycles, fixture.expected_active_cycles());
        }
    }

    #[test]
    fn summary_statistics_cover_even_samples_and_population_deviation() {
        let summary = summarize_values(vec![4.0, 1.0, 3.0, 2.0]).unwrap();
        assert_eq!(summary.count, 4);
        assert_eq!(summary.min, 1.0);
        assert_eq!(summary.median, 2.5);
        assert_eq!(summary.mean, 2.5);
        assert_eq!(summary.max, 4.0);
        assert!((summary.standard_deviation - 1.25_f64.sqrt()).abs() < f64::EPSILON);
        assert!(summarize_values(Vec::new()).is_err());
        assert!(summarize_values(vec![f64::NAN]).is_err());
    }

    #[test]
    fn multi_run_baseline_aggregates_samples_and_detects_regression() {
        let mut args = sample_args();
        args.measurement_runs = 3;
        args.warmup_runs = 1;
        let samples = vec![
            sample_performance_artifact(LookupBackend::Transcript, 90, 100, 900.0, 100.0, 10),
            sample_performance_artifact(LookupBackend::Transcript, 100, 110, 1_000.0, 90.0, 20),
            sample_performance_artifact(LookupBackend::Transcript, 110, 120, 1_100.0, 80.0, 30),
        ];
        let baseline = build_multi_run_baseline(&args, samples).unwrap();
        assert_eq!(baseline.schema_version, MULTI_RUN_SCHEMA_VERSION);
        assert_eq!(baseline.measurement_runs, 3);
        assert_eq!(baseline.warmup_runs, 1);
        assert_eq!(baseline.stats.nova_fold_and_spartan_report_ms.mean, 100.0);
        assert_eq!(baseline.stats.measured_total_ms.median, 110.0);
        assert_eq!(baseline.stats.peak_delta_bytes.as_ref().unwrap().mean, 20.0);

        let encoded = serde_json::to_vec(&baseline).unwrap();
        let decoded: MultiRunBaselineArtifact = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(decoded, baseline);

        let mut current = baseline.clone();
        current.stats.nova_fold_and_spartan_report_ms.mean = 112.0;
        let comparison = compare_multi_run_baselines(
            &current,
            &baseline,
            Path::new("baseline.aggregate.json"),
            10.0,
        )
        .unwrap();
        assert!(!comparison.passed);
        let proving = comparison
            .metrics
            .iter()
            .find(|metric| metric.metric == "nova_fold_and_spartan_report_ms")
            .unwrap();
        assert_eq!(proving.regression_percent, 12.0);
        assert!(!proving.passed);

        let mut different_backend = current.clone();
        different_backend.lookup_backend = "logup".to_string();
        assert!(compare_multi_run_baselines(
            &different_backend,
            &baseline,
            Path::new("baseline.aggregate.json"),
            10.0,
        )
        .unwrap_err()
        .to_string()
        .contains("lookup_backend"));

        let mut different_workload = current;
        different_workload.workload_sha3_256 = "cd".repeat(32);
        assert!(compare_multi_run_baselines(
            &different_workload,
            &baseline,
            Path::new("baseline.aggregate.json"),
            10.0,
        )
        .unwrap_err()
        .to_string()
        .contains("workload identity"));
    }

    #[test]
    fn lookup_backend_comparison_tracks_same_workload_across_lasso_and_logup() {
        let transcript = sample_lookup_backend_baseline(LookupBackend::Transcript);
        let logup = sample_lookup_backend_baseline(LookupBackend::LogUp);

        let comparison = compare_lookup_backend_baselines(&transcript, &logup).unwrap();
        assert_eq!(
            comparison.schema_version,
            LOOKUP_BACKEND_COMPARISON_SCHEMA_VERSION
        );
        assert_eq!(comparison.left_lookup_backend, "transcript");
        assert_eq!(comparison.right_lookup_backend, "logup");
        assert!(comparison.audit.distinct_lookup_backends);
        assert!(comparison.audit.matching_workload_identity);
        assert!(comparison.audit.matching_block_shape);
        assert!(comparison.audit.matching_execution_volume);
        assert!(comparison.audit.passed);
        assert_eq!(comparison.metrics.len(), 4);
        assert_eq!(
            comparison.metrics[0].metric,
            "nova_fold_and_spartan_report_ms"
        );
        assert!(comparison.metrics[0].better_lookup_backend.is_some());

        let encoded = serde_json::to_vec(&comparison).unwrap();
        let decoded: LookupBackendComparisonArtifact = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(decoded, comparison);
    }

    #[test]
    fn lookup_backend_comparison_rejects_same_backend_or_workload_mismatch() {
        let transcript = sample_lookup_backend_baseline(LookupBackend::Transcript);
        let same_backend = sample_lookup_backend_baseline(LookupBackend::Transcript);
        assert!(compare_lookup_backend_baselines(&transcript, &same_backend)
            .unwrap_err()
            .to_string()
            .contains("distinct lookup_backend"));

        let mut mismatched_workload = sample_lookup_backend_baseline(LookupBackend::LogUp);
        mismatched_workload.workload_sha3_256 = "cd".repeat(32);
        assert!(
            compare_lookup_backend_baselines(&transcript, &mismatched_workload)
                .unwrap_err()
                .to_string()
                .contains("workload_sha3_256")
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
            measurement_runs: 1,
            warmup_runs: 0,
            aggregate_output: None,
            baseline_input: None,
            max_regression_percent: 10.0,
            fixture: ProductionFixture::CpuLookup64k,
            lookup_backend: LookupBackend::Transcript,
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

        let mut no_measurements = sample_args();
        no_measurements.measurement_runs = 0;
        assert!(validate_runner_args(&no_measurements)
            .unwrap_err()
            .contains("measurement-runs"));

        let mut invalid_threshold = sample_args();
        invalid_threshold.max_regression_percent = f64::NAN;
        assert!(validate_runner_args(&invalid_threshold)
            .unwrap_err()
            .contains("finite non-negative"));
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
            measurement_runs: 1,
            warmup_runs: 0,
            aggregate_output: None,
            baseline_input: None,
            max_regression_percent: 10.0,
            fixture: ProductionFixture::CpuLookup64k,
            lookup_backend: LookupBackend::Transcript,
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
            measurement_runs: 1,
            warmup_runs: 0,
            aggregate_output: None,
            baseline_input: None,
            max_regression_percent: 10.0,
            fixture: ProductionFixture::CpuLookup64k,
            lookup_backend: LookupBackend::Transcript,
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
            measurement_runs: 1,
            warmup_runs: 0,
            aggregate_output: None,
            baseline_input: None,
            max_regression_percent: 10.0,
            fixture: ProductionFixture::CpuLookup64k,
            lookup_backend: LookupBackend::Transcript,
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
            measurement_runs: 1,
            warmup_runs: 0,
            aggregate_output: None,
            baseline_input: None,
            max_regression_percent: 10.0,
            fixture: ProductionFixture::CpuLookup64k,
            lookup_backend: LookupBackend::Transcript,
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
        assert_eq!(metadata.storage_format, TraceStorageFormat::Json);
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
            measurement_runs: 1,
            warmup_runs: 0,
            aggregate_output: None,
            baseline_input: None,
            max_regression_percent: 10.0,
            fixture: ProductionFixture::CpuLookup64k,
            lookup_backend: LookupBackend::Transcript,
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
        assert_eq!(metadata.storage_format, TraceStorageFormat::Json);
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
    fn binary_trace_bundle_streams_roundtrip_and_rejects_corruption() {
        assert_eq!(
            trace_output_format(Path::new("trace.jnvtrace")),
            TraceStorageFormat::Binary
        );
        assert_eq!(
            trace_output_format(Path::new("trace.JSON")),
            TraceStorageFormat::Json
        );
        let elf_path = temp_manifest_path("binary-trace", "tiny-rv64.elf");
        let directory = elf_path.parent().unwrap().to_path_buf();
        let trace_path = directory.join("tiny-rv64.trace.bin");
        std::fs::create_dir_all(&directory).unwrap();
        let elf = tiny_rv64_elf();
        std::fs::write(&elf_path, &elf).unwrap();

        let mut args = sample_args();
        args.trace_source = TraceSource::Elf;
        args.elf_input = Some(elf_path.clone());
        args.trace_output = Some(trace_path.clone());
        args.trace_block_size = 1;
        args.block_counts.clear();
        let loaded = load_trace_blocks(&args).unwrap();
        let metadata = loaded.file_metadata.as_ref().unwrap();

        assert_eq!(loaded.blocks.len(), 2);
        assert_eq!(metadata.storage_format, TraceStorageFormat::Binary);
        assert_eq!(metadata.schema_version, TRACE_BUNDLE_SCHEMA_VERSION);
        assert_eq!(
            metadata.sha3_256,
            <[u8; 32]>::from(Sha3_256::digest(std::fs::read(&trace_path).unwrap()))
        );
        assert_eq!(
            loaded.program_digest,
            Some(bytecode_digest(&loaded.bytecode).unwrap())
        );

        let valid_bytes = std::fs::read(&trace_path).unwrap();
        assert_eq!(&valid_bytes[..BINARY_TRACE_MAGIC.len()], BINARY_TRACE_MAGIC);

        let corrupted_header_path = directory.join("corrupted-header.trace.bin");
        let mut corrupted_header_bytes = valid_bytes.clone();
        corrupted_header_bytes[BINARY_TRACE_MAGIC.len() + 4 + 32] ^= 0x01;
        std::fs::write(&corrupted_header_path, corrupted_header_bytes).unwrap();
        assert!(load_trace_file(&corrupted_header_path)
            .unwrap_err()
            .contains("header digest mismatch"));

        let corrupted_path = directory.join("corrupted.trace.bin");
        let mut corrupted_bytes = valid_bytes.clone();
        *corrupted_bytes.last_mut().unwrap() ^= 0x01;
        std::fs::write(&corrupted_path, corrupted_bytes).unwrap();
        assert!(load_trace_file(&corrupted_path)
            .unwrap_err()
            .contains("block 1 digest mismatch"));

        let truncated_path = directory.join("truncated.trace.bin");
        std::fs::write(&truncated_path, &valid_bytes[..valid_bytes.len() - 1]).unwrap();
        assert!(load_trace_file(&truncated_path)
            .unwrap_err()
            .contains("block 1 payload"));

        let trailing_path = directory.join("trailing.trace.bin");
        let mut trailing_bytes = valid_bytes;
        trailing_bytes.push(0xff);
        std::fs::write(&trailing_path, trailing_bytes).unwrap();
        assert!(load_trace_file(&trailing_path)
            .unwrap_err()
            .contains("trailing bytes"));

        for path in [
            trailing_path,
            truncated_path,
            corrupted_path,
            corrupted_header_path,
            trace_path,
            elf_path,
        ] {
            std::fs::remove_file(path).unwrap();
        }
        std::fs::remove_dir(directory).unwrap();
    }

    #[test]
    fn production_fixture_binary_trace_roundtrips_all_blocks() {
        let trace_path = temp_manifest_path("binary-production", "cpu-lookup-64k.trace.bin");
        let directory = trace_path.parent().unwrap().to_path_buf();
        std::fs::create_dir_all(&directory).unwrap();
        let mut args = sample_args();
        args.trace_source = TraceSource::Fixture;
        args.trace_output = Some(trace_path.clone());
        args.trace_block_size = 1024;
        args.block_counts.clear();

        let loaded = load_trace_blocks(&args).unwrap();
        let active_cycles = loaded
            .blocks
            .iter()
            .map(|block| block.active_cycles)
            .sum::<usize>();

        assert_eq!(
            active_cycles,
            ProductionFixture::CpuLookup64k.expected_active_cycles()
        );
        assert_eq!(loaded.blocks.len(), 49);
        assert_eq!(
            loaded.file_metadata.unwrap().storage_format,
            TraceStorageFormat::Binary
        );
        assert!(trace_path.metadata().unwrap().len() > 0);

        std::fs::remove_file(trace_path).unwrap();
        std::fs::remove_dir(directory).unwrap();
    }

    #[test]
    fn elf_runner_reaches_nova_folding_and_spartan_report() {
        let elf_path = temp_manifest_path("elf-e2e", "tiny-rv64.elf");
        let directory = elf_path.parent().unwrap().to_path_buf();
        let trace_path = directory.join("tiny-rv64.trace.bin");
        let report_path = directory.join("report.json");
        let manifest_path = directory.join("report.manifest.json");
        let performance_path = directory.join("report.performance.json");
        let aggregate_path = directory.join("report.aggregate.json");
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
            block_counts: vec![1, 2],
            output: report_path.clone(),
            manifest_output: Some(manifest_path.clone()),
            performance_output: None,
            memory_sample_interval_ms: 1,
            program_digest_byte: 9,
            measurement_runs: 2,
            warmup_runs: 0,
            aggregate_output: Some(aggregate_path.clone()),
            baseline_input: None,
            max_regression_percent: 10.0,
            fixture: ProductionFixture::CpuLookup64k,
            lookup_backend: LookupBackend::LogUp,
        };

        run(args).unwrap();

        let report = std::fs::read_to_string(&report_path).unwrap();
        let manifest = std::fs::read_to_string(&manifest_path).unwrap();
        let performance_json = std::fs::read_to_string(&performance_path).unwrap();
        let performance: PerformanceBaselineArtifact =
            serde_json::from_str(&performance_json).unwrap();
        let aggregate_json = std::fs::read_to_string(&aggregate_path).unwrap();
        let aggregate: MultiRunBaselineArtifact = serde_json::from_str(&aggregate_json).unwrap();
        assert!(report.contains("\"block_count\":1"));
        assert!(report.contains("\"block_count\":2"));
        assert!(report.contains("\"recursive_snark_bytes_len\":"));
        assert!(manifest.contains("\"trace_source\":\"elf\""));
        assert!(manifest.contains("\"lookup_backend\":\"logup\""));
        assert!(manifest.contains("\"trace_block_size\":1"));
        assert!(manifest.contains(&format!(
            "\"trace_file_schema_version\":\"{TRACE_BUNDLE_SCHEMA_VERSION}\""
        )));
        assert!(manifest.contains("\"trace_storage_format\":\"binary\""));
        assert_eq!(performance.schema_version, PERFORMANCE_SCHEMA_VERSION);
        assert_eq!(performance.trace_source, "elf");
        assert_eq!(performance.lookup_backend, "logup");
        assert_eq!(performance.source_block_count, 2);
        assert_eq!(performance.reported_block_counts, vec![1, 2]);
        assert_eq!(performance.largest_reported_block_count, 2);
        assert_eq!(performance.largest_reported_active_cycles, 2);
        assert_eq!(performance.processed_block_count, 3);
        assert_eq!(performance.processed_active_cycles, 3);
        assert!(performance
            .throughput
            .proving_processed_active_cycles_per_second
            .is_some());
        assert_eq!(performance.memory.sample_interval_ms, 1);
        assert_eq!(aggregate.schema_version, MULTI_RUN_SCHEMA_VERSION);
        assert_eq!(aggregate.lookup_backend, "logup");
        assert_eq!(aggregate.measurement_runs, 2);
        assert_eq!(aggregate.samples.len(), 2);
        assert_eq!(aggregate.source_block_count, 2);
        assert_eq!(aggregate.reported_block_counts, vec![1, 2]);

        for path in [
            aggregate_path,
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
            measurement_runs: 1,
            warmup_runs: 0,
            aggregate_output: None,
            baseline_input: None,
            max_regression_percent: 10.0,
            fixture: ProductionFixture::CpuLookup64k,
            lookup_backend: LookupBackend::Transcript,
        };
        let block_counts = normalized_block_counts(&args, 2, false).unwrap();
        let artifact = sample_benchmark_artifact();
        let manifest_path = normalized_manifest_output(&args.output, args.manifest_output.as_ref());
        let json = build_manifest_json(&args, &block_counts, 2, None, &artifact, &manifest_path);

        assert!(json.contains("\"schema_version\":\"jolt-nova-benchmark-runner-v5\""));
        assert!(json.contains("\"lookup_backend\":\"transcript\""));
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
            json.contains("\"performance_schema_version\":\"jolt-nova-performance-baseline-v2\"")
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
            measurement_runs: 1,
            warmup_runs: 0,
            aggregate_output: None,
            baseline_input: None,
            max_regression_percent: 10.0,
            fixture: ProductionFixture::CpuLookup64k,
            lookup_backend: LookupBackend::Transcript,
        };
        let artifact = sample_benchmark_artifact();
        let manifest_path = normalized_manifest_output(&args.output, args.manifest_output.as_ref());
        let metadata = TraceFileMetadata {
            schema_version: TRACE_FILE_SCHEMA_VERSION.to_string(),
            storage_format: TraceStorageFormat::Json,
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
        assert!(json.contains("\"trace_storage_format\":\"json\""));
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
            measurement_runs: 1,
            warmup_runs: 0,
            aggregate_output: None,
            baseline_input: None,
            max_regression_percent: 10.0,
            fixture: ProductionFixture::CpuLookup64k,
            lookup_backend: LookupBackend::Transcript,
        };
        let artifact = sample_benchmark_artifact();
        let manifest_path = temp_manifest_path("manifest-write", "report.manifest.json");

        write_manifest(&args, &[1], 1, None, &artifact, &manifest_path).unwrap();
        let manifest = std::fs::read_to_string(&manifest_path).unwrap();
        std::fs::remove_file(&manifest_path).unwrap();
        std::fs::remove_dir(manifest_path.parent().unwrap()).unwrap();

        assert!(manifest.contains("\"schema_version\":\"jolt-nova-benchmark-runner-v5\""));
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
            measurement_runs: 1,
            warmup_runs: 0,
            aggregate_output: None,
            baseline_input: None,
            max_regression_percent: 10.0,
            fixture: ProductionFixture::CpuLookup64k,
            lookup_backend: LookupBackend::Transcript,
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
            [0xabu8; 32],
        );

        assert_eq!(artifact.schema_version, PERFORMANCE_SCHEMA_VERSION);
        assert_eq!(artifact.source_block_count, 2);
        assert_eq!(artifact.largest_reported_block_count, 2);
        assert_eq!(artifact.largest_reported_active_cycles, 6);
        assert_eq!(artifact.processed_block_count, 3);
        assert_eq!(artifact.processed_active_cycles, 9);
        assert_eq!(artifact.workload_sha3_256, "ab".repeat(32));
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

    fn sample_args() -> Args {
        Args {
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
            memory_sample_interval_ms: 10,
            program_digest_byte: 9,
            measurement_runs: 1,
            warmup_runs: 0,
            aggregate_output: None,
            baseline_input: None,
            max_regression_percent: 10.0,
            fixture: ProductionFixture::CpuLookup64k,
            lookup_backend: LookupBackend::Transcript,
        }
    }

    fn sample_performance_artifact(
        lookup_backend: LookupBackend,
        prove_ms: u64,
        total_ms: u64,
        end_to_end_throughput: f64,
        proving_throughput: f64,
        peak_delta_bytes: u64,
    ) -> PerformanceBaselineArtifact {
        PerformanceBaselineArtifact {
            schema_version: PERFORMANCE_SCHEMA_VERSION.to_string(),
            runner: RUNNER_NAME.to_string(),
            trace_source: "synthetic".to_string(),
            lookup_backend: lookup_backend.as_str().to_string(),
            build_profile: "debug".to_string(),
            target_os: std::env::consts::OS.to_string(),
            target_arch: std::env::consts::ARCH.to_string(),
            workload_sha3_256: "ab".repeat(32),
            source_block_count: 2,
            reported_block_counts: vec![1, 2],
            largest_reported_block_count: 2,
            largest_reported_active_cycles: 6,
            processed_block_count: 3,
            processed_active_cycles: 9,
            timings_ms: PerformanceTimings {
                trace_load: 5,
                nova_fold_and_spartan_report: prove_ms,
                manifest_write: 1,
                measured_total: total_ms,
            },
            throughput: PerformanceThroughput {
                proving_processed_active_cycles_per_second: Some(proving_throughput),
                end_to_end_processed_active_cycles_per_second: Some(end_to_end_throughput),
                end_to_end_processed_blocks_per_second: Some(end_to_end_throughput / 3.0),
            },
            memory: PerformanceMemory {
                sample_interval_ms: 10,
                initial_physical_bytes: Some(1_000),
                peak_physical_bytes: Some(1_000 + peak_delta_bytes),
                peak_delta_bytes: Some(peak_delta_bytes),
            },
            report_path: "benchmark-runs/jolt-nova/report.json".to_string(),
            manifest_path: "benchmark-runs/jolt-nova/report.manifest.json".to_string(),
        }
    }

    fn sample_lookup_backend_baseline(lookup_backend: LookupBackend) -> MultiRunBaselineArtifact {
        let mut args = sample_args();
        args.lookup_backend = lookup_backend;
        let (prove_ms, total_ms, end_to_end_throughput, proving_throughput, peak_delta_bytes) =
            match lookup_backend {
                LookupBackend::Transcript => (100, 120, 1_000.0, 90.0, 20),
                LookupBackend::LogUp => (125, 145, 830.0, 72.0, 28),
            };
        let sample = sample_performance_artifact_for_lookup_backend(
            lookup_backend,
            prove_ms,
            total_ms,
            end_to_end_throughput,
            proving_throughput,
            peak_delta_bytes,
        );
        build_multi_run_baseline(&args, vec![sample]).unwrap()
    }

    fn sample_performance_artifact_for_lookup_backend(
        lookup_backend: LookupBackend,
        prove_ms: u64,
        total_ms: u64,
        end_to_end_throughput: f64,
        proving_throughput: f64,
        peak_delta_bytes: u64,
    ) -> PerformanceBaselineArtifact {
        sample_performance_artifact(
            lookup_backend,
            prove_ms,
            total_ms,
            end_to_end_throughput,
            proving_throughput,
            peak_delta_bytes,
        )
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
