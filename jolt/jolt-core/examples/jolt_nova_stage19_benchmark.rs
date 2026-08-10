extern crate jolt_inlines_keccak256;

use std::{
    cell::RefCell,
    error::Error,
    fs::File,
    io::{self, BufWriter, Write},
    path::{Path, PathBuf},
    rc::Rc,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
    thread,
    time::{Duration, Instant},
};

use clap::{Parser, ValueEnum};
use common::jolt_device::MemoryConfig;
use jolt_core::{
    curve::Bn254Curve,
    host::Program,
    poly::commitment::dory::DoryCommitmentScheme,
    zkvm::{
        block::{
            stage19_hex_digest, BlockProofPipeline, NovaFoldingBackend,
            RecursiveBlindFoldProfileObserver, RecursiveBlindFoldProfilePhase,
            Stage18ReleaseParameters, Stage18ZkEndToEndProof, Stage19BenchmarkArtifact,
            Stage19BenchmarkSample, Stage19MemoryBytes, Stage19Platform, Stage19ProofSizes,
            Stage19RelationMeasurement, Stage19TimingsMicros, JOLT_NOVA_STAGE19_LOOKUP_BACKEND,
            JOLT_NOVA_STAGE19_PROFILE_METHOD,
        },
        program::ProgramPreprocessing,
        prover::JoltProverPreprocessing,
        verifier::{JoltSharedPreprocessing, JoltVerifierPreprocessing},
        RV64IMACProof, RV64IMACProver, RV64IMACVerifier, Serializable,
    },
};
use serde::Serialize;
use serde_json::Value;
use sha3::{Digest, Sha3_256};

const DEFAULT_OUTPUT: &str = "benchmark-runs/stage19/fibonacci-32.json";
const MAX_MEASUREMENT_RUNS: usize = 20;
const MAX_MEMORY_SAMPLE_INTERVAL_MS: u64 = 1_000;

#[derive(Clone, Debug, Parser)]
struct Args {
    #[arg(long, value_enum, default_value_t = Workload::Fibonacci)]
    workload: Workload,

    /// Workload scale: Fibonacci n, SHA3 iterations, or Collatz input.
    #[arg(long, default_value_t = 32)]
    scale: u32,

    /// Comma-separated production block target sizes.
    #[arg(long, value_delimiter = ',', default_value = "64,256")]
    block_sizes: Vec<usize>,

    #[arg(long, default_value_t = 1)]
    measurement_runs: usize,

    #[arg(long, default_value_t = 0)]
    warmup_runs: usize,

    #[arg(long, default_value_t = 10)]
    memory_sample_interval_ms: u64,

    #[arg(long, default_value = DEFAULT_OUTPUT)]
    output: PathBuf,

    #[arg(long)]
    markdown_output: Option<PathBuf>,

    #[arg(long)]
    block_profile_output: Option<PathBuf>,

    #[arg(long)]
    blindfold_profile_output: Option<PathBuf>,

    #[arg(long)]
    baseline_input: Option<PathBuf>,

    #[arg(long, default_value_t = 15.0)]
    max_regression_percent: f64,

    #[arg(long, default_value = "target-stage-19-guest")]
    guest_target: PathBuf,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum Workload {
    Fibonacci,
    Sha3Chain,
    MemoryOps,
    Collatz,
}

impl Workload {
    fn guest_name(self) -> &'static str {
        match self {
            Self::Fibonacci => "fibonacci-guest",
            Self::Sha3Chain => "sha3-chain-guest",
            Self::MemoryOps => "memory-ops-guest",
            Self::Collatz => "collatz-guest",
        }
    }

    fn function(self) -> Option<&'static str> {
        match self {
            Self::Collatz => Some("collatz_convergence"),
            _ => None,
        }
    }

    fn max_trace_length(self) -> usize {
        match self {
            Self::Fibonacci | Self::MemoryOps => 1 << 16,
            Self::Collatz => 1 << 20,
            Self::Sha3Chain => 1 << 22,
        }
    }

    fn configure_program(self, program: &mut Program) {
        if let Some(function) = self.function() {
            program.set_func(function);
        }
        if self == Self::MemoryOps {
            program.set_heap_size(1 << 16);
        }
    }

    fn input_bytes(self, scale: u32) -> Result<Vec<u8>, String> {
        let mut inputs = Vec::new();
        match self {
            Self::Fibonacci => {
                inputs.extend(postcard::to_stdvec(&scale).map_err(|e| e.to_string())?)
            }
            Self::Sha3Chain => {
                inputs.extend(postcard::to_stdvec(&[5u8; 32]).map_err(|e| e.to_string())?);
                inputs.extend(postcard::to_stdvec(&scale).map_err(|e| e.to_string())?);
            }
            Self::MemoryOps => {}
            Self::Collatz => inputs
                .extend(postcard::to_stdvec(&(scale as u128).max(2)).map_err(|e| e.to_string())?),
        }
        Ok(inputs)
    }

    fn id(self, scale: u32) -> String {
        match self {
            Self::Fibonacci => format!("fibonacci-{scale}"),
            Self::Sha3Chain => format!("sha3-chain-{scale}-iterations"),
            Self::MemoryOps => "memory-ops".to_string(),
            Self::Collatz => format!("collatz-{scale}"),
        }
    }
}

struct PreparedBenchmark {
    program: Program,
    elf: Vec<u8>,
    inputs: Vec<u8>,
    memory_config: MemoryConfig,
    preprocessing:
        JoltProverPreprocessing<jolt_core::ark_bn254::Fr, Bn254Curve, DoryCommitmentScheme>,
    preprocessing_micros: u64,
    workload: String,
    workload_digest: [u8; 32],
}

#[derive(Clone, Debug, Default)]
struct MemoryMeasurement {
    start: Option<u64>,
    peak: Option<u64>,
    peak_delta: Option<u64>,
}

struct PeakMemorySampler {
    start: Option<u64>,
    peak: Arc<AtomicU64>,
    stop: Arc<AtomicBool>,
    handle: Option<thread::JoinHandle<()>>,
}

impl PeakMemorySampler {
    fn start(interval_ms: u64) -> Self {
        let start = physical_memory_bytes();
        let peak = Arc::new(AtomicU64::new(start.unwrap_or_default()));
        let stop = Arc::new(AtomicBool::new(false));
        let thread_peak = Arc::clone(&peak);
        let thread_stop = Arc::clone(&stop);
        let handle = thread::spawn(move || {
            while !thread_stop.load(Ordering::Relaxed) {
                if let Some(memory) = physical_memory_bytes() {
                    thread_peak.fetch_max(memory, Ordering::Relaxed);
                }
                thread::sleep(Duration::from_millis(interval_ms));
            }
            if let Some(memory) = physical_memory_bytes() {
                thread_peak.fetch_max(memory, Ordering::Relaxed);
            }
        });
        Self {
            start,
            peak,
            stop,
            handle: Some(handle),
        }
    }

    fn finish(mut self) -> MemoryMeasurement {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
        let peak = self
            .start
            .map(|start| self.peak.load(Ordering::Relaxed).max(start));
        MemoryMeasurement {
            start: self.start,
            peak,
            peak_delta: self
                .start
                .zip(peak)
                .map(|(start, peak)| peak.saturating_sub(start)),
        }
    }
}

impl Drop for PeakMemorySampler {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

#[derive(Debug, Serialize)]
struct BlindFoldPhaseMeasurement {
    phase: &'static str,
    elapsed_micros: u64,
    memory_start: Option<u64>,
    memory_peak: Option<u64>,
    memory_peak_delta: Option<u64>,
}

struct ActiveBlindFoldPhase {
    phase: RecursiveBlindFoldProfilePhase,
    started: Instant,
    memory_sampler: PeakMemorySampler,
}

struct BlindFoldProfileCollector {
    memory_sample_interval_ms: u64,
    active: Option<ActiveBlindFoldPhase>,
    measurements: Vec<BlindFoldPhaseMeasurement>,
}

impl BlindFoldProfileCollector {
    fn new(memory_sample_interval_ms: u64) -> Self {
        Self {
            memory_sample_interval_ms,
            active: None,
            measurements: Vec::new(),
        }
    }

    fn finish(mut self) -> Vec<BlindFoldPhaseMeasurement> {
        if let Some(active) = self.active.take() {
            self.record_finished(active);
        }
        self.measurements
    }

    fn record_finished(&mut self, active: ActiveBlindFoldPhase) {
        let elapsed_micros = elapsed_micros(active.started);
        let memory = active.memory_sampler.finish();
        self.measurements.push(BlindFoldPhaseMeasurement {
            phase: active.phase.as_str(),
            elapsed_micros,
            memory_start: memory.start,
            memory_peak: memory.peak,
            memory_peak_delta: memory.peak_delta,
        });
    }
}

impl RecursiveBlindFoldProfileObserver for BlindFoldProfileCollector {
    fn phase_started(&mut self, phase: RecursiveBlindFoldProfilePhase) {
        if let Some(active) = self.active.take() {
            self.record_finished(active);
        }
        self.active = Some(ActiveBlindFoldPhase {
            phase,
            started: Instant::now(),
            memory_sampler: PeakMemorySampler::start(self.memory_sample_interval_ms),
        });
    }

    fn phase_finished(&mut self, phase: RecursiveBlindFoldProfilePhase) {
        let Some(active) = self.active.take() else {
            return;
        };
        if active.phase != phase {
            self.record_finished(active);
            return;
        }
        self.record_finished(active);
    }
}

fn main() {
    let args = Args::parse();
    let result = thread::Builder::new()
        .name("jolt-nova-stage19-benchmark".to_string())
        .stack_size(128 * 1024 * 1024)
        .spawn(move || run(args).map_err(|error| error.to_string()))
        .and_then(|handle| {
            handle
                .join()
                .map_err(|_| io::Error::other("Stage-19 benchmark worker panicked"))
        })
        .and_then(|result| result.map_err(io::Error::other));
    if let Err(error) = result {
        eprintln!("stage19 benchmark failed: {error}");
        std::process::exit(1);
    }
}

fn run(mut args: Args) -> Result<(), Box<dyn Error>> {
    normalize_and_validate_args(&mut args).map_err(invalid_input)?;
    let mut prepared = prepare(&args)?;
    let workload_digest = stage19_hex_digest(prepared.workload_digest);
    let profile_path = args
        .block_profile_output
        .clone()
        .unwrap_or_else(|| sibling_path(&args.output, "blocks.jsonl"));
    let blindfold_profile_path = args
        .blindfold_profile_output
        .clone()
        .unwrap_or_else(|| sibling_path(&args.output, "blindfold.jsonl"));
    ensure_parent(&profile_path)?;
    ensure_parent(&blindfold_profile_path)?;
    let profile_writer = Rc::new(RefCell::new(BufWriter::new(File::create(&profile_path)?)));
    let profile_error = Rc::new(RefCell::new(None::<io::Error>));
    let mut blindfold_profile_writer = BufWriter::new(File::create(&blindfold_profile_path)?);

    for warmup in 0..args.warmup_runs {
        println!("stage19 warmup {}/{}", warmup + 1, args.warmup_runs);
        for block_size in &args.block_sizes {
            let _ = run_cell(&mut prepared, &args, *block_size, 1, None, None)?;
        }
    }

    let mut samples = Vec::with_capacity(args.measurement_runs * args.block_sizes.len());
    for run_index in 1..=args.measurement_runs {
        for block_size in &args.block_sizes {
            println!(
                "stage19 measurement run={run_index}/{} block_size={block_size}",
                args.measurement_runs
            );
            let (sample, blindfold_phases) = run_cell(
                &mut prepared,
                &args,
                *block_size,
                run_index,
                Some(Rc::clone(&profile_writer)),
                Some(Rc::clone(&profile_error)),
            )?;
            serde_json::to_writer(
                &mut blindfold_profile_writer,
                &serde_json::json!({
                    "run_index": run_index,
                    "block_target_size": block_size,
                    "phases": blindfold_phases,
                }),
            )?;
            blindfold_profile_writer.write_all(b"\n")?;
            samples.push(sample);
        }
    }
    profile_writer.borrow_mut().flush()?;
    blindfold_profile_writer.flush()?;
    if let Some(error) = profile_error.borrow_mut().take() {
        return Err(error.into());
    }

    let rust_toolchain =
        std::env::var("RUSTUP_TOOLCHAIN").unwrap_or_else(|_| "rust-1.95".to_string());
    let mut artifact = Stage19BenchmarkArtifact::build(
        Stage19Platform::current(rust_toolchain),
        prepared.workload.clone(),
        workload_digest,
        prepared.preprocessing_micros,
        args.measurement_runs,
        samples,
    )
    .map_err(invalid_input)?;

    if let Some(path) = &args.baseline_input {
        let baseline: Stage19BenchmarkArtifact = serde_json::from_slice(&std::fs::read(path)?)?;
        let comparison = artifact
            .compare_with_baseline(&baseline, args.max_regression_percent)
            .map_err(invalid_input)?;
        println!("stage19_regression_passed={}", comparison.passed);
    }

    let markdown_path = args
        .markdown_output
        .clone()
        .unwrap_or_else(|| sibling_path(&args.output, "md"));
    ensure_parent(&args.output)?;
    ensure_parent(&markdown_path)?;
    std::fs::write(
        &args.output,
        format!("{}\n", artifact.to_json_pretty().map_err(invalid_input)?),
    )?;
    std::fs::write(
        &markdown_path,
        artifact.to_markdown().map_err(invalid_input)?,
    )?;

    println!("stage19_json={}", args.output.display());
    println!("stage19_markdown={}", markdown_path.display());
    println!("stage19_block_profiles={}", profile_path.display());
    println!(
        "stage19_blindfold_profiles={}",
        blindfold_profile_path.display()
    );
    println!("stage19_matrix_digest={}", artifact.matrix_digest);
    println!("stage19_bottleneck={}", artifact.bottleneck.phase);
    if artifact
        .regression
        .as_ref()
        .is_some_and(|comparison| !comparison.passed)
    {
        return Err(invalid_input("Stage-19 regression threshold exceeded".to_string()).into());
    }
    Ok(())
}

fn prepare(args: &Args) -> Result<PreparedBenchmark, Box<dyn Error>> {
    let started = Instant::now();
    let mut program = Program::new(args.workload.guest_name());
    args.workload.configure_program(&mut program);
    program.build(path_string(&args.guest_target)?);
    let inputs = args
        .workload
        .input_bytes(args.scale)
        .map_err(invalid_input)?;
    let (bytecode, init_memory_state, program_size, entry_address) = program.decode();
    let (_, _, _, io_device) = program.trace(&inputs, &[], &[]);
    let program_preprocessing = ProgramPreprocessing::<DoryCommitmentScheme>::preprocess(
        bytecode,
        init_memory_state,
        entry_address,
    )?;
    let shared = JoltSharedPreprocessing::new(
        program_preprocessing,
        io_device.memory_layout,
        args.workload.max_trace_length(),
    );
    let preprocessing = JoltProverPreprocessing::new(shared);
    let elf = program
        .get_elf_contents()
        .ok_or_else(|| invalid_input("guest ELF is unavailable after build".to_string()))?;
    let mut hasher = Sha3_256::new();
    hasher.update(b"JOLT_NOVA_STAGE19_WORKLOAD_V1");
    hasher.update(args.workload.id(args.scale).as_bytes());
    hasher.update(&elf);
    hasher.update(&inputs);
    let workload_digest = hasher.finalize().into();
    Ok(PreparedBenchmark {
        program,
        elf,
        inputs,
        memory_config: MemoryConfig {
            program_size: Some(program_size),
            ..Default::default()
        },
        preprocessing,
        preprocessing_micros: elapsed_micros(started),
        workload: args.workload.id(args.scale),
        workload_digest,
    })
}

fn run_cell(
    prepared: &mut PreparedBenchmark,
    args: &Args,
    block_size: usize,
    run_index: usize,
    profile_writer: Option<Rc<RefCell<BufWriter<File>>>>,
    profile_error: Option<Rc<RefCell<Option<io::Error>>>>,
) -> Result<(Stage19BenchmarkSample, Vec<BlindFoldPhaseMeasurement>), Box<dyn Error>> {
    let total_started = Instant::now();
    let trace_started = Instant::now();
    let block_iterator = tracer::trace_blocks(
        &prepared.elf,
        prepared.program.elf.as_ref(),
        &prepared.inputs,
        &[],
        &[],
        &prepared.memory_config,
        None,
        block_size,
    );
    let trace_micros = elapsed_micros(trace_started);

    let native_memory_sampler = PeakMemorySampler::start(args.memory_sample_interval_ms);
    let native_prove_started = Instant::now();
    let prover = RV64IMACProver::gen_from_elf(
        &prepared.preprocessing,
        &prepared.elf,
        &prepared.inputs,
        &[],
        &[],
        None,
        None,
        None,
    );
    let public_io = prover.program_io.clone();
    let padded_trace_length = prover.padded_trace_len;
    let (proof, debug_info) = prover.prove();
    let native_jolt_prove = elapsed_micros(native_prove_started);
    let serialized_proof = proof.serialize_to_bytes()?;
    let proof_for_verifier = RV64IMACProof::deserialize_from_bytes(&serialized_proof)?;

    let native_verify_started = Instant::now();
    let verifier_preprocessing = JoltVerifierPreprocessing::from(&prepared.preprocessing);
    let verifier = RV64IMACVerifier::new(
        &verifier_preprocessing,
        proof_for_verifier,
        public_io,
        None,
        debug_info,
    )?;
    let (_verified, artifacts) = verifier.verify_with_recursive_zk_complete_artifacts()?;
    let native_jolt_verify_and_export = elapsed_micros(native_verify_started);
    let native_memory = native_memory_sampler.finish();
    let receipt = artifacts.verified_jolt_receipt().clone();
    let statement = artifacts.statement();
    let parameters = Stage18ReleaseParameters::production(block_size, statement.shape_id)?;

    let recursive_memory_sampler = PeakMemorySampler::start(args.memory_sample_interval_ms);
    let pipeline = BlockProofPipeline::<
        _,
        jolt_core::ark_bn254::Fr,
        NovaFoldingBackend,
    >::with_backend_and_verified_jolt_lookup_receipt(
        receipt.digest(),
        NovaFoldingBackend::new(parameters.nova_config().clone()),
        receipt.clone(),
    );
    let streaming_started = Instant::now();
    let streaming = pipeline.prove_streaming_blocks_with_final_proof_and_profile(
        prepared
            .preprocessing
            .materialized_program()
            .bytecode
            .as_ref(),
        block_iterator,
        &parameters,
        |profile| {
            if let (Some(writer), Some(error_slot)) = (&profile_writer, &profile_error) {
                if error_slot.borrow().is_some() {
                    return;
                }
                let result = (|| -> io::Result<()> {
                    let profile_value: Value =
                        serde_json::from_str(&profile.to_json_line().map_err(io::Error::other)?)
                            .map_err(io::Error::other)?;
                    let row = serde_json::json!({
                        "run_index": run_index,
                        "block_target_size": block_size,
                        "profile": profile_value,
                    });
                    serde_json::to_writer(&mut *writer.borrow_mut(), &row)
                        .map_err(io::Error::other)?;
                    writer.borrow_mut().write_all(b"\n")
                })();
                if let Err(error) = result {
                    *error_slot.borrow_mut() = Some(error);
                }
            }
        },
    )?;
    streaming.verify(&parameters, &receipt)?;
    let nova_spartan_stream = elapsed_micros(streaming_started);
    let metrics = streaming.metrics().clone();

    let blindfold_started = Instant::now();
    let mut blindfold_profile = BlindFoldProfileCollector::new(args.memory_sample_interval_ms);
    let (end_to_end, verification_key, blindfold_baseline) = Stage18ZkEndToEndProof::<
        Bn254Curve,
        DoryCommitmentScheme,
        [u8; 32],
    >::prove_from_verified_artifacts_with_observer(
        streaming,
        artifacts,
        &parameters,
        &mut blindfold_profile,
    )?;
    let recursive_blindfold_prove = elapsed_micros(blindfold_started);
    let blindfold_phases = blindfold_profile.finish();

    let final_verify_started = Instant::now();
    end_to_end.verify(&parameters, &receipt, &verification_key)?;
    let final_end_to_end_verify = elapsed_micros(final_verify_started);
    let recursive_memory = recursive_memory_sampler.finish();
    let measured_total = elapsed_micros(total_started);
    let recursive_payload = metrics
        .final_folded_proof_bytes
        .saturating_add(blindfold_baseline.proof_bytes);
    let throughput = metrics.total_active_cycles as f64 / (measured_total as f64 / 1_000_000.0);

    let sample = Stage19BenchmarkSample {
        run_index,
        workload: prepared.workload.clone(),
        workload_sha3_256: stage19_hex_digest(prepared.workload_digest),
        block_target_size: block_size,
        block_count: metrics.block_count,
        active_cycles: metrics.total_active_cycles,
        padded_trace_length,
        max_resident_trace_blocks: metrics.max_resident_trace_blocks,
        max_resident_trace_cycles: metrics.max_resident_trace_cycles,
        receipt_is_production_zk: receipt.zk_mode(),
        native_jolt_verified: true,
        streaming_proof_verified: true,
        complete_zk_verified: true,
        lookup_backend: JOLT_NOVA_STAGE19_LOOKUP_BACKEND.to_string(),
        receipt_digest: stage19_hex_digest(receipt.digest()),
        jolt_statement_id: stage19_hex_digest(receipt.recursive_execution_statement_id()),
        timings_micros: Stage19TimingsMicros {
            trace: trace_micros.max(1),
            native_jolt_prove,
            native_jolt_verify_and_export,
            nova_spartan_stream,
            recursive_blindfold_prove,
            final_end_to_end_verify,
            measured_total,
        },
        throughput_active_cycles_per_second: throughput,
        memory_bytes: Stage19MemoryBytes {
            native_jolt_start: native_memory.start,
            native_jolt_peak: native_memory.peak,
            native_jolt_peak_delta: native_memory.peak_delta,
            recursive_start: recursive_memory.start,
            recursive_peak: recursive_memory.peak,
            recursive_peak_delta: recursive_memory.peak_delta,
            estimated_peak_trace: metrics.estimated_peak_trace_bytes as u64,
            peak_tracked_ram_addresses: metrics.peak_tracked_ram_addresses,
        },
        proof_sizes: Stage19ProofSizes {
            native_jolt_proof: serialized_proof.len(),
            folded_spartan_proof: metrics.final_folded_proof_bytes,
            blindfold_spartan_proof: blindfold_baseline.proof_bytes,
            recursive_proof_payload: recursive_payload,
        },
        relation_profiles: metrics
            .relation_profiles
            .iter()
            .map(|profile| Stage19RelationMeasurement {
                relation: profile.relation.to_string(),
                attribution_method: JOLT_NOVA_STAGE19_PROFILE_METHOD.to_string(),
                calls: profile.calls,
                total_nanos: profile.total_nanos,
                max_nanos: profile.max_nanos,
            })
            .collect(),
    };
    sample.validate().map_err(invalid_input)?;
    println!(
        "stage19_cell block_size={} blocks={} cycles={} total_ms={:.3}",
        block_size,
        sample.block_count,
        sample.active_cycles,
        sample.timings_micros.measured_total as f64 / 1_000.0
    );
    Ok((sample, blindfold_phases))
}

fn normalize_and_validate_args(args: &mut Args) -> Result<(), String> {
    if !(1..=MAX_MEASUREMENT_RUNS).contains(&args.measurement_runs) {
        return Err(format!(
            "measurement-runs must be in 1..={MAX_MEASUREMENT_RUNS}"
        ));
    }
    if args.warmup_runs > MAX_MEASUREMENT_RUNS {
        return Err(format!("warmup-runs must be in 0..={MAX_MEASUREMENT_RUNS}"));
    }
    if !(1..=MAX_MEMORY_SAMPLE_INTERVAL_MS).contains(&args.memory_sample_interval_ms) {
        return Err(format!(
            "memory-sample-interval-ms must be in 1..={MAX_MEMORY_SAMPLE_INTERVAL_MS}"
        ));
    }
    if !args.max_regression_percent.is_finite() || args.max_regression_percent < 0.0 {
        return Err("max-regression-percent must be finite and non-negative".to_string());
    }
    if args.workload != Workload::MemoryOps && args.scale == 0 {
        return Err("workload scale must be non-zero".to_string());
    }
    args.block_sizes.sort_unstable();
    args.block_sizes.dedup();
    if args.block_sizes.is_empty()
        || args
            .block_sizes
            .iter()
            .any(|size| *size == 0 || !size.is_power_of_two())
    {
        return Err("block sizes must be non-empty powers of two".to_string());
    }
    let mut paths = vec![args.output.clone()];
    paths.push(
        args.markdown_output
            .clone()
            .unwrap_or_else(|| sibling_path(&args.output, "md")),
    );
    paths.push(
        args.block_profile_output
            .clone()
            .unwrap_or_else(|| sibling_path(&args.output, "blocks.jsonl")),
    );
    paths.push(
        args.blindfold_profile_output
            .clone()
            .unwrap_or_else(|| sibling_path(&args.output, "blindfold.jsonl")),
    );
    let unique = paths.iter().collect::<std::collections::BTreeSet<_>>();
    if unique.len() != paths.len() {
        return Err(
            "JSON, Markdown, block-profile, and BlindFold-profile outputs must be distinct"
                .to_string(),
        );
    }
    Ok(())
}

fn physical_memory_bytes() -> Option<u64> {
    memory_stats::memory_stats().map(|stats| stats.physical_mem as u64)
}

fn elapsed_micros(started: Instant) -> u64 {
    started.elapsed().as_micros().max(1).min(u64::MAX as u128) as u64
}

fn ensure_parent(path: &Path) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    Ok(())
}

fn sibling_path(path: &Path, suffix: &str) -> PathBuf {
    let stem = path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("stage19");
    path.with_file_name(format!("{stem}.{suffix}"))
}

fn path_string(path: &Path) -> Result<&str, io::Error> {
    path.to_str()
        .ok_or_else(|| invalid_input(format!("path is not valid UTF-8: {}", path.display())))
}

fn invalid_input(message: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args() -> Args {
        Args {
            workload: Workload::Fibonacci,
            scale: 32,
            block_sizes: vec![256, 64, 64],
            measurement_runs: 1,
            warmup_runs: 0,
            memory_sample_interval_ms: 10,
            output: PathBuf::from("stage19.json"),
            markdown_output: None,
            block_profile_output: None,
            blindfold_profile_output: None,
            baseline_input: None,
            max_regression_percent: 15.0,
            guest_target: PathBuf::from("target-stage-19-guest"),
        }
    }

    #[test]
    fn stage19_runner_normalizes_block_matrix() {
        let mut args = args();
        normalize_and_validate_args(&mut args).unwrap();
        assert_eq!(args.block_sizes, vec![64, 256]);
    }

    #[test]
    fn stage19_runner_rejects_invalid_scales_and_outputs() {
        let mut zero = args();
        zero.scale = 0;
        assert!(normalize_and_validate_args(&mut zero).is_err());

        let mut duplicate = args();
        duplicate.markdown_output = Some(duplicate.output.clone());
        assert!(normalize_and_validate_args(&mut duplicate).is_err());
    }

    #[test]
    fn stage19_workloads_have_versioned_inputs_and_bounds() {
        for workload in [
            Workload::Fibonacci,
            Workload::Sha3Chain,
            Workload::MemoryOps,
            Workload::Collatz,
        ] {
            assert!(workload.max_trace_length().is_power_of_two());
            assert!(!workload.id(32).is_empty());
            assert!(workload.input_bytes(32).is_ok());
        }
    }

    #[test]
    fn stage19_committed_release_baselines_validate() {
        let baseline_root =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../benchmark-baselines/stage19");
        for (name, profile_name) in [
            (
                "fibonacci-32.windows-rust-1.95.json",
                "fibonacci-32.windows-rust-1.95.blocks.jsonl",
            ),
            (
                "fibonacci-1024.windows-rust-1.95.json",
                "fibonacci-1024.windows-rust-1.95.blocks.jsonl",
            ),
        ] {
            let encoded = std::fs::read_to_string(baseline_root.join(name)).unwrap();
            let artifact: Stage19BenchmarkArtifact = serde_json::from_str(&encoded).unwrap();
            artifact.validate().unwrap();
            assert_eq!(artifact.platform.build_profile, "release");
            assert!(artifact.samples.iter().all(|sample| {
                sample.native_jolt_verified
                    && sample.streaming_proof_verified
                    && sample.complete_zk_verified
                    && sample.max_resident_trace_blocks <= 2
            }));

            let profiles = std::fs::read_to_string(baseline_root.join(profile_name)).unwrap();
            let rows = profiles
                .lines()
                .map(|line| serde_json::from_str::<Value>(line).unwrap())
                .collect::<Vec<_>>();
            assert_eq!(
                rows.len(),
                artifact
                    .samples
                    .iter()
                    .map(|sample| sample.block_count)
                    .sum::<usize>()
            );
            assert!(rows.iter().all(|row| {
                row["profile"]["resident_trace_blocks"]
                    .as_u64()
                    .is_some_and(|blocks| blocks <= 2)
            }));
        }
    }
}
