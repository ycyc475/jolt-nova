use std::{
    error::Error,
    fmt, io,
    path::{Path, PathBuf},
};

use clap::{Parser, ValueEnum};
use common::constants::REGISTER_COUNT;
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
use serde::Deserialize;
use sha3::{Digest as ShaDigest, Sha3_256};
use tracer::{instruction::Cycle, MachineBoundaryState, TraceBlock};

const DEFAULT_OUTPUT_PATH: &str = "benchmark-runs/jolt-nova/final-proof-size-scaling.json";
const RUNNER_NAME: &str = "jolt_nova_final_proof_size_benchmark";
const MANIFEST_SCHEMA_VERSION: &str = "jolt-nova-benchmark-runner-v1";
const TRACE_FILE_SCHEMA_VERSION: &str = "jolt-nova-trace-blocks-v1";

/// Synthetic smoke runner for Jolt-Nova final proof size scaling artifacts.
///
/// This runner is intentionally small: it builds contiguous no-op trace blocks,
/// runs the real Nova folding/final-proof-size reporting path, and writes the
/// stage-8 JSON artifact under `benchmark-runs/` by default.
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
}

impl TraceSource {
    fn as_str(self) -> &'static str {
        match self {
            Self::Synthetic => "synthetic",
            Self::TraceFile => "trace-file",
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
    file_metadata: Option<TraceFileMetadata>,
}

#[derive(Debug)]
struct TraceFileMetadata {
    schema_version: String,
    sha3_256: [u8; 32],
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TraceFileDocument {
    schema_version: String,
    blocks: Vec<SerializedTraceBlock>,
}

#[derive(Debug, Deserialize)]
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

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SerializedMachineBoundaryState {
    global_cycle: usize,
    emulator_trace_len: usize,
    pc: u64,
    registers: Vec<i64>,
    terminated: bool,
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

fn main() {
    if let Err(error) = run(Args::parse()) {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run(args: Args) -> Result<(), Box<dyn Error>> {
    let loaded_trace = load_trace_blocks(&args).map_err(invalid_input)?;
    let blocks = &loaded_trace.blocks;
    let block_counts = normalized_block_counts(&args, blocks.len()).map_err(invalid_input)?;
    let manifest_output = normalized_manifest_output(&args.output, args.manifest_output.as_ref());
    let bytecode = BytecodePreprocessing::default();
    let pipeline = BlockProofPipeline::<_, ark_bn254::Fr, NovaFoldingBackend>::with_backend(
        [args.program_digest_byte; 32],
        NovaFoldingBackend::default(),
    );

    let artifact = pipeline.prove_block_prefixes_and_write_final_proof_size_benchmark_artifact(
        &bytecode,
        blocks,
        &block_counts,
        JoltNovaReportOutputFormat::Json,
        &args.output,
    )?;
    write_manifest(
        &args,
        &block_counts,
        blocks.len(),
        loaded_trace.file_metadata.as_ref(),
        &artifact,
        &manifest_output,
    )?;

    println!("wrote {}", manifest_path_string(&args.output));
    println!("manifest {}", manifest_path_string(&manifest_output));
    println!("trace_source {}", args.trace_source);
    println!(
        "trace_profile {}",
        selected_trace_profile(&args).unwrap_or("none")
    );
    if let Some(metadata) = &loaded_trace.file_metadata {
        println!("trace_file_schema {}", metadata.schema_version);
        println!("trace_input_sha3_256 {}", hex_digest(&metadata.sha3_256));
    }
    println!("format {}", artifact.output_format);
    println!("rows {}", artifact.report.rows.len());
    println!("bytes {}", artifact.serialized_bytes().len());
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

fn normalized_manifest_output(report_output: &Path, manifest_output: Option<&PathBuf>) -> PathBuf {
    manifest_output
        .cloned()
        .unwrap_or_else(|| default_manifest_output_path(report_output))
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

fn normalized_block_counts(args: &Args, loaded_blocks: usize) -> Result<Vec<usize>, String> {
    if loaded_blocks == 0 {
        return Err("loaded trace must contain at least one block".to_string());
    }

    let block_counts = if args.block_counts.is_empty() {
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
            if let Some(trace_input) = &args.trace_input {
                return Err(format!(
                    "trace-input {} is only valid with --trace-source trace-file",
                    manifest_path_string(trace_input)
                ));
            }
            Ok(LoadedTraceBlocks {
                blocks: build_trace_blocks(args.trace_profile, args.blocks, args.cycles_per_block)?,
                file_metadata: None,
            })
        }
        TraceSource::TraceFile => {
            let trace_input = args.trace_input.as_ref().ok_or_else(|| {
                "trace-source trace-file requires --trace-input <path>".to_string()
            })?;
            load_trace_file(trace_input)
        }
    }
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
    if document.schema_version != TRACE_FILE_SCHEMA_VERSION {
        return Err(format!(
            "unsupported trace schema version {:?}; expected {:?}",
            document.schema_version, TRACE_FILE_SCHEMA_VERSION
        ));
    }

    let blocks = document
        .blocks
        .into_iter()
        .map(TraceBlock::try_from)
        .collect::<Result<Vec<_>, _>>()?;
    validate_loaded_trace_blocks(&blocks)?;

    Ok(LoadedTraceBlocks {
        blocks,
        file_metadata: Some(TraceFileMetadata {
            schema_version: document.schema_version,
            sha3_256: Sha3_256::digest(&input_bytes).into(),
        }),
    })
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
    append_json_optional_path_field(&mut json, "trace_input_path", args.trace_input.as_ref());
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
    append_json_usize_array_field(&mut json, "block_counts", block_counts);
    json.push(',');
    append_json_usize_field(
        &mut json,
        "program_digest_byte",
        args.program_digest_byte as usize,
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

    #[test]
    fn default_block_counts_cover_every_prefix() {
        let args = Args {
            trace_source: TraceSource::Synthetic,
            trace_profile: TraceProfile::SyntheticNoop,
            trace_input: None,
            blocks: 3,
            cycles_per_block: 2,
            block_counts: Vec::new(),
            output: PathBuf::from(DEFAULT_OUTPUT_PATH),
            manifest_output: None,
            program_digest_byte: 9,
        };

        assert_eq!(normalized_block_counts(&args, 3).unwrap(), vec![1, 2, 3]);
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
    fn noop_trace_blocks_are_contiguous() {
        let args = Args {
            trace_source: TraceSource::Synthetic,
            trace_profile: TraceProfile::SyntheticNoop,
            trace_input: None,
            blocks: 3,
            cycles_per_block: 2,
            block_counts: Vec::new(),
            output: PathBuf::from(DEFAULT_OUTPUT_PATH),
            manifest_output: None,
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
            blocks: 1,
            cycles_per_block: 2,
            block_counts: Vec::new(),
            output: PathBuf::from(DEFAULT_OUTPUT_PATH),
            manifest_output: None,
            program_digest_byte: 9,
        };
        assert!(load_trace_blocks(&synthetic_with_input)
            .unwrap_err()
            .contains("only valid with --trace-source trace-file"));

        let trace_file_without_input = Args {
            trace_source: TraceSource::TraceFile,
            trace_profile: TraceProfile::SyntheticNoop,
            trace_input: None,
            blocks: 1,
            cycles_per_block: 2,
            block_counts: Vec::new(),
            output: PathBuf::from(DEFAULT_OUTPUT_PATH),
            manifest_output: None,
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
            blocks: 1,
            cycles_per_block: 2,
            block_counts: Vec::new(),
            output: PathBuf::from(DEFAULT_OUTPUT_PATH),
            manifest_output: None,
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
        assert_eq!(normalized_block_counts(&args, 2).unwrap(), vec![1, 2]);
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
            blocks: 2,
            cycles_per_block: 3,
            block_counts: vec![1, 2],
            output: PathBuf::from("benchmark-runs/jolt-nova/report.json"),
            manifest_output: Some(PathBuf::from(
                "benchmark-runs/jolt-nova/report.manifest.json",
            )),
            program_digest_byte: 11,
        };
        let block_counts = normalized_block_counts(&args, 2).unwrap();
        let artifact = sample_benchmark_artifact();
        let manifest_path = normalized_manifest_output(&args.output, args.manifest_output.as_ref());
        let json = build_manifest_json(&args, &block_counts, 2, None, &artifact, &manifest_path);

        assert!(json.contains("\"schema_version\":\"jolt-nova-benchmark-runner-v1\""));
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
            blocks: 2,
            cycles_per_block: 2,
            block_counts: vec![1, 2],
            output: PathBuf::from("benchmark-runs/jolt-nova/report.json"),
            manifest_output: None,
            program_digest_byte: 9,
        };
        let artifact = sample_benchmark_artifact();
        let manifest_path = normalized_manifest_output(&args.output, args.manifest_output.as_ref());
        let metadata = TraceFileMetadata {
            schema_version: TRACE_FILE_SCHEMA_VERSION.to_string(),
            sha3_256: [0xabu8; 32],
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
            blocks: 1,
            cycles_per_block: 2,
            block_counts: vec![1],
            output: PathBuf::from("report.json"),
            manifest_output: None,
            program_digest_byte: 9,
        };
        let artifact = sample_benchmark_artifact();
        let manifest_path = temp_manifest_path("manifest-write", "report.manifest.json");

        write_manifest(&args, &[1], 1, None, &artifact, &manifest_path).unwrap();
        let manifest = std::fs::read_to_string(&manifest_path).unwrap();
        std::fs::remove_file(&manifest_path).unwrap();
        std::fs::remove_dir(manifest_path.parent().unwrap()).unwrap();

        assert!(manifest.contains("\"schema_version\":\"jolt-nova-benchmark-runner-v1\""));
        assert!(manifest.contains("\"block_counts\":[1]"));
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
