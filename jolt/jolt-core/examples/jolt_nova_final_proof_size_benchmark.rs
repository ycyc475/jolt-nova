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
            BlockProofPipeline, JoltNovaReportOutputFormat,
            NovaBlockProofPipelineFinalProofSizeBenchmarkArtifact, NovaFoldingBackend,
            JOLT_NOVA_FINAL_PROOF_SIZE_SCALING_REPORT_KIND, JOLT_NOVA_REPORT_SCHEMA_VERSION,
        },
        bytecode::BytecodePreprocessing,
    },
};
use tracer::{instruction::Cycle, MachineBoundaryState, TraceBlock};

const DEFAULT_OUTPUT_PATH: &str = "benchmark-runs/jolt-nova/final-proof-size-scaling.json";
const RUNNER_NAME: &str = "jolt_nova_final_proof_size_benchmark";
const MANIFEST_SCHEMA_VERSION: &str = "jolt-nova-benchmark-runner-v1";

/// Synthetic smoke runner for Jolt-Nova final proof size scaling artifacts.
///
/// This runner is intentionally small: it builds contiguous no-op trace blocks,
/// runs the real Nova folding/final-proof-size reporting path, and writes the
/// stage-8 JSON artifact under `benchmark-runs/` by default.
#[derive(Debug, Clone, Parser)]
struct Args {
    /// Synthetic trace profile to generate.
    #[arg(long, value_enum, default_value_t = TraceProfile::SyntheticNoop)]
    trace_profile: TraceProfile,

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

fn main() {
    if let Err(error) = run(Args::parse()) {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run(args: Args) -> Result<(), Box<dyn Error>> {
    let block_counts = normalized_block_counts(&args).map_err(invalid_input)?;
    let blocks = build_trace_blocks(args.trace_profile, args.blocks, args.cycles_per_block)
        .map_err(invalid_input)?;
    let manifest_output = normalized_manifest_output(&args.output, args.manifest_output.as_ref());
    let bytecode = BytecodePreprocessing::default();
    let pipeline = BlockProofPipeline::<_, ark_bn254::Fr, NovaFoldingBackend>::with_backend(
        [args.program_digest_byte; 32],
        NovaFoldingBackend::default(),
    );

    let artifact = pipeline.prove_block_prefixes_and_write_final_proof_size_benchmark_artifact(
        &bytecode,
        &blocks,
        &block_counts,
        JoltNovaReportOutputFormat::Json,
        &args.output,
    )?;
    write_manifest(&args, &block_counts, &artifact, &manifest_output)?;

    println!("wrote {}", manifest_path_string(&args.output));
    println!("manifest {}", manifest_path_string(&manifest_output));
    println!("trace_profile {}", args.trace_profile);
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

fn normalized_block_counts(args: &Args) -> Result<Vec<usize>, String> {
    if args.blocks == 0 {
        return Err("blocks must be greater than zero".to_string());
    }
    if args.cycles_per_block == 0 {
        return Err("cycles-per-block must be greater than zero".to_string());
    }

    let block_counts = if args.block_counts.is_empty() {
        (1..=args.blocks).collect::<Vec<_>>()
    } else {
        args.block_counts.clone()
    };

    validate_block_counts(args.blocks, &block_counts)?;
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
        build_manifest_json(args, block_counts, artifact, manifest_output),
    )
}

fn build_manifest_json(
    args: &Args,
    block_counts: &[usize],
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

    let mut json = String::new();
    json.push('{');
    append_json_string_field(&mut json, "schema_version", MANIFEST_SCHEMA_VERSION);
    json.push(',');
    append_json_string_field(&mut json, "runner", RUNNER_NAME);
    json.push(',');
    append_json_string_field(&mut json, "trace_profile", args.trace_profile.as_str());
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
    append_json_usize_field(&mut json, "blocks", args.blocks);
    json.push(',');
    append_json_usize_field(&mut json, "cycles_per_block", args.cycles_per_block);
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

fn append_json_string_field(json: &mut String, name: &str, value: &str) {
    append_json_string(json, name);
    json.push(':');
    append_json_string(json, value);
}

fn append_json_usize_field(json: &mut String, name: &str, value: usize) {
    append_json_string(json, name);
    json.push(':');
    json.push_str(&value.to_string());
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
            trace_profile: TraceProfile::SyntheticNoop,
            blocks: 3,
            cycles_per_block: 2,
            block_counts: Vec::new(),
            output: PathBuf::from(DEFAULT_OUTPUT_PATH),
            manifest_output: None,
            program_digest_byte: 9,
        };

        assert_eq!(normalized_block_counts(&args).unwrap(), vec![1, 2, 3]);
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
        let blocks = build_trace_blocks(TraceProfile::SyntheticNoop, 3, 2).unwrap();

        assert_eq!(blocks.len(), 3);
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
    fn manifest_json_records_runner_inputs_and_outputs() {
        let args = Args {
            trace_profile: TraceProfile::SyntheticNoop,
            blocks: 2,
            cycles_per_block: 3,
            block_counts: vec![1, 2],
            output: PathBuf::from("benchmark-runs/jolt-nova/report.json"),
            manifest_output: Some(PathBuf::from(
                "benchmark-runs/jolt-nova/report.manifest.json",
            )),
            program_digest_byte: 11,
        };
        let block_counts = normalized_block_counts(&args).unwrap();
        let artifact = sample_benchmark_artifact();
        let manifest_path = normalized_manifest_output(&args.output, args.manifest_output.as_ref());
        let json = build_manifest_json(&args, &block_counts, &artifact, &manifest_path);

        assert!(json.contains("\"schema_version\":\"jolt-nova-benchmark-runner-v1\""));
        assert!(json.contains("\"runner\":\"jolt_nova_final_proof_size_benchmark\""));
        assert!(json.contains("\"trace_profile\":\"synthetic-noop\""));
        assert!(json.contains("\"report_schema_version\":\"jolt-nova-report-v1\""));
        assert!(json.contains("\"report_kind\":\"final-proof-size-scaling\""));
        assert!(json.contains("\"report_output_format\":\"json\""));
        assert!(json.contains("\"report_path\":\"benchmark-runs/jolt-nova/report.json\""));
        assert!(
            json.contains("\"manifest_path\":\"benchmark-runs/jolt-nova/report.manifest.json\"")
        );
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
    fn write_manifest_creates_parent_directory() {
        let args = Args {
            trace_profile: TraceProfile::SyntheticNoop,
            blocks: 1,
            cycles_per_block: 2,
            block_counts: vec![1],
            output: PathBuf::from("report.json"),
            manifest_output: None,
            program_digest_byte: 9,
        };
        let artifact = sample_benchmark_artifact();
        let manifest_path = temp_manifest_path("manifest-write", "report.manifest.json");

        write_manifest(&args, &[1], &artifact, &manifest_path).unwrap();
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
