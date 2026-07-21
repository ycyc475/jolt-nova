use std::{error::Error, io, path::PathBuf};

use clap::Parser;
use common::constants::REGISTER_COUNT;
use jolt_core::{
    ark_bn254,
    zkvm::{
        block::{BlockProofPipeline, JoltNovaReportOutputFormat, NovaFoldingBackend},
        bytecode::BytecodePreprocessing,
    },
};
use tracer::{instruction::Cycle, MachineBoundaryState, TraceBlock};

const DEFAULT_OUTPUT_PATH: &str = "benchmark-runs/jolt-nova/final-proof-size-scaling.json";

/// Synthetic smoke runner for Jolt-Nova final proof size scaling artifacts.
///
/// This runner is intentionally small: it builds contiguous no-op trace blocks,
/// runs the real Nova folding/final-proof-size reporting path, and writes the
/// stage-8 JSON artifact under `benchmark-runs/` by default.
#[derive(Debug, Clone, Parser)]
struct Args {
    /// Number of synthetic no-op trace blocks to generate.
    #[arg(long, default_value_t = 2)]
    blocks: usize,

    /// Active no-op cycles per synthetic block.
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

    /// Repeated byte used to form the synthetic program digest.
    #[arg(long, default_value_t = 9)]
    program_digest_byte: u8,
}

fn main() {
    if let Err(error) = run(Args::parse()) {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run(args: Args) -> Result<(), Box<dyn Error>> {
    let block_counts = normalized_block_counts(&args).map_err(invalid_input)?;
    let blocks =
        build_noop_trace_blocks(args.blocks, args.cycles_per_block).map_err(invalid_input)?;
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

    println!("wrote {}", args.output.display());
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

    #[test]
    fn default_block_counts_cover_every_prefix() {
        let args = Args {
            blocks: 3,
            cycles_per_block: 2,
            block_counts: Vec::new(),
            output: PathBuf::from(DEFAULT_OUTPUT_PATH),
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
    fn noop_trace_blocks_are_contiguous() {
        let blocks = build_noop_trace_blocks(3, 2).unwrap();

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
}
