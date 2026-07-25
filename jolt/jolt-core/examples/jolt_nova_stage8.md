# Jolt-Nova Stage 8 runner and baseline guide

Stage 8 turns the block proof pipeline into a reproducible command-line
experiment. The runner accepts synthetic blocks, a versioned trace file, or a
real RV64 ELF program and produces three linked artifacts:

1. a final-proof-size scaling report;
2. a manifest that binds the inputs, trace schema, and output paths;
3. a performance baseline with phase timings, throughput, and best-effort peak
   physical memory.

The real-program path is:

```text
RV64 ELF
  -> Jolt program decode and inline expansion
  -> bytecode preprocessing
  -> tracer blocks at tick boundaries
  -> versioned, program-bound trace bundle
  -> Nova folding
  -> final Spartan proof
  -> size, timing, throughput, and memory artifacts
```

## Run a real ELF baseline

Run release builds for meaningful measurements:

```bash
cargo run --release \
  -p jolt-core \
  --example jolt_nova_final_proof_size_benchmark \
  --no-default-features \
  --features nova \
  -- \
  --trace-source elf \
  --elf-input path/to/guest.elf \
  --trace-output benchmark-runs/jolt-nova/guest.trace.json \
  --trace-block-size 1024 \
  --output benchmark-runs/jolt-nova/guest.report.json \
  --manifest-output benchmark-runs/jolt-nova/guest.manifest.json \
  --performance-output benchmark-runs/jolt-nova/guest.performance.json \
  --memory-sample-interval-ms 10
```

Optional `--guest-input`, `--untrusted-advice`, and `--trusted-advice` files are
passed directly to the tracer. The runner exports the trace bundle and reads it
back before proving, so the same validation path is exercised as an independently
loaded bundle.

To rerun an exported bundle without tracing the ELF again:

```bash
cargo run --release \
  -p jolt-core \
  --example jolt_nova_final_proof_size_benchmark \
  --no-default-features \
  --features nova \
  -- \
  --trace-source trace-file \
  --trace-input benchmark-runs/jolt-nova/guest.trace.json \
  --output benchmark-runs/jolt-nova/guest-replay.report.json
```

## Performance artifact

The `jolt-nova-performance-baseline-v1` artifact records:

- build profile, operating system, and architecture;
- source block count, selected block counts, largest-prefix size, and the sum of
  blocks/cycles actually processed across all independently proved prefixes;
- trace loading or generation time;
- Nova folding plus Spartan report time;
- manifest-write and measured end-to-end time;
- proving and end-to-end throughput;
- initial, sampled peak, and peak-delta physical memory.

Memory is sampled in-process with `memory-stats`. It is a practical regression
signal, not an exact allocator profile: short-lived peaks between samples can be
missed, and operating-system accounting differs. Use the same release build,
machine, input, block size, and sample interval when comparing runs. Run at least
three times and compare the median timing and maximum observed peak memory. The
sample interval must be in `1..=1000` milliseconds so shutdown cannot be delayed
indefinitely by a sleeping sampler.

The measured total intentionally stops before writing the performance artifact,
so serializing the measurement does not measure itself.

When several `--block-counts` are requested, each prefix is proved
independently. Throughput therefore uses `processed_block_count` and
`processed_active_cycles` (the sums across all requested prefixes), while the
largest-prefix fields describe the final scaling point.

## Stage 8 completion status

Stage 8 now provides:

- final folded-instance and Spartan proof-size reporting;
- stable JSON report and manifest schemas;
- synthetic, versioned trace-file, and real ELF trace sources;
- bytecode and ELF digest binding for real trace bundles;
- trace-bundle readback and chain validation;
- terminal-block CPU/R1CS lookahead handling;
- end-to-end ELF-to-Nova-to-Spartan tests;
- repeatable wall-clock, throughput, and memory baseline artifacts;
- CI coverage for the complete runner.

## Current boundaries

- Stage 9.1 now supports sound program-bound partial prefixes by committing the
  external next-cycle lookahead in both the CPU proof and folded state. See the
  [Stage 9 guide](./jolt_nova_stage9.md).
- Stage 9.3 adds per-block binary serialization; the proving pipeline still
  materializes decoded blocks for repeated prefix experiments.
- Peak-memory sampling is best effort and does not replace heap profiling.
- The runner measures the current Nova backend and Spartan final proof; it is not
  yet a comparative LogUp/Lasso benchmark.

## Next-stage directions

Stage 9 should move from a validated experimental pipeline toward scalable,
comparative research:

1. ~~bind external lookahead data so real traces can be folded and reported by
   prefixes~~ (completed in Stage 9.1);
2. ~~add production-size guest fixtures and automated multi-run baseline
   comparisons~~ (completed in Stage 9.2);
3. ~~binary-encode trace blocks one record at a time to reduce JSON memory and
   I/O overhead~~ (completed in Stage 9.3);
4. introduce a real LogUp lookup backend behind the existing replaceable lookup
   interface and compare it with Lasso;
5. profile per-relation witness generation and Nova step-circuit costs;
6. perform adversarial trace, boundary, and final-proof soundness review before
   claiming a production proof system.
