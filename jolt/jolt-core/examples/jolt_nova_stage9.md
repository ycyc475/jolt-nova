# Jolt-Nova Stage 9

Stage 9 moves the validated Stage 8 runner toward scalable, comparative
experiments. Stage 9.1 closes the soundness gap that previously prevented a
program-bound trace from proving a strict block prefix.

## Stage 9.1: externally bound CPU lookahead

The final CPU/R1CS row of a non-terminal prefix needs the first cycle after the
prefix to enforce its next-PC relation. Before Stage 9.1, the pipeline carried
only a Boolean saying whether a lookahead existed. The benchmark runner
therefore rejected every program-bound block count except the complete trace.

Stage 9.1 adds the following data flow:

1. The prefix runner obtains the first cycle of the next block from the complete,
   validated trace bundle.
2. The CPU prover validates the final row against that external cycle.
3. The CPU proof stores a domain-separated SHA3-256 commitment to the canonical
   postcard serialization of the exact cycle.
4. Witness verification recomputes the commitment from the verifier-supplied
   cycle and rejects omission or replacement.
5. The commitment is copied into `FoldableBlockState` and included in
   `state_digest`, so the Nova recursive statement and final folded accumulator
   bind the same lookahead.

Internal block boundaries still use the next block's first cycle. A terminated
final block still uses the canonical no-op lookahead. Existing complete-trace
APIs remain wrappers with no external lookahead.

## Runner usage

Real ELF and program-bound trace sources now accept explicit partial prefixes:

```text
cargo run -p jolt-core --features nova --example \
  jolt_nova_final_proof_size_benchmark -- \
  --trace-source elf \
  --elf-input path/to/program.elf \
  --trace-block-size 1024 \
  --block-counts 1,2,4,8
```

If `--block-counts` is omitted for a program-bound trace, the runner continues
to select only the complete trace by default. This keeps the default real-trace
run bounded while allowing explicit scaling experiments.

## Verification coverage

Stage 9.1 tests cover:

- exact lookahead commitment generation;
- rejection when a prefix verifier omits the external lookahead;
- rejection after commitment tampering;
- propagation into the foldable state;
- generic partial-prefix pipeline proving and verification;
- a real two-block ELF run reporting both the one-block prefix and full trace.

## Stage 9.2: production fixtures and multi-run baselines

Stage 9.2 adds two versioned, deterministic RV64 guest fixtures directly to the
runner:

- `cpu-lookup-64k`: 49,156 cycles of arithmetic, register updates, branches,
  and instruction lookups;
- `ram-lookup-80k`: 81,925 cycles including stack loads/stores, register
  updates, branches, and instruction lookups.

The fixtures are real RV64 ELF programs. Tests decode and execute both programs
to termination and check their exact cycle counts. They do not require an
external guest toolchain, which makes workload identity stable across machines.

Run a repeatable production-size experiment with:

```text
cargo run --release -p jolt-core --no-default-features --features nova \
  --example jolt_nova_final_proof_size_benchmark -- \
  --trace-source fixture \
  --fixture cpu-lookup-64k \
  --trace-output benchmark-runs/jolt-nova/cpu-lookup-64k.trace.json \
  --trace-block-size 1024 \
  --block-counts 1,4,16,49 \
  --warmup-runs 1 \
  --measurement-runs 5 \
  --output benchmark-runs/jolt-nova/cpu-lookup-64k.report.json
```

The measured runs still write the canonical report, manifest, and latest
single-run performance artifact. With more than one measured run, the runner
also writes `<report-stem>.aggregate.json`. The versioned aggregate records:

- every raw performance sample;
- min, median, mean, max, and population standard deviation;
- proving and end-to-end timing/throughput;
- best-effort peak physical-memory delta;
- the exact trace/workload SHA3-256 digest, fixture, build profile, target,
  source block count, and reported prefixes used to establish workload
  compatibility.

An existing aggregate can act as an automated regression gate:

```text
cargo run --release -p jolt-core --no-default-features --features nova \
  --example jolt_nova_final_proof_size_benchmark -- \
  --trace-source fixture \
  --fixture cpu-lookup-64k \
  --trace-output benchmark-runs/jolt-nova/current.trace.json \
  --trace-block-size 1024 \
  --block-counts 1,4,16,49 \
  --warmup-runs 1 \
  --measurement-runs 5 \
  --baseline-input benchmark-runs/jolt-nova/baseline.aggregate.json \
  --aggregate-output benchmark-runs/jolt-nova/current.aggregate.json \
  --max-regression-percent 10 \
  --output benchmark-runs/jolt-nova/current.report.json
```

The comparison rejects incompatible workloads before comparing them. It exits
non-zero when mean proving time, end-to-end time, throughput, or available
peak-memory metrics regress beyond the configured threshold. Faster results are
not treated as regressions.

## Remaining Stage 9 work

Stage 9.2 completes the repeatable baseline infrastructure. Later Stage 9 work
will address streaming/binary trace storage, real LogUp integration,
per-relation profiling, and adversarial soundness review.
