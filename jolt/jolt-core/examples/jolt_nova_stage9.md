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

## Remaining Stage 9 work

The next step is Stage 9.2: add production-size guest fixtures and automated,
repeatable multi-run baseline comparison. Later Stage 9 work will address
streaming/binary trace storage, real LogUp integration, per-relation profiling,
and adversarial soundness review.
