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

## Stage 9.3: versioned binary trace storage

Stage 9.3 adds a binary trace container for large trace bundles while retaining
full compatibility with the Stage 8/9 JSON files. Output format selection is
extension-based:

- `.bin` and `.jnvtrace` write the binary container;
- every other extension, including `.json`, writes the existing JSON document.

Input format detection uses the binary magic bytes rather than the file
extension, so renamed trace files still load correctly.

The `jolt-nova-binary-trace-v1` container contains:

1. fixed magic bytes and a length-delimited postcard header;
2. a SHA3-256 digest of the header;
3. the logical trace schema, program binding, and declared block count;
4. one length-delimited postcard record per trace block;
5. a separate SHA3-256 digest for every block record.

The loader checks size limits before allocation, verifies every digest before
decoding, rejects truncation and trailing bytes, validates the bytecode/program
digest, and finally runs the unchanged trace-block chain validation. The
manifest records `trace_storage_format` as either `json` or `binary`, while the
workload digest remains the SHA3-256 digest of the exact trace file. This new
field advances the runner manifest schema to `jolt-nova-benchmark-runner-v4`.

Use the binary format by changing the trace output extension:

```text
cargo run --release -p jolt-core --no-default-features --features nova \
  --example jolt_nova_final_proof_size_benchmark -- \
  --trace-source fixture \
  --fixture cpu-lookup-64k \
  --trace-output benchmark-runs/jolt-nova/cpu-lookup-64k.trace.bin \
  --trace-block-size 1024 \
  --block-counts 1,4,16,49 \
  --output benchmark-runs/jolt-nova/cpu-lookup-64k.report.json
```

Serialization and deserialization now allocate at most one encoded block at a
time instead of constructing a second in-memory representation of the complete
trace document. The current prover still materializes the decoded
`Vec<TraceBlock>` because the prefix-report pipeline revisits the source blocks;
fully streaming trace-to-fold execution is a later optimization.

Stage 9.3 tests cover JSON backward compatibility, binary real-ELF and
49,156-cycle production-fixture round trips, manifest format reporting,
per-block digest corruption, truncated payloads, trailing bytes, and the binary
trace path through Nova folding and the Spartan report.

## Stage 9.4: real block LogUp relation

Stage 9.4 replaces the former LogUp naming/interface placeholder with an actual
finite-field fractional-sum relation. For every block, the prover now:

1. derives a domain-separated Fiat-Shamir tuple-compression challenge from the
   block identity and canonical lookup digests;
2. compresses each tuple
   `(left_operand, right_operand, lookup_index, lookup_output)` into one BN254
   field element;
3. derives a separate denominator challenge `beta`;
4. deterministically increments `beta` if any denominator is zero;
5. computes the query-side sum
   `sum_i 1 / (beta + query_i)`;
6. independently reconstructs executed table entries from the trace cycles and
   computes `sum_j multiplicity_j / (beta + table_j)`;
7. binds the challenges, retry count, cardinalities, sums, and a
   domain-separated proof digest into `BlockLookupClaim` and
   `FoldableBlockState`.

The implementation uses batch inversion, so each side needs one field inversion
instead of one inversion per lookup.

The Nova step circuit supports two fixed-shape lookup subclaim backends:

- `transcript` retains the Stage 6 digest/count fingerprint baseline;
- `log-up` selects `logup-subclaim-v1`.

The circuit constrains the selector to be Boolean. With LogUp selected, it
enforces equality of the query and table fractional sums and selects a
LogUp-specific fingerprint that includes every challenge and proof field. The
selected fingerprint advances both the lookup accumulator and semantic
accumulator, so it is covered by the recursive Nova proof and final Spartan
proof.

Run the real LogUp path with:

```text
cargo run --release -p jolt-core --no-default-features --features nova \
  --example jolt_nova_final_proof_size_benchmark -- \
  --trace-source fixture \
  --fixture cpu-lookup-64k \
  --lookup-backend log-up \
  --trace-output benchmark-runs/jolt-nova/cpu-lookup-64k.trace.bin \
  --trace-block-size 1024 \
  --block-counts 1,4,16,49 \
  --measurement-runs 5 \
  --output benchmark-runs/jolt-nova/cpu-lookup-64k.logup.report.json
```

The manifest, single-run performance artifact, and aggregate baseline record
the selected lookup backend. Baseline comparison rejects results from different
backends. Their schemas advance to runner manifest v5, performance baseline v2,
and multi-run baseline v2.

### Security boundary

This is a real LogUp multiset equality over the executed block tuples, not a
renamed transcript fingerprint. The host block verifier recomputes the
challenges and both fractional sums, and Nova internally enforces the selected
sum equality and binds the resulting proof.

It is not yet a replacement for the complete Jolt instruction-lookup polynomial
commitment argument: the full virtual table commitment, opening proofs, and
sumcheck reduction still belong to Jolt's main lookup proof. Connecting those
commitments directly to each folded block remains later Stage 9 work and is
required before describing the experimental block pipeline as an independent
production lookup proof.

## Stage 9.5: verified full-Jolt lookup receipt bridge

Stage 9.5 connects the existing block pipeline to the complete original Jolt
lookup argument without placing the variable-size Dory verifier inside the Nova
step circuit.

The original Jolt lookup argument is distributed across the proof rather than
stored in one `lookup_proof` field. The bridge therefore commits to:

- the preprocessing and verifier PCS setup;
- public program I/O and the trusted-advice commitment;
- all polynomial commitments;
- the Stage 2 lookup claim-reduction sumcheck;
- the Stage 5 read/RAF sumcheck;
- the Stage 6b RA-virtualization sumcheck;
- the Stage 7 reduction closure;
- the joint PCS opening proof and the complete serialized Jolt proof.

`JoltVerifier::verify_with_lookup_receipt` returns an opaque
`VerifiedJoltLookupProofReceipt` only after the complete normal Jolt verifier
accepts. A receipt-aware block pipeline then:

1. copies the receipt digest and shape metadata into every
   `FoldableBlockState`;
2. derives a per-block digest that also binds the program digest, block
   boundaries, local lookup claims, and block LogUp proof;
3. includes those fields in the Nova statement digest and both lookup
   subclaim backends;
4. constrains receipt presence to be Boolean and requires every receipt field
   to be zero when the receipt is absent;
5. exposes strict verification entry points that require the original opaque
   receipt and reject a self-asserted digest.

The receipt-aware Nova path is covered through recursive folding and the final
folded proof envelope. Tampered receipts, substituted receipts, and altered
per-block bindings are rejected.

### Security boundary

This stage provides a verified-receipt bridge: the complete Jolt sumchecks and
Dory joint opening are verified outside Nova, and Nova cryptographically binds
the accepted result into every recursive step. It is stronger than merely
copying host-generated lookup data into the witness.

It does not make the final Spartan proof independently re-execute the Jolt/Dory
verifier. It also does not yet prove that each block's local lookup tuples are
openings of the same witness polynomials committed by that Jolt proof: the
per-block digest binds the two statements together, but a digest is not a
polynomial-opening equality argument. Achieving the stronger composition
requires either an in-circuit verifier for the relevant Jolt algebraic claims,
per-block opening/reduction claims tied to the original commitments, or a
proof-carrying recursive composition whose verifier is represented inside the
folding relation. That is the next cryptographic integration boundary.

## Remaining Stage 9 work

Stage 9.5 completes the externally verified full-Jolt lookup receipt bridge.
Later Stage 9 work will internalize or recursively compose the Jolt lookup
verifier, build controlled LogUp/Lasso comparisons, add per-relation profiling,
finish streaming trace-to-fold execution, and perform adversarial soundness
review.
