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

## Stage 9.6: authenticated block lookup openings

Stage 9.6 closes the polynomial-opening gap left by Stage 9.5 for instruction
lookup indices. It reuses openings that already belong to the original Jolt
proof instead of introducing a second commitment scheme.

For every committed `InstructionRa(i)` one-hot polynomial, Jolt already opens
the polynomial at the `HammingWeightClaimReduction` point. Those claims are
included in the joint Dory opening. After the complete Jolt verifier accepts,
`JoltVerifier::verify_with_lookup_opening_receipt` returns an opaque receipt
containing exactly those authenticated points and claims.

The block bridge then reconstructs every global opening as

```text
sum over blocks and cycles:
    eq(r_address, lookup_index_chunk_i)
  * eq(r_cycle, global_cycle)
+ deterministic Jolt NoOp padding
```

and requires exact field equality with the authenticated Jolt claim. Only a
successful reconstruction produces
`VerifiedJoltLookupBlockOpeningReceipt`. The receipt commits to:

- the complete Stage 9.5 verifier receipt;
- all authenticated `InstructionRa` opening points and claims;
- every block boundary and local lookup-claim digest;
- every block's contribution to every global opening.

The opening receipt, opening count, and per-block binding are added to the
foldable state, Nova statement transcript, and both lookup subclaim backends.
The Nova circuit constrains receipt presence to be Boolean, requires an opening
receipt to imply a base Jolt receipt, and zeroes all opening fields when absent.
Strict pipeline and final-proof verification APIs require the opaque receipt;
the older Stage 9.5 verifier intentionally rejects opening-enriched states.

Tests cover correct block reconstruction, corrupted opening claims, substituted
or altered per-block bindings, folding through the real Nova backend, and a
real Fibonacci Jolt/Dory proof that emits authenticated openings only after the
complete verifier succeeds.

### Security boundary

Stage 9.6 probabilistically binds each block's lookup-index chunks to the same
`InstructionRa` polynomials committed by the original Jolt proof. Soundness
inherits the Fiat-Shamir-selected multilinear opening point and Jolt's joint
Dory opening. It also handles Jolt's power-of-two trace padding explicitly.

The Nova Pallas circuit does not execute BN254/Dory verification internally.
It folds an opaque receipt produced by the host-side complete Jolt verifier.
This therefore assumes the composition entry point actually runs that verifier
and supplies its returned receipt.

This stage authenticates lookup indices, not yet each block's lookup operands
and output against all corresponding Jolt witness polynomials. It also does not
replace Jolt's Lasso argument with the experimental block LogUp relation. Those
are separate future relations and must be completed before the block pipeline
can stand alone without the original Jolt proof.

The current opening-receipt API is available for the non-ZK Jolt proof format,
where opening claims are retained in `JoltProof`. Supporting the `zk` proof
format requires deriving equivalent verified claims from the BlindFold path
without exposing hidden witness information.

## Stage 9.7: authenticated complete lookup tuples

Stage 9.7 extends the Stage 9.6 bridge from lookup indices to every value in
the block LogUp tuple:

```text
(left_lookup_operand, right_lookup_operand, lookup_index, lookup_output)
```

The lookup index remains authenticated by the committed
`InstructionRa(i)` Dory openings. Jolt represents the two operands and output
as virtual polynomials rather than separate committed polynomials. The complete
Jolt verifier nevertheless authenticates their claims at the shared
`InstructionClaimReduction` point: the instruction claim reduction, Read/RAF,
Spartan instruction-input virtualization, and R1CS relations constrain the
claims before the proof is reduced to the joint PCS opening.

After complete verification,
`verify_with_lookup_opening_receipt` now retains that shared point and the
three verified claims in receipt version 2. The block bridge reconstructs each
claim as

```text
sum over blocks and cycles:
    eq(r_cycle, global_cycle) * tuple_component(cycle)
+ deterministic Jolt NoOp padding
```

and compares all three values exactly in the Jolt field. The per-block
contribution digest now covers the `InstructionRa` contributions and the three
tuple-component contributions. Consequently the existing receipt digest,
per-block binding, Nova statement, lookup fingerprint, recursive SNARK, and
final Spartan envelope bind the complete tuple instead of only its index.

Tests cover valid complete-tuple reconstruction, independent corruption of an
authenticated index opening, independent corruption of an operand/output
claim, binding tampering, Nova folding, and extraction from a serialized real
Jolt/Dory proof.

### Security boundary

This stage closes the previously documented operand/output gap for the
host-verified non-ZK composition path. The claims are virtual-polynomial claims,
so their soundness comes from the complete chain of Jolt sumchecks and R1CS
relations rather than three new standalone polynomial commitments.

The Pallas Nova circuit still does not re-execute the BN254 Jolt/Dory verifier.
It proves a fixed-width binding to an opaque receipt that can only be emitted by
the host-side complete verifier. A self-contained recursive composition still
requires internalizing or recursively verifying that verifier relation.

As in Stage 9.6, the opening receipt is currently unavailable in Jolt's `zk`
proof format. Stage 9.7 also does not replace Jolt Lasso with block LogUp; it
only ensures that both statements refer to the same executed lookup tuples.

## Stage 9.8: authenticated register openings

Stage 9.8 extends the same verified-opening bridge to Jolt's register relation.
After the complete original Jolt verifier accepts, receipt version 3 retains
seven additional authenticated claims:

- `Rs1Value`, `Rs2Value`, and `RdWriteValue` at the shared
  `RegistersClaimReduction` point;
- `Rs1Ra`, `Rs2Ra`, and `RdWa` at the shared
  `RegistersReadWriteChecking` address-cycle point;
- the committed `RdInc` opening produced by `IncClaimReduction`.

For each block, the bridge reconstructs the three value claims as

```text
sum over block cycles:
    eq(r_cycle, global_cycle) * register_value(cycle)
```

and the address claims as

```text
sum over block cycles:
    eq(r_address, register_index) * eq(r_cycle, global_cycle)
```

It reconstructs `RdInc` from each cycle's post-write value minus pre-write
value, using the authenticated `IncClaimReduction` point. Deterministic Jolt
padding is included in each global reconstruction.

Only exact field equality for all seven claims produces the version 3 block
opening receipt. Its per-block contribution digest now covers lookup indices,
complete lookup tuples, register values, register addresses, and register
increments. The existing receipt fields are bound into both Nova's lookup
fingerprint and register fingerprint, so tampering changes the recursive
statement and is rejected by the Nova circuit and final proof verifier.

Tests include a nonzero two-cycle ADD trace, corrupted register claims, all
minimal and Nova block regressions, the ZK feature compile surface, and a real
Fibonacci Jolt/Dory proof.

### Security boundary

`RdInc` is a committed polynomial whose claim is closed by Jolt's joint Dory
opening. Register values and address selectors are virtual-polynomial claims:
their soundness comes from the complete Jolt register sumchecks, bytecode and
Spartan/R1CS reductions, and Twist read/write relations that the normal
verifier checks before emitting the opaque receipt.

Nova does not execute the BN254/Dory verifier. It folds a fixed-width binding
to the host-verified receipt, so the composition entry point must run the
complete Jolt verifier and use the returned opaque value. The public type,
method, and state-field names still contain `Lookup` for source compatibility,
but receipt version 3 authenticates both lookup and register execution data.

This bridge remains available only for the non-ZK Jolt proof format. A
privacy-preserving BlindFold/ZK extraction path and authenticated CPU/RAM
openings remain future work.

## Remaining Stage 9 work

After Stage 9.8, the main cryptographic gaps are:

- compose or internalize the relevant Jolt/Dory verifier rather than relying
  on an opaque host-verified receipt;
- connect the block CPU and RAM relations to their original Jolt polynomial
  openings with the same strength;
- derive a privacy-preserving equivalent receipt from the BlindFold/ZK proof
  path;
- build controlled Lasso/LogUp comparisons and perform adversarial soundness
  review;
- add per-relation profiling and finish streaming trace-to-fold execution.
