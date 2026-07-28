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

## Stage 9.9: authenticated RAM openings

Stage 9.9 extends the verified execution-opening receipt from lookup and
register data to the complete per-cycle RAM access tuple. Receipt version 4
retains:

- every committed `RamRa(i)` opening at
  `HammingWeightClaimReduction`;
- `RamAddress`, `RamReadValue`, and `RamWriteValue` at their shared
  `SpartanOuter` point;
- the committed `RamInc` opening produced by `IncClaimReduction`;
- the authenticated RAM domain size and lowest memory-layout address needed to
  reproduce Jolt's address remapping.

The block verifier remaps each nonzero byte address exactly as Jolt does:

```text
remapped_address = (address - lowest_address) / 8
```

It rejects addresses below the authenticated layout, unaligned addresses, and
addresses outside the authenticated RAM domain. For each `RamRa(i)` it then
reconstructs:

```text
sum over RAM-access cycles:
    eq(r_chunk, chunk_i(remapped_address))
  * eq(r_cycle, global_cycle)
```

NoOp cycles contribute zero because Jolt's RAM one-hot polynomial has no active
address on those cycles. At the shared Spartan point, each cycle contributes:

```text
Read:  (address, value, value)
Write: (address, pre_value, post_value)
NoOp:  (0, 0, 0)
```

Finally, the bridge reconstructs `RamInc` as `post_value - pre_value` on write
cycles and zero otherwise. Exact equality is required for all committed and
virtual claims before a version 4 block receipt can be produced.

The per-block contribution digest now covers instruction lookup, register, and
RAM openings. The same opaque receipt fields are included in Nova's lookup,
register, and RAM fingerprints, so the recursive RAM accumulator and final
Spartan envelope bind the authenticated original Jolt RAM witness.

Tests cover a nonzero `LD`/`SD` trace, corrupted `RamInc`, one-hot address
reconstruction, complete minimal and Nova block regressions, the ZK compile
surface, runner integration, and extraction from a real Fibonacci Jolt/Dory
proof.

### Security boundary

`RamRa(i)` and `RamInc` are committed polynomials closed by Jolt's joint Dory
opening. The address/read/write tuple consists of virtual-polynomial claims,
whose soundness comes from the complete Spartan/R1CS, RAM RAF, RAM read/write,
value-check, claim-reduction, and Twist relations run by the original verifier.

As in Stages 9.6–9.8, the Pallas Nova circuit does not re-execute the BN254/Dory
verifier. It folds a fixed-width binding to the opaque host-verified receipt.
The public API and state fields retain their historical lookup-oriented names
for compatibility, while receipt version 4 authenticates lookup, register, and
RAM execution data.

The non-ZK proof format remains required for opening extraction. A
privacy-preserving BlindFold/ZK path remains future work.

## Stage 9.10: authenticated CPU/R1CS openings

Stage 9.10 extends the verified execution-opening receipt to Jolt's CPU/R1CS
outer sumcheck. After the complete original Jolt verifier accepts, receipt
version 5 retains the shared Spartan outer opening point plus one authenticated
claim for every entry in `ALL_R1CS_INPUTS`.

The block verifier reconstructs each CPU claim from the local trace by
materializing `R1CSCycleInputs::from_cycle_with_next` for every cycle and for
the deterministic NoOp padding used after the final active cycle:

```text
sum over blocks and cycles:
    eq(r_cycle, global_cycle) * r1cs_input_value(cycle)
+ deterministic Jolt NoOp padding
```

The receipt and block contribution digests now include those 35 CPU claims, and
Nova's CPU fingerprint binds the opening receipt metadata alongside the CPU
row-count fields. Tests cover nonzero ADD/LD/SD traces, corrupted CPU claims,
lookup/register/RAM regressions, and Nova folding of the authenticated CPU
receipt.

### Security boundary

This stage authenticates the CPU/R1CS claims already checked by the complete
host-side Jolt verifier. It still folds an opaque receipt; the Nova circuit does
not re-execute BN254/Dory verification internally.

## Stage 9.11: full verifier transcript capsule

Stage 9.11 starts the verifier-internalization path by upgrading the base Jolt
receipt from a lookup-oriented digest into a full verifier transcript capsule.
`VerifiedJoltLookupProofReceipt` keeps its historical name for API
compatibility, but receipt version 2 now exposes fixed-size digests for every
clear verifier stage that the complete host-side Jolt verifier checks:

- Stage 1 UniSkip and Spartan outer sumcheck;
- Stage 2 UniSkip and lookup claim-reduction sumcheck;
- Stage 3 instruction/bytecode relation sumcheck;
- Stage 4 register/RAM relation sumcheck;
- Stage 5 read/RAF sumcheck;
- Stage 6a bytecode virtualization sumcheck;
- Stage 6b Spartan instruction-input virtualization sumcheck;
- Stage 7 reduction closure;
- the joint Dory opening proof;
- the full serialized Jolt proof.

The receipt digest domain advances to
`jolt-nova/verified-jolt-verifier-transcript-receipt/v2`. All later execution
opening receipts continue to bind the base receipt digest, so the block/Nova
pipeline now carries a fixed-width commitment to the whole accepted verifier
transcript rather than only the lookup-specific transcript slice.

The candidate capsule is still computed before verification and returned only
after the complete verifier accepts. This preserves the opaque-receipt safety
boundary while giving the next stage a stable input surface for a recursive
verifier gadget or proof-carrying verifier composition.

Tests cover the upgraded receipt version and the added per-stage verifier
digests, plus the existing authenticated lookup/register/RAM/CPU folding
regressions.

### Security boundary

This stage does not yet execute the BN254/Dory verifier inside Nova. It makes
the verifier relation more explicit and fixed-width, which is necessary for
internalization, but the soundness boundary is still: the host runs the complete
Jolt verifier, obtains the opaque capsule, and Nova folds a binding to that
accepted capsule.

## Stage 9.12: explicit verifier-stage relation binding

Stage 9.12 splits the Stage 9.11 capsule into an explicit verifier-stage
relation digest that the block/Nova pipeline can bind independently from the
opaque receipt digest. The base receipt now exposes:

- `verifier_stage_relation_count`, currently 11;
- `verifier_stage_relation_digest`, a domain-separated digest over the verifier
  preamble, commitments, Stage 1/2 UniSkip proofs, Stage 1–7 sumchecks, and the
  joint Dory opening proof.

Every receipt-aware foldable block state records that relation digest and
count. The strict receipt binder populates them from the opaque receipt, the
strict verifier rejects mismatches, and the per-block receipt binding digest
now absorbs the verifier-stage relation plus the individual verifier-stage
digests.

Nova's private step statement and witness now carry the verifier-stage relation
fields explicitly. The step circuit:

- includes them in the in-circuit statement transcript;
- includes them in the lookup/verifier fingerprint path, including the LogUp
  selected backend path;
- requires them to be zero when the verified Jolt receipt is absent.

This gives later work a stable replacement seam: individual verifier stages can
be recursively verified or internalized one at a time while preserving the
fixed-width block-folding interface.

Tests cover receipt relation construction, per-block population, tamper
rejection, existing authenticated execution-opening regressions, and Nova
folding through the explicit verifier-stage relation fields.

### Security boundary

This stage still does not prove the BN254/Dory verifier inside the Pallas Nova
step circuit. It separates and binds the verifier-stage relation as explicit
Nova-visible metadata, but the relation is still trusted only because the
complete host-side Jolt verifier emitted the opaque receipt.

## Stage 9.13: minimal verifier-stage internalization prototype

Stage 9.13 introduces the first explicit verifier-stage object instead of
carrying only one opaque receipt digest. An accepted receipt now exposes eleven
fixed-width transcript components: the verifier preamble, Stages 1--7, the
joint opening, and the complete proof binding.

`VerifierStageInternalizationClaim` selects one component and binds its index,
stage digest, parent verifier-relation digest, and relation width into a
domain-separated claim digest. `verify_against` reconstructs that claim from
the accepted receipt and rejects a changed stage, relation, count, or digest.

The same component list feeds an append-only
`RecursiveVerifierTranscriptCapsule`. Its transition requires stages to be
absorbed in order and hashes the previous root, stage index, and stage digest.
This is the fixed-width transition primitive needed by Stage 9.14.

### Security boundary

The prototype internalizes the *shape and consistency* of one verifier-stage
claim. It does not yet execute the BN254 sumcheck or Dory equations inside the
Pallas Nova circuit. The complete Jolt verifier must still accept before these
objects are created.

## Stage 9.14: recursive verifier transcript capsule binding

Stage 9.14 carries the complete `RecursiveVerifierTranscriptCapsule` output
through the block and Nova folding interfaces. Each receipt-aware
`FoldableBlockState` now stores the capsule transcript root and absorbed stage
count derived from the accepted Jolt receipt. The strict receipt binder
populates those fields, the strict verifier rejects mismatches, and the
per-block receipt binding digest absorbs them with the local lookup statement.

Nova's block statement and step witness now carry the capsule root and stage
count explicitly. The step circuit:

- includes both values in the in-circuit statement transcript;
- includes both values in the lookup verifier fingerprint, including the LogUp
  backend-selected path;
- requires both values to be zero when no verified Jolt receipt is present.

This makes the verifier transcript capsule a recursive, fixed-width object in
the block-folding relation. Later stages can replace the current digest-only
capsule transition with in-circuit verifier gadgets without changing the
external block/Nova interface again.

### Security boundary

This stage still binds a digest-level recursive verifier object. It does not
yet execute the BN254 sumcheck, Dory opening, or Spartan verifier equations
inside the Pallas Nova circuit. The complete host-side Jolt verifier remains
the source of acceptance for the capsule.

## Stage 9.15: BlindFold/ZK receipt capsule

Stage 9.15 adds the privacy-preserving receipt surface for Jolt's ZK proof
path. `VerifiedJoltLookupProofReceipt` now records whether the accepted proof
was verified in ZK mode and binds a fixed-width digest of the underlying
`BlindFoldProof`. The new `VerifiedJoltBlindFoldReceipt` is a compact
counterpart to the non-ZK receipt: it commits to the accepted lookup receipt,
the verifier-stage relation, the recursive transcript root, and the BlindFold
proof digest, without exposing any hidden opening claims.

The block/Nova pipeline now carries the ZK-mode bit and BlindFold-receipt
digest explicitly, so recursive folding can distinguish the privacy-preserving
path from the clear opening path while still binding the accepted host-side
verifier output.

### Security boundary

This stage still does not reconstruct the opening claims in ZK mode. It
introduces a fixed-width, privacy-preserving receipt capsule that can be bound
recursively, but the non-ZK opening extraction path remains separate.

## Stage 9.16: Lasso/LogUp comparison and audit

Stage 9.16 turns the existing lookup-backend switch into a controlled
comparison harness. The benchmark/example code can now compare the original
Lasso-style transcript path against the LogUp backend on the same workload,
recording the relative timing, throughput, and memory deltas in a dedicated
comparison artifact. The comparison helper also rejects mismatched workloads,
measurement shapes, and identical backends, so it is safe to use as an
apples-to-apples experiment surface.

Security-wise, the stage does not change the lookup proof itself. Instead it
packages the already-existing LogUp soundness checks and the transcript
baseline into a repeatable audit flow, so later profiling and streaming work
can reuse the same comparison input.

### Security boundary

This stage compares two already-verified backends and records their
performance/audit metadata. It does not yet alter the cryptographic lookup
argument or internalize more of the Jolt verifier.

## Stage 9.17: per-relation profiling

Stage 9.17 adds a stable per-relation profiling surface to the benchmark runner.
Every single-run performance artifact now records a `relation_profiles` array
under schema `jolt-nova-relation-profile-v1`, and the multi-run aggregate
summarizes those profiles across samples. The tracked relations are:

- `cpu-r1cs`;
- `register`;
- `ram`;
- `lookup`;
- `nova-fold`;
- `spartan-final-report`.

The profile currently combines real trace-derived event counts with a stable
static attribution model:

- RAM event counts come from each trace cycle's `ram_access()`;
- register event counts come from observed `rs1`, `rs2`, and `rd` accesses;
- lookup event counts and distinct instruction counts come from the processed
  trace prefixes;
- CPU, Nova fold, and Spartan final-report costs are attributed from active
  cycles, processed prefix blocks, and report rows.

The performance baseline schema advances to
`jolt-nova-performance-baseline-v3`, the aggregate schema advances to
`jolt-nova-multi-run-baseline-v3`, and the runner manifest advances to
`jolt-nova-benchmark-runner-v6`. The comparison helpers remain compatible with
the Stage 9.16 v2 aggregate schema, but when both inputs contain relation
profiles they also compare `relation_profile.<name>.estimated_ms`.

### Security boundary

This stage does not change the cryptographic proof relation. The per-relation
times are attribution estimates over the already-measured prove/report duration;
they are meant to make CPU/RAM/lookup/fold hotspots visible and comparable.
Later instrumentation can replace the static attribution model with real
internal prover spans without changing the consumer-facing artifact shape.

## Stage 9.18: streaming trace-to-fold surface

Stage 9.18 adds a streaming prefix-fold surface for the final-proof-size
benchmark path. The block pipeline now accepts `IntoIterator<Item = TraceBlock>`
inputs for the prefix scaling helpers, so callers can feed blocks as they are
decoded instead of first materializing the entire trace window. The streaming
path keeps only the requested prefix plus a one-block lookahead in memory, and
it preserves the same final-proof-size rows and artifact schema as the
existing batch API.

The new surface is intended as the first practical trace-to-fold bridge for the
benchmark/reporting pipeline:

- it validates the requested prefix counts before folding starts;
- it emits an error if the source ends before a requested prefix is available;
- it still reuses the existing Nova folding and Spartan final-proof assembly
  logic for the actual proof work;
- it keeps the batch helpers intact so the existing non-streaming callers stay
  compatible.

### Security boundary

This stage changes the dataflow shape, not the cryptographic relation. It is a
memory-leaner trace ingestion path for final-proof-size reporting, but it does
not yet replace the remaining verifier internalization work or the later
profiling refinements.

## Remaining Stage 9 work

After Stage 9.18, the main cryptographic gaps are:

- replace digest-only recursive verifier capsule transitions with in-circuit
  verifier gadgets;
- wire real internal profiling spans into the Stage 9.17 relation profile
  artifact.

## Stage 10.1: explicit Nova step relation boundary

Stage 10 starts turning the block-to-Nova handoff from host-side bookkeeping
into an auditable recursive relation surface. Stage 10.1 fixes the public
input/output shape for one Nova step and binds the first set of continuity
fields inside the Nova step circuit.

The recursive `z` state now has 11 scalar words:

- semantic accumulator;
- next block index;
- total active cycles;
- register, RAM, lookup, and CPU claim accumulators;
- program digest;
- next global cycle;
- machine-state digest;
- register-state digest.

`JoltNovaStepRelationBoundary` exposes this as named storage-level input and
output states. The Nova backend builds this boundary before proving each step,
validates that the previous output matches the next witness, and then proves
the same transition in-circuit. The circuit now enforces program continuity,
block-index continuity, global-cycle continuity, machine-state continuity,
register-state continuity, and the relation
`global_cycle_end = global_cycle_start + active_cycles`.

The final compressed Spartan path also derives its initial Nova input from the
fold metadata instead of using an all-zero state, so the final proof is tied to
the same first-block program/state/cycle boundary as the recursive fold.

### Security boundary

Stage 10.1 still folds authenticated Jolt receipts and digest/fingerprint
claims. It does not yet execute the full CPU, RAM, register, or lookup verifier
equations inside Nova. Those remain the next Stage 10 milestones.

## Stage 10.2: explicit CPU/R1CS relation boundary

Stage 10.2 turns the CPU-side proof metadata into its own auditable recursive
boundary. The new `JoltCpuR1csRelationBoundary` exposes a storage-level
snapshot of the CPU/R1CS claim, including the optional lookahead-cycle digest
that binds the proof to the exact next-cycle commitment when one is present.

This stage wires the same CPU fields into three places:

- the fold statement digest;
- the CPU claim fingerprint used by Nova folding;
- the step-circuit witness and its boolean/zero-digest consistency checks.

That makes the CPU/R1CS claim an explicit recursive object instead of a host-
only side condition.

### Security boundary

Stage 10.2 still relies on host-side validation for the lookahead-cycle shape
and does not yet internalize the full CPU verifier gadget. What it does add is
a stable, inspectable CPU/R1CS boundary that later verifier internalization
work can reuse without changing the recursive state layout again.
