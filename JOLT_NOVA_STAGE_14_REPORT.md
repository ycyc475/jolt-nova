# Jolt-Nova Stage 14: Recursive Verifier Internalization

Stage 14 removes opaque host receipts from the new recursive-verifier trust
boundary. Receipt digests remain compatibility identifiers only.

## Implemented relation

The Stage-14 verifier path is:

1. Retain the complete typed Jolt proof, preprocessing, public I/O, and advice
   commitment in `RecursiveJoltVerifierObject`.
2. Run Jolt's exact BN254 Circom-compatible Poseidon transcript transition
   `Poseidon(state, n_rounds, data)` inside the Nova circuit.
3. Reconstruct each omitted sumcheck linear coefficient, enforce every round
   polynomial evaluation, and derive every challenge from the constrained
   transcript.
4. Enforce Jolt's verifier R1CS row-by-row. The same R1CS is used by BlindFold
   and contains Lasso lookup, register, RAM, CPU/Spartan, and joint-opening
   endpoint constraints.
5. Bind the transcript-verified full polynomial coefficient rows to the exact
   verifier-R1CS witness grid.
6. Prove this joined relation with Nova and compress it with Spartan.
7. Carry Dory as a typed deferred PCS obligation and execute its real BN254
   multi-pairing once during final acceptance.
8. Accept only if the Spartan verifier relation, deferred PCS pairing, object
   identifier, and PCS-obligation identifier all agree.

## Security boundary

`object_id`, `obligation_id`, and legacy receipt digests are commitments and
continuity identifiers. None of them is an acceptance certificate. There is no
`host_verified` or caller-provided `accepted` witness. The circuit derives its
acceptance output after satisfying all transcript, sumcheck, coefficient-grid,
and verifier-R1CS constraints; the outer verifier separately executes Dory.

## Validation coverage

- Native `light-poseidon` parity and complete sumcheck absorption parity.
- Forged Fiat-Shamir challenge rejection even when polynomial arithmetic is
  adjusted to remain valid.
- Sumcheck endpoint and shape rejection.
- Verifier-R1CS row mutation rejection.
- Cross-layer coefficient-grid mutation rejection.
- Real Nova recursive proof and Spartan compression/verification.
- Spartan public-output mutation rejection.
- Real Dory pairing acceptance and forged opening rejection.
- Final object-ID mismatch rejection.
- Structural/proof-size baseline through `RecursiveJoltVerifierBaseline`.

## Engine choice

The primary Nova engine is `Bn256EngineIPA` and the secondary engine is
`GrumpkinEngine`. This aligns the Nova circuit field with Jolt's BN254 scalar
field and avoids non-native arithmetic for verifier relations. IPA keeps the
recursive layer free of a KZG trusted setup.
