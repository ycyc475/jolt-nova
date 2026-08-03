use super::*;

use std::{
    collections::HashMap,
    error::Error,
    fmt,
    mem::size_of,
    time::{Duration, Instant},
};

pub const JOLT_NOVA_STAGE18_VERSION: &str = "jolt-nova-stage18-v1";
const STAGE18_MAX_RESIDENT_TRACE_BLOCKS: usize = 2;
const STAGE18_SECURITY_BITS: usize = 128;

const CPU_RELATION: &str = "cpu-r1cs";
const IO_RELATION: &str = "trace-io-claims";
const REGISTER_RELATION: &str = "register-read-write";
const RAM_RELATION: &str = "ram-read-write";
const LOOKUP_RELATION: &str = "jolt-lasso-lookup-claim";
const RECEIPT_RELATION: &str = "verified-jolt-receipt-binding";
const NOVA_RELATION: &str = "nova-block-fold";
const SPARTAN_RELATION: &str = "spartan-final-compression";

/// Fixed production parameters for the Stage-18 streaming path.
///
/// Keeping the complete Nova configuration and recursive-verifier shape in a
/// single digest prevents a proof produced under an experimental LogUp or
/// placeholder backend from being accepted as a production Lasso proof.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage18ReleaseParameters {
    block_target_size: usize,
    max_resident_trace_blocks: usize,
    security_bits: usize,
    nova_config: NovaFoldConfig,
    recursive_verifier_shape_id: [u8; 32],
}

impl Stage18ReleaseParameters {
    pub fn production(
        block_target_size: usize,
        recursive_verifier_shape_id: [u8; 32],
    ) -> Result<Self, Stage18Error> {
        let mut nova_config = NovaFoldConfig::default();
        nova_config.subclaim_backend_name = NOVA_JOLT_LASSO_SUBCLAIM_BACKEND_NAME;
        nova_config.final_proof_backend_name = SPARTAN_FINAL_PROOF_SYSTEM_NAME;
        nova_config.use_zero_knowledge = true;
        let parameters = Self {
            block_target_size,
            max_resident_trace_blocks: STAGE18_MAX_RESIDENT_TRACE_BLOCKS,
            security_bits: STAGE18_SECURITY_BITS,
            nova_config,
            recursive_verifier_shape_id,
        };
        parameters.validate()?;
        Ok(parameters)
    }

    pub fn validate(&self) -> Result<(), Stage18Error> {
        if self.block_target_size == 0 || !self.block_target_size.is_power_of_two() {
            return Err(Stage18Error::InvalidReleaseParameters(
                "block target size must be a non-zero power of two",
            ));
        }
        if self.max_resident_trace_blocks != STAGE18_MAX_RESIDENT_TRACE_BLOCKS {
            return Err(Stage18Error::InvalidReleaseParameters(
                "production streaming must retain exactly two trace blocks at most",
            ));
        }
        if self.security_bits < STAGE18_SECURITY_BITS {
            return Err(Stage18Error::InvalidReleaseParameters(
                "production security level must be at least 128 bits",
            ));
        }
        if self.nova_config.backend_name != "nova-recursive-snark"
            || self.nova_config.relation_name != NOVA_BLOCK_FOLD_RELATION_NAME
            || self.nova_config.subclaim_backend_name != NOVA_JOLT_LASSO_SUBCLAIM_BACKEND_NAME
            || self.nova_config.final_proof_backend_name != SPARTAN_FINAL_PROOF_SYSTEM_NAME
            || !self.nova_config.use_zero_knowledge
        {
            return Err(Stage18Error::InvalidReleaseParameters(
                "production parameters require Nova, native Jolt Lasso, Spartan, and ZK",
            ));
        }
        if self.recursive_verifier_shape_id == [0; 32] {
            return Err(Stage18Error::InvalidReleaseParameters(
                "recursive verifier shape identifier must be non-zero",
            ));
        }
        Ok(())
    }

    pub fn block_target_size(&self) -> usize {
        self.block_target_size
    }

    pub fn max_resident_trace_blocks(&self) -> usize {
        self.max_resident_trace_blocks
    }

    pub fn security_bits(&self) -> usize {
        self.security_bits
    }

    pub fn nova_config(&self) -> &NovaFoldConfig {
        &self.nova_config
    }

    pub fn recursive_verifier_shape_id(&self) -> [u8; 32] {
        self.recursive_verifier_shape_id
    }

    pub fn digest(&self) -> [u8; 32] {
        let mut hasher = Sha3_256::new();
        hasher.update(b"JOLT_NOVA_STAGE18_RELEASE_PARAMETERS_V1");
        hasher.update(JOLT_NOVA_STAGE18_VERSION.as_bytes());
        hasher.update((self.block_target_size as u64).to_le_bytes());
        hasher.update((self.max_resident_trace_blocks as u64).to_le_bytes());
        hasher.update((self.security_bits as u64).to_le_bytes());
        hasher.update(self.nova_config.backend_name.as_bytes());
        hasher.update(self.nova_config.relation_name.as_bytes());
        hasher.update(self.nova_config.subclaim_backend_name.as_bytes());
        hasher.update(self.nova_config.final_proof_backend_name.as_bytes());
        hasher.update([u8::from(self.nova_config.use_zero_knowledge)]);
        hasher.update(self.recursive_verifier_shape_id);
        finalize_stage18_digest(hasher)
    }

    pub fn to_json_pretty(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(&serde_json::json!({
            "version": JOLT_NOVA_STAGE18_VERSION,
            "block_target_size": self.block_target_size,
            "max_resident_trace_blocks": self.max_resident_trace_blocks,
            "security_bits": self.security_bits,
            "nova_backend": self.nova_config.backend_name,
            "nova_relation": self.nova_config.relation_name,
            "lookup_backend": self.nova_config.subclaim_backend_name,
            "final_proof_backend": self.nova_config.final_proof_backend_name,
            "zero_knowledge": self.nova_config.use_zero_knowledge,
            "recursive_verifier_shape_id": stage18_hex(self.recursive_verifier_shape_id),
            "release_parameters_digest": stage18_hex(self.digest()),
        }))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage18RelationProfile {
    pub relation: &'static str,
    pub calls: usize,
    pub total_nanos: u128,
    pub max_nanos: u128,
}

impl Stage18RelationProfile {
    fn new(relation: &'static str) -> Self {
        Self {
            relation,
            calls: 0,
            total_nanos: 0,
            max_nanos: 0,
        }
    }

    fn record(&mut self, elapsed: Duration) {
        let nanos = elapsed.as_nanos();
        self.calls += 1;
        self.total_nanos = self.total_nanos.saturating_add(nanos);
        self.max_nanos = self.max_nanos.max(nanos);
    }
}

/// One block's profiling row. The streaming prover sends this value to a
/// caller-provided sink and then drops it, so profiling does not require a
/// `Vec` proportional to the number of blocks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage18BlockProfile {
    pub block_index: usize,
    pub active_cycles: usize,
    pub resident_trace_blocks: usize,
    pub resident_trace_cycles: usize,
    pub tracked_ram_addresses: usize,
    pub sampled_physical_memory_bytes: Option<usize>,
    pub relation_profiles: Vec<Stage18RelationProfile>,
}

impl Stage18BlockProfile {
    fn new(current: &TraceBlock, next: Option<&TraceBlock>) -> Self {
        Self {
            block_index: current.block_index,
            active_cycles: current.active_cycles,
            resident_trace_blocks: 1 + usize::from(next.is_some()),
            resident_trace_cycles: current.cycles.len()
                + next.map_or(0, |block| block.cycles.len()),
            tracked_ram_addresses: 0,
            sampled_physical_memory_bytes: None,
            relation_profiles: Vec::with_capacity(7),
        }
    }

    fn record(&mut self, relation: &'static str, elapsed: Duration) {
        let nanos = elapsed.as_nanos();
        self.relation_profiles.push(Stage18RelationProfile {
            relation,
            calls: 1,
            total_nanos: nanos,
            max_nanos: nanos,
        });
    }

    pub fn to_json_line(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(&self.as_json_value())
    }

    fn as_json_value(&self) -> serde_json::Value {
        serde_json::json!({
            "block_index": self.block_index,
            "active_cycles": self.active_cycles,
            "resident_trace_blocks": self.resident_trace_blocks,
            "resident_trace_cycles": self.resident_trace_cycles,
            "tracked_ram_addresses": self.tracked_ram_addresses,
            "sampled_physical_memory_bytes": self.sampled_physical_memory_bytes,
            "relations": relation_profiles_json(&self.relation_profiles),
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage18StreamingMetrics {
    pub block_count: usize,
    pub total_active_cycles: usize,
    pub max_resident_trace_blocks: usize,
    pub max_resident_trace_cycles: usize,
    pub peak_tracked_ram_addresses: usize,
    pub start_physical_memory_bytes: Option<usize>,
    pub peak_physical_memory_bytes: Option<usize>,
    pub end_physical_memory_bytes: Option<usize>,
    pub estimated_peak_trace_bytes: usize,
    pub final_folded_proof_bytes: usize,
    pub relation_profiles: Vec<Stage18RelationProfile>,
}

impl Stage18StreamingMetrics {
    fn new() -> Self {
        let start = sample_physical_memory();
        Self {
            block_count: 0,
            total_active_cycles: 0,
            max_resident_trace_blocks: 0,
            max_resident_trace_cycles: 0,
            peak_tracked_ram_addresses: 0,
            start_physical_memory_bytes: start,
            peak_physical_memory_bytes: start,
            end_physical_memory_bytes: None,
            estimated_peak_trace_bytes: 0,
            final_folded_proof_bytes: 0,
            relation_profiles: [
                CPU_RELATION,
                IO_RELATION,
                REGISTER_RELATION,
                RAM_RELATION,
                LOOKUP_RELATION,
                RECEIPT_RELATION,
                NOVA_RELATION,
                SPARTAN_RELATION,
            ]
            .into_iter()
            .map(Stage18RelationProfile::new)
            .collect(),
        }
    }

    fn profile_mut(&mut self, relation: &'static str) -> &mut Stage18RelationProfile {
        self.relation_profiles
            .iter_mut()
            .find(|profile| profile.relation == relation)
            .expect("Stage-18 profile relation is fixed by the release manifest")
    }

    fn observe_memory(&mut self) {
        if let Some(memory) = sample_physical_memory() {
            self.peak_physical_memory_bytes = Some(
                self.peak_physical_memory_bytes
                    .map_or(memory, |peak| peak.max(memory)),
            );
        }
    }

    fn observe_resident_blocks(&mut self, current: &TraceBlock, next: Option<&TraceBlock>) {
        let resident_blocks = 1 + usize::from(next.is_some());
        let resident_cycles = current.cycles.len() + next.map_or(0, |block| block.cycles.len());
        self.max_resident_trace_blocks = self.max_resident_trace_blocks.max(resident_blocks);
        self.max_resident_trace_cycles = self.max_resident_trace_cycles.max(resident_cycles);
        let trace_bytes = resident_blocks
            .saturating_mul(size_of::<TraceBlock>())
            .saturating_add(resident_cycles.saturating_mul(size_of::<Cycle>()));
        self.estimated_peak_trace_bytes = self.estimated_peak_trace_bytes.max(trace_bytes);
        self.observe_memory();
    }

    pub fn physical_memory_delta_bytes(&self) -> Option<isize> {
        Some(self.peak_physical_memory_bytes? as isize - self.start_physical_memory_bytes? as isize)
    }

    pub fn to_json_pretty(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(&self.as_json_value())
    }

    fn as_json_value(&self) -> serde_json::Value {
        serde_json::json!({
            "block_count": self.block_count,
            "total_active_cycles": self.total_active_cycles,
            "max_resident_trace_blocks": self.max_resident_trace_blocks,
            "max_resident_trace_cycles": self.max_resident_trace_cycles,
            "peak_tracked_ram_addresses": self.peak_tracked_ram_addresses,
            "start_physical_memory_bytes": self.start_physical_memory_bytes,
            "peak_physical_memory_bytes": self.peak_physical_memory_bytes,
            "end_physical_memory_bytes": self.end_physical_memory_bytes,
            "physical_memory_delta_bytes": self.physical_memory_delta_bytes(),
            "estimated_peak_trace_bytes": self.estimated_peak_trace_bytes,
            "final_folded_proof_bytes": self.final_folded_proof_bytes,
            "relations": relation_profiles_json(&self.relation_profiles),
        })
    }
}

/// Verifier-facing output of the bounded-memory Stage-18 execution pipeline.
///
/// This type deliberately omits per-block bundles and fold inputs. They are
/// checked and discarded immediately after each Nova step; only the recursive
/// accumulator and its real Spartan compression remain resident.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage18StreamingNovaProof<Digest = [u8; 32]> {
    accumulator: NovaFoldAccumulator<Digest>,
    final_proof: FinalFoldedProof<Digest>,
    metrics: Stage18StreamingMetrics,
    release_parameters_digest: [u8; 32],
    verified_jolt_receipt_digest: [u8; 32],
    jolt_statement_id: [u8; 32],
    linkage_digest: [u8; 32],
}

impl<Digest> Stage18StreamingNovaProof<Digest>
where
    Digest: Clone + PartialEq + AsRef<[u8]>,
{
    pub fn accumulator(&self) -> &NovaFoldAccumulator<Digest> {
        &self.accumulator
    }

    pub fn final_proof(&self) -> &FinalFoldedProof<Digest> {
        &self.final_proof
    }

    pub fn metrics(&self) -> &Stage18StreamingMetrics {
        &self.metrics
    }

    pub fn jolt_statement_id(&self) -> [u8; 32] {
        self.jolt_statement_id
    }

    pub fn verified_jolt_receipt_digest(&self) -> [u8; 32] {
        self.verified_jolt_receipt_digest
    }

    pub fn linkage_digest(&self) -> [u8; 32] {
        self.linkage_digest
    }

    pub fn report_json_pretty(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(&serde_json::json!({
            "version": JOLT_NOVA_STAGE18_VERSION,
            "release_parameters_digest": stage18_hex(self.release_parameters_digest),
            "verified_jolt_receipt_digest": stage18_hex(self.verified_jolt_receipt_digest),
            "jolt_statement_id": stage18_hex(self.jolt_statement_id),
            "streaming_linkage_digest": stage18_hex(self.linkage_digest),
            "folded_instance_digest": stage18_hex(self.final_proof.instance.instance_digest),
            "folded_proof_digest": stage18_hex(self.final_proof.proof_digest),
            "metrics": self.metrics.as_json_value(),
        }))
    }

    pub fn verify(
        &self,
        parameters: &Stage18ReleaseParameters,
        receipt: &VerifiedJoltLookupProofReceipt,
    ) -> Result<(), Stage18Error> {
        parameters.validate()?;
        if self.release_parameters_digest != parameters.digest() {
            return Err(Stage18Error::LinkageMismatch(
                "release parameter digest mismatch",
            ));
        }
        if !receipt.zk_mode() {
            return Err(Stage18Error::ExpectedZkReceipt);
        }
        if self.verified_jolt_receipt_digest != receipt.digest()
            || self.jolt_statement_id != receipt.recursive_execution_statement_id()
        {
            return Err(Stage18Error::LinkageMismatch(
                "verified Jolt receipt or statement mismatch",
            ));
        }
        if self.accumulator.config != *parameters.nova_config() {
            return Err(Stage18Error::LinkageMismatch("Nova configuration mismatch"));
        }
        if self.accumulator.metadata.absorbed_blocks != self.metrics.block_count
            || self.accumulator.metadata.total_active_cycles != self.metrics.total_active_cycles
        {
            return Err(Stage18Error::LinkageMismatch(
                "streaming metrics do not match folded metadata",
            ));
        }
        if self.metrics.block_count == 0
            || self.metrics.max_resident_trace_blocks > parameters.max_resident_trace_blocks()
        {
            return Err(Stage18Error::StreamingBoundExceeded {
                observed: self.metrics.max_resident_trace_blocks,
                allowed: parameters.max_resident_trace_blocks(),
            });
        }
        verify_configured_final_folded_proof(&self.accumulator, &self.final_proof)?;
        if self.linkage_digest
            != digest_streaming_proof_linkage(
                &self.accumulator,
                &self.final_proof,
                &self.metrics,
                self.release_parameters_digest,
                self.verified_jolt_receipt_digest,
                self.jolt_statement_id,
            )
        {
            return Err(Stage18Error::LinkageMismatch(
                "streaming final-proof linkage mismatch",
            ));
        }
        Ok(())
    }
}

/// Production Stage-18 acceptance envelope for one execution statement.
///
/// `streaming_execution` proves the block chain and final folded state. The
/// complete ZK acceptance proves the original Jolt/BlindFold verifier relation
/// and executes both typed commitment-group obligations. Both halves are tied
/// to the same mode-independent Jolt execution statement.
#[cfg(feature = "zk")]
pub struct Stage18ZkEndToEndProof<C, PCS, Digest = [u8; 32]>
where
    C: crate::curve::JoltCurve<F = ark_bn254::Fr>,
    PCS: crate::poly::commitment::commitment_scheme::CommitmentScheme<Field = ark_bn254::Fr>,
{
    streaming_execution: Stage18StreamingNovaProof<Digest>,
    recursive_statement: RecursiveBlindFoldStatement,
    recursive_acceptance:
        RecursiveJoltZkCompleteFinalAcceptance<C, PCS, crate::transcripts::PoseidonTranscript>,
    deferred_pcs_id: [u8; 32],
    linkage_digest: [u8; 32],
}

#[cfg(feature = "zk")]
impl<C, PCS, Digest> Stage18ZkEndToEndProof<C, PCS, Digest>
where
    C: crate::curve::JoltCurve<F = ark_bn254::Fr>,
    PCS: crate::poly::commitment::commitment_scheme::CommitmentScheme<Field = ark_bn254::Fr>,
    Digest: Clone + PartialEq + AsRef<[u8]>,
{
    /// Builds the final object only from the lossless typed artifacts exported
    /// by one successful production Jolt verifier invocation.
    pub fn prove_from_verified_artifacts(
        streaming_execution: Stage18StreamingNovaProof<Digest>,
        artifacts: RecursiveJoltZkVerifiedArtifacts<C, PCS>,
        parameters: &Stage18ReleaseParameters,
    ) -> Result<
        (
            Self,
            RecursiveBlindFoldVerifierVerificationKey,
            RecursiveBlindFoldBaseline,
        ),
        Stage18Error,
    > {
        let receipt = artifacts.verified_jolt_receipt().clone();
        streaming_execution.verify(parameters, &receipt)?;
        if streaming_execution.verified_jolt_receipt_digest() != receipt.digest() {
            return Err(Stage18Error::LinkageMismatch(
                "streaming proof was not produced from this verifier invocation",
            ));
        }
        let recursive_statement = artifacts.statement();
        if recursive_statement.jolt_statement_id != streaming_execution.jolt_statement_id() {
            return Err(Stage18Error::LinkageMismatch(
                "block folding and recursive verifier use different Jolt statements",
            ));
        }
        if recursive_statement.shape_id != parameters.recursive_verifier_shape_id() {
            return Err(Stage18Error::LinkageMismatch(
                "recursive verifier shape is not the release-pinned shape",
            ));
        }

        let deferred_pcs_id = artifacts.deferred_pcs_id();
        let (blindfold_artifact, deferred_pcs_opening) = artifacts.into_parts();
        let (circuit, group_obligation) = blindfold_artifact.into_parts();
        if circuit.statement() != recursive_statement {
            return Err(Stage18Error::LinkageMismatch(
                "recursive circuit statement changed during artifact decomposition",
            ));
        }
        let (prover_parameters, verification_key) = circuit
            .setup_pinned()
            .map_err(Stage18Error::RecursiveVerification)?;
        let recursive_proof = prover_parameters
            .prove(&circuit)
            .map_err(Stage18Error::RecursiveVerification)?;
        let baseline = circuit.baseline(&recursive_proof);
        let recursive_acceptance = RecursiveJoltZkCompleteFinalAcceptance {
            blindfold: RecursiveJoltZkRecursiveFinalAcceptance {
                recursive_proof,
                group_obligation,
            },
            deferred_pcs_opening,
        };
        let linkage_digest = digest_stage18_zk_end_to_end_linkage(
            &streaming_execution,
            &recursive_statement,
            &recursive_acceptance,
            deferred_pcs_id,
            parameters.digest(),
        );
        let proof = Self {
            streaming_execution,
            recursive_statement,
            recursive_acceptance,
            deferred_pcs_id,
            linkage_digest,
        };
        proof.verify(parameters, &receipt, &verification_key)?;
        Ok((proof, verification_key, baseline))
    }

    pub fn verify(
        &self,
        parameters: &Stage18ReleaseParameters,
        receipt: &VerifiedJoltLookupProofReceipt,
        verification_key: &RecursiveBlindFoldVerifierVerificationKey,
    ) -> Result<(), Stage18Error> {
        self.streaming_execution.verify(parameters, receipt)?;
        if self.recursive_statement.jolt_statement_id
            != self.streaming_execution.jolt_statement_id()
            || self.recursive_statement.jolt_statement_id
                != receipt.recursive_execution_statement_id()
        {
            return Err(Stage18Error::LinkageMismatch(
                "recursive verifier statement does not match folded execution",
            ));
        }
        if self.recursive_statement.shape_id != parameters.recursive_verifier_shape_id() {
            return Err(Stage18Error::LinkageMismatch(
                "recursive verifier shape/setup substitution",
            ));
        }
        if self.linkage_digest
            != digest_stage18_zk_end_to_end_linkage(
                &self.streaming_execution,
                &self.recursive_statement,
                &self.recursive_acceptance,
                self.deferred_pcs_id,
                parameters.digest(),
            )
        {
            return Err(Stage18Error::LinkageMismatch(
                "complete ZK end-to-end linkage digest mismatch",
            ));
        }
        self.recursive_acceptance
            .verify(
                verification_key,
                &self.recursive_statement,
                self.recursive_statement.object_id,
                self.streaming_execution.jolt_statement_id(),
                self.deferred_pcs_id,
            )
            .map_err(Stage18Error::RecursiveVerification)
    }

    pub fn streaming_execution(&self) -> &Stage18StreamingNovaProof<Digest> {
        &self.streaming_execution
    }

    pub fn recursive_statement(&self) -> &RecursiveBlindFoldStatement {
        &self.recursive_statement
    }

    pub fn deferred_pcs_id(&self) -> [u8; 32] {
        self.deferred_pcs_id
    }

    pub fn linkage_digest(&self) -> [u8; 32] {
        self.linkage_digest
    }

    #[cfg(test)]
    pub(crate) fn toggle_recursive_transcript_for_test(&mut self) {
        self.recursive_statement.final_transcript_round ^= 1;
    }

    #[cfg(test)]
    pub(crate) fn toggle_deferred_pcs_id_for_test(&mut self) {
        self.deferred_pcs_id[0] ^= 1;
    }

    #[cfg(test)]
    pub(crate) fn toggle_recursive_proof_for_test(&mut self) {
        if let Some(byte) = self
            .recursive_acceptance
            .blindfold
            .recursive_proof
            .proof_bytes
            .first_mut()
        {
            *byte ^= 1;
        }
    }

    #[cfg(test)]
    pub(crate) fn toggle_linkage_digest_for_test(&mut self) {
        self.linkage_digest[0] ^= 1;
    }
}

impl<Digest, F> BlockProofPipeline<Digest, F, NovaFoldingBackend>
where
    Digest: Clone + PartialEq + AsRef<[u8]>,
    F: JoltField,
{
    /// Consumes a lazy trace with one-block lookahead, proves and folds each
    /// block immediately, and discards all block-local witness material before
    /// requesting the next block.
    pub fn prove_streaming_blocks_with_final_proof<I>(
        &self,
        bytecode_preprocessing: &BytecodePreprocessing,
        blocks: I,
        parameters: &Stage18ReleaseParameters,
    ) -> Result<Stage18StreamingNovaProof<Digest>, Stage18Error>
    where
        I: IntoIterator<Item = TraceBlock>,
    {
        self.prove_streaming_blocks_with_final_proof_and_profile(
            bytecode_preprocessing,
            blocks,
            parameters,
            |_| {},
        )
    }

    /// Profiling variant of [`Self::prove_streaming_blocks_with_final_proof`].
    /// Each row is emitted after its block has been folded and before the next
    /// iteration, allowing callers to serialize it without retaining history.
    pub fn prove_streaming_blocks_with_final_proof_and_profile<I, Sink>(
        &self,
        bytecode_preprocessing: &BytecodePreprocessing,
        blocks: I,
        parameters: &Stage18ReleaseParameters,
        mut profile_sink: Sink,
    ) -> Result<Stage18StreamingNovaProof<Digest>, Stage18Error>
    where
        I: IntoIterator<Item = TraceBlock>,
        Sink: FnMut(Stage18BlockProfile),
    {
        parameters.validate()?;
        if self.folding_backend.config != *parameters.nova_config() {
            return Err(Stage18Error::InvalidReleaseParameters(
                "pipeline Nova configuration differs from the release manifest",
            ));
        }
        let receipt = self
            .verified_jolt_lookup_receipt
            .as_ref()
            .ok_or(Stage18Error::MissingVerifiedJoltReceipt)?;
        if !receipt.zk_mode() {
            return Err(Stage18Error::ExpectedZkReceipt);
        }

        let mut iterator = blocks.into_iter();
        let mut current = iterator.next().ok_or(Stage18Error::EmptyTrace)?;
        let mut next = iterator.next();
        let mut accumulator =
            <NovaFoldingBackend as BlockFoldingBackend<Digest, F>>::new_accumulator(
                &self.folding_backend,
            );
        let mut previous_public_input: Option<BlockPublicInput<Digest>> = None;
        let mut latest_ram_values = HashMap::<u64, u64>::new();
        let mut metrics = Stage18StreamingMetrics::new();

        loop {
            let mut block_profile = Stage18BlockProfile::new(&current, next.as_ref());
            metrics.observe_resident_blocks(&current, next.as_ref());
            if metrics.max_resident_trace_blocks > parameters.max_resident_trace_blocks() {
                return Err(Stage18Error::StreamingBoundExceeded {
                    observed: metrics.max_resident_trace_blocks,
                    allowed: parameters.max_resident_trace_blocks(),
                });
            }
            if current.target_size != parameters.block_target_size() {
                return Err(Stage18Error::BlockTargetSizeMismatch {
                    block_index: current.block_index,
                    expected: parameters.block_target_size(),
                    actual: current.target_size,
                });
            }
            if current.end_state.terminated && next.is_some() {
                return Err(Stage18Error::TraceContinuesAfterTermination {
                    block_index: current.block_index,
                });
            }

            let lookahead = next
                .as_ref()
                .and_then(|block| block.cycles.first())
                .or_else(|| {
                    current
                        .end_state
                        .terminated
                        .then_some(&TERMINAL_LOOKAHEAD_CYCLE)
                });
            if lookahead.is_none() {
                return Err(Stage18Error::UnterminatedTrace {
                    block_index: current.block_index,
                });
            }

            let started = Instant::now();
            let cpu_proof = self.bundle_prover.cpu_prover.prove_block(
                bytecode_preprocessing,
                &current,
                lookahead,
            )?;
            let elapsed = started.elapsed();
            metrics.profile_mut(CPU_RELATION).record(elapsed);
            block_profile.record(CPU_RELATION, elapsed);

            let started = Instant::now();
            let io_claims = extract_block_io_claims(&current)?;
            let elapsed = started.elapsed();
            metrics.profile_mut(IO_RELATION).record(elapsed);
            block_profile.record(IO_RELATION, elapsed);

            let started = Instant::now();
            let register_claim = build_block_register_claim(&current, &io_claims)?;
            let elapsed = started.elapsed();
            metrics.profile_mut(REGISTER_RELATION).record(elapsed);
            block_profile.record(REGISTER_RELATION, elapsed);

            let started = Instant::now();
            let ram_claim = build_block_ram_claim(&current, &io_claims)?;
            absorb_streaming_ram_continuity(&mut latest_ram_values, &io_claims)?;
            metrics.peak_tracked_ram_addresses = metrics
                .peak_tracked_ram_addresses
                .max(latest_ram_values.len());
            block_profile.tracked_ram_addresses = latest_ram_values.len();
            let elapsed = started.elapsed();
            metrics.profile_mut(RAM_RELATION).record(elapsed);
            block_profile.record(RAM_RELATION, elapsed);

            let started = Instant::now();
            let lookup_claim = build_block_lookup_claim(&current, &io_claims)?;
            let elapsed = started.elapsed();
            metrics.profile_mut(LOOKUP_RELATION).record(elapsed);
            block_profile.record(LOOKUP_RELATION, elapsed);

            let bundle = BlockProofBundle {
                cpu_proof,
                io_claims,
                register_claim,
                ram_claim,
                lookup_claim,
            };
            let mut fold_input = build_block_fold_input(&bundle);
            verify_block_fold_input(
                bytecode_preprocessing,
                &current,
                lookahead,
                &bundle,
                &fold_input,
            )?;

            if let Some(previous) = previous_public_input.as_ref() {
                previous.validate_contiguous_with(bundle.public_input())?;
            }
            previous_public_input = Some(bundle.public_input().clone());

            let started = Instant::now();
            bind_verified_jolt_lookup_receipt_to_fold_inputs(
                std::slice::from_mut(&mut fold_input),
                receipt,
            )?;
            verify_verified_jolt_lookup_receipt_bindings(
                std::slice::from_ref(&fold_input),
                receipt,
            )?;
            let elapsed = started.elapsed();
            metrics.profile_mut(RECEIPT_RELATION).record(elapsed);
            block_profile.record(RECEIPT_RELATION, elapsed);

            let started = Instant::now();
            self.folding_backend.absorb(&mut accumulator, &fold_input)?;
            let elapsed = started.elapsed();
            metrics.profile_mut(NOVA_RELATION).record(elapsed);
            block_profile.record(NOVA_RELATION, elapsed);

            metrics.block_count += 1;
            metrics.total_active_cycles += current.active_cycles;
            metrics.observe_memory();
            block_profile.sampled_physical_memory_bytes = sample_physical_memory();
            profile_sink(block_profile);

            drop(fold_input);
            drop(bundle);
            match next.take() {
                Some(next_block) => {
                    current = next_block;
                    next = iterator.next();
                }
                None => break,
            }
        }

        if !current.end_state.terminated {
            return Err(Stage18Error::UnterminatedTrace {
                block_index: current.block_index,
            });
        }

        let started = Instant::now();
        let final_proof = prove_configured_final_folded_accumulator(&accumulator)?;
        metrics
            .profile_mut(SPARTAN_RELATION)
            .record(started.elapsed());
        metrics.final_folded_proof_bytes =
            final_proof.spartan_proof_bytes.as_ref().map_or(0, Vec::len);
        metrics.end_physical_memory_bytes = sample_physical_memory();
        metrics.observe_memory();

        let release_parameters_digest = parameters.digest();
        let verified_jolt_receipt_digest = receipt.digest();
        let jolt_statement_id = receipt.recursive_execution_statement_id();
        let linkage_digest = digest_streaming_proof_linkage(
            &accumulator,
            &final_proof,
            &metrics,
            release_parameters_digest,
            verified_jolt_receipt_digest,
            jolt_statement_id,
        );
        let proof = Stage18StreamingNovaProof {
            accumulator,
            final_proof,
            metrics,
            release_parameters_digest,
            verified_jolt_receipt_digest,
            jolt_statement_id,
            linkage_digest,
        };
        proof.verify(parameters, receipt)?;
        Ok(proof)
    }
}

#[derive(Debug)]
pub enum Stage18Error {
    Block(BlockTraceError),
    InvalidReleaseParameters(&'static str),
    MissingVerifiedJoltReceipt,
    ExpectedZkReceipt,
    EmptyTrace,
    UnterminatedTrace {
        block_index: usize,
    },
    TraceContinuesAfterTermination {
        block_index: usize,
    },
    BlockTargetSizeMismatch {
        block_index: usize,
        expected: usize,
        actual: usize,
    },
    StreamingBoundExceeded {
        observed: usize,
        allowed: usize,
    },
    LinkageMismatch(&'static str),
    RecursiveVerification(String),
}

impl From<BlockTraceError> for Stage18Error {
    fn from(error: BlockTraceError) -> Self {
        Self::Block(error)
    }
}

impl From<BlockPublicInputError> for Stage18Error {
    fn from(error: BlockPublicInputError) -> Self {
        Self::Block(BlockTraceError::PublicInput(error))
    }
}

impl fmt::Display for Stage18Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Block(error) => error.fmt(f),
            Self::InvalidReleaseParameters(reason) => {
                write!(f, "invalid Stage-18 release parameters: {reason}")
            }
            Self::MissingVerifiedJoltReceipt => {
                write!(f, "Stage-18 production pipeline requires a verified Jolt receipt")
            }
            Self::ExpectedZkReceipt => write!(f, "Stage-18 production pipeline requires ZK mode"),
            Self::EmptyTrace => write!(f, "Stage-18 streaming trace is empty"),
            Self::UnterminatedTrace { block_index } => write!(
                f,
                "Stage-18 trace ended at block {block_index} without a terminal machine state"
            ),
            Self::TraceContinuesAfterTermination { block_index } => write!(
                f,
                "Stage-18 trace continues after terminal block {block_index}"
            ),
            Self::BlockTargetSizeMismatch {
                block_index,
                expected,
                actual,
            } => write!(
                f,
                "Stage-18 block {block_index} target size mismatch: expected {expected}, got {actual}"
            ),
            Self::StreamingBoundExceeded { observed, allowed } => write!(
                f,
                "Stage-18 resident trace block bound exceeded: observed {observed}, allowed {allowed}"
            ),
            Self::LinkageMismatch(reason) => {
                write!(f, "Stage-18 proof linkage mismatch: {reason}")
            }
            Self::RecursiveVerification(reason) => {
                write!(f, "Stage-18 recursive verification failed: {reason}")
            }
        }
    }
}

impl Error for Stage18Error {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Block(error) => Some(error),
            _ => None,
        }
    }
}

fn sample_physical_memory() -> Option<usize> {
    memory_stats::memory_stats().map(|stats| stats.physical_mem)
}

fn stage18_hex(bytes: impl AsRef<[u8]>) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let bytes = bytes.as_ref();
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        encoded.push(HEX[(byte >> 4) as usize] as char);
        encoded.push(HEX[(byte & 0x0f) as usize] as char);
    }
    encoded
}

fn relation_profiles_json(profiles: &[Stage18RelationProfile]) -> serde_json::Value {
    serde_json::Value::Array(
        profiles
            .iter()
            .map(|profile| {
                serde_json::json!({
                    "relation": profile.relation,
                    "calls": profile.calls,
                    "total_nanos": profile.total_nanos,
                    "max_nanos": profile.max_nanos,
                })
            })
            .collect(),
    )
}

fn absorb_streaming_ram_continuity(
    latest_values: &mut HashMap<u64, u64>,
    io_claims: &BlockIOClaims,
) -> Result<(), BlockTraceError> {
    for summary in ram_address_summaries(&io_claims.ram_accesses) {
        if let Some(expected) = latest_values.get(&summary.address).copied() {
            if summary.first_value != expected {
                return Err(BlockTraceError::BlockRamClaimContinuityMismatch {
                    block_index: io_claims.block_index,
                    address: summary.address,
                    expected,
                    actual: summary.first_value,
                });
            }
        }
        latest_values.insert(summary.address, summary.final_value);
    }
    Ok(())
}

fn digest_streaming_proof_linkage<Digest>(
    accumulator: &NovaFoldAccumulator<Digest>,
    final_proof: &FinalFoldedProof<Digest>,
    metrics: &Stage18StreamingMetrics,
    release_parameters_digest: [u8; 32],
    receipt_digest: [u8; 32],
    jolt_statement_id: [u8; 32],
) -> [u8; 32]
where
    Digest: AsRef<[u8]>,
{
    let mut hasher = Sha3_256::new();
    hasher.update(b"JOLT_NOVA_STAGE18_STREAMING_PROOF_LINKAGE_V1");
    hasher.update(JOLT_NOVA_STAGE18_VERSION.as_bytes());
    hasher.update(release_parameters_digest);
    hasher.update(receipt_digest);
    hasher.update(jolt_statement_id);
    hasher.update((accumulator.metadata.absorbed_blocks as u64).to_le_bytes());
    hasher.update((accumulator.metadata.total_active_cycles as u64).to_le_bytes());
    hasher.update(accumulator.metadata.accumulator_digest);
    if let Some(program_digest) = accumulator.metadata.program_digest.as_ref() {
        hasher.update(program_digest.as_ref());
    }
    hasher.update(final_proof.instance.instance_digest);
    hasher.update(final_proof.proof_digest);
    hasher.update(digest_streaming_metrics(metrics));
    finalize_stage18_digest(hasher)
}

fn digest_streaming_metrics(metrics: &Stage18StreamingMetrics) -> [u8; 32] {
    let mut hasher = Sha3_256::new();
    hasher.update(b"JOLT_NOVA_STAGE18_STREAMING_METRICS_V1");
    for value in [
        metrics.block_count,
        metrics.total_active_cycles,
        metrics.max_resident_trace_blocks,
        metrics.max_resident_trace_cycles,
        metrics.peak_tracked_ram_addresses,
        metrics.estimated_peak_trace_bytes,
        metrics.final_folded_proof_bytes,
    ] {
        hasher.update((value as u64).to_le_bytes());
    }
    for value in [
        metrics.start_physical_memory_bytes,
        metrics.peak_physical_memory_bytes,
        metrics.end_physical_memory_bytes,
    ] {
        hasher.update([u8::from(value.is_some())]);
        hasher.update((value.unwrap_or(0) as u64).to_le_bytes());
    }
    hasher.update((metrics.relation_profiles.len() as u64).to_le_bytes());
    for profile in &metrics.relation_profiles {
        hasher.update((profile.relation.len() as u64).to_le_bytes());
        hasher.update(profile.relation.as_bytes());
        hasher.update((profile.calls as u64).to_le_bytes());
        hasher.update(profile.total_nanos.to_le_bytes());
        hasher.update(profile.max_nanos.to_le_bytes());
    }
    finalize_stage18_digest(hasher)
}

#[cfg(feature = "zk")]
fn digest_stage18_zk_end_to_end_linkage<C, PCS, Digest>(
    streaming_execution: &Stage18StreamingNovaProof<Digest>,
    statement: &RecursiveBlindFoldStatement,
    acceptance: &RecursiveJoltZkCompleteFinalAcceptance<
        C,
        PCS,
        crate::transcripts::PoseidonTranscript,
    >,
    deferred_pcs_id: [u8; 32],
    release_parameters_digest: [u8; 32],
) -> [u8; 32]
where
    C: crate::curve::JoltCurve<F = ark_bn254::Fr>,
    PCS: crate::poly::commitment::commitment_scheme::CommitmentScheme<Field = ark_bn254::Fr>,
    Digest: Clone + PartialEq + AsRef<[u8]>,
{
    let mut hasher = Sha3_256::new();
    hasher.update(b"JOLT_NOVA_STAGE18_ZK_END_TO_END_LINKAGE_V1");
    hasher.update(JOLT_NOVA_STAGE18_VERSION.as_bytes());
    hasher.update(release_parameters_digest);
    hasher.update(streaming_execution.linkage_digest());
    hasher.update(statement.object_id);
    hasher.update(statement.deferred_group_id);
    hasher.update(statement.jolt_statement_id);
    hasher.update(statement.shape_id);
    hasher.update(statement.initial_transcript_state.canonical_le_bytes);
    hasher.update(statement.initial_transcript_round.to_le_bytes());
    hasher.update(statement.final_transcript_state.canonical_le_bytes);
    hasher.update(statement.final_transcript_round.to_le_bytes());
    hasher.update(deferred_pcs_id);
    hasher.update((acceptance.blindfold.recursive_proof.proof_bytes.len() as u64).to_le_bytes());
    hasher.update(&acceptance.blindfold.recursive_proof.proof_bytes);
    hasher.update((acceptance.blindfold.recursive_proof.public_output.len() as u64).to_le_bytes());
    for word in &acceptance.blindfold.recursive_proof.public_output {
        hasher.update(word);
    }
    finalize_stage18_digest(hasher)
}

fn finalize_stage18_digest(hasher: Sha3_256) -> [u8; 32] {
    let bytes = hasher.finalize();
    let mut digest = [0u8; 32];
    digest.copy_from_slice(&bytes);
    digest
}

#[cfg(test)]
mod tests {
    use super::*;

    fn boundary(global_cycle: usize, terminated: bool) -> MachineBoundaryState {
        MachineBoundaryState {
            global_cycle,
            emulator_trace_len: global_cycle,
            pc: 0,
            registers: [0; REGISTER_COUNT as usize],
            terminated,
        }
    }

    fn noop_block(
        block_index: usize,
        start: MachineBoundaryState,
        end: MachineBoundaryState,
        target_size: usize,
    ) -> TraceBlock {
        let active_cycles = end.global_cycle - start.global_cycle;
        TraceBlock {
            block_index,
            global_cycle_start: start.global_cycle,
            active_cycles,
            target_size,
            start_state: start,
            end_state: end,
            cycles: vec![Cycle::NoOp; active_cycles],
            ended_at_tick_boundary: true,
        }
    }

    fn two_block_fixture() -> Vec<TraceBlock> {
        let block0 = noop_block(0, boundary(0, false), boundary(2, false), 2);
        let block1 = noop_block(1, block0.end_state.clone(), boundary(4, true), 2);
        vec![block0, block1]
    }

    #[test]
    fn stage18_release_parameters_pin_production_backends() {
        let parameters = Stage18ReleaseParameters::production(2, [7; 32]).unwrap();
        assert_eq!(parameters.block_target_size(), 2);
        assert_eq!(parameters.max_resident_trace_blocks(), 2);
        assert_eq!(parameters.security_bits(), 128);
        assert_eq!(
            parameters.nova_config().subclaim_backend_name,
            NOVA_JOLT_LASSO_SUBCLAIM_BACKEND_NAME
        );
        assert_eq!(
            parameters.nova_config().final_proof_backend_name,
            SPARTAN_FINAL_PROOF_SYSTEM_NAME
        );
        assert_ne!(parameters.digest(), [0; 32]);

        assert!(matches!(
            Stage18ReleaseParameters::production(3, [7; 32]),
            Err(Stage18Error::InvalidReleaseParameters(_))
        ));
        assert!(matches!(
            Stage18ReleaseParameters::production(2, [0; 32]),
            Err(Stage18Error::InvalidReleaseParameters(_))
        ));
        let mut substituted_backend = parameters.clone();
        substituted_backend.nova_config.backend_name = "nova-lookalike";
        assert!(matches!(
            substituted_backend.validate(),
            Err(Stage18Error::InvalidReleaseParameters(_))
        ));
    }

    #[test]
    fn stage18_streams_two_blocks_into_real_nova_spartan_proof() {
        let blocks = two_block_fixture();
        let parameters = Stage18ReleaseParameters::production(2, [7; 32]).unwrap();
        let receipt = VerifiedJoltLookupProofReceipt::new_zk_for_test(61, 8);
        let backend = NovaFoldingBackend::new(parameters.nova_config().clone());
        let pipeline = BlockProofPipeline::<_, ark_bn254::Fr, NovaFoldingBackend>::
            with_backend_and_verified_jolt_lookup_receipt(
                [9; 32],
                backend,
                receipt.clone(),
            );

        let mut block_profiles = Vec::new();
        let proof = pipeline
            .prove_streaming_blocks_with_final_proof_and_profile(
                &BytecodePreprocessing::default(),
                blocks,
                &parameters,
                |profile| block_profiles.push(profile),
            )
            .unwrap();

        proof.verify(&parameters, &receipt).unwrap();
        assert_eq!(proof.metrics().block_count, 2);
        assert_eq!(proof.metrics().total_active_cycles, 4);
        assert_eq!(proof.metrics().max_resident_trace_blocks, 2);
        assert_eq!(block_profiles.len(), 2);
        assert_eq!(block_profiles[0].block_index, 0);
        assert_eq!(block_profiles[1].block_index, 1);
        assert_eq!(block_profiles[0].resident_trace_blocks, 2);
        assert_eq!(block_profiles[1].resident_trace_blocks, 1);
        assert!(block_profiles
            .iter()
            .all(|profile| profile.relation_profiles.len() == 7));
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&block_profiles[0].to_json_line().unwrap())
                .unwrap()["block_index"],
            0
        );
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&proof.report_json_pretty().unwrap())
                .unwrap()["metrics"]["block_count"],
            2
        );
        assert!(parameters
            .to_json_pretty()
            .unwrap()
            .contains(SPARTAN_FINAL_PROOF_SYSTEM_NAME));
        println!(
            "STAGE18_RELEASE_PARAMETERS={}",
            parameters.to_json_pretty().unwrap()
        );
        for profile in &block_profiles {
            println!("STAGE18_BLOCK_PROFILE={}", profile.to_json_line().unwrap());
        }
        println!(
            "STAGE18_STREAMING_REPORT={}",
            proof.report_json_pretty().unwrap()
        );
        assert!(proof.metrics().final_folded_proof_bytes > 0);
        assert_eq!(
            proof.final_proof().proof_system,
            SPARTAN_FINAL_PROOF_SYSTEM_NAME
        );
        assert!(proof
            .metrics()
            .relation_profiles
            .iter()
            .filter(|profile| profile.relation != SPARTAN_RELATION)
            .all(|profile| profile.calls == 2));
        assert_eq!(
            proof
                .metrics()
                .relation_profiles
                .iter()
                .find(|profile| profile.relation == SPARTAN_RELATION)
                .unwrap()
                .calls,
            1
        );

        let wrong_receipt = VerifiedJoltLookupProofReceipt::new_zk_for_test(62, 8);
        assert!(matches!(
            proof.verify(&parameters, &wrong_receipt),
            Err(Stage18Error::LinkageMismatch(_))
        ));

        let mut tampered = proof.clone();
        tampered.final_proof.proof_digest[0] ^= 1;
        assert!(tampered.verify(&parameters, &receipt).is_err());

        let mut tampered_metrics = proof.clone();
        tampered_metrics.metrics.estimated_peak_trace_bytes += 1;
        assert!(matches!(
            tampered_metrics.verify(&parameters, &receipt),
            Err(Stage18Error::LinkageMismatch(_))
        ));
    }

    #[test]
    fn stage18_rejects_truncation_reordering_and_wrong_block_size_before_acceptance() {
        let parameters = Stage18ReleaseParameters::production(2, [7; 32]).unwrap();
        let receipt = VerifiedJoltLookupProofReceipt::new_zk_for_test(63, 8);
        let pipeline = BlockProofPipeline::<_, ark_bn254::Fr, NovaFoldingBackend>::
            with_backend_and_verified_jolt_lookup_receipt(
                [9; 32],
                NovaFoldingBackend::new(parameters.nova_config().clone()),
                receipt,
            );

        let mut truncated = two_block_fixture();
        truncated.pop();
        assert!(matches!(
            pipeline.prove_streaming_blocks_with_final_proof(
                &BytecodePreprocessing::default(),
                truncated,
                &parameters,
            ),
            Err(Stage18Error::UnterminatedTrace { block_index: 0 })
        ));

        let mut reordered = two_block_fixture();
        reordered.swap(0, 1);
        assert!(pipeline
            .prove_streaming_blocks_with_final_proof(
                &BytecodePreprocessing::default(),
                reordered,
                &parameters,
            )
            .is_err());

        let mut wrong_size = two_block_fixture();
        wrong_size[0].target_size = 4;
        assert!(matches!(
            pipeline.prove_streaming_blocks_with_final_proof(
                &BytecodePreprocessing::default(),
                wrong_size,
                &parameters,
            ),
            Err(Stage18Error::BlockTargetSizeMismatch { block_index: 0, .. })
        ));

        let mut spliced_state = two_block_fixture();
        spliced_state[1].start_state.pc ^= 4;
        assert!(pipeline
            .prove_streaming_blocks_with_final_proof(
                &BytecodePreprocessing::default(),
                spliced_state,
                &parameters,
            )
            .is_err());
    }

    #[test]
    fn stage18_streaming_ram_state_rejects_cross_block_value_splice() {
        let mut latest = HashMap::new();
        let first = BlockIOClaims {
            block_index: 0,
            global_cycle_start: 0,
            active_cycles: 1,
            register_reads: Vec::new(),
            register_writes: Vec::new(),
            ram_accesses: vec![RamAccessClaim::Write {
                local_cycle: 0,
                global_cycle: 0,
                address: 0x1000,
                pre_value: 7,
                post_value: 11,
            }],
            lookup_claims: Vec::new(),
        };
        absorb_streaming_ram_continuity(&mut latest, &first).unwrap();
        assert_eq!(latest.get(&0x1000), Some(&11));

        let spliced = BlockIOClaims {
            block_index: 1,
            global_cycle_start: 1,
            active_cycles: 1,
            register_reads: Vec::new(),
            register_writes: Vec::new(),
            ram_accesses: vec![RamAccessClaim::Read {
                local_cycle: 0,
                global_cycle: 1,
                address: 0x1000,
                value: 12,
            }],
            lookup_claims: Vec::new(),
        };
        assert!(matches!(
            absorb_streaming_ram_continuity(&mut latest, &spliced),
            Err(BlockTraceError::BlockRamClaimContinuityMismatch {
                block_index: 1,
                address: 0x1000,
                expected: 11,
                actual: 12,
            })
        ));
    }
}
