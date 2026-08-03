//! Stage-19 reproducible benchmark artifacts and regression gates.
//!
//! Stage 18 established the production cryptographic path.  This module keeps
//! Stage-19 measurements honest: every row records the native Jolt proof and
//! the authenticated streaming/Nova/Spartan path for the same execution
//! statement, and artifact validation rejects mixed workloads, synthetic
//! receipts, incomplete verification, or violations of the two-block bound.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};

pub const JOLT_NOVA_STAGE19_SCHEMA_VERSION: &str = "jolt-nova-stage19-benchmark-v1";
pub const JOLT_NOVA_STAGE19_RUNNER_VERSION: &str = "jolt-nova-stage19-runner-v1";
pub const JOLT_NOVA_STAGE19_PROFILE_METHOD: &str = "measured-stage18-instrumentation-v1";
pub const JOLT_NOVA_STAGE19_LOOKUP_BACKEND: &str = "jolt-lasso-subclaim-v1";

const REQUIRED_RELATIONS: [&str; 8] = [
    "cpu-r1cs",
    "trace-io-claims",
    "register-read-write",
    "ram-read-write",
    "jolt-lasso-lookup-claim",
    "verified-jolt-receipt-binding",
    "nova-block-fold",
    "spartan-final-compression",
];

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Stage19Platform {
    pub os: String,
    pub arch: String,
    pub build_profile: String,
    pub rust_toolchain: String,
}

impl Stage19Platform {
    pub fn current(rust_toolchain: impl Into<String>) -> Self {
        Self {
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
            build_profile: if cfg!(debug_assertions) {
                "debug".to_string()
            } else {
                "release".to_string()
            },
            rust_toolchain: rust_toolchain.into(),
        }
    }

    fn fingerprint(&self) -> String {
        format!(
            "{}/{}/{}/{}",
            self.os, self.arch, self.build_profile, self.rust_toolchain
        )
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Stage19TimingsMicros {
    /// Time to construct the lazy trace-block iterator. Actual block generation
    /// is consumed and measured inside `nova_spartan_stream`.
    pub trace: u64,
    pub native_jolt_prove: u64,
    pub native_jolt_verify_and_export: u64,
    pub nova_spartan_stream: u64,
    pub recursive_blindfold_prove: u64,
    pub final_end_to_end_verify: u64,
    pub measured_total: u64,
}

impl Stage19TimingsMicros {
    pub fn recursive_extension(&self) -> u64 {
        self.nova_spartan_stream
            .saturating_add(self.recursive_blindfold_prove)
            .saturating_add(self.final_end_to_end_verify)
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Stage19MemoryBytes {
    pub native_jolt_start: Option<u64>,
    pub native_jolt_peak: Option<u64>,
    pub native_jolt_peak_delta: Option<u64>,
    pub recursive_start: Option<u64>,
    pub recursive_peak: Option<u64>,
    pub recursive_peak_delta: Option<u64>,
    pub estimated_peak_trace: u64,
    pub peak_tracked_ram_addresses: usize,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Stage19ProofSizes {
    /// Canonically serialized native Jolt ZK proof.
    pub native_jolt_proof: usize,
    /// Final Spartan compression of the block-fold accumulator.
    pub folded_spartan_proof: usize,
    /// Spartan proof of the recursive BlindFold verifier circuit.
    pub blindfold_spartan_proof: usize,
    /// Sum of recursive proof byte strings. Typed group/PCS obligations are
    /// deliberately reported separately by verification status, not hidden in
    /// this byte count.
    pub recursive_proof_payload: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Stage19RelationMeasurement {
    pub relation: String,
    pub attribution_method: String,
    pub calls: usize,
    pub total_nanos: u128,
    pub max_nanos: u128,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Stage19BenchmarkSample {
    pub run_index: usize,
    pub workload: String,
    pub workload_sha3_256: String,
    pub block_target_size: usize,
    pub block_count: usize,
    pub active_cycles: usize,
    pub padded_trace_length: usize,
    pub max_resident_trace_blocks: usize,
    pub max_resident_trace_cycles: usize,
    pub receipt_is_production_zk: bool,
    pub native_jolt_verified: bool,
    pub streaming_proof_verified: bool,
    pub complete_zk_verified: bool,
    pub lookup_backend: String,
    pub receipt_digest: String,
    pub jolt_statement_id: String,
    pub timings_micros: Stage19TimingsMicros,
    pub throughput_active_cycles_per_second: f64,
    pub memory_bytes: Stage19MemoryBytes,
    pub proof_sizes: Stage19ProofSizes,
    pub relation_profiles: Vec<Stage19RelationMeasurement>,
}

/// Canonical digest view. Floating-point rates and all aggregate/bottleneck
/// fields are derived from these integer measurements and deliberately omitted
/// so a JSON round trip or build-profile change cannot alter artifact identity.
#[derive(Serialize)]
struct Stage19DigestSample<'a> {
    run_index: usize,
    workload: &'a str,
    workload_sha3_256: &'a str,
    block_target_size: usize,
    block_count: usize,
    active_cycles: usize,
    padded_trace_length: usize,
    max_resident_trace_blocks: usize,
    max_resident_trace_cycles: usize,
    receipt_is_production_zk: bool,
    native_jolt_verified: bool,
    streaming_proof_verified: bool,
    complete_zk_verified: bool,
    lookup_backend: &'a str,
    receipt_digest: &'a str,
    jolt_statement_id: &'a str,
    timings_micros: &'a Stage19TimingsMicros,
    memory_bytes: &'a Stage19MemoryBytes,
    proof_sizes: &'a Stage19ProofSizes,
    relation_profiles: &'a [Stage19RelationMeasurement],
}

impl<'a> From<&'a Stage19BenchmarkSample> for Stage19DigestSample<'a> {
    fn from(sample: &'a Stage19BenchmarkSample) -> Self {
        Self {
            run_index: sample.run_index,
            workload: &sample.workload,
            workload_sha3_256: &sample.workload_sha3_256,
            block_target_size: sample.block_target_size,
            block_count: sample.block_count,
            active_cycles: sample.active_cycles,
            padded_trace_length: sample.padded_trace_length,
            max_resident_trace_blocks: sample.max_resident_trace_blocks,
            max_resident_trace_cycles: sample.max_resident_trace_cycles,
            receipt_is_production_zk: sample.receipt_is_production_zk,
            native_jolt_verified: sample.native_jolt_verified,
            streaming_proof_verified: sample.streaming_proof_verified,
            complete_zk_verified: sample.complete_zk_verified,
            lookup_backend: &sample.lookup_backend,
            receipt_digest: &sample.receipt_digest,
            jolt_statement_id: &sample.jolt_statement_id,
            timings_micros: &sample.timings_micros,
            memory_bytes: &sample.memory_bytes,
            proof_sizes: &sample.proof_sizes,
            relation_profiles: &sample.relation_profiles,
        }
    }
}

impl Stage19BenchmarkSample {
    pub fn validate(&self) -> Result<(), String> {
        if self.run_index == 0 {
            return Err("run_index must be one-based".to_string());
        }
        if self.workload.trim().is_empty() || self.workload_sha3_256.len() != 64 {
            return Err(
                "workload identity must be non-empty and use a SHA3-256 digest".to_string(),
            );
        }
        if self.block_target_size == 0 || !self.block_target_size.is_power_of_two() {
            return Err("block_target_size must be a non-zero power of two".to_string());
        }
        if self.block_count == 0 || self.active_cycles == 0 || self.padded_trace_length == 0 {
            return Err("benchmark samples require a non-empty execution".to_string());
        }
        if self.max_resident_trace_blocks == 0 || self.max_resident_trace_blocks > 2 {
            return Err("production streaming exceeded the two-block resident bound".to_string());
        }
        if !self.receipt_is_production_zk
            || !self.native_jolt_verified
            || !self.streaming_proof_verified
            || !self.complete_zk_verified
        {
            return Err(
                "all native and recursive cryptographic verification gates must pass".to_string(),
            );
        }
        if self.lookup_backend != JOLT_NOVA_STAGE19_LOOKUP_BACKEND {
            return Err("Stage-19 production measurements require native Jolt Lasso".to_string());
        }
        if self.receipt_digest.len() != 64 || self.jolt_statement_id.len() != 64 {
            return Err("receipt and statement identifiers must be full-width digests".to_string());
        }
        if self.timings_micros.native_jolt_prove == 0
            || self.timings_micros.native_jolt_verify_and_export == 0
            || self.timings_micros.nova_spartan_stream == 0
            || self.timings_micros.recursive_blindfold_prove == 0
            || self.timings_micros.final_end_to_end_verify == 0
            || self.timings_micros.measured_total == 0
        {
            return Err(
                "cryptographic phase timings must be measured, not zero-filled".to_string(),
            );
        }
        if !self.throughput_active_cycles_per_second.is_finite()
            || self.throughput_active_cycles_per_second <= 0.0
        {
            return Err("throughput must be finite and positive".to_string());
        }
        let expected_throughput =
            self.active_cycles as f64 / (self.timings_micros.measured_total as f64 / 1_000_000.0);
        let throughput_error =
            (self.throughput_active_cycles_per_second - expected_throughput).abs();
        if throughput_error > expected_throughput.abs().max(1.0) * 1e-12 {
            return Err("throughput is inconsistent with cycles and measured time".to_string());
        }
        if self.proof_sizes.native_jolt_proof == 0
            || self.proof_sizes.folded_spartan_proof == 0
            || self.proof_sizes.blindfold_spartan_proof == 0
            || self.proof_sizes.recursive_proof_payload
                != self
                    .proof_sizes
                    .folded_spartan_proof
                    .saturating_add(self.proof_sizes.blindfold_spartan_proof)
        {
            return Err("proof-size accounting is incomplete or inconsistent".to_string());
        }

        let observed = self
            .relation_profiles
            .iter()
            .map(|profile| profile.relation.as_str())
            .collect::<BTreeSet<_>>();
        let required = REQUIRED_RELATIONS.into_iter().collect::<BTreeSet<_>>();
        if observed != required {
            return Err(
                "relation profile does not cover the complete Stage-18 relation set".to_string(),
            );
        }
        for profile in &self.relation_profiles {
            if profile.attribution_method != JOLT_NOVA_STAGE19_PROFILE_METHOD
                || profile.calls == 0
                || profile.total_nanos == 0
                || profile.max_nanos == 0
                || profile.max_nanos > profile.total_nanos
            {
                return Err(format!(
                    "relation {} is estimated, empty, or internally inconsistent",
                    profile.relation
                ));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Stage19SummaryStatistics {
    pub count: usize,
    pub min: f64,
    pub median: f64,
    pub mean: f64,
    pub max: f64,
    pub standard_deviation: f64,
}

impl Stage19SummaryStatistics {
    fn from_values(mut values: Vec<f64>) -> Result<Self, String> {
        if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
            return Err("summary statistics require finite samples".to_string());
        }
        values.sort_by(f64::total_cmp);
        let count = values.len();
        let mean = values.iter().sum::<f64>() / count as f64;
        let median = if count % 2 == 0 {
            (values[count / 2 - 1] + values[count / 2]) / 2.0
        } else {
            values[count / 2]
        };
        let variance = values
            .iter()
            .map(|value| {
                let delta = value - mean;
                delta * delta
            })
            .sum::<f64>()
            / count as f64;
        Ok(Self {
            count,
            min: values[0],
            median,
            mean,
            max: values[count - 1],
            standard_deviation: variance.sqrt(),
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Stage19BlockAggregate {
    pub block_target_size: usize,
    pub samples: usize,
    pub block_count: Stage19SummaryStatistics,
    pub active_cycles: Stage19SummaryStatistics,
    pub native_jolt_prove_micros: Stage19SummaryStatistics,
    pub recursive_extension_micros: Stage19SummaryStatistics,
    pub measured_total_micros: Stage19SummaryStatistics,
    pub throughput_active_cycles_per_second: Stage19SummaryStatistics,
    pub native_jolt_peak_delta_bytes: Option<Stage19SummaryStatistics>,
    pub recursive_peak_delta_bytes: Option<Stage19SummaryStatistics>,
    pub native_jolt_proof_bytes: Stage19SummaryStatistics,
    pub recursive_proof_payload_bytes: Stage19SummaryStatistics,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Stage19Bottleneck {
    pub phase: String,
    pub mean_micros: f64,
    pub percent_of_measured_total: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Stage19RegressionMetric {
    pub block_target_size: usize,
    pub metric: String,
    pub baseline_mean: f64,
    pub current_mean: f64,
    pub regression_percent: f64,
    pub passed: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Stage19RegressionComparison {
    pub baseline_matrix_digest: String,
    pub max_regression_percent: f64,
    pub passed: bool,
    pub metrics: Vec<Stage19RegressionMetric>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Stage19BenchmarkArtifact {
    pub schema_version: String,
    pub runner_version: String,
    pub platform: Stage19Platform,
    pub workload: String,
    pub workload_sha3_256: String,
    /// One-time guest build, I/O-layout trace, and Jolt preprocessing cost.
    pub preprocessing_micros: u64,
    pub measurement_runs: usize,
    pub block_target_sizes: Vec<usize>,
    pub samples: Vec<Stage19BenchmarkSample>,
    pub aggregates: Vec<Stage19BlockAggregate>,
    pub bottleneck: Stage19Bottleneck,
    pub matrix_digest: String,
    pub regression: Option<Stage19RegressionComparison>,
}

impl Stage19BenchmarkArtifact {
    pub fn build(
        platform: Stage19Platform,
        workload: impl Into<String>,
        workload_sha3_256: impl Into<String>,
        preprocessing_micros: u64,
        measurement_runs: usize,
        samples: Vec<Stage19BenchmarkSample>,
    ) -> Result<Self, String> {
        let workload = workload.into();
        let workload_sha3_256 = workload_sha3_256.into();
        let aggregates = aggregate_samples(&samples)?;
        let block_target_sizes = aggregates
            .iter()
            .map(|aggregate| aggregate.block_target_size)
            .collect::<Vec<_>>();
        let bottleneck = identify_bottleneck(&samples)?;
        let mut artifact = Self {
            schema_version: JOLT_NOVA_STAGE19_SCHEMA_VERSION.to_string(),
            runner_version: JOLT_NOVA_STAGE19_RUNNER_VERSION.to_string(),
            platform,
            workload,
            workload_sha3_256,
            preprocessing_micros,
            measurement_runs,
            block_target_sizes,
            samples,
            aggregates,
            bottleneck,
            matrix_digest: String::new(),
            regression: None,
        };
        artifact.matrix_digest = artifact.compute_matrix_digest()?;
        artifact.validate()?;
        Ok(artifact)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != JOLT_NOVA_STAGE19_SCHEMA_VERSION
            || self.runner_version != JOLT_NOVA_STAGE19_RUNNER_VERSION
        {
            return Err("unsupported Stage-19 artifact version".to_string());
        }
        if self.preprocessing_micros == 0 || self.measurement_runs == 0 {
            return Err("preprocessing and measurement count must be non-zero".to_string());
        }
        if self.workload.trim().is_empty() || self.workload_sha3_256.len() != 64 {
            return Err("artifact workload identity is invalid".to_string());
        }
        if self.samples.len()
            != self
                .measurement_runs
                .checked_mul(self.block_target_sizes.len())
                .ok_or_else(|| "sample matrix size overflow".to_string())?
        {
            return Err("benchmark matrix is incomplete".to_string());
        }
        let mut cells = BTreeSet::new();
        for sample in &self.samples {
            sample.validate()?;
            if sample.workload != self.workload
                || sample.workload_sha3_256 != self.workload_sha3_256
            {
                return Err("sample workload does not match artifact workload".to_string());
            }
            if !cells.insert((sample.block_target_size, sample.run_index)) {
                return Err("benchmark matrix contains a duplicate cell".to_string());
            }
        }
        let aggregate_sizes = self
            .aggregates
            .iter()
            .map(|aggregate| aggregate.block_target_size)
            .collect::<Vec<_>>();
        if aggregate_sizes != self.block_target_sizes {
            return Err("aggregate block-size order does not match the matrix".to_string());
        }
        let expected_aggregates = aggregate_samples(&self.samples)?;
        if self.aggregates.len() != expected_aggregates.len()
            || self
                .aggregates
                .iter()
                .zip(&expected_aggregates)
                .any(|(stored, expected)| !aggregate_matches(stored, expected))
        {
            return Err("stored aggregates do not match the measured samples".to_string());
        }
        let expected_bottleneck = identify_bottleneck(&self.samples)?;
        let bottleneck_mean_error =
            (self.bottleneck.mean_micros - expected_bottleneck.mean_micros).abs();
        let bottleneck_percent_error = (self.bottleneck.percent_of_measured_total
            - expected_bottleneck.percent_of_measured_total)
            .abs();
        if self.bottleneck.phase != expected_bottleneck.phase
            || bottleneck_mean_error > expected_bottleneck.mean_micros.abs().max(1.0) * 1e-12
            || bottleneck_percent_error
                > expected_bottleneck.percent_of_measured_total.abs().max(1.0) * 1e-12
        {
            return Err("stored bottleneck does not match the measured samples".to_string());
        }
        let computed_matrix_digest = self.compute_matrix_digest()?;
        if self.matrix_digest != computed_matrix_digest {
            return Err(format!(
                "benchmark matrix digest mismatch: artifact={} computed={computed_matrix_digest}",
                self.matrix_digest
            ));
        }
        if let Some(regression) = &self.regression {
            if regression.passed != regression.metrics.iter().all(|metric| metric.passed) {
                return Err("regression summary does not match metric results".to_string());
            }
        }
        Ok(())
    }

    pub fn compare_with_baseline(
        &mut self,
        baseline: &Self,
        max_regression_percent: f64,
    ) -> Result<&Stage19RegressionComparison, String> {
        self.validate()?;
        baseline.validate()?;
        if !max_regression_percent.is_finite() || max_regression_percent < 0.0 {
            return Err("max regression percent must be finite and non-negative".to_string());
        }
        if self.platform != baseline.platform
            || self.workload != baseline.workload
            || self.workload_sha3_256 != baseline.workload_sha3_256
            || self.block_target_sizes != baseline.block_target_sizes
        {
            return Err("baseline is not comparable to the current matrix".to_string());
        }
        let mut metrics = Vec::new();
        for (current, previous) in self.aggregates.iter().zip(&baseline.aggregates) {
            for (name, current_mean, baseline_mean) in [
                (
                    "measured-total-micros",
                    current.measured_total_micros.mean,
                    previous.measured_total_micros.mean,
                ),
                (
                    "recursive-extension-micros",
                    current.recursive_extension_micros.mean,
                    previous.recursive_extension_micros.mean,
                ),
                (
                    "recursive-proof-payload-bytes",
                    current.recursive_proof_payload_bytes.mean,
                    previous.recursive_proof_payload_bytes.mean,
                ),
            ] {
                metrics.push(regression_metric(
                    current.block_target_size,
                    name,
                    baseline_mean,
                    current_mean,
                    max_regression_percent,
                    false,
                )?);
            }
            metrics.push(regression_metric(
                current.block_target_size,
                "throughput-active-cycles-per-second",
                previous.throughput_active_cycles_per_second.mean,
                current.throughput_active_cycles_per_second.mean,
                max_regression_percent,
                true,
            )?);
            if let (Some(current_memory), Some(previous_memory)) = (
                &current.recursive_peak_delta_bytes,
                &previous.recursive_peak_delta_bytes,
            ) {
                metrics.push(regression_metric(
                    current.block_target_size,
                    "recursive-peak-delta-bytes",
                    previous_memory.mean,
                    current_memory.mean,
                    max_regression_percent,
                    false,
                )?);
            }
        }
        let comparison = Stage19RegressionComparison {
            baseline_matrix_digest: baseline.matrix_digest.clone(),
            max_regression_percent,
            passed: metrics.iter().all(|metric| metric.passed),
            metrics,
        };
        self.regression = Some(comparison);
        Ok(self.regression.as_ref().expect("comparison was inserted"))
    }

    pub fn to_json_pretty(&self) -> Result<String, String> {
        self.validate()?;
        serde_json::to_string_pretty(self).map_err(|error| error.to_string())
    }

    pub fn to_markdown(&self) -> Result<String, String> {
        self.validate()?;
        let mut output = String::new();
        output.push_str("# Jolt-Nova Stage 19 benchmark report\n\n");
        output.push_str(&format!(
            "- Workload: `{}`\n- Workload SHA3-256: `{}`\n- Platform: `{}`\n- One-time preprocessing: {:.2} ms\n- Runs per block size: {}\n- Matrix digest: `{}`\n\n",
            self.workload,
            self.workload_sha3_256,
            self.platform.fingerprint(),
            self.preprocessing_micros as f64 / 1_000.0,
            self.measurement_runs,
            self.matrix_digest
        ));
        output.push_str("All rows use a verifier-exported production ZK receipt and passed native Jolt, streaming Spartan, recursive BlindFold, group, and Dory PCS verification.\n\n");
        output.push_str("| block size | blocks | cycles | native Jolt prove (ms) | recursive extension (ms) | total (ms) | cycles/s | native peak delta (MiB) | recursive peak delta (MiB) | Jolt proof bytes | recursive payload bytes |\n");
        output.push_str("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n");
        for aggregate in &self.aggregates {
            let native_peak = aggregate
                .native_jolt_peak_delta_bytes
                .as_ref()
                .map(|stats| format!("{:.2}", stats.mean / 1_048_576.0))
                .unwrap_or_else(|| "n/a".to_string());
            let recursive_peak = aggregate
                .recursive_peak_delta_bytes
                .as_ref()
                .map(|stats| format!("{:.2}", stats.mean / 1_048_576.0))
                .unwrap_or_else(|| "n/a".to_string());
            output.push_str(&format!(
                "| {} | {:.0} | {:.0} | {:.2} | {:.2} | {:.2} | {:.2} | {} | {} | {:.0} | {:.0} |\n",
                aggregate.block_target_size,
                aggregate.block_count.mean,
                aggregate.active_cycles.mean,
                aggregate.native_jolt_prove_micros.mean / 1_000.0,
                aggregate.recursive_extension_micros.mean / 1_000.0,
                aggregate.measured_total_micros.mean / 1_000.0,
                aggregate.throughput_active_cycles_per_second.mean,
                native_peak,
                recursive_peak,
                aggregate.native_jolt_proof_bytes.mean,
                aggregate.recursive_proof_payload_bytes.mean,
            ));
        }

        let mut relation_totals = BTreeMap::<&str, (usize, u128, u128)>::new();
        for profile in self
            .samples
            .iter()
            .flat_map(|sample| sample.relation_profiles.iter())
        {
            let entry = relation_totals
                .entry(profile.relation.as_str())
                .or_insert((0, 0, 0));
            entry.0 = entry.0.saturating_add(profile.calls);
            entry.1 = entry.1.saturating_add(profile.total_nanos);
            entry.2 = entry.2.max(profile.max_nanos);
        }
        output.push_str("\n## Measured relation attribution\n\n");
        output.push_str("| relation | calls | total (ms) | max call (ms) |\n");
        output.push_str("|---|---:|---:|---:|\n");
        for (relation, (calls, total_nanos, max_nanos)) in relation_totals {
            output.push_str(&format!(
                "| `{relation}` | {calls} | {:.3} | {:.3} |\n",
                total_nanos as f64 / 1_000_000.0,
                max_nanos as f64 / 1_000_000.0,
            ));
        }
        output.push_str(&format!(
            "\nDominant measured phase: **{}** ({:.2} ms, {:.2}% of measured total).\n",
            self.bottleneck.phase,
            self.bottleneck.mean_micros / 1_000.0,
            self.bottleneck.percent_of_measured_total
        ));
        if let Some(regression) = &self.regression {
            output.push_str(&format!(
                "\nRegression gate: **{}** (limit {:.2}%).\n",
                if regression.passed {
                    "passed"
                } else {
                    "failed"
                },
                regression.max_regression_percent
            ));
        }
        output.push_str("\nTiming note: `trace` records lazy iterator setup; streamed trace generation is included in `nova_spartan_stream`. Memory values are sampled process physical-memory deltas. Recursive payload bytes are the folded Spartan and BlindFold Spartan byte strings; typed group and Dory PCS objects are verified but not serialized into that subtotal.\n");
        Ok(output)
    }

    fn compute_matrix_digest(&self) -> Result<String, String> {
        let digest_samples = self
            .samples
            .iter()
            .map(Stage19DigestSample::from)
            .collect::<Vec<_>>();
        let encoded = serde_json::to_vec(&(
            &self.schema_version,
            &self.runner_version,
            &self.platform,
            &self.workload,
            &self.workload_sha3_256,
            self.preprocessing_micros,
            self.measurement_runs,
            &self.block_target_sizes,
            &digest_samples,
        ))
        .map_err(|error| error.to_string())?;
        Ok(hex_digest(Sha3_256::digest(encoded).into()))
    }
}

fn aggregate_samples(
    samples: &[Stage19BenchmarkSample],
) -> Result<Vec<Stage19BlockAggregate>, String> {
    if samples.is_empty() {
        return Err("benchmark artifact requires samples".to_string());
    }
    let mut groups = BTreeMap::<usize, Vec<&Stage19BenchmarkSample>>::new();
    for sample in samples {
        sample.validate()?;
        groups
            .entry(sample.block_target_size)
            .or_default()
            .push(sample);
    }
    groups
        .into_iter()
        .map(|(block_target_size, group)| {
            let stats = |values: Vec<f64>| Stage19SummaryStatistics::from_values(values);
            let optional_stats =
                |values: Vec<Option<u64>>| -> Result<Option<Stage19SummaryStatistics>, String> {
                    let collected = values
                        .into_iter()
                        .flatten()
                        .map(|value| value as f64)
                        .collect::<Vec<_>>();
                    if collected.is_empty() {
                        Ok(None)
                    } else {
                        stats(collected).map(Some)
                    }
                };
            Ok(Stage19BlockAggregate {
                block_target_size,
                samples: group.len(),
                block_count: stats(group.iter().map(|s| s.block_count as f64).collect())?,
                active_cycles: stats(group.iter().map(|s| s.active_cycles as f64).collect())?,
                native_jolt_prove_micros: stats(
                    group
                        .iter()
                        .map(|s| s.timings_micros.native_jolt_prove as f64)
                        .collect(),
                )?,
                recursive_extension_micros: stats(
                    group
                        .iter()
                        .map(|s| s.timings_micros.recursive_extension() as f64)
                        .collect(),
                )?,
                measured_total_micros: stats(
                    group
                        .iter()
                        .map(|s| s.timings_micros.measured_total as f64)
                        .collect(),
                )?,
                throughput_active_cycles_per_second: stats(
                    group
                        .iter()
                        .map(|s| s.throughput_active_cycles_per_second)
                        .collect(),
                )?,
                native_jolt_peak_delta_bytes: optional_stats(
                    group
                        .iter()
                        .map(|s| s.memory_bytes.native_jolt_peak_delta)
                        .collect(),
                )?,
                recursive_peak_delta_bytes: optional_stats(
                    group
                        .iter()
                        .map(|s| s.memory_bytes.recursive_peak_delta)
                        .collect(),
                )?,
                native_jolt_proof_bytes: stats(
                    group
                        .iter()
                        .map(|s| s.proof_sizes.native_jolt_proof as f64)
                        .collect(),
                )?,
                recursive_proof_payload_bytes: stats(
                    group
                        .iter()
                        .map(|s| s.proof_sizes.recursive_proof_payload as f64)
                        .collect(),
                )?,
            })
        })
        .collect()
}

fn approximately_equal(left: f64, right: f64) -> bool {
    (left - right).abs() <= left.abs().max(right.abs()).max(1.0) * 1e-12
}

fn summary_matches(left: &Stage19SummaryStatistics, right: &Stage19SummaryStatistics) -> bool {
    left.count == right.count
        && approximately_equal(left.min, right.min)
        && approximately_equal(left.median, right.median)
        && approximately_equal(left.mean, right.mean)
        && approximately_equal(left.max, right.max)
        && approximately_equal(left.standard_deviation, right.standard_deviation)
}

fn optional_summary_matches(
    left: &Option<Stage19SummaryStatistics>,
    right: &Option<Stage19SummaryStatistics>,
) -> bool {
    match (left, right) {
        (Some(left), Some(right)) => summary_matches(left, right),
        (None, None) => true,
        _ => false,
    }
}

fn aggregate_matches(left: &Stage19BlockAggregate, right: &Stage19BlockAggregate) -> bool {
    left.block_target_size == right.block_target_size
        && left.samples == right.samples
        && summary_matches(&left.block_count, &right.block_count)
        && summary_matches(&left.active_cycles, &right.active_cycles)
        && summary_matches(
            &left.native_jolt_prove_micros,
            &right.native_jolt_prove_micros,
        )
        && summary_matches(
            &left.recursive_extension_micros,
            &right.recursive_extension_micros,
        )
        && summary_matches(&left.measured_total_micros, &right.measured_total_micros)
        && summary_matches(
            &left.throughput_active_cycles_per_second,
            &right.throughput_active_cycles_per_second,
        )
        && optional_summary_matches(
            &left.native_jolt_peak_delta_bytes,
            &right.native_jolt_peak_delta_bytes,
        )
        && optional_summary_matches(
            &left.recursive_peak_delta_bytes,
            &right.recursive_peak_delta_bytes,
        )
        && summary_matches(
            &left.native_jolt_proof_bytes,
            &right.native_jolt_proof_bytes,
        )
        && summary_matches(
            &left.recursive_proof_payload_bytes,
            &right.recursive_proof_payload_bytes,
        )
}

fn identify_bottleneck(samples: &[Stage19BenchmarkSample]) -> Result<Stage19Bottleneck, String> {
    if samples.is_empty() {
        return Err("cannot identify a bottleneck without samples".to_string());
    }
    let phases: [(&str, fn(&Stage19BenchmarkSample) -> u64); 5] = [
        ("native-jolt-prove", |s: &Stage19BenchmarkSample| {
            s.timings_micros.native_jolt_prove
        }),
        (
            "native-jolt-verify-and-export",
            |s: &Stage19BenchmarkSample| s.timings_micros.native_jolt_verify_and_export,
        ),
        ("nova-spartan-stream", |s: &Stage19BenchmarkSample| {
            s.timings_micros.nova_spartan_stream
        }),
        ("recursive-blindfold-prove", |s: &Stage19BenchmarkSample| {
            s.timings_micros.recursive_blindfold_prove
        }),
        ("final-end-to-end-verify", |s: &Stage19BenchmarkSample| {
            s.timings_micros.final_end_to_end_verify
        }),
    ];
    let (phase, mean_micros) = phases
        .into_iter()
        .map(|(name, value)| {
            let mean =
                samples.iter().map(value).map(|v| v as f64).sum::<f64>() / samples.len() as f64;
            (name, mean)
        })
        .max_by(|left, right| left.1.total_cmp(&right.1))
        .expect("phase set is non-empty");
    let total_mean = samples
        .iter()
        .map(|sample| sample.timings_micros.measured_total as f64)
        .sum::<f64>()
        / samples.len() as f64;
    Ok(Stage19Bottleneck {
        phase: phase.to_string(),
        mean_micros,
        percent_of_measured_total: if total_mean > 0.0 {
            mean_micros / total_mean * 100.0
        } else {
            0.0
        },
    })
}

fn regression_metric(
    block_target_size: usize,
    metric: &str,
    baseline_mean: f64,
    current_mean: f64,
    max_regression_percent: f64,
    higher_is_better: bool,
) -> Result<Stage19RegressionMetric, String> {
    if !baseline_mean.is_finite()
        || baseline_mean <= 0.0
        || !current_mean.is_finite()
        || current_mean < 0.0
    {
        return Err(format!(
            "metric {metric} has an invalid baseline or current value"
        ));
    }
    let regression_percent = if higher_is_better {
        (baseline_mean - current_mean) / baseline_mean * 100.0
    } else {
        (current_mean - baseline_mean) / baseline_mean * 100.0
    };
    Ok(Stage19RegressionMetric {
        block_target_size,
        metric: metric.to_string(),
        baseline_mean,
        current_mean,
        regression_percent,
        passed: regression_percent <= max_regression_percent,
    })
}

pub fn hex_digest(digest: [u8; 32]) -> String {
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(block_size: usize, run_index: usize, total: u64) -> Stage19BenchmarkSample {
        Stage19BenchmarkSample {
            run_index,
            workload: "fibonacci-32".to_string(),
            workload_sha3_256: "11".repeat(32),
            block_target_size: block_size,
            block_count: 4,
            active_cycles: 256,
            padded_trace_length: 256,
            max_resident_trace_blocks: 2,
            max_resident_trace_cycles: block_size * 2,
            receipt_is_production_zk: true,
            native_jolt_verified: true,
            streaming_proof_verified: true,
            complete_zk_verified: true,
            lookup_backend: JOLT_NOVA_STAGE19_LOOKUP_BACKEND.to_string(),
            receipt_digest: "22".repeat(32),
            jolt_statement_id: "33".repeat(32),
            timings_micros: Stage19TimingsMicros {
                trace: 10,
                native_jolt_prove: 100,
                native_jolt_verify_and_export: 20,
                nova_spartan_stream: 50,
                recursive_blindfold_prove: 30,
                final_end_to_end_verify: 10,
                measured_total: total,
            },
            throughput_active_cycles_per_second: 256.0 / (total as f64 / 1_000_000.0),
            memory_bytes: Stage19MemoryBytes {
                native_jolt_start: Some(100),
                native_jolt_peak: Some(200),
                native_jolt_peak_delta: Some(100),
                recursive_start: Some(200),
                recursive_peak: Some(250),
                recursive_peak_delta: Some(50),
                estimated_peak_trace: 4_096,
                peak_tracked_ram_addresses: 8,
            },
            proof_sizes: Stage19ProofSizes {
                native_jolt_proof: 2_000,
                folded_spartan_proof: 1_000,
                blindfold_spartan_proof: 500,
                recursive_proof_payload: 1_500,
            },
            relation_profiles: REQUIRED_RELATIONS
                .iter()
                .map(|relation| Stage19RelationMeasurement {
                    relation: (*relation).to_string(),
                    attribution_method: JOLT_NOVA_STAGE19_PROFILE_METHOD.to_string(),
                    calls: 1,
                    total_nanos: 10,
                    max_nanos: 10,
                })
                .collect(),
        }
    }

    #[test]
    fn stage19_builds_complete_matrix_and_markdown() {
        let artifact = Stage19BenchmarkArtifact::build(
            Stage19Platform::current("1.95"),
            "fibonacci-32",
            "11".repeat(32),
            50,
            2,
            vec![
                sample(64, 1, 220),
                sample(64, 2, 240),
                sample(256, 1, 200),
                sample(256, 2, 210),
            ],
        )
        .unwrap();
        assert_eq!(artifact.aggregates.len(), 2);
        assert_eq!(artifact.samples.len(), 4);
        assert_eq!(artifact.matrix_digest.len(), 64);
        let markdown = artifact.to_markdown().unwrap();
        assert!(markdown.contains("native Jolt prove"));
        assert!(markdown.contains("recursive peak delta"));
        assert!(markdown.contains("Measured relation attribution"));
        assert!(markdown.contains("`jolt-lasso-lookup-claim`"));
        let json = artifact.to_json_pretty().unwrap();
        assert!(json.contains(JOLT_NOVA_STAGE19_SCHEMA_VERSION));
        let round_trip: Stage19BenchmarkArtifact = serde_json::from_str(&json).unwrap();
        round_trip.validate().unwrap();
        assert_eq!(round_trip.matrix_digest, artifact.matrix_digest);
    }

    #[test]
    fn stage19_rejects_unverified_or_non_streaming_samples() {
        let mut invalid = sample(64, 1, 220);
        invalid.complete_zk_verified = false;
        assert!(invalid.validate().is_err());
        invalid.complete_zk_verified = true;
        invalid.max_resident_trace_blocks = 3;
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn stage19_rejects_estimated_incomplete_or_misaccounted_samples() {
        let mut estimated = sample(64, 1, 220);
        estimated.relation_profiles[0].attribution_method = "static-estimate".to_string();
        assert!(estimated.validate().is_err());

        let mut incomplete = sample(64, 1, 220);
        incomplete.relation_profiles.pop();
        assert!(incomplete.validate().is_err());

        let mut misaccounted = sample(64, 1, 220);
        misaccounted.proof_sizes.recursive_proof_payload += 1;
        assert!(misaccounted.validate().is_err());
    }

    #[test]
    fn stage19_regression_gate_detects_time_and_throughput_regressions() {
        let platform = Stage19Platform::current("1.95");
        let baseline = Stage19BenchmarkArtifact::build(
            platform.clone(),
            "fibonacci-32",
            "11".repeat(32),
            50,
            1,
            vec![sample(64, 1, 200)],
        )
        .unwrap();
        let current_sample = sample(64, 1, 260);
        let mut current = Stage19BenchmarkArtifact::build(
            platform,
            "fibonacci-32",
            "11".repeat(32),
            50,
            1,
            vec![current_sample],
        )
        .unwrap();
        let comparison = current.compare_with_baseline(&baseline, 10.0).unwrap();
        assert!(!comparison.passed);
        assert!(comparison.metrics.iter().any(|metric| !metric.passed));
    }

    #[test]
    fn stage19_digest_detects_sample_tampering() {
        let mut artifact = Stage19BenchmarkArtifact::build(
            Stage19Platform::current("1.95"),
            "fibonacci-32",
            "11".repeat(32),
            50,
            1,
            vec![sample(64, 1, 200)],
        )
        .unwrap();
        artifact.samples[0].active_cycles += 1;
        assert!(artifact.validate().is_err());
    }
}
