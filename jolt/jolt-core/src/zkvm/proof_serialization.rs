#[cfg(not(feature = "zk"))]
use std::collections::BTreeMap;
use std::io::{Read, Write};

use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, SerializationError, Valid, Validate,
};
use num::FromPrimitive;
use sha3::{Digest as ShaDigest, Sha3_256};
use strum::EnumCount;

#[cfg(not(feature = "zk"))]
use crate::poly::opening_proof::{OpeningPoint, Openings};
#[cfg(feature = "zk")]
use crate::subprotocols::blindfold::BlindFoldProof;
use crate::{
    curve::JoltCurve,
    field::JoltField,
    poly::{
        commitment::{commitment_scheme::CommitmentScheme, dory::DoryLayout},
        opening_proof::{OpeningId, PolynomialId, SumcheckId},
    },
    utils::errors::ProofVerifyError,
};
use crate::{
    subprotocols::{
        sumcheck::SumcheckInstanceProof, univariate_skip::UniSkipFirstRoundProofVariant,
    },
    transcripts::Transcript,
    zkvm::{
        config::{OneHotConfig, ReadWriteConfig},
        instruction::{CircuitFlags, InstructionFlags},
        witness::{CommittedPolynomial, VirtualPolynomial},
    },
};

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct JoltProof<
    F: JoltField,
    C: JoltCurve<F = F>,
    PCS: CommitmentScheme<Field = F>,
    FS: Transcript,
> {
    pub commitments: Vec<PCS::Commitment>,
    pub stage1_uni_skip_first_round_proof: UniSkipFirstRoundProofVariant<F, C, FS>,
    pub stage1_sumcheck_proof: SumcheckInstanceProof<F, C, FS>,
    pub stage2_uni_skip_first_round_proof: UniSkipFirstRoundProofVariant<F, C, FS>,
    pub stage2_sumcheck_proof: SumcheckInstanceProof<F, C, FS>,
    pub stage3_sumcheck_proof: SumcheckInstanceProof<F, C, FS>,
    pub stage4_sumcheck_proof: SumcheckInstanceProof<F, C, FS>,
    pub stage5_sumcheck_proof: SumcheckInstanceProof<F, C, FS>,
    pub stage6a_sumcheck_proof: SumcheckInstanceProof<F, C, FS>,
    pub stage6b_sumcheck_proof: SumcheckInstanceProof<F, C, FS>,
    pub stage7_sumcheck_proof: SumcheckInstanceProof<F, C, FS>,
    #[cfg(feature = "zk")]
    pub blindfold_proof: BlindFoldProof<F, C>,
    pub joint_opening_proof: PCS::Proof,
    pub untrusted_advice_commitment: Option<PCS::Commitment>,
    #[cfg(not(feature = "zk"))]
    pub opening_claims: Claims<F>,
    pub trace_length: usize,
    pub ram_K: usize,
    pub rw_config: ReadWriteConfig,
    pub one_hot_config: OneHotConfig,
    pub dory_layout: DoryLayout,
}

/// A compact commitment to the verifier transcript of a fully verified Jolt
/// proof.
///
/// This receipt is deliberately not constructible by downstream callers. It
/// is returned by `JoltVerifier::verify_with_lookup_receipt` only after the
/// complete Jolt verifier has accepted all sumchecks and the joint PCS
/// opening. Nova can bind this fixed-size capsule into every recursive step
/// without carrying the variable-size Jolt proof as circuit witness. The
/// historical lookup-oriented type name is retained for API compatibility,
/// but version 2 binds the complete clear verifier transcript, not only the
/// lookup-specific reductions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedJoltLookupProofReceipt {
    version: u16,
    trace_length: u64,
    commitment_count: u64,
    zk_mode: bool,
    blindfold_proof_digest: [u8; 32],
    preprocessing_digest: [u8; 32],
    verifier_setup_digest: [u8; 32],
    public_io_digest: [u8; 32],
    trusted_advice_commitment_digest: [u8; 32],
    commitments_digest: [u8; 32],
    stage1_uni_skip_first_round_proof_digest: [u8; 32],
    stage1_sumcheck_digest: [u8; 32],
    stage2_uni_skip_first_round_proof_digest: [u8; 32],
    stage2_sumcheck_digest: [u8; 32],
    stage3_sumcheck_digest: [u8; 32],
    stage4_sumcheck_digest: [u8; 32],
    stage5_sumcheck_digest: [u8; 32],
    stage6a_sumcheck_digest: [u8; 32],
    stage6b_sumcheck_digest: [u8; 32],
    stage7_sumcheck_digest: [u8; 32],
    joint_opening_proof_digest: [u8; 32],
    full_proof_digest: [u8; 32],
    receipt_digest: [u8; 32],
}

impl VerifiedJoltLookupProofReceipt {
    pub const VERSION: u16 = 3;
    pub const VERIFIER_STAGE_RELATION_COUNT: usize = 11;

    pub fn trace_length(&self) -> usize {
        self.trace_length as usize
    }

    pub fn commitment_count(&self) -> usize {
        self.commitment_count as usize
    }

    pub fn zk_mode(&self) -> bool {
        self.zk_mode
    }

    pub fn blindfold_proof_digest(&self) -> [u8; 32] {
        self.blindfold_proof_digest
    }

    pub fn preprocessing_digest(&self) -> [u8; 32] {
        self.preprocessing_digest
    }

    pub fn verifier_setup_digest(&self) -> [u8; 32] {
        self.verifier_setup_digest
    }

    pub fn public_io_digest(&self) -> [u8; 32] {
        self.public_io_digest
    }

    pub fn commitments_digest(&self) -> [u8; 32] {
        self.commitments_digest
    }

    pub fn stage1_uni_skip_first_round_proof_digest(&self) -> [u8; 32] {
        self.stage1_uni_skip_first_round_proof_digest
    }

    pub fn stage1_sumcheck_digest(&self) -> [u8; 32] {
        self.stage1_sumcheck_digest
    }

    pub fn stage2_uni_skip_first_round_proof_digest(&self) -> [u8; 32] {
        self.stage2_uni_skip_first_round_proof_digest
    }

    pub fn stage2_sumcheck_digest(&self) -> [u8; 32] {
        self.stage2_sumcheck_digest
    }

    pub fn stage3_sumcheck_digest(&self) -> [u8; 32] {
        self.stage3_sumcheck_digest
    }

    pub fn stage4_sumcheck_digest(&self) -> [u8; 32] {
        self.stage4_sumcheck_digest
    }

    pub fn stage5_sumcheck_digest(&self) -> [u8; 32] {
        self.stage5_sumcheck_digest
    }

    pub fn stage6a_sumcheck_digest(&self) -> [u8; 32] {
        self.stage6a_sumcheck_digest
    }

    pub fn stage6b_sumcheck_digest(&self) -> [u8; 32] {
        self.stage6b_sumcheck_digest
    }

    pub fn stage7_sumcheck_digest(&self) -> [u8; 32] {
        self.stage7_sumcheck_digest
    }

    pub fn joint_opening_proof_digest(&self) -> [u8; 32] {
        self.joint_opening_proof_digest
    }

    pub fn full_proof_digest(&self) -> [u8; 32] {
        self.full_proof_digest
    }

    pub fn digest(&self) -> [u8; 32] {
        self.receipt_digest
    }

    pub fn verifier_stage_relation_count(&self) -> usize {
        Self::VERIFIER_STAGE_RELATION_COUNT
    }

    pub fn verifier_stage_relation_digest(&self) -> [u8; 32] {
        let mut hasher = Sha3_256::new();
        hasher.update(b"jolt-nova/verifier-stage-relation-capsule/v2");
        hasher.update(self.version.to_le_bytes());
        hasher.update(self.trace_length.to_le_bytes());
        hasher.update(self.commitment_count.to_le_bytes());
        hasher.update([u8::from(self.zk_mode)]);
        hasher.update(self.blindfold_proof_digest);
        hasher.update(self.preprocessing_digest);
        hasher.update(self.verifier_setup_digest);
        hasher.update(self.public_io_digest);
        hasher.update(self.trusted_advice_commitment_digest);
        hasher.update(self.commitments_digest);
        hasher.update((Self::VERIFIER_STAGE_RELATION_COUNT as u64).to_le_bytes());
        hasher.update(self.stage1_uni_skip_first_round_proof_digest);
        hasher.update(self.stage1_sumcheck_digest);
        hasher.update(self.stage2_uni_skip_first_round_proof_digest);
        hasher.update(self.stage2_sumcheck_digest);
        hasher.update(self.stage3_sumcheck_digest);
        hasher.update(self.stage4_sumcheck_digest);
        hasher.update(self.stage5_sumcheck_digest);
        hasher.update(self.stage6a_sumcheck_digest);
        hasher.update(self.stage6b_sumcheck_digest);
        hasher.update(self.stage7_sumcheck_digest);
        hasher.update(self.joint_opening_proof_digest);
        hasher.finalize().into()
    }

    /// Returns the fixed-width verifier transcript components used by the
    /// recursive capsule prototype. The first component binds the verifier
    /// preamble and commitments; the remaining components bind the clear
    /// verifier stages, the joint opening, and the complete proof encoding.
    pub fn recursive_transcript_stage_digests(&self) -> [[u8; 32]; 11] {
        [
            digest_bytes(
                b"jolt-nova/verifier-transcript-preamble/v2",
                &[
                    self.proof_mode_digest(),
                    self.preprocessing_digest,
                    self.verifier_setup_digest,
                    self.public_io_digest,
                    self.trusted_advice_commitment_digest,
                    self.commitments_digest,
                ],
            ),
            digest_bytes(
                b"jolt-nova/verifier-transcript-stage-1/v1",
                &[
                    self.stage1_uni_skip_first_round_proof_digest,
                    self.stage1_sumcheck_digest,
                ],
            ),
            digest_bytes(
                b"jolt-nova/verifier-transcript-stage-2/v1",
                &[
                    self.stage2_uni_skip_first_round_proof_digest,
                    self.stage2_sumcheck_digest,
                ],
            ),
            self.stage3_sumcheck_digest,
            self.stage4_sumcheck_digest,
            self.stage5_sumcheck_digest,
            self.stage6a_sumcheck_digest,
            self.stage6b_sumcheck_digest,
            self.stage7_sumcheck_digest,
            self.joint_opening_proof_digest,
            self.full_proof_digest,
        ]
    }

    /// Builds the smallest useful verifier-stage claim for the staged
    /// internalization path. The claim is still produced from the host-side
    /// accepted receipt; the next stages will replace this digest check with
    /// an in-circuit verifier relation one stage at a time.
    pub fn verifier_stage_internalization_claim(
        &self,
        stage_index: usize,
    ) -> Option<VerifierStageInternalizationClaim> {
        let stage_digests = self.recursive_transcript_stage_digests();
        let stage_digest = stage_digests.get(stage_index)?;
        Some(VerifierStageInternalizationClaim::from_parts(
            stage_index,
            *stage_digest,
            self.verifier_stage_relation_digest(),
            self.verifier_stage_relation_count(),
        ))
    }

    /// Converts the accepted receipt into the fixed-width recursive capsule
    /// used by Stage 9.14.
    pub fn recursive_transcript_capsule(&self) -> RecursiveVerifierTranscriptCapsule {
        RecursiveVerifierTranscriptCapsule::from_receipt(self)
    }

    /// Returns a fixed-width privacy-preserving BlindFold receipt when the
    /// accepted proof uses Jolt's ZK proof format.
    pub fn blindfold_receipt(&self) -> Option<VerifiedJoltBlindFoldReceipt> {
        VerifiedJoltBlindFoldReceipt::from_lookup_receipt(self)
    }

    #[cfg(test)]
    pub(crate) fn new_for_test(seed: u8, trace_length: usize) -> Self {
        let digest = |domain: u8| [seed.wrapping_add(domain); 32];
        let mut receipt = Self {
            version: Self::VERSION,
            trace_length: trace_length as u64,
            commitment_count: 4,
            zk_mode: false,
            blindfold_proof_digest: [0; 32],
            preprocessing_digest: digest(1),
            verifier_setup_digest: digest(2),
            public_io_digest: digest(3),
            trusted_advice_commitment_digest: digest(4),
            commitments_digest: digest(5),
            stage1_uni_skip_first_round_proof_digest: digest(6),
            stage1_sumcheck_digest: digest(7),
            stage2_uni_skip_first_round_proof_digest: digest(8),
            stage2_sumcheck_digest: digest(9),
            stage3_sumcheck_digest: digest(10),
            stage4_sumcheck_digest: digest(11),
            stage5_sumcheck_digest: digest(12),
            stage6a_sumcheck_digest: digest(13),
            stage6b_sumcheck_digest: digest(14),
            stage7_sumcheck_digest: digest(15),
            joint_opening_proof_digest: digest(16),
            full_proof_digest: digest(17),
            receipt_digest: [0; 32],
        };
        receipt.receipt_digest = receipt.compute_digest();
        receipt
    }

    #[cfg(test)]
    pub(crate) fn new_zk_for_test(seed: u8, trace_length: usize) -> Self {
        let mut receipt = Self::new_for_test(seed, trace_length);
        receipt.zk_mode = true;
        receipt.blindfold_proof_digest = [seed.wrapping_add(18); 32];
        receipt.receipt_digest = receipt.compute_digest();
        receipt
    }

    fn compute_digest(&self) -> [u8; 32] {
        let mut hasher = Sha3_256::new();
        hasher.update(b"jolt-nova/verified-jolt-verifier-transcript-receipt/v3");
        hasher.update(self.version.to_le_bytes());
        hasher.update(self.trace_length.to_le_bytes());
        hasher.update(self.commitment_count.to_le_bytes());
        hasher.update([u8::from(self.zk_mode)]);
        hasher.update(self.blindfold_proof_digest);
        hasher.update(self.preprocessing_digest);
        hasher.update(self.verifier_setup_digest);
        hasher.update(self.public_io_digest);
        hasher.update(self.trusted_advice_commitment_digest);
        hasher.update(self.commitments_digest);
        hasher.update(self.stage1_uni_skip_first_round_proof_digest);
        hasher.update(self.stage1_sumcheck_digest);
        hasher.update(self.stage2_uni_skip_first_round_proof_digest);
        hasher.update(self.stage2_sumcheck_digest);
        hasher.update(self.stage3_sumcheck_digest);
        hasher.update(self.stage4_sumcheck_digest);
        hasher.update(self.stage5_sumcheck_digest);
        hasher.update(self.stage6a_sumcheck_digest);
        hasher.update(self.stage6b_sumcheck_digest);
        hasher.update(self.stage7_sumcheck_digest);
        hasher.update(self.joint_opening_proof_digest);
        hasher.update(self.full_proof_digest);
        hasher.finalize().into()
    }

    fn proof_mode_digest(&self) -> [u8; 32] {
        let mut hasher = Sha3_256::new();
        hasher.update(b"jolt-nova/verified-jolt-proof-mode/v1");
        hasher.update([u8::from(self.zk_mode)]);
        hasher.update(self.blindfold_proof_digest);
        hasher.finalize().into()
    }
}

fn digest_bytes(domain: &'static [u8], values: &[[u8; 32]]) -> [u8; 32] {
    let mut hasher = Sha3_256::new();
    hasher.update(domain);
    hasher.update((values.len() as u64).to_le_bytes());
    for value in values {
        hasher.update(value);
    }
    hasher.finalize().into()
}

/// A fixed-width claim for one verifier stage.
///
/// This is the Stage 9.13 seam: a Nova step can carry one selected verifier
/// stage as an explicit relation claim without carrying the variable-size Jolt
/// proof. It deliberately does not claim to re-run BN254/Dory verification.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VerifierStageInternalizationClaim {
    version: u16,
    stage_index: u16,
    stage_digest: [u8; 32],
    verifier_stage_relation_digest: [u8; 32],
    verifier_stage_relation_count: u16,
    claim_digest: [u8; 32],
}

impl VerifierStageInternalizationClaim {
    pub const VERSION: u16 = 1;

    fn from_parts(
        stage_index: usize,
        stage_digest: [u8; 32],
        verifier_stage_relation_digest: [u8; 32],
        verifier_stage_relation_count: usize,
    ) -> Self {
        let mut claim = Self {
            version: Self::VERSION,
            stage_index: stage_index as u16,
            stage_digest,
            verifier_stage_relation_digest,
            verifier_stage_relation_count: verifier_stage_relation_count as u16,
            claim_digest: [0; 32],
        };
        claim.claim_digest = claim.compute_digest();
        claim
    }

    pub fn stage_index(&self) -> usize {
        self.stage_index as usize
    }

    pub fn stage_digest(&self) -> [u8; 32] {
        self.stage_digest
    }

    pub fn verifier_stage_relation_digest(&self) -> [u8; 32] {
        self.verifier_stage_relation_digest
    }

    pub fn verifier_stage_relation_count(&self) -> usize {
        self.verifier_stage_relation_count as usize
    }

    pub fn digest(&self) -> [u8; 32] {
        self.claim_digest
    }

    /// Checks the fixed-width claim against the accepted receipt and its
    /// deterministic claim digest.
    pub fn verify_against(&self, receipt: &VerifiedJoltLookupProofReceipt) -> bool {
        receipt
            .verifier_stage_internalization_claim(self.stage_index())
            .map(|expected| expected == *self)
            .unwrap_or(false)
    }

    fn compute_digest(&self) -> [u8; 32] {
        let mut hasher = Sha3_256::new();
        hasher.update(b"jolt-nova/verifier-stage-internalization-claim/v1");
        hasher.update(self.version.to_le_bytes());
        hasher.update(self.stage_index.to_le_bytes());
        hasher.update(self.stage_digest);
        hasher.update(self.verifier_stage_relation_digest);
        hasher.update(self.verifier_stage_relation_count.to_le_bytes());
        hasher.finalize().into()
    }
}

/// A fixed-width, append-only transcript accumulator suitable for recursive
/// folding. Each `absorb_stage` transition hashes the previous root together
/// with the next stage index and digest. A complete capsule can be checked
/// against a host-verified receipt, while an incomplete capsule can be passed
/// through intermediate recursive steps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RecursiveVerifierTranscriptCapsule {
    version: u16,
    relation_digest: [u8; 32],
    stage_count: u16,
    absorbed_stage_count: u16,
    transcript_root: [u8; 32],
}

impl RecursiveVerifierTranscriptCapsule {
    pub const VERSION: u16 = 1;

    pub fn new(relation_digest: [u8; 32], stage_count: usize) -> Self {
        let mut hasher = Sha3_256::new();
        hasher.update(b"jolt-nova/recursive-verifier-transcript/v1");
        hasher.update(Self::VERSION.to_le_bytes());
        hasher.update(relation_digest);
        hasher.update((stage_count as u16).to_le_bytes());
        hasher.update(0u16.to_le_bytes());
        Self {
            version: Self::VERSION,
            relation_digest,
            stage_count: stage_count as u16,
            absorbed_stage_count: 0,
            transcript_root: hasher.finalize().into(),
        }
    }

    pub fn from_receipt(receipt: &VerifiedJoltLookupProofReceipt) -> Self {
        let mut capsule = Self::new(
            receipt.verifier_stage_relation_digest(),
            receipt.verifier_stage_relation_count(),
        );
        for (stage_index, stage_digest) in receipt
            .recursive_transcript_stage_digests()
            .into_iter()
            .enumerate()
        {
            capsule = capsule
                .absorb_stage(stage_index, stage_digest)
                .expect("receipt transcript stages must match the fixed capsule width");
        }
        capsule
    }

    pub fn relation_digest(&self) -> [u8; 32] {
        self.relation_digest
    }

    pub fn stage_count(&self) -> usize {
        self.stage_count as usize
    }

    pub fn absorbed_stage_count(&self) -> usize {
        self.absorbed_stage_count as usize
    }

    pub fn transcript_root(&self) -> [u8; 32] {
        self.transcript_root
    }

    pub fn is_complete(&self) -> bool {
        self.absorbed_stage_count == self.stage_count
    }

    pub fn absorb_stage(
        mut self,
        stage_index: usize,
        stage_digest: [u8; 32],
    ) -> Result<Self, &'static str> {
        if stage_index != self.absorbed_stage_count as usize {
            return Err("recursive verifier stages must be absorbed in order");
        }
        if self.is_complete() {
            return Err("recursive verifier transcript capsule is already complete");
        }

        let mut hasher = Sha3_256::new();
        hasher.update(b"jolt-nova/recursive-verifier-transcript/step/v1");
        hasher.update(self.version.to_le_bytes());
        hasher.update(self.relation_digest);
        hasher.update(self.transcript_root);
        hasher.update((stage_index as u16).to_le_bytes());
        hasher.update(stage_digest);
        self.transcript_root = hasher.finalize().into();
        self.absorbed_stage_count += 1;
        Ok(self)
    }

    pub fn verify_against(&self, receipt: &VerifiedJoltLookupProofReceipt) -> bool {
        self.is_complete() && *self == Self::from_receipt(receipt)
    }
}

/// A fixed-width receipt for Jolt's BlindFold/ZK verifier path.
///
/// It binds to an accepted `VerifiedJoltLookupProofReceipt` and to the
/// BlindFold proof digest without exposing hidden opening claims. This is the
/// Stage 9.15 privacy-preserving counterpart to the non-ZK opening-heavy
/// receipt surface.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VerifiedJoltBlindFoldReceipt {
    version: u16,
    trace_length: u64,
    commitment_count: u64,
    lookup_receipt_digest: [u8; 32],
    verifier_stage_relation_digest: [u8; 32],
    recursive_transcript_root: [u8; 32],
    blindfold_proof_digest: [u8; 32],
    receipt_digest: [u8; 32],
}

impl VerifiedJoltBlindFoldReceipt {
    pub const VERSION: u16 = 1;

    pub fn from_lookup_receipt(receipt: &VerifiedJoltLookupProofReceipt) -> Option<Self> {
        if !receipt.zk_mode() {
            return None;
        }

        let capsule = receipt.recursive_transcript_capsule();
        let mut blindfold_receipt = Self {
            version: Self::VERSION,
            trace_length: receipt.trace_length as u64,
            commitment_count: receipt.commitment_count as u64,
            lookup_receipt_digest: receipt.digest(),
            verifier_stage_relation_digest: receipt.verifier_stage_relation_digest(),
            recursive_transcript_root: capsule.transcript_root(),
            blindfold_proof_digest: receipt.blindfold_proof_digest(),
            receipt_digest: [0; 32],
        };
        blindfold_receipt.receipt_digest = blindfold_receipt.compute_digest();
        Some(blindfold_receipt)
    }

    pub fn trace_length(&self) -> usize {
        self.trace_length as usize
    }

    pub fn commitment_count(&self) -> usize {
        self.commitment_count as usize
    }

    pub fn lookup_receipt_digest(&self) -> [u8; 32] {
        self.lookup_receipt_digest
    }

    pub fn verifier_stage_relation_digest(&self) -> [u8; 32] {
        self.verifier_stage_relation_digest
    }

    pub fn recursive_transcript_root(&self) -> [u8; 32] {
        self.recursive_transcript_root
    }

    pub fn blindfold_proof_digest(&self) -> [u8; 32] {
        self.blindfold_proof_digest
    }

    pub fn digest(&self) -> [u8; 32] {
        self.receipt_digest
    }

    pub fn verify_against(&self, receipt: &VerifiedJoltLookupProofReceipt) -> bool {
        Self::from_lookup_receipt(receipt)
            .map(|expected| expected == *self)
            .unwrap_or(false)
    }

    fn compute_digest(&self) -> [u8; 32] {
        let mut hasher = Sha3_256::new();
        hasher.update(b"jolt-nova/verified-jolt-blindfold-receipt/v1");
        hasher.update(self.version.to_le_bytes());
        hasher.update(self.trace_length.to_le_bytes());
        hasher.update(self.commitment_count.to_le_bytes());
        hasher.update(self.lookup_receipt_digest);
        hasher.update(self.verifier_stage_relation_digest);
        hasher.update(self.recursive_transcript_root);
        hasher.update(self.blindfold_proof_digest);
        hasher.finalize().into()
    }
}

#[cfg(test)]
mod tests {
    use super::{
        RecursiveVerifierTranscriptCapsule, VerifiedJoltBlindFoldReceipt,
        VerifiedJoltLookupProofReceipt,
    };

    #[test]
    fn verified_jolt_lookup_receipt_captures_full_verifier_transcript() {
        let receipt = VerifiedJoltLookupProofReceipt::new_for_test(7, 16);
        assert_eq!(receipt.version, VerifiedJoltLookupProofReceipt::VERSION);
        assert!(!receipt.zk_mode());
        assert_eq!(receipt.blindfold_proof_digest(), [0; 32]);
        assert!(receipt.blindfold_receipt().is_none());
        assert_ne!(receipt.stage1_uni_skip_first_round_proof_digest(), [0; 32]);
        assert_ne!(receipt.stage1_sumcheck_digest(), [0; 32]);
        assert_ne!(receipt.stage2_uni_skip_first_round_proof_digest(), [0; 32]);
        assert_ne!(receipt.stage3_sumcheck_digest(), [0; 32]);
        assert_ne!(receipt.stage4_sumcheck_digest(), [0; 32]);
        assert_ne!(receipt.stage6a_sumcheck_digest(), [0; 32]);
        assert_ne!(receipt.stage7_sumcheck_digest(), [0; 32]);
        assert_eq!(
            receipt.verifier_stage_relation_count(),
            VerifiedJoltLookupProofReceipt::VERIFIER_STAGE_RELATION_COUNT
        );
        assert_ne!(receipt.verifier_stage_relation_digest(), [0; 32]);
        assert_ne!(receipt.digest(), [0; 32]);
    }

    #[test]
    fn verified_jolt_blindfold_receipt_binds_zk_proof_path_without_openings() {
        let receipt = VerifiedJoltLookupProofReceipt::new_zk_for_test(19, 32);
        assert!(receipt.zk_mode());
        assert_ne!(receipt.blindfold_proof_digest(), [0; 32]);

        let blindfold_receipt = receipt
            .blindfold_receipt()
            .expect("ZK receipts expose a BlindFold receipt capsule");
        assert_eq!(blindfold_receipt.lookup_receipt_digest(), receipt.digest());
        assert_eq!(
            blindfold_receipt.verifier_stage_relation_digest(),
            receipt.verifier_stage_relation_digest()
        );
        assert_eq!(
            blindfold_receipt.recursive_transcript_root(),
            receipt.recursive_transcript_capsule().transcript_root()
        );
        assert_eq!(
            blindfold_receipt.blindfold_proof_digest(),
            receipt.blindfold_proof_digest()
        );
        assert!(blindfold_receipt.verify_against(&receipt));
        assert_ne!(blindfold_receipt.digest(), [0; 32]);

        let mut tampered = blindfold_receipt;
        tampered.blindfold_proof_digest[0] ^= 1;
        assert!(!tampered.verify_against(&receipt));
        assert!(VerifiedJoltBlindFoldReceipt::from_lookup_receipt(
            &VerifiedJoltLookupProofReceipt::new_for_test(19, 32)
        )
        .is_none());
    }

    #[test]
    fn verifier_stage_internalization_claim_is_bound_to_receipt() {
        let receipt = VerifiedJoltLookupProofReceipt::new_for_test(31, 32);
        let claim = receipt
            .verifier_stage_internalization_claim(8)
            .expect("stage 8 is inside the fixed transcript capsule");

        assert_eq!(claim.stage_index(), 8);
        assert!(claim.verify_against(&receipt));
        assert_ne!(claim.digest(), [0; 32]);

        let mut tampered = claim;
        tampered.stage_digest[0] ^= 1;
        assert!(!tampered.verify_against(&receipt));
    }

    #[test]
    fn recursive_verifier_transcript_capsule_absorbs_stages_in_order() {
        let receipt = VerifiedJoltLookupProofReceipt::new_for_test(47, 64);
        let capsule = receipt.recursive_transcript_capsule();

        assert_eq!(capsule.stage_count(), 11);
        assert_eq!(capsule.absorbed_stage_count(), 11);
        assert!(capsule.is_complete());
        assert!(capsule.verify_against(&receipt));

        let mut partial = RecursiveVerifierTranscriptCapsule::new(
            receipt.verifier_stage_relation_digest(),
            receipt.verifier_stage_relation_count(),
        );
        let digests = receipt.recursive_transcript_stage_digests();
        assert!(partial.absorb_stage(1, digests[0]).is_err());
        for (stage_index, stage_digest) in digests.into_iter().enumerate() {
            partial = partial.absorb_stage(stage_index, stage_digest).unwrap();
        }
        assert_eq!(partial, capsule);
    }
}

/// The instruction-lookup, register, and RAM openings authenticated by a
/// complete Jolt verification.
///
/// For every committed `InstructionRa(i)` polynomial, this receipt retains the
/// opening at `HammingWeightClaimReduction`. Those claims are part of Jolt's
/// joint PCS opening. It also retains the `LeftLookupOperand`,
/// `RightLookupOperand`, and `LookupOutput` virtual-polynomial claims at
/// `InstructionClaimReduction`. The complete verifier constrains those three
/// claims through the instruction claim-reduction, Read/RAF, Spartan, and R1CS
/// chains before reducing the proof to the joint PCS opening.
///
/// The register side retains the shared cycle opening and claims for
/// `Rs1Value`, `Rs2Value`, and `RdWriteValue`; the shared address-cycle opening
/// and claims for `Rs1Ra`, `Rs2Ra`, and `RdWa`; and the final committed `RdInc`
/// opening produced by `IncClaimReduction`.
///
/// The RAM side retains all committed `RamRa(i)` openings, the shared Spartan
/// opening and claims for `RamAddress`, `RamReadValue`, and `RamWriteValue`,
/// and the final committed `RamInc` opening produced by `IncClaimReduction`.
///
/// The CPU/R1CS side retains the shared Spartan outer opening point and one
/// authenticated claim for each entry in `ALL_R1CS_INPUTS`.
///
/// The opening point and claims remain private: callers can only obtain this
/// type from `JoltVerifier::verify_with_lookup_opening_receipt`.
#[cfg(not(feature = "zk"))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedJoltLookupOpeningReceipt<F: JoltField> {
    version: u16,
    lookup_receipt: VerifiedJoltLookupProofReceipt,
    log_k_chunk: u8,
    opening_points: Vec<Vec<F::Challenge>>,
    opening_claims: Vec<F>,
    tuple_opening_point: Vec<F::Challenge>,
    left_lookup_operand_claim: F,
    right_lookup_operand_claim: F,
    lookup_output_claim: F,
    register_value_opening_point: Vec<F::Challenge>,
    rs1_value_claim: F,
    rs2_value_claim: F,
    rd_write_value_claim: F,
    register_address_opening_point: Vec<F::Challenge>,
    rs1_ra_claim: F,
    rs2_ra_claim: F,
    rd_wa_claim: F,
    rd_inc_opening_point: Vec<F::Challenge>,
    rd_inc_claim: F,
    ram_start_address: u64,
    ram_k: usize,
    ram_opening_points: Vec<Vec<F::Challenge>>,
    ram_opening_claims: Vec<F>,
    ram_tuple_opening_point: Vec<F::Challenge>,
    ram_address_claim: F,
    ram_read_value_claim: F,
    ram_write_value_claim: F,
    ram_inc_opening_point: Vec<F::Challenge>,
    ram_inc_claim: F,
    cpu_opening_point: Vec<F::Challenge>,
    cpu_opening_claims: Vec<F>,
    receipt_digest: [u8; 32],
}

#[cfg(not(feature = "zk"))]
impl<F: JoltField> VerifiedJoltLookupOpeningReceipt<F> {
    pub const VERSION: u16 = 5;

    pub fn lookup_receipt(&self) -> &VerifiedJoltLookupProofReceipt {
        &self.lookup_receipt
    }

    pub fn trace_length(&self) -> usize {
        self.lookup_receipt.trace_length()
    }

    pub fn instruction_opening_count(&self) -> usize {
        self.opening_claims.len()
    }

    pub fn authenticated_opening_count(&self) -> usize {
        self.opening_claims.len()
            + 3
            + self.register_opening_count()
            + self.ram_opening_count()
            + self.cpu_opening_count()
    }

    pub fn register_opening_count(&self) -> usize {
        7
    }

    pub fn ram_opening_count(&self) -> usize {
        self.ram_opening_claims.len() + 4
    }

    pub fn cpu_opening_count(&self) -> usize {
        self.cpu_opening_claims.len()
    }

    pub fn digest(&self) -> [u8; 32] {
        self.receipt_digest
    }

    pub(crate) fn log_k_chunk(&self) -> usize {
        self.log_k_chunk as usize
    }

    pub(crate) fn opening_points(&self) -> &[Vec<F::Challenge>] {
        &self.opening_points
    }

    pub(crate) fn opening_claims(&self) -> &[F] {
        &self.opening_claims
    }

    pub(crate) fn tuple_opening_point(&self) -> &[F::Challenge] {
        &self.tuple_opening_point
    }

    pub(crate) fn tuple_opening_claims(&self) -> [F; 3] {
        [
            self.left_lookup_operand_claim,
            self.right_lookup_operand_claim,
            self.lookup_output_claim,
        ]
    }

    pub(crate) fn register_value_opening_point(&self) -> &[F::Challenge] {
        &self.register_value_opening_point
    }

    pub(crate) fn register_value_opening_claims(&self) -> [F; 3] {
        [
            self.rs1_value_claim,
            self.rs2_value_claim,
            self.rd_write_value_claim,
        ]
    }

    pub(crate) fn register_address_opening_point(&self) -> &[F::Challenge] {
        &self.register_address_opening_point
    }

    pub(crate) fn register_address_opening_claims(&self) -> [F; 3] {
        [self.rs1_ra_claim, self.rs2_ra_claim, self.rd_wa_claim]
    }

    pub(crate) fn rd_inc_opening(&self) -> (&[F::Challenge], F) {
        (&self.rd_inc_opening_point, self.rd_inc_claim)
    }

    pub(crate) fn ram_start_address(&self) -> u64 {
        self.ram_start_address
    }

    pub(crate) fn ram_k(&self) -> usize {
        self.ram_k
    }

    pub(crate) fn ram_opening_points(&self) -> &[Vec<F::Challenge>] {
        &self.ram_opening_points
    }

    pub(crate) fn ram_opening_claims(&self) -> &[F] {
        &self.ram_opening_claims
    }

    pub(crate) fn ram_tuple_opening_point(&self) -> &[F::Challenge] {
        &self.ram_tuple_opening_point
    }

    pub(crate) fn ram_tuple_opening_claims(&self) -> [F; 3] {
        [
            self.ram_address_claim,
            self.ram_read_value_claim,
            self.ram_write_value_claim,
        ]
    }

    pub(crate) fn ram_inc_opening(&self) -> (&[F::Challenge], F) {
        (&self.ram_inc_opening_point, self.ram_inc_claim)
    }

    pub(crate) fn cpu_opening_point(&self) -> &[F::Challenge] {
        &self.cpu_opening_point
    }

    pub(crate) fn cpu_opening_claims(&self) -> &[F] {
        &self.cpu_opening_claims
    }

    pub(crate) fn from_verified_openings(
        lookup_receipt: VerifiedJoltLookupProofReceipt,
        log_k_chunk: usize,
        opening_points: Vec<Vec<F::Challenge>>,
        opening_claims: Vec<F>,
        tuple_opening_point: Vec<F::Challenge>,
        left_lookup_operand_claim: F,
        right_lookup_operand_claim: F,
        lookup_output_claim: F,
        register_value_opening_point: Vec<F::Challenge>,
        rs1_value_claim: F,
        rs2_value_claim: F,
        rd_write_value_claim: F,
        register_address_opening_point: Vec<F::Challenge>,
        rs1_ra_claim: F,
        rs2_ra_claim: F,
        rd_wa_claim: F,
        rd_inc_opening_point: Vec<F::Challenge>,
        rd_inc_claim: F,
        ram_start_address: u64,
        ram_k: usize,
        ram_opening_points: Vec<Vec<F::Challenge>>,
        ram_opening_claims: Vec<F>,
        ram_tuple_opening_point: Vec<F::Challenge>,
        ram_address_claim: F,
        ram_read_value_claim: F,
        ram_write_value_claim: F,
        ram_inc_opening_point: Vec<F::Challenge>,
        ram_inc_claim: F,
        cpu_opening_point: Vec<F::Challenge>,
        cpu_opening_claims: Vec<F>,
    ) -> Self {
        let mut receipt = Self {
            version: Self::VERSION,
            lookup_receipt,
            log_k_chunk: log_k_chunk
                .try_into()
                .expect("test lookup chunk size must fit in u8"),
            opening_points,
            opening_claims,
            tuple_opening_point,
            left_lookup_operand_claim,
            right_lookup_operand_claim,
            lookup_output_claim,
            register_value_opening_point,
            rs1_value_claim,
            rs2_value_claim,
            rd_write_value_claim,
            register_address_opening_point,
            rs1_ra_claim,
            rs2_ra_claim,
            rd_wa_claim,
            rd_inc_opening_point,
            rd_inc_claim,
            ram_start_address,
            ram_k,
            ram_opening_points,
            ram_opening_claims,
            ram_tuple_opening_point,
            ram_address_claim,
            ram_read_value_claim,
            ram_write_value_claim,
            ram_inc_opening_point,
            ram_inc_claim,
            cpu_opening_point,
            cpu_opening_claims,
            receipt_digest: [0; 32],
        };
        receipt.receipt_digest = receipt.compute_digest();
        receipt
    }

    #[cfg(test)]
    pub(crate) fn new_for_test(
        lookup_receipt: VerifiedJoltLookupProofReceipt,
        log_k_chunk: usize,
        opening_points: Vec<Vec<F::Challenge>>,
        opening_claims: Vec<F>,
        tuple_opening_point: Vec<F::Challenge>,
        tuple_opening_claims: [F; 3],
        register_value_opening_point: Vec<F::Challenge>,
        register_value_opening_claims: [F; 3],
        register_address_opening_point: Vec<F::Challenge>,
        register_address_opening_claims: [F; 3],
        rd_inc_opening_point: Vec<F::Challenge>,
        rd_inc_claim: F,
        ram_start_address: u64,
        ram_k: usize,
        ram_opening_points: Vec<Vec<F::Challenge>>,
        ram_opening_claims: Vec<F>,
        ram_tuple_opening_point: Vec<F::Challenge>,
        ram_tuple_opening_claims: [F; 3],
        ram_inc_opening_point: Vec<F::Challenge>,
        ram_inc_claim: F,
        cpu_opening_point: Vec<F::Challenge>,
        cpu_opening_claims: Vec<F>,
    ) -> Self {
        Self::from_verified_openings(
            lookup_receipt,
            log_k_chunk,
            opening_points,
            opening_claims,
            tuple_opening_point,
            tuple_opening_claims[0],
            tuple_opening_claims[1],
            tuple_opening_claims[2],
            register_value_opening_point,
            register_value_opening_claims[0],
            register_value_opening_claims[1],
            register_value_opening_claims[2],
            register_address_opening_point,
            register_address_opening_claims[0],
            register_address_opening_claims[1],
            register_address_opening_claims[2],
            rd_inc_opening_point,
            rd_inc_claim,
            ram_start_address,
            ram_k,
            ram_opening_points,
            ram_opening_claims,
            ram_tuple_opening_point,
            ram_tuple_opening_claims[0],
            ram_tuple_opening_claims[1],
            ram_tuple_opening_claims[2],
            ram_inc_opening_point,
            ram_inc_claim,
            cpu_opening_point,
            cpu_opening_claims,
        )
    }

    fn compute_digest(&self) -> [u8; 32] {
        let mut hasher = Sha3_256::new();
        hasher.update(b"jolt-nova/verified-jolt-execution-opening-receipt/v5");
        hasher.update(self.version.to_le_bytes());
        hasher.update(self.lookup_receipt.digest());
        hasher.update([self.log_k_chunk]);
        hasher.update((self.opening_claims.len() as u64).to_le_bytes());
        hasher.update(canonical_digest(
            b"instruction-ra-opening-points",
            &self.opening_points,
        ));
        hasher.update(canonical_digest(
            b"instruction-ra-opening-claims",
            &self.opening_claims,
        ));
        hasher.update(canonical_digest(
            b"instruction-lookup-tuple-opening-point",
            &self.tuple_opening_point,
        ));
        hasher.update(canonical_digest(
            b"instruction-lookup-left-operand-claim",
            &self.left_lookup_operand_claim,
        ));
        hasher.update(canonical_digest(
            b"instruction-lookup-right-operand-claim",
            &self.right_lookup_operand_claim,
        ));
        hasher.update(canonical_digest(
            b"instruction-lookup-output-claim",
            &self.lookup_output_claim,
        ));
        hasher.update(canonical_digest(
            b"register-value-opening-point",
            &self.register_value_opening_point,
        ));
        hasher.update(canonical_digest(
            b"register-value-opening-claims",
            &[
                self.rs1_value_claim,
                self.rs2_value_claim,
                self.rd_write_value_claim,
            ],
        ));
        hasher.update(canonical_digest(
            b"register-address-opening-point",
            &self.register_address_opening_point,
        ));
        hasher.update(canonical_digest(
            b"register-address-opening-claims",
            &[self.rs1_ra_claim, self.rs2_ra_claim, self.rd_wa_claim],
        ));
        hasher.update(canonical_digest(
            b"register-rd-inc-opening-point",
            &self.rd_inc_opening_point,
        ));
        hasher.update(canonical_digest(
            b"register-rd-inc-opening-claim",
            &self.rd_inc_claim,
        ));
        hasher.update(self.ram_start_address.to_le_bytes());
        hasher.update((self.ram_k as u64).to_le_bytes());
        hasher.update((self.ram_opening_claims.len() as u64).to_le_bytes());
        hasher.update(canonical_digest(
            b"ram-ra-opening-points",
            &self.ram_opening_points,
        ));
        hasher.update(canonical_digest(
            b"ram-ra-opening-claims",
            &self.ram_opening_claims,
        ));
        hasher.update(canonical_digest(
            b"ram-tuple-opening-point",
            &self.ram_tuple_opening_point,
        ));
        hasher.update(canonical_digest(
            b"ram-tuple-opening-claims",
            &[
                self.ram_address_claim,
                self.ram_read_value_claim,
                self.ram_write_value_claim,
            ],
        ));
        hasher.update(canonical_digest(
            b"ram-inc-opening-point",
            &self.ram_inc_opening_point,
        ));
        hasher.update(canonical_digest(
            b"ram-inc-opening-claim",
            &self.ram_inc_claim,
        ));
        hasher.update((self.cpu_opening_claims.len() as u64).to_le_bytes());
        hasher.update(canonical_digest(
            b"cpu-r1cs-opening-point",
            &self.cpu_opening_point,
        ));
        hasher.update(canonical_digest(
            b"cpu-r1cs-opening-claims",
            &self.cpu_opening_claims,
        ));
        hasher.finalize().into()
    }
}

fn canonical_digest<T: CanonicalSerialize>(domain: &'static [u8], item: &T) -> [u8; 32] {
    let mut bytes = Vec::new();
    item.serialize_compressed(&mut bytes)
        .expect("canonical serialization into a Vec cannot fail");
    let mut hasher = Sha3_256::new();
    hasher.update(b"jolt-nova/canonical-digest/v1");
    hasher.update((domain.len() as u64).to_le_bytes());
    hasher.update(domain);
    hasher.update((bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
    hasher.finalize().into()
}

impl<F: JoltField, C: JoltCurve<F = F>, PCS: CommitmentScheme<Field = F>, FS: Transcript>
    JoltProof<F, C, PCS, FS>
{
    pub(crate) fn lookup_receipt_candidate<PublicIo: CanonicalSerialize>(
        &self,
        public_io: &PublicIo,
        preprocessing_digest: [u8; 32],
        verifier_setup: &PCS::VerifierSetup,
        trusted_advice_commitment: &Option<PCS::Commitment>,
    ) -> VerifiedJoltLookupProofReceipt {
        let zk_mode = self.stage1_sumcheck_proof.is_zk();
        #[cfg(feature = "zk")]
        let blindfold_proof_digest = if zk_mode {
            canonical_digest(b"blindfold-proof", &self.blindfold_proof)
        } else {
            [0; 32]
        };
        #[cfg(not(feature = "zk"))]
        let blindfold_proof_digest = [0; 32];
        let mut receipt = VerifiedJoltLookupProofReceipt {
            version: VerifiedJoltLookupProofReceipt::VERSION,
            trace_length: self.trace_length as u64,
            commitment_count: self.commitments.len() as u64,
            zk_mode,
            blindfold_proof_digest,
            preprocessing_digest,
            verifier_setup_digest: canonical_digest(b"verifier-setup", verifier_setup),
            public_io_digest: canonical_digest(b"public-io", public_io),
            trusted_advice_commitment_digest: canonical_digest(
                b"trusted-advice-commitment",
                trusted_advice_commitment,
            ),
            commitments_digest: canonical_digest(b"commitments", &self.commitments),
            stage1_uni_skip_first_round_proof_digest: canonical_digest(
                b"verifier-stage-1-uni-skip",
                &self.stage1_uni_skip_first_round_proof,
            ),
            stage1_sumcheck_digest: canonical_digest(
                b"verifier-stage-1-sumcheck",
                &self.stage1_sumcheck_proof,
            ),
            stage2_uni_skip_first_round_proof_digest: canonical_digest(
                b"verifier-stage-2-uni-skip",
                &self.stage2_uni_skip_first_round_proof,
            ),
            stage2_sumcheck_digest: canonical_digest(
                b"verifier-stage-2-sumcheck",
                &self.stage2_sumcheck_proof,
            ),
            stage3_sumcheck_digest: canonical_digest(
                b"verifier-stage-3-sumcheck",
                &self.stage3_sumcheck_proof,
            ),
            stage4_sumcheck_digest: canonical_digest(
                b"verifier-stage-4-sumcheck",
                &self.stage4_sumcheck_proof,
            ),
            stage5_sumcheck_digest: canonical_digest(
                b"verifier-stage-5-sumcheck",
                &self.stage5_sumcheck_proof,
            ),
            stage6a_sumcheck_digest: canonical_digest(
                b"verifier-stage-6a-sumcheck",
                &self.stage6a_sumcheck_proof,
            ),
            stage6b_sumcheck_digest: canonical_digest(
                b"verifier-stage-6b-sumcheck",
                &self.stage6b_sumcheck_proof,
            ),
            stage7_sumcheck_digest: canonical_digest(
                b"verifier-stage-7-sumcheck",
                &self.stage7_sumcheck_proof,
            ),
            joint_opening_proof_digest: canonical_digest(
                b"joint-opening-proof",
                &self.joint_opening_proof,
            ),
            full_proof_digest: canonical_digest(b"full-jolt-proof", self),
            receipt_digest: [0; 32],
        };
        receipt.receipt_digest = receipt.compute_digest();
        receipt
    }

    /// Verifies all sumcheck and uniskip proofs use the same ZK variant.
    /// Returns the ZK mode if consistent, or an error if any stage disagrees.
    pub fn verify_zk_consistency(&self) -> Result<bool, ProofVerifyError> {
        let zk_mode = self.stage1_sumcheck_proof.is_zk();

        let consistent = self.stage1_uni_skip_first_round_proof.is_zk() == zk_mode
            && self.stage2_uni_skip_first_round_proof.is_zk() == zk_mode
            && self.stage2_sumcheck_proof.is_zk() == zk_mode
            && self.stage3_sumcheck_proof.is_zk() == zk_mode
            && self.stage4_sumcheck_proof.is_zk() == zk_mode
            && self.stage5_sumcheck_proof.is_zk() == zk_mode
            && self.stage6a_sumcheck_proof.is_zk() == zk_mode
            && self.stage6b_sumcheck_proof.is_zk() == zk_mode
            && self.stage7_sumcheck_proof.is_zk() == zk_mode;

        if !consistent {
            return Err(ProofVerifyError::SumcheckVerificationError);
        }
        Ok(zk_mode)
    }
}

#[cfg(not(feature = "zk"))]
pub struct Claims<F: JoltField>(pub Openings<F>);

#[cfg(not(feature = "zk"))]
impl<F: JoltField> CanonicalSerialize for Claims<F> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.0.len().serialize_with_mode(&mut writer, compress)?;
        for (key, (_opening_point, claim)) in self.0.iter() {
            key.serialize_with_mode(&mut writer, compress)?;
            claim.serialize_with_mode(&mut writer, compress)?;
        }
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        let mut size = self.0.len().serialized_size(compress);
        for (key, (_opening_point, claim)) in self.0.iter() {
            size += key.serialized_size(compress);
            size += claim.serialized_size(compress);
        }
        size
    }
}

#[cfg(not(feature = "zk"))]
impl<F: JoltField> Valid for Claims<F> {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

#[cfg(not(feature = "zk"))]
impl<F: JoltField> CanonicalDeserialize for Claims<F> {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let size = usize::deserialize_with_mode(&mut reader, compress, validate)?;
        let mut claims = BTreeMap::new();
        for _ in 0..size {
            let key = OpeningId::deserialize_with_mode(&mut reader, compress, validate)?;
            let claim = F::deserialize_with_mode(&mut reader, compress, validate)?;
            claims.insert(key, (OpeningPoint::default(), claim));
        }
        Ok(Claims(claims))
    }
}

impl CanonicalSerialize for DoryLayout {
    fn serialize_with_mode<W: Write>(
        &self,
        writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        u8::from(*self).serialize_with_mode(writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        u8::from(*self).serialized_size(compress)
    }
}

impl Valid for DoryLayout {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl CanonicalDeserialize for DoryLayout {
    fn deserialize_with_mode<R: Read>(
        reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let value = u8::deserialize_with_mode(reader, compress, validate)?;
        if value > 1 {
            return Err(SerializationError::InvalidData);
        }
        Ok(DoryLayout::from(value))
    }
}

// Compact encoding for OpeningId:
// Each variant uses a fused byte = BASE + sumcheck_id (1 byte total for advice, 2 bytes for committed/virtual)
// - [0, NUM_SUMCHECKS) = UntrustedAdvice(sumcheck_id)
// - [NUM_SUMCHECKS, 2*NUM_SUMCHECKS) = TrustedAdvice(sumcheck_id)
// - [2*NUM_SUMCHECKS, 3*NUM_SUMCHECKS) + poly_index = Committed(poly, sumcheck_id)
// - [3*NUM_SUMCHECKS, 4*NUM_SUMCHECKS) + poly_index = Virtual(poly, sumcheck_id)
const OPENING_ID_UNTRUSTED_ADVICE_BASE: u8 = 0;
const OPENING_ID_TRUSTED_ADVICE_BASE: u8 =
    OPENING_ID_UNTRUSTED_ADVICE_BASE + SumcheckId::COUNT as u8;
const OPENING_ID_COMMITTED_BASE: u8 = OPENING_ID_TRUSTED_ADVICE_BASE + SumcheckId::COUNT as u8;
const OPENING_ID_VIRTUAL_BASE: u8 = OPENING_ID_COMMITTED_BASE + SumcheckId::COUNT as u8;

impl CanonicalSerialize for OpeningId {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        match self {
            OpeningId::UntrustedAdvice(sumcheck_id) => {
                let fused = OPENING_ID_UNTRUSTED_ADVICE_BASE + (*sumcheck_id as u8);
                fused.serialize_with_mode(&mut writer, compress)
            }
            OpeningId::TrustedAdvice(sumcheck_id) => {
                let fused = OPENING_ID_TRUSTED_ADVICE_BASE + (*sumcheck_id as u8);
                fused.serialize_with_mode(&mut writer, compress)
            }
            OpeningId::Polynomial(PolynomialId::Committed(committed_polynomial), sumcheck_id) => {
                let fused = OPENING_ID_COMMITTED_BASE + (*sumcheck_id as u8);
                fused.serialize_with_mode(&mut writer, compress)?;
                committed_polynomial.serialize_with_mode(&mut writer, compress)
            }
            OpeningId::Polynomial(PolynomialId::Virtual(virtual_polynomial), sumcheck_id) => {
                let fused = OPENING_ID_VIRTUAL_BASE + (*sumcheck_id as u8);
                fused.serialize_with_mode(&mut writer, compress)?;
                virtual_polynomial.serialize_with_mode(&mut writer, compress)
            }
        }
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        match self {
            OpeningId::UntrustedAdvice(_) | OpeningId::TrustedAdvice(_) => 1,
            OpeningId::Polynomial(PolynomialId::Committed(committed_polynomial), _) => {
                1 + committed_polynomial.serialized_size(compress)
            }
            OpeningId::Polynomial(PolynomialId::Virtual(virtual_polynomial), _) => {
                1 + virtual_polynomial.serialized_size(compress)
            }
        }
    }
}

impl Valid for OpeningId {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl CanonicalDeserialize for OpeningId {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let fused = u8::deserialize_with_mode(&mut reader, compress, validate)?;
        match fused {
            _ if fused < OPENING_ID_TRUSTED_ADVICE_BASE => {
                let sumcheck_id = fused - OPENING_ID_UNTRUSTED_ADVICE_BASE;
                Ok(OpeningId::UntrustedAdvice(
                    SumcheckId::from_u8(sumcheck_id).ok_or(SerializationError::InvalidData)?,
                ))
            }
            _ if fused < OPENING_ID_COMMITTED_BASE => {
                let sumcheck_id = fused - OPENING_ID_TRUSTED_ADVICE_BASE;
                Ok(OpeningId::TrustedAdvice(
                    SumcheckId::from_u8(sumcheck_id).ok_or(SerializationError::InvalidData)?,
                ))
            }
            _ if fused < OPENING_ID_VIRTUAL_BASE => {
                let sumcheck_id = fused - OPENING_ID_COMMITTED_BASE;
                let polynomial =
                    CommittedPolynomial::deserialize_with_mode(&mut reader, compress, validate)?;
                Ok(OpeningId::committed(
                    polynomial,
                    SumcheckId::from_u8(sumcheck_id).ok_or(SerializationError::InvalidData)?,
                ))
            }
            _ => {
                let sumcheck_id = fused - OPENING_ID_VIRTUAL_BASE;
                let polynomial =
                    VirtualPolynomial::deserialize_with_mode(&mut reader, compress, validate)?;
                Ok(OpeningId::virt(
                    polynomial,
                    SumcheckId::from_u8(sumcheck_id).ok_or(SerializationError::InvalidData)?,
                ))
            }
        }
    }
}

impl CanonicalSerialize for CommittedPolynomial {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        match self {
            Self::RdInc => 0u8.serialize_with_mode(writer, compress),
            Self::RamInc => 1u8.serialize_with_mode(writer, compress),
            Self::InstructionRa(i) => {
                2u8.serialize_with_mode(&mut writer, compress)?;
                (u8::try_from(*i).unwrap()).serialize_with_mode(writer, compress)
            }
            Self::BytecodeRa(i) => {
                3u8.serialize_with_mode(&mut writer, compress)?;
                (u8::try_from(*i).unwrap()).serialize_with_mode(writer, compress)
            }
            Self::RamRa(i) => {
                4u8.serialize_with_mode(&mut writer, compress)?;
                (u8::try_from(*i).unwrap()).serialize_with_mode(writer, compress)
            }
            Self::TrustedAdvice => 5u8.serialize_with_mode(writer, compress),
            Self::UntrustedAdvice => 6u8.serialize_with_mode(writer, compress),
            Self::BytecodeChunk(i) => {
                7u8.serialize_with_mode(&mut writer, compress)?;
                (u8::try_from(*i).unwrap()).serialize_with_mode(writer, compress)
            }
            Self::ProgramImageInit => 8u8.serialize_with_mode(writer, compress),
        }
    }

    fn serialized_size(&self, _compress: Compress) -> usize {
        match self {
            Self::RdInc
            | Self::RamInc
            | Self::TrustedAdvice
            | Self::UntrustedAdvice
            | Self::ProgramImageInit => 1,
            Self::InstructionRa(_)
            | Self::BytecodeRa(_)
            | Self::RamRa(_)
            | Self::BytecodeChunk(_) => 2,
        }
    }
}

impl Valid for CommittedPolynomial {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl CanonicalDeserialize for CommittedPolynomial {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        Ok(
            match u8::deserialize_with_mode(&mut reader, compress, validate)? {
                0 => Self::RdInc,
                1 => Self::RamInc,
                2 => {
                    let i = u8::deserialize_with_mode(reader, compress, validate)?;
                    Self::InstructionRa(i as usize)
                }
                3 => {
                    let i = u8::deserialize_with_mode(reader, compress, validate)?;
                    Self::BytecodeRa(i as usize)
                }
                4 => {
                    let i = u8::deserialize_with_mode(reader, compress, validate)?;
                    Self::RamRa(i as usize)
                }
                5 => Self::TrustedAdvice,
                6 => Self::UntrustedAdvice,
                7 => {
                    let i = u8::deserialize_with_mode(reader, compress, validate)?;
                    Self::BytecodeChunk(i as usize)
                }
                8 => Self::ProgramImageInit,
                _ => return Err(SerializationError::InvalidData),
            },
        )
    }
}

impl CanonicalSerialize for VirtualPolynomial {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        match self {
            Self::PC => 0u8.serialize_with_mode(&mut writer, compress),
            Self::UnexpandedPC => 1u8.serialize_with_mode(&mut writer, compress),
            Self::NextPC => 2u8.serialize_with_mode(&mut writer, compress),
            Self::NextUnexpandedPC => 3u8.serialize_with_mode(&mut writer, compress),
            Self::NextIsNoop => 4u8.serialize_with_mode(&mut writer, compress),
            Self::NextIsVirtual => 5u8.serialize_with_mode(&mut writer, compress),
            Self::NextIsFirstInSequence => 6u8.serialize_with_mode(&mut writer, compress),
            Self::LeftLookupOperand => 7u8.serialize_with_mode(&mut writer, compress),
            Self::RightLookupOperand => 8u8.serialize_with_mode(&mut writer, compress),
            Self::LeftInstructionInput => 9u8.serialize_with_mode(&mut writer, compress),
            Self::RightInstructionInput => 10u8.serialize_with_mode(&mut writer, compress),
            Self::Product => 11u8.serialize_with_mode(&mut writer, compress),
            Self::ShouldJump => 12u8.serialize_with_mode(&mut writer, compress),
            Self::ShouldBranch => 13u8.serialize_with_mode(&mut writer, compress),
            Self::Rd => 14u8.serialize_with_mode(&mut writer, compress),
            Self::Imm => 15u8.serialize_with_mode(&mut writer, compress),
            Self::Rs1Value => 16u8.serialize_with_mode(&mut writer, compress),
            Self::Rs2Value => 17u8.serialize_with_mode(&mut writer, compress),
            Self::RdWriteValue => 18u8.serialize_with_mode(&mut writer, compress),
            Self::Rs1Ra => 19u8.serialize_with_mode(&mut writer, compress),
            Self::Rs2Ra => 20u8.serialize_with_mode(&mut writer, compress),
            Self::RdWa => 21u8.serialize_with_mode(&mut writer, compress),
            Self::LookupOutput => 22u8.serialize_with_mode(&mut writer, compress),
            Self::InstructionRaf => 23u8.serialize_with_mode(&mut writer, compress),
            Self::InstructionRafFlag => 24u8.serialize_with_mode(&mut writer, compress),
            Self::InstructionRa(i) => {
                25u8.serialize_with_mode(&mut writer, compress)?;
                (u8::try_from(*i).unwrap()).serialize_with_mode(&mut writer, compress)
            }
            Self::RegistersVal => 26u8.serialize_with_mode(&mut writer, compress),
            Self::RamAddress => 27u8.serialize_with_mode(&mut writer, compress),
            Self::RamRa => 28u8.serialize_with_mode(&mut writer, compress),
            Self::RamReadValue => 29u8.serialize_with_mode(&mut writer, compress),
            Self::RamWriteValue => 30u8.serialize_with_mode(&mut writer, compress),
            Self::RamVal => 31u8.serialize_with_mode(&mut writer, compress),
            Self::RamValInit => 32u8.serialize_with_mode(&mut writer, compress),
            Self::RamValFinal => 33u8.serialize_with_mode(&mut writer, compress),
            Self::RamHammingWeight => 34u8.serialize_with_mode(&mut writer, compress),
            Self::UnivariateSkip => 35u8.serialize_with_mode(&mut writer, compress),
            Self::OpFlags(flags) => {
                36u8.serialize_with_mode(&mut writer, compress)?;
                (u8::try_from(*flags as usize).unwrap()).serialize_with_mode(&mut writer, compress)
            }
            Self::InstructionFlags(flags) => {
                37u8.serialize_with_mode(&mut writer, compress)?;
                (u8::try_from(*flags as usize).unwrap()).serialize_with_mode(&mut writer, compress)
            }
            Self::LookupTableFlag(flag) => {
                38u8.serialize_with_mode(&mut writer, compress)?;
                (u8::try_from(*flag).unwrap()).serialize_with_mode(&mut writer, compress)
            }
            Self::BytecodeReadRafAddrClaim => 39u8.serialize_with_mode(&mut writer, compress),
            Self::BooleanityAddrClaim => 40u8.serialize_with_mode(&mut writer, compress),
            Self::BytecodeValStage(i) => {
                41u8.serialize_with_mode(&mut writer, compress)?;
                (u8::try_from(*i).unwrap()).serialize_with_mode(&mut writer, compress)
            }
            Self::BytecodeClaimReductionIntermediate => {
                42u8.serialize_with_mode(&mut writer, compress)
            }
            Self::ProgramImageInitContributionRw => 43u8.serialize_with_mode(&mut writer, compress),
        }
    }

    fn serialized_size(&self, _compress: Compress) -> usize {
        match self {
            Self::PC
            | Self::UnexpandedPC
            | Self::NextPC
            | Self::NextUnexpandedPC
            | Self::NextIsNoop
            | Self::NextIsVirtual
            | Self::NextIsFirstInSequence
            | Self::LeftLookupOperand
            | Self::RightLookupOperand
            | Self::LeftInstructionInput
            | Self::RightInstructionInput
            | Self::Product
            | Self::ShouldJump
            | Self::ShouldBranch
            | Self::Rd
            | Self::Imm
            | Self::Rs1Value
            | Self::Rs2Value
            | Self::RdWriteValue
            | Self::Rs1Ra
            | Self::Rs2Ra
            | Self::RdWa
            | Self::LookupOutput
            | Self::InstructionRaf
            | Self::InstructionRafFlag
            | Self::RegistersVal
            | Self::RamAddress
            | Self::RamRa
            | Self::RamReadValue
            | Self::RamWriteValue
            | Self::RamVal
            | Self::RamValInit
            | Self::RamValFinal
            | Self::RamHammingWeight
            | Self::UnivariateSkip
            | Self::BytecodeReadRafAddrClaim
            | Self::BooleanityAddrClaim
            | Self::BytecodeClaimReductionIntermediate
            | Self::ProgramImageInitContributionRw => 1,
            Self::InstructionRa(_)
            | Self::OpFlags(_)
            | Self::InstructionFlags(_)
            | Self::LookupTableFlag(_)
            | Self::BytecodeValStage(_) => 2,
        }
    }
}

impl Valid for VirtualPolynomial {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl CanonicalDeserialize for VirtualPolynomial {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        Ok(
            match u8::deserialize_with_mode(&mut reader, compress, validate)? {
                0 => Self::PC,
                1 => Self::UnexpandedPC,
                2 => Self::NextPC,
                3 => Self::NextUnexpandedPC,
                4 => Self::NextIsNoop,
                5 => Self::NextIsVirtual,
                6 => Self::NextIsFirstInSequence,
                7 => Self::LeftLookupOperand,
                8 => Self::RightLookupOperand,
                9 => Self::LeftInstructionInput,
                10 => Self::RightInstructionInput,
                11 => Self::Product,
                12 => Self::ShouldJump,
                13 => Self::ShouldBranch,
                14 => Self::Rd,
                15 => Self::Imm,
                16 => Self::Rs1Value,
                17 => Self::Rs2Value,
                18 => Self::RdWriteValue,
                19 => Self::Rs1Ra,
                20 => Self::Rs2Ra,
                21 => Self::RdWa,
                22 => Self::LookupOutput,
                23 => Self::InstructionRaf,
                24 => Self::InstructionRafFlag,
                25 => {
                    let i = u8::deserialize_with_mode(&mut reader, compress, validate)?;
                    Self::InstructionRa(i as usize)
                }
                26 => Self::RegistersVal,
                27 => Self::RamAddress,
                28 => Self::RamRa,
                29 => Self::RamReadValue,
                30 => Self::RamWriteValue,
                31 => Self::RamVal,
                32 => Self::RamValInit,
                33 => Self::RamValFinal,
                34 => Self::RamHammingWeight,
                35 => Self::UnivariateSkip,
                36 => {
                    let discriminant = u8::deserialize_with_mode(&mut reader, compress, validate)?;
                    let flags = CircuitFlags::from_repr(discriminant)
                        .ok_or(SerializationError::InvalidData)?;
                    Self::OpFlags(flags)
                }
                37 => {
                    let discriminant = u8::deserialize_with_mode(&mut reader, compress, validate)?;
                    let flags = InstructionFlags::from_repr(discriminant)
                        .ok_or(SerializationError::InvalidData)?;
                    Self::InstructionFlags(flags)
                }
                38 => {
                    let flag = u8::deserialize_with_mode(&mut reader, compress, validate)?;
                    Self::LookupTableFlag(flag as usize)
                }
                39 => Self::BytecodeReadRafAddrClaim,
                40 => Self::BooleanityAddrClaim,
                41 => {
                    let i = u8::deserialize_with_mode(&mut reader, compress, validate)?;
                    Self::BytecodeValStage(i as usize)
                }
                42 => Self::BytecodeClaimReductionIntermediate,
                43 => Self::ProgramImageInitContributionRw,
                _ => return Err(SerializationError::InvalidData),
            },
        )
    }
}

pub fn serialize_and_print_size(
    item_name: &str,
    file_name: &str,
    item: &impl CanonicalSerialize,
) -> Result<(), SerializationError> {
    use std::fs::File;
    let mut file = File::create(file_name)?;
    item.serialize_compressed(&mut file)?;
    let file_size_bytes = file.metadata()?.len();
    let file_size_kb = file_size_bytes as f64 / 1024.0;
    tracing::info!("{item_name} Written to {file_name}");
    tracing::info!("{item_name} size: {file_size_kb:.1} kB");
    Ok(())
}
