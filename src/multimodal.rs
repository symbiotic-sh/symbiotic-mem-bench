//! Provider-neutral multimodal recall experiment contract.
//!
//! The harness owns fixtures, experiment arms, scoring, provenance, and proof that a requested
//! arm actually executed. A memory-system adapter owns ingestion and recall. This boundary keeps
//! benchmark code from copying a product's parser, blob store, index, or recall implementation.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::path::{Component, Path, PathBuf};

use anyhow::Context;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const FIXTURE_SCHEMA: &str = "membench.multimodal_fixture.v1";
pub const ANNOTATION_SCHEMA: &str = "membench.longmemeval_v2_image_annotations.v1";
pub const RESULT_SCHEMA: &str = "membench.multimodal_result.v1";
pub const PINNED_PRODUCT_GIT_SHA: &str = "022c22af37d9ba166e347dcd54f4db85142f8cea";
pub const PINNED_PRODUCT_CONTRACT_SHA256: &str =
    "d44178f863361aecc5c2208399ba2a9b7cc5cb37d0bd09cd75e0419ebb96c529";
const PRODUCT_CONTRACT_PATH: &str = "contracts/multimodal-recall-contract.v1.json";
const PRODUCT_CONTRACT_ID: &str = "symbiotic-memory.multimodal-recall-collapse";

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProductMultimodalContract {
    contract_id: String,
    contract_version: u32,
    schema: ProductContractSchema,
    source_files_sha256: BTreeMap<String, String>,
    wire_specimens: Vec<CapturedArtifactEvidence>,
    collapse_vectors: Vec<ProductCollapseVector>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProductContractSchema {
    envelope: String,
    artifact_evidence_wire_type: String,
    collapse_function: String,
    source_hash_algorithm: String,
    default_overlap_threshold_millionths: u32,
    vector_output_semantics: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProductCollapseVector {
    case_id: String,
    input: Vec<CapturedArtifactEvidence>,
    actual_collapsed_output: Vec<CapturedArtifactEvidence>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MultimodalFixtureSet {
    pub schema: String,
    pub fixture_set_id: String,
    pub split: FixtureSplit,
    pub source: DatasetProvenance,
    /// Full retrieval corpora, including distractors. Oracle annotations never construct this list.
    pub corpora: Vec<MultimodalCorpus>,
    pub cases: Vec<MultimodalCase>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MultimodalCorpus {
    pub corpus_id: String,
    pub evidence: Vec<EvidenceItem>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FixtureSplit {
    Development,
    HeldOut,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DatasetProvenance {
    pub dataset_id: String,
    pub version: String,
    /// SHA-256 over the source files used to build this fixture set.
    pub source_digest: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_uri: Option<String>,
    /// Whether this fixture is identical to an official benchmark release and scoring protocol.
    pub official_equivalent: bool,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MultimodalCase {
    pub case_id: String,
    pub origin: FixtureOrigin,
    pub corpus_id: String,
    pub question: ContentSpec,
    pub gold: GoldAnswer,
    /// Human-reviewed label. The harness never infers this from question text or file extension.
    pub media_dependent: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub media_dependence_review: Option<MediaDependenceReview>,
    pub oracle_lane: RecallLane,
    pub oracle_evidence: Vec<EvidenceRegion>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MediaDependenceReview {
    pub reviewed_by: String,
    pub rationale: String,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FixtureOrigin {
    LongMemEvalV2,
    HeldOutPdf,
    HeldOutSpreadsheet,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ContentSpec {
    pub text: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub media: Vec<MediaRef>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Ord, PartialOrd, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MediaRef {
    pub asset_id: String,
    /// Dataset-relative locator or a content-addressed adapter locator. Never inline base64.
    pub locator: String,
    pub sha256: String,
    pub media_type: String,
    pub size_bytes: u64,
}

/// Product-wire-compatible blob identity returned after fixture import.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Ord, PartialOrd, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CapturedContentDigest {
    pub algorithm: String,
    pub value: String,
}

/// Mirrors the product `BlobRef` JSON shape. The product's `BlobId` newtype serializes as its
/// inner `ContentDigest`, so `id` is deliberately nested here.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Ord, PartialOrd, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CapturedBlobRef {
    pub id: CapturedContentDigest,
    pub size_bytes: u64,
    pub detected_media_type: String,
}

/// Mirrors the product `RegionLocator` JSON shape.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CapturedRegionLocator {
    Whole,
    Page {
        index: u32,
    },
    Sheet {
        name: String,
    },
    CellRange {
        sheet: String,
        a1: String,
    },
    Rectangle {
        anchor_region_id: String,
        coordinate_space: CapturedCoordinateSpace,
        x: i64,
        y: i64,
        width: u64,
        height: u64,
        rotation_millidegrees: i32,
    },
    TimeRange {
        start_ms: u64,
        end_ms: u64,
    },
    ByteRange {
        start: u64,
        end_exclusive: u64,
    },
    Named {
        scheme: String,
        value: String,
    },
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CapturedCoordinateSpace {
    PdfPoints,
    Pixels,
    NormalizedMillionths,
    OfficeEmu,
}

/// Mirrors the product `RegionRef` JSON shape.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Hash, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CapturedRegionRef {
    pub region_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_region_id: Option<String>,
    pub locator: CapturedRegionLocator,
}

/// Mirrors the product `RetrievalScores` JSON shape.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CapturedRetrievalScores {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text_projection: Option<f32>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub native_profiles: BTreeMap<String, f32>,
    pub fused: f32,
}

/// Product-wire-compatible captured pointer. The fixture locator is import input only; this binding
/// is the sole authority presented to recall and reader stages.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CapturedArtifact {
    /// Benchmark envelope identity; not part of product `ArtifactEvidence`.
    pub evidence_id: String,
    /// Bench-only integrity witness for deterministic reader bytes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub projection_output_sha256: Option<String>,
    /// Bench-only verified metadata for the projection output. Text branch candidates are
    /// materialized from this record and therefore carry the output binding/blob, while native
    /// candidates continue to carry the raw source binding/blob.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub projection_output_blob: Option<CapturedBlobRef>,
    pub product: CapturedArtifactEvidence,
}

/// Exact public wire mirror of the product `ArtifactEvidence` shape. A checked drift fixture pins
/// this independent copy until the product publishes a stable crate dependency.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CapturedArtifactEvidence {
    pub binding_id: String,
    pub blob: CapturedBlobRef,
    pub region: CapturedRegionRef,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub projection: Option<CapturedProjectionRef>,
    pub truth_tier: TruthTier,
    pub retrieval: CapturedRetrievalScores,
}

impl std::ops::Deref for CapturedArtifact {
    type Target = CapturedArtifactEvidence;

    fn deref(&self) -> &Self::Target {
        &self.product
    }
}

impl std::ops::DerefMut for CapturedArtifact {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.product
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CapturedProjectionRef {
    pub execution_id: String,
    pub source_binding_id: String,
    pub source_region_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_binding_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transformed_at: Option<String>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TruthTier {
    Raw,
    DeterministicProjection,
    ModelProjection,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceItem {
    pub evidence_id: String,
    pub source_media: MediaRef,
    pub region: EvidenceRegion,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text_projection: Option<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum EvidenceRegion {
    Screenshot {
        trajectory_id: String,
        state_index: u32,
        locator: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        bounds: Option<NormalizedBounds>,
    },
    Page {
        document_id: String,
        page_number: u32,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        bounds: Option<NormalizedBounds>,
    },
    Sheet {
        workbook_id: String,
        sheet_name: String,
        range: String,
    },
    Cell {
        workbook_id: String,
        sheet_name: String,
        cell: String,
    },
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NormalizedBounds {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GoldAnswer {
    pub value: String,
    pub scoring: ScoringRule,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LongMemEvalV2ImageAnnotations {
    pub schema: String,
    pub fixture_set_id: String,
    pub dataset_version: String,
    pub haystack_file: String,
    pub annotations: Vec<LongMemEvalV2ImageAnnotation>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LongMemEvalV2ImageAnnotation {
    pub question_id: String,
    /// Must be explicitly true. The loader never guesses image dependence from file names or text.
    pub image_dependent: bool,
    pub reviewed_by: String,
    pub rationale: String,
    pub oracle_lane: RecallLane,
    pub evidence: Vec<LongMemEvalV2ScreenshotEvidence>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LongMemEvalV2ScreenshotEvidence {
    pub trajectory_id: String,
    pub state_index: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bounds: Option<NormalizedBounds>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ScoringRule {
    ExactNormalized,
    ContainsAll {
        #[serde(default)]
        case_sensitive: bool,
    },
    /// An upstream benchmark-owned evaluator. Core refuses this unless a scorer plug-in supports it.
    External {
        evaluator: String,
    },
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RecallLane {
    TextProjection,
    Native,
    HybridCollapsed,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ExperimentCell {
    ATextControl,
    BNative,
    CHybridCollapsed,
    DOracleLane,
    EReaderModality,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ReaderMode {
    TextProjection,
    SourceBlob,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExperimentArm {
    pub cell: ExperimentCell,
    pub reader_mode: ReaderMode,
}

impl ExperimentArm {
    pub fn text_control() -> Self {
        Self {
            cell: ExperimentCell::ATextControl,
            reader_mode: ReaderMode::TextProjection,
        }
    }

    pub fn native() -> Self {
        Self {
            cell: ExperimentCell::BNative,
            reader_mode: ReaderMode::TextProjection,
        }
    }

    pub fn hybrid() -> Self {
        Self {
            cell: ExperimentCell::CHybridCollapsed,
            reader_mode: ReaderMode::TextProjection,
        }
    }

    pub fn oracle() -> Self {
        Self {
            cell: ExperimentCell::DOracleLane,
            reader_mode: ReaderMode::TextProjection,
        }
    }

    pub fn reader_modality(reader_mode: ReaderMode) -> Self {
        Self {
            cell: ExperimentCell::EReaderModality,
            reader_mode,
        }
    }

    fn validate(&self) -> anyhow::Result<()> {
        if self.cell != ExperimentCell::EReaderModality {
            anyhow::ensure!(
                self.reader_mode == ReaderMode::TextProjection,
                "source-blob reader mode is only valid for cell E"
            );
        }
        Ok(())
    }

    fn requested_lane(&self, case: &MultimodalCase) -> RecallLane {
        match self.cell {
            ExperimentCell::ATextControl => RecallLane::TextProjection,
            ExperimentCell::BNative => RecallLane::Native,
            ExperimentCell::CHybridCollapsed | ExperimentCell::EReaderModality => {
                RecallLane::HybridCollapsed
            }
            ExperimentCell::DOracleLane => case.oracle_lane,
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MultimodalCapabilities {
    pub text_projection_recall: bool,
    pub native_image_recall: bool,
    pub native_document_recall: bool,
    pub hybrid_collapse: bool,
    pub text_projection_reader: bool,
    pub source_blob_reader: bool,
    pub oracle_region_filter: bool,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AdapterDescriptor {
    pub adapter_id: String,
    pub adapter_version: String,
    pub capabilities: MultimodalCapabilities,
}

pub trait MultimodalRecallAdapter {
    fn descriptor(&self) -> AdapterDescriptor;

    /// Import the complete verified corpus and return product-compatible captured pointers. The
    /// adapter may use fixture locators only during this import step.
    fn import_corpus(
        &mut self,
        request: &CorpusImportRequest,
        spend: &mut SpendJournal,
    ) -> Result<CapturedCorpus, AdapterFailure>;

    /// The request contains no gold answer. Cell D contains gold *evidence regions* and is stamped
    /// as oracle-only in provenance; it can never be mistaken for a normal system score.
    fn recall(
        &mut self,
        request: &RecallRequest,
        spend: &mut SpendJournal,
        reader: &mut ReaderJournal,
    ) -> Result<RecallResponse, AdapterFailure>;

    /// Retrieval-only Cell-E boundary. It cannot receive reader bytes and cannot return an answer.
    fn retrieve(
        &mut self,
        _request: &RetrievalRequest,
        _spend: &mut SpendJournal,
    ) -> Result<RetrievalResponse, AdapterFailure> {
        Err(AdapterFailure::CapabilityUnavailable(
            "retrieval-only cell E boundary".to_string(),
        ))
    }
}

pub trait MultimodalReaderAdapter {
    fn descriptor(&self) -> ReaderDescriptor;
    fn read(
        &mut self,
        request: &BoundReaderRequest,
        spend: &mut SpendJournal,
    ) -> Result<BoundReaderResponse, AdapterFailure>;
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ReaderDescriptor {
    pub reader_id: String,
    pub reader_version: String,
    pub mode: ReaderMode,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RetrievalRequest {
    pub case_id: String,
    pub corpus_id: String,
    pub question: ContentSpec,
    pub corpus: CapturedCorpus,
    pub lane: RecallLane,
    pub budget: RemainingBudget,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RetrievalResponse {
    pub retrieved: Vec<RetrievedEvidence>,
    pub execution: RetrievalExecutionProof,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RetrievalExecutionProof {
    pub effective_lane: RecallLane,
    pub text_branch_candidates: Vec<BranchCandidate>,
    pub native_branch_candidates: Vec<BranchCandidate>,
    pub collapse_clusters: Vec<CollapseCluster>,
    pub oracle_regions_applied: Vec<EvidenceRegion>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BoundReaderInput {
    pub evidence_id: String,
    pub mode: ReaderMode,
    pub binding_id: String,
    pub media_type: String,
    pub verified_sha256: String,
    pub bytes: Vec<u8>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BoundReaderInvocation {
    pub case_id: String,
    pub question: ContentSpec,
    pub frozen_retrieval_sha256: String,
    pub inputs: Vec<BoundReaderInput>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BoundReaderRequest {
    pub request_sha256: String,
    pub invocation: BoundReaderInvocation,
    pub budget: RemainingBudget,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BoundReaderResponse {
    pub answer: String,
    pub effective_request_sha256: String,
    pub effective_input_sha256: Vec<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CorpusImportRequest {
    pub corpus_id: String,
    /// Exact verified bytes plus their fixture metadata. Paths never cross the adapter boundary.
    pub items: Vec<VerifiedEvidenceImport>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct VerifiedEvidenceImport {
    pub evidence: EvidenceItem,
    pub source_bytes: Vec<u8>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CapturedCorpus {
    pub corpus_id: String,
    pub artifacts: Vec<CapturedArtifact>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RecallRequest {
    pub case_id: String,
    pub corpus_id: String,
    pub question: ContentSpec,
    pub corpus: CapturedCorpus,
    pub lane: RecallLane,
    pub reader_mode: ReaderMode,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub oracle_regions: Vec<EvidenceRegion>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frozen_retrieval: Option<FrozenRetrieval>,
    pub budget: RemainingBudget,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RemainingBudget {
    pub provider_calls: u64,
    pub cost_micro_usd: u64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FrozenRetrieval {
    pub fingerprint: String,
    pub retrieved: Vec<FrozenRetrievedEvidence>,
}

/// Reader-neutral retrieval selection. Reader materialization is deliberately absent so the
/// frozen artifact cannot leak a prior reader's projection text.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FrozenRetrievedEvidence {
    pub evidence_id: String,
    pub lane: RecallLane,
    pub artifact: CapturedArtifact,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RecallResponse {
    pub answer: String,
    pub retrieved: Vec<RetrievedEvidence>,
    pub execution: AdapterExecutionProof,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SpendTrace {
    pub call_id: String,
    pub provider: String,
    pub model: String,
    pub status: SpendStatus,
    pub input_units: u64,
    pub output_units: u64,
    pub cost_micro_usd: u64,
    pub pricing_table_version: String,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SpendStatus {
    Reserved,
    Succeeded,
    Failed,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SpendReservation {
    call_id: String,
    operation_id: String,
}

const SPEND_LEDGER_SCHEMA: &str = "membench.multimodal_spend_ledger.v2";

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct SpendLedgerPayload {
    run_instance_id: String,
    effective_config_sha256: String,
    generation: u64,
    operations: BTreeMap<String, PersistedSpendOperation>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct SpendLedgerEnvelope {
    schema: String,
    payload_sha256: String,
    payload: SpendLedgerPayload,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct PersistedSpendOperation {
    operation_id: String,
    case_id: String,
    reservation: SpendTrace,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    terminal: Option<SpendTrace>,
}

/// Harness-owned provider budget and durable journal. The ledger is a checksummed, atomically
/// replaced state file guarded by a cross-process create-new lock. Each operation has one stable
/// id and at most one authoritative terminal, including after an ambiguous directory-fsync error.
#[derive(Debug)]
pub struct SpendJournal {
    provider_calls_allowed: bool,
    run_instance_id: String,
    effective_config_sha256: String,
    case_id: String,
    ledger_path: Option<PathBuf>,
    remaining: RemainingBudget,
    traces: Vec<SpendTrace>,
    open: BTreeMap<String, SpendTrace>,
    reserved_calls: u64,
    reserved_cost_micro_usd: u64,
    #[cfg(test)]
    injected_persist_failure: Option<SpendPersistFailurePoint>,
}

#[cfg_attr(not(test), allow(dead_code))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SpendPersistFailurePoint {
    BeforeRename,
    AfterRenameBeforeDirectoryFsync,
}

impl SpendJournal {
    fn new(
        run_instance_id: &str,
        effective_config_sha256: &str,
        case_id: &str,
        ledger_path: Option<&Path>,
        remaining: RemainingBudget,
        provider_calls_allowed: bool,
    ) -> Self {
        Self {
            provider_calls_allowed,
            run_instance_id: run_instance_id.to_string(),
            effective_config_sha256: effective_config_sha256.to_string(),
            case_id: case_id.to_string(),
            ledger_path: ledger_path.map(Path::to_path_buf),
            remaining,
            traces: Vec::new(),
            open: BTreeMap::new(),
            reserved_calls: 0,
            reserved_cost_micro_usd: 0,
            #[cfg(test)]
            injected_persist_failure: None,
        }
    }

    fn injected_failure(&mut self) -> Option<SpendPersistFailurePoint> {
        #[cfg(test)]
        {
            self.injected_persist_failure.take()
        }
        #[cfg(not(test))]
        {
            None
        }
    }

    fn operation_id(&self, call_id: &str) -> String {
        sha256(
            format!(
                "{}\0{}\0{}\0{}",
                self.run_instance_id, self.effective_config_sha256, self.case_id, call_id
            )
            .as_bytes(),
        )
    }

    fn persist_reservation(
        &mut self,
        operation_id: &str,
        trace: &SpendTrace,
    ) -> anyhow::Result<()> {
        let failure = self.injected_failure();
        update_spend_ledger(
            self.ledger_path.as_deref(),
            &self.run_instance_id,
            &self.effective_config_sha256,
            operation_id,
            &self.case_id,
            SpendTransition::Reserve(trace.clone()),
            failure,
        )?;
        Ok(())
    }

    fn persist_terminal(
        &mut self,
        operation_id: &str,
        terminal: &SpendTrace,
    ) -> anyhow::Result<()> {
        let failure = self.injected_failure();
        update_spend_ledger(
            self.ledger_path.as_deref(),
            &self.run_instance_id,
            &self.effective_config_sha256,
            operation_id,
            &self.case_id,
            SpendTransition::Finish(terminal.clone()),
            failure,
        )?;
        Ok(())
    }

    fn authoritative_terminal(&self, operation_id: &str) -> anyhow::Result<Option<SpendTrace>> {
        read_spend_operation(
            self.ledger_path.as_deref(),
            &self.run_instance_id,
            &self.effective_config_sha256,
            operation_id,
        )
        .map(|operation| operation.and_then(|operation| operation.terminal))
    }

    pub fn reserve(
        &mut self,
        call_id: impl Into<String>,
        provider: impl Into<String>,
        model: impl Into<String>,
        max_cost_micro_usd: u64,
        pricing_table_version: impl Into<String>,
    ) -> Result<SpendReservation, AdapterFailure> {
        if !self.provider_calls_allowed {
            return Err(AdapterFailure::Failed(
                "cost ladder categorically prohibits provider dispatch".to_string(),
            ));
        }
        let call_id = call_id.into();
        let provider = provider.into();
        let model = model.into();
        let pricing_table_version = pricing_table_version.into();
        validate_id("call_id", &call_id).map_err(adapter_budget_error)?;
        non_empty("provider", &provider).map_err(adapter_budget_error)?;
        non_empty("model", &model).map_err(adapter_budget_error)?;
        non_empty("pricing table version", &pricing_table_version).map_err(adapter_budget_error)?;
        if self.open.contains_key(&call_id)
            || self.traces.iter().any(|trace| trace.call_id == call_id)
        {
            return Err(AdapterFailure::Failed(format!(
                "duplicate provider call id '{call_id}'"
            )));
        }
        let next_calls = self.reserved_calls.saturating_add(1);
        let next_cost = self
            .reserved_cost_micro_usd
            .saturating_add(max_cost_micro_usd);
        if next_calls > self.remaining.provider_calls || next_cost > self.remaining.cost_micro_usd {
            return Err(AdapterFailure::Failed(
                "provider reservation exceeds remaining budget".to_string(),
            ));
        }
        let trace = SpendTrace {
            call_id: call_id.clone(),
            provider,
            model,
            status: SpendStatus::Reserved,
            input_units: 0,
            output_units: 0,
            cost_micro_usd: max_cost_micro_usd,
            pricing_table_version,
        };
        let operation_id = self.operation_id(&call_id);
        self.persist_reservation(&operation_id, &trace)
            .map_err(adapter_budget_error)?;
        self.reserved_calls = next_calls;
        self.reserved_cost_micro_usd = next_cost;
        self.open.insert(call_id.clone(), trace.clone());
        self.traces.push(trace);
        Ok(SpendReservation {
            call_id,
            operation_id,
        })
    }

    pub fn finish(
        &mut self,
        reservation: SpendReservation,
        succeeded: bool,
        input_units: u64,
        output_units: u64,
        actual_cost_micro_usd: u64,
    ) -> Result<(), AdapterFailure> {
        let reserved = self
            .open
            .get(&reservation.call_id)
            .cloned()
            .ok_or_else(|| {
                AdapterFailure::Failed(
                    "provider reservation is unknown or already finished".to_string(),
                )
            })?;
        if actual_cost_micro_usd > reserved.cost_micro_usd {
            return Err(AdapterFailure::Failed(
                "provider terminal cost exceeded its pre-dispatch reservation".to_string(),
            ));
        }
        let terminal = SpendTrace {
            status: if succeeded {
                SpendStatus::Succeeded
            } else {
                SpendStatus::Failed
            },
            input_units,
            output_units,
            cost_micro_usd: actual_cost_micro_usd,
            ..reserved
        };
        if reservation.operation_id != self.operation_id(&reservation.call_id) {
            return Err(AdapterFailure::Failed(
                "provider reservation operation id mismatch".to_string(),
            ));
        }
        self.persist_terminal(&reservation.operation_id, &terminal)
            .map_err(adapter_budget_error)?;
        self.open.remove(&reservation.call_id);
        self.traces.push(terminal);
        Ok(())
    }

    fn close_unfinished_as_failed(&mut self) -> anyhow::Result<bool> {
        let had_unfinished = !self.open.is_empty();
        let unfinished: Vec<_> = self.open.values().cloned().collect();
        for reserved in unfinished {
            let operation_id = self.operation_id(&reserved.call_id);
            if let Some(authoritative) = self.authoritative_terminal(&operation_id)? {
                self.open.remove(&authoritative.call_id);
                self.traces.push(authoritative);
                continue;
            }
            let terminal = SpendTrace {
                status: SpendStatus::Failed,
                input_units: 0,
                output_units: 0,
                // Unknown usage is charged at the reserved ceiling; it is never treated as zero.
                cost_micro_usd: reserved.cost_micro_usd,
                ..reserved
            };
            self.persist_terminal(&operation_id, &terminal)?;
            self.open.remove(&terminal.call_id);
            self.traces.push(terminal);
        }
        Ok(had_unfinished)
    }
}

fn adapter_budget_error(error: anyhow::Error) -> AdapterFailure {
    AdapterFailure::Failed(error.to_string())
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RetrievedEvidence {
    pub evidence_id: String,
    pub lane: RecallLane,
    pub artifact: CapturedArtifact,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rendered_text: Option<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AdapterExecutionProof {
    pub effective_lane: RecallLane,
    pub effective_reader_mode: ReaderMode,
    pub text_branch_candidates: Vec<BranchCandidate>,
    pub native_branch_candidates: Vec<BranchCandidate>,
    pub collapse_clusters: Vec<CollapseCluster>,
    pub oracle_regions_applied: Vec<EvidenceRegion>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BranchCandidate {
    pub candidate_id: String,
    pub evidence_id: String,
    pub lane: RecallLane,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub profile_id: Option<String>,
    pub score: f32,
    pub artifact: CapturedArtifact,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CollapseCluster {
    pub representative_candidate_id: String,
    pub member_candidate_ids: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ReaderInputProof {
    pub evidence_id: String,
    pub mode: ReaderMode,
    pub binding_id: String,
    pub verified_sha256: String,
    pub byte_len: u64,
}

#[derive(Clone, Debug)]
struct VerifiedEvidenceMaterial {
    source_bytes: Vec<u8>,
    text_projection: Option<Vec<u8>>,
}

#[derive(Clone, Debug)]
struct VerifiedCorpusMaterial {
    by_evidence: BTreeMap<String, VerifiedEvidenceMaterial>,
}

impl VerifiedCorpusMaterial {
    fn from_items(items: &[VerifiedEvidenceImport]) -> Self {
        Self {
            by_evidence: items
                .iter()
                .map(|item| {
                    (
                        item.evidence.evidence_id.clone(),
                        VerifiedEvidenceMaterial {
                            source_bytes: item.source_bytes.clone(),
                            text_projection: item
                                .evidence
                                .text_projection
                                .as_ref()
                                .map(|text| text.as_bytes().to_vec()),
                        },
                    )
                })
                .collect(),
        }
    }
}

/// Harness-owned binding resolver. Reader bytes can only enter an adapter through these methods;
/// each successful read records the exact authorized binding, byte length, and digest.
#[derive(Debug)]
pub struct ReaderJournal {
    mode: ReaderMode,
    artifacts: BTreeMap<String, CapturedArtifact>,
    material: VerifiedCorpusMaterial,
    reads: Vec<ReaderInputProof>,
}

impl ReaderJournal {
    fn new(mode: ReaderMode, corpus: &CapturedCorpus, material: &VerifiedCorpusMaterial) -> Self {
        Self {
            mode,
            artifacts: corpus
                .artifacts
                .iter()
                .map(|artifact| (artifact.evidence_id.clone(), artifact.clone()))
                .collect(),
            material: material.clone(),
            reads: Vec::new(),
        }
    }

    pub fn read_text_projection(
        &mut self,
        evidence_id: &str,
        output_binding_id: &str,
    ) -> Result<Vec<u8>, AdapterFailure> {
        if self.mode != ReaderMode::TextProjection {
            return Err(AdapterFailure::Failed(
                "text projection read attempted in source-blob mode".to_string(),
            ));
        }
        let artifact = self.authorized_artifact(evidence_id)?;
        let projection = artifact.projection.as_ref().ok_or_else(|| {
            AdapterFailure::Failed("text reader artifact has no projection".to_string())
        })?;
        if projection.output_binding_id.as_deref() != Some(output_binding_id) {
            return Err(AdapterFailure::Failed(
                "text reader requested an unauthorized projection binding".to_string(),
            ));
        }
        let bytes = self
            .material
            .by_evidence
            .get(evidence_id)
            .and_then(|material| material.text_projection.clone())
            .ok_or_else(|| AdapterFailure::Failed("projection bytes unavailable".to_string()))?;
        self.record_read(evidence_id, output_binding_id, &bytes)?;
        Ok(bytes)
    }

    pub fn read_source_blob(
        &mut self,
        evidence_id: &str,
        source_binding_id: &str,
    ) -> Result<Vec<u8>, AdapterFailure> {
        if self.mode != ReaderMode::SourceBlob {
            return Err(AdapterFailure::Failed(
                "source blob read attempted in text-projection mode".to_string(),
            ));
        }
        let artifact = self.authorized_artifact(evidence_id)?;
        if artifact.binding_id != source_binding_id {
            return Err(AdapterFailure::Failed(
                "blob reader requested an unauthorized source binding".to_string(),
            ));
        }
        let bytes = self
            .material
            .by_evidence
            .get(evidence_id)
            .map(|material| material.source_bytes.clone())
            .ok_or_else(|| AdapterFailure::Failed("source bytes unavailable".to_string()))?;
        self.record_read(evidence_id, source_binding_id, &bytes)?;
        Ok(bytes)
    }

    fn authorized_artifact(&self, evidence_id: &str) -> Result<&CapturedArtifact, AdapterFailure> {
        self.artifacts.get(evidence_id).ok_or_else(|| {
            AdapterFailure::Failed("reader requested evidence outside captured corpus".to_string())
        })
    }

    fn record_read(
        &mut self,
        evidence_id: &str,
        binding_id: &str,
        bytes: &[u8],
    ) -> Result<(), AdapterFailure> {
        if self
            .reads
            .iter()
            .any(|read| read.evidence_id == evidence_id)
        {
            return Err(AdapterFailure::Failed(
                "reader resolved the same evidence more than once".to_string(),
            ));
        }
        self.reads.push(ReaderInputProof {
            evidence_id: evidence_id.to_string(),
            mode: self.mode,
            binding_id: binding_id.to_string(),
            verified_sha256: sha256(bytes),
            byte_len: bytes.len() as u64,
        });
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AdapterFailure {
    CapabilityUnavailable(String),
    Failed(String),
}

impl std::fmt::Display for AdapterFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CapabilityUnavailable(message) => {
                write!(formatter, "adapter capability unavailable: {message}")
            }
            Self::Failed(message) => write!(formatter, "adapter recall failed: {message}"),
        }
    }
}

impl std::error::Error for AdapterFailure {}

pub trait MultimodalScorer {
    fn scorer_id(&self) -> &str;
    fn score(&self, rule: &ScoringRule, gold: &str, answer: &str) -> anyhow::Result<bool>;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct DeterministicScorer;

impl MultimodalScorer for DeterministicScorer {
    fn scorer_id(&self) -> &str {
        "membench-deterministic-v1"
    }

    fn score(&self, rule: &ScoringRule, gold: &str, answer: &str) -> anyhow::Result<bool> {
        match rule {
            ScoringRule::ExactNormalized => Ok(normalize_text(gold) == normalize_text(answer)),
            ScoringRule::ContainsAll { case_sensitive } => {
                let (gold, answer) = if *case_sensitive {
                    (gold.to_string(), answer.to_string())
                } else {
                    (gold.to_lowercase(), answer.to_lowercase())
                };
                let phrases: Vec<_> = gold
                    .split([';', ','])
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                    .collect();
                Ok(!phrases.is_empty()
                    && phrases
                        .iter()
                        .all(|phrase| contains_word_bounded(&answer, phrase)))
            }
            ScoringRule::External { evaluator } => anyhow::bail!(
                "scorer '{}' does not support external evaluator '{evaluator}'",
                self.scorer_id()
            ),
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CostLadderStep {
    ValidateApparatus,
    ProjectionControl,
    OfflineNative,
    ProviderPilot,
    StratifiedMedium,
    FullBenchmark,
}

impl CostLadderStep {
    pub fn permits_provider_calls(self) -> bool {
        self >= Self::ProviderPilot
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionBudget {
    pub max_step: CostLadderStep,
    pub max_provider_calls: u64,
    pub max_cost_micro_usd: u64,
}

impl ExecutionBudget {
    pub fn offline() -> Self {
        Self {
            max_step: CostLadderStep::OfflineNative,
            max_provider_calls: 0,
            max_cost_micro_usd: 0,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PreregisteredHypothesis {
    pub statement: String,
    pub mechanism: String,
    pub decision_gate: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MultimodalRunPlan {
    pub arm: ExperimentArm,
    pub budget: ExecutionBudget,
    pub hypothesis: PreregisteredHypothesis,
    /// Caller-controlled timestamp. Excluded from the deterministic run id.
    pub started_at: String,
    /// Unique caller-minted execution identity. Replicates must never collide.
    pub run_instance_id: String,
    /// Exact source revision; `unknown` is rejected.
    pub git_sha: String,
    /// Hash of effective model/provider/concurrency/judge configuration.
    pub effective_config_sha256: String,
    /// Local root used only for verified fixture import.
    pub asset_root: PathBuf,
    /// Durable JSONL ledger required before any provider-backed dispatch.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spend_ledger_path: Option<PathBuf>,
    /// Harness-owned durable registry. `run_instance_id` is claimed with create-new semantics.
    pub run_registry_root: PathBuf,
    /// Maximum verified source bytes accepted for one imported asset.
    pub max_import_asset_bytes: u64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MultimodalRunResult {
    pub schema: String,
    pub run_id: String,
    pub experiment_id: String,
    pub fixture_set_id: String,
    pub fixture_digest: String,
    pub arm: ExperimentArm,
    pub adapter: AdapterDescriptor,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reader: Option<ReaderDescriptor>,
    pub scorer_id: String,
    pub hypothesis: PreregisteredHypothesis,
    pub provenance: RunProvenance,
    pub metrics: MultimodalMetrics,
    pub cases: Vec<CaseResult>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RunProvenance {
    pub git_sha: String,
    pub started_at: String,
    pub provider_calls: u64,
    pub cost_micro_usd: u64,
    pub oracle_gold: bool,
    pub leaderboard_eligible: bool,
    pub budget: ExecutionBudget,
    pub effective_config_sha256: String,
    pub spend: Vec<SpendTrace>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MultimodalMetrics {
    pub correct: usize,
    pub total: usize,
    pub media_dependent_correct: usize,
    pub media_dependent_total: usize,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CaseResult {
    pub case_id: String,
    pub correct: bool,
    pub answer: String,
    pub requested_lane: RecallLane,
    pub fired_proof: FiredProof,
    pub retrieved: Vec<RetrievedEvidence>,
    pub reader_inputs: Vec<ReaderInputProof>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FiredProof {
    pub request_sha256: String,
    pub response_sha256: String,
    pub effective_lane: RecallLane,
    pub effective_reader_mode: ReaderMode,
    pub text_branch_fingerprint: String,
    pub native_branch_fingerprint: String,
    pub collapse_fingerprint: String,
    pub oracle_region_count: usize,
    pub reader_input_fingerprint: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bound_reader_request_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bound_reader_effective_request_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bound_reader_effective_input_fingerprint: Option<String>,
    pub retrieved_region_fingerprint: String,
}

pub fn run_experiment<A: MultimodalRecallAdapter, S: MultimodalScorer>(
    fixture: &MultimodalFixtureSet,
    plan: &MultimodalRunPlan,
    adapter: &mut A,
    scorer: &S,
) -> anyhow::Result<MultimodalRunResult> {
    anyhow::ensure!(
        plan.arm.cell != ExperimentCell::EReaderModality,
        "cell E must run through run_reader_modality_pair"
    );
    run_experiment_internal(fixture, plan, adapter, scorer, None, true)
}

/// Run cell E with one frozen hybrid retrieval artifact shared byte-identically by both readers.
pub fn run_reader_modality_pair<
    A: MultimodalRecallAdapter,
    T: MultimodalReaderAdapter,
    B: MultimodalReaderAdapter,
    S: MultimodalScorer,
>(
    fixture: &MultimodalFixtureSet,
    plan: &MultimodalRunPlan,
    adapter: &mut A,
    text_reader: &mut T,
    blob_reader: &mut B,
    scorer: &S,
) -> anyhow::Result<(MultimodalRunResult, MultimodalRunResult)> {
    anyhow::ensure!(
        plan.arm.cell == ExperimentCell::EReaderModality,
        "paired runner requires cell E"
    );
    run_reader_modality_pair_internal(fixture, plan, adapter, text_reader, blob_reader, scorer)
}

fn validate_run_plan(plan: &MultimodalRunPlan) -> anyhow::Result<()> {
    plan.arm.validate()?;
    chrono::DateTime::parse_from_rfc3339(&plan.started_at)
        .with_context(|| "started_at must be an RFC3339 timestamp")?;
    validate_id("run_instance_id", &plan.run_instance_id)?;
    anyhow::ensure!(plan.git_sha != "unknown", "git_sha must be exact");
    validate_sha256("effective config digest", &plan.effective_config_sha256)?;
    anyhow::ensure!(
        plan.max_import_asset_bytes > 0,
        "max_import_asset_bytes must be positive"
    );
    if !plan.budget.max_step.permits_provider_calls() {
        anyhow::ensure!(
            plan.budget.max_provider_calls == 0 && plan.budget.max_cost_micro_usd == 0,
            "offline cost-ladder steps categorically require zero provider-call and cost maxima"
        );
    } else {
        anyhow::ensure!(
            plan.budget.max_provider_calls > 0 && plan.budget.max_cost_micro_usd > 0,
            "provider cost-ladder steps require positive call and cost maxima"
        );
        anyhow::ensure!(
            plan.spend_ledger_path.is_some(),
            "provider-backed runs require a durable spend_ledger_path"
        );
    }
    for (name, value) in [
        ("hypothesis statement", &plan.hypothesis.statement),
        ("hypothesis mechanism", &plan.hypothesis.mechanism),
        ("hypothesis decision gate", &plan.hypothesis.decision_gate),
    ] {
        anyhow::ensure!(!value.trim().is_empty(), "{name} must be non-empty");
    }
    Ok(())
}

fn claim_run_instance(plan: &MultimodalRunPlan) -> anyhow::Result<()> {
    std::fs::create_dir_all(&plan.run_registry_root).with_context(|| {
        format!(
            "create multimodal run registry {}",
            plan.run_registry_root.display()
        )
    })?;
    let metadata = std::fs::symlink_metadata(&plan.run_registry_root).with_context(|| {
        format!(
            "stat multimodal run registry {}",
            plan.run_registry_root.display()
        )
    })?;
    anyhow::ensure!(
        metadata.file_type().is_dir() && !metadata.file_type().is_symlink(),
        "run registry root must be a real directory"
    );
    let claim_path = plan
        .run_registry_root
        .join(format!("{}.json", plan.run_instance_id));
    use std::io::Write;
    let mut claim = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&claim_path)
        .with_context(|| {
            format!(
                "run instance '{}' is already claimed or cannot be claimed at {}",
                plan.run_instance_id,
                claim_path.display()
            )
        })?;
    serde_json::to_writer(
        &mut claim,
        &serde_json::json!({
            "run_instance_id": plan.run_instance_id,
            "started_at": plan.started_at,
            "git_sha": plan.git_sha,
            "effective_config_sha256": plan.effective_config_sha256,
        }),
    )?;
    claim.write_all(b"\n")?;
    claim.sync_all()?;
    Ok(())
}

fn validate_reader_descriptor(
    descriptor: &ReaderDescriptor,
    expected_mode: ReaderMode,
) -> anyhow::Result<()> {
    validate_id("reader_id", &descriptor.reader_id)?;
    non_empty("reader version", &descriptor.reader_version)?;
    anyhow::ensure!(
        descriptor.mode == expected_mode,
        "reader '{}' advertises {:?}, expected {:?}",
        descriptor.reader_id,
        descriptor.mode,
        expected_mode
    );
    Ok(())
}

fn resolve_bound_reader_request(
    case: &MultimodalCase,
    corpus: &CapturedCorpus,
    material: &VerifiedCorpusMaterial,
    retrieved: &[RetrievedEvidence],
    frozen_retrieval_sha256: &str,
    mode: ReaderMode,
    budget: RemainingBudget,
) -> anyhow::Result<(
    BoundReaderRequest,
    Vec<ReaderInputProof>,
    Vec<Option<String>>,
)> {
    let mut journal = ReaderJournal::new(mode, corpus, material);
    let mut inputs = Vec::with_capacity(retrieved.len());
    let mut rendered = Vec::with_capacity(retrieved.len());
    for item in retrieved {
        match mode {
            ReaderMode::TextProjection => {
                let projection = item
                    .artifact
                    .projection
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("text reader artifact has no projection"))?;
                let binding_id = projection
                    .output_binding_id
                    .as_deref()
                    .ok_or_else(|| anyhow::anyhow!("projection output binding missing"))?;
                let bytes = journal
                    .read_text_projection(&item.evidence_id, binding_id)
                    .map_err(anyhow::Error::new)?;
                let text = String::from_utf8(bytes.clone())
                    .with_context(|| "registered text projection is not UTF-8")?;
                inputs.push(BoundReaderInput {
                    evidence_id: item.evidence_id.clone(),
                    mode,
                    binding_id: binding_id.to_string(),
                    media_type: "text/plain; charset=utf-8".to_string(),
                    verified_sha256: sha256(&bytes),
                    bytes,
                });
                rendered.push(Some(text));
            }
            ReaderMode::SourceBlob => {
                let bytes = journal
                    .read_source_blob(&item.evidence_id, &item.artifact.binding_id)
                    .map_err(anyhow::Error::new)?;
                inputs.push(BoundReaderInput {
                    evidence_id: item.evidence_id.clone(),
                    mode,
                    binding_id: item.artifact.binding_id.clone(),
                    media_type: item.artifact.blob.detected_media_type.clone(),
                    verified_sha256: sha256(&bytes),
                    bytes,
                });
                rendered.push(None);
            }
        }
    }
    let invocation = BoundReaderInvocation {
        case_id: case.case_id.clone(),
        question: ContentSpec {
            text: case.question.text.clone(),
            media: Vec::new(),
        },
        frozen_retrieval_sha256: frozen_retrieval_sha256.to_string(),
        inputs,
    };
    let request_sha256 = stable_json_hash(&(invocation.clone(), budget.clone()))?;
    Ok((
        BoundReaderRequest {
            request_sha256,
            invocation,
            budget,
        },
        journal.reads,
        rendered,
    ))
}

fn validate_bound_reader_response(
    request: &BoundReaderRequest,
    response: &BoundReaderResponse,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        !response.answer.trim().is_empty(),
        "reader returned empty answer"
    );
    anyhow::ensure!(
        request.request_sha256
            == stable_json_hash(&(request.invocation.clone(), request.budget.clone()))?,
        "harness reader request hash is invalid"
    );
    anyhow::ensure!(
        response.effective_request_sha256 == request.request_sha256,
        "reader effective request hash does not match invocation"
    );
    let expected_inputs: Vec<_> = request
        .invocation
        .inputs
        .iter()
        .map(|input| sha256(&input.bytes))
        .collect();
    anyhow::ensure!(
        request
            .invocation
            .inputs
            .iter()
            .zip(&expected_inputs)
            .all(|(input, digest)| input.verified_sha256 == *digest),
        "harness reader input digest is invalid"
    );
    anyhow::ensure!(
        response.effective_input_sha256 == expected_inputs,
        "reader effective input hashes do not match resolved bytes"
    );
    Ok(())
}

fn run_reader_modality_pair_internal<
    A: MultimodalRecallAdapter,
    T: MultimodalReaderAdapter,
    B: MultimodalReaderAdapter,
    S: MultimodalScorer,
>(
    fixture: &MultimodalFixtureSet,
    plan: &MultimodalRunPlan,
    adapter: &mut A,
    text_reader: &mut T,
    blob_reader: &mut B,
    scorer: &S,
) -> anyhow::Result<(MultimodalRunResult, MultimodalRunResult)> {
    validate_fixture(fixture)?;
    validate_run_plan(plan)?;
    claim_run_instance(plan)?;
    let descriptor = adapter.descriptor();
    validate_descriptor(&descriptor)?;
    preflight_capabilities(fixture, &plan.arm, &descriptor.capabilities)?;
    let text_reader_descriptor = text_reader.descriptor();
    let blob_reader_descriptor = blob_reader.descriptor();
    validate_reader_descriptor(&text_reader_descriptor, ReaderMode::TextProjection)?;
    validate_reader_descriptor(&blob_reader_descriptor, ReaderMode::SourceBlob)?;

    let mut pair_calls = 0u64;
    let mut pair_cost = 0u64;
    let mut shared_calls = 0u64;
    let mut shared_cost = 0u64;
    let mut text_calls = 0u64;
    let mut text_cost = 0u64;
    let mut blob_calls = 0u64;
    let mut blob_cost = 0u64;
    let mut shared_spend = Vec::new();
    let mut text_spend = Vec::new();
    let mut blob_spend = Vec::new();
    let mut captured_by_corpus = BTreeMap::new();
    let mut material_by_corpus = BTreeMap::new();
    for corpus in &fixture.corpora {
        let verified_items =
            verify_corpus_assets(corpus, &plan.asset_root, plan.max_import_asset_bytes)?;
        let material = VerifiedCorpusMaterial::from_items(&verified_items);
        let remaining = RemainingBudget {
            provider_calls: plan.budget.max_provider_calls.saturating_sub(pair_calls),
            cost_micro_usd: plan.budget.max_cost_micro_usd.saturating_sub(pair_cost),
        };
        let mut journal = SpendJournal::new(
            &plan.run_instance_id,
            &plan.effective_config_sha256,
            &format!("import: {}", corpus.corpus_id),
            plan.spend_ledger_path.as_deref(),
            remaining.clone(),
            plan.budget.max_step.permits_provider_calls(),
        );
        let captured = adapter.import_corpus(
            &CorpusImportRequest {
                corpus_id: corpus.corpus_id.clone(),
                items: verified_items,
            },
            &mut journal,
        );
        let had_unfinished = journal.close_unfinished_as_failed()?;
        let spent = validate_spend(&journal.traces, &remaining)?;
        pair_calls = pair_calls.saturating_add(spent.0);
        pair_cost = pair_cost.saturating_add(spent.1);
        shared_calls = shared_calls.saturating_add(spent.0);
        shared_cost = shared_cost.saturating_add(spent.1);
        shared_spend.extend(journal.traces);
        enforce_budget(&plan.budget, pair_calls, pair_cost)?;
        let captured = captured
            .map_err(|error| anyhow::anyhow!("import corpus '{}': {error}", corpus.corpus_id))?;
        anyhow::ensure!(
            !had_unfinished,
            "import returned with unfinished reservation"
        );
        validate_captured_corpus(corpus, &captured)?;
        captured_by_corpus.insert(corpus.corpus_id.as_str(), captured);
        material_by_corpus.insert(corpus.corpus_id.as_str(), material);
    }

    let mut text_cases = Vec::with_capacity(fixture.cases.len());
    let mut blob_cases = Vec::with_capacity(fixture.cases.len());
    for case in &fixture.cases {
        let corpus = captured_by_corpus
            .get(case.corpus_id.as_str())
            .ok_or_else(|| anyhow::anyhow!("case references absent corpus"))?;
        let material = material_by_corpus
            .get(case.corpus_id.as_str())
            .ok_or_else(|| anyhow::anyhow!("case has no verified material"))?;
        let retrieval_budget = RemainingBudget {
            provider_calls: plan.budget.max_provider_calls.saturating_sub(pair_calls),
            cost_micro_usd: plan.budget.max_cost_micro_usd.saturating_sub(pair_cost),
        };
        let retrieval_request = RetrievalRequest {
            case_id: case.case_id.clone(),
            corpus_id: case.corpus_id.clone(),
            question: ContentSpec {
                text: case.question.text.clone(),
                media: Vec::new(),
            },
            corpus: corpus.clone(),
            lane: RecallLane::HybridCollapsed,
            budget: retrieval_budget.clone(),
        };
        let mut journal = SpendJournal::new(
            &plan.run_instance_id,
            &plan.effective_config_sha256,
            &format!("retrieve: {}", case.case_id),
            plan.spend_ledger_path.as_deref(),
            retrieval_budget.clone(),
            plan.budget.max_step.permits_provider_calls(),
        );
        let retrieval = adapter.retrieve(&retrieval_request, &mut journal);
        let had_unfinished = journal.close_unfinished_as_failed()?;
        let spent = validate_spend(&journal.traces, &retrieval_budget)?;
        pair_calls = pair_calls.saturating_add(spent.0);
        pair_cost = pair_cost.saturating_add(spent.1);
        shared_calls = shared_calls.saturating_add(spent.0);
        shared_cost = shared_cost.saturating_add(spent.1);
        shared_spend.extend(journal.traces);
        enforce_budget(&plan.budget, pair_calls, pair_cost)?;
        let retrieval = retrieval
            .map_err(|error| anyhow::anyhow!("case '{}' retrieval: {error}", case.case_id))?;
        anyhow::ensure!(
            !had_unfinished,
            "retrieval returned with unfinished reservation"
        );
        anyhow::ensure!(
            retrieval
                .retrieved
                .iter()
                .all(|item| item.rendered_text.is_none()),
            "retrieval-only adapter attempted to return reader material"
        );
        let frozen_items = frozen_retrieval_identity(&retrieval.retrieved);
        let frozen = FrozenRetrieval {
            fingerprint: stable_json_hash(&frozen_items)?,
            retrieved: frozen_items,
        };

        let text_budget = RemainingBudget {
            provider_calls: plan.budget.max_provider_calls.saturating_sub(pair_calls),
            cost_micro_usd: plan.budget.max_cost_micro_usd.saturating_sub(pair_cost),
        };
        let (text_request, text_inputs, text_rendered) = resolve_bound_reader_request(
            case,
            corpus,
            material,
            &retrieval.retrieved,
            &frozen.fingerprint,
            ReaderMode::TextProjection,
            text_budget.clone(),
        )?;
        let mut validation_retrieved = retrieval.retrieved.clone();
        for (item, rendered_text) in validation_retrieved.iter_mut().zip(&text_rendered) {
            item.rendered_text = rendered_text.clone();
        }
        let validation_request = RecallRequest {
            case_id: case.case_id.clone(),
            corpus_id: case.corpus_id.clone(),
            question: ContentSpec {
                text: case.question.text.clone(),
                media: Vec::new(),
            },
            corpus: corpus.clone(),
            lane: RecallLane::HybridCollapsed,
            reader_mode: ReaderMode::TextProjection,
            oracle_regions: Vec::new(),
            frozen_retrieval: Some(frozen.clone()),
            budget: RemainingBudget {
                provider_calls: 0,
                cost_micro_usd: 0,
            },
        };
        validate_response(
            case,
            &validation_request,
            &RecallResponse {
                answer: "retrieval-only validation".to_string(),
                retrieved: validation_retrieved,
                execution: AdapterExecutionProof {
                    effective_lane: retrieval.execution.effective_lane,
                    effective_reader_mode: ReaderMode::TextProjection,
                    text_branch_candidates: retrieval.execution.text_branch_candidates.clone(),
                    native_branch_candidates: retrieval.execution.native_branch_candidates.clone(),
                    collapse_clusters: retrieval.execution.collapse_clusters.clone(),
                    oracle_regions_applied: retrieval.execution.oracle_regions_applied.clone(),
                },
            },
            &text_inputs,
        )?;
        let mut text_journal = SpendJournal::new(
            &plan.run_instance_id,
            &plan.effective_config_sha256,
            &format!("reader:text:{}", case.case_id),
            plan.spend_ledger_path.as_deref(),
            text_budget.clone(),
            plan.budget.max_step.permits_provider_calls(),
        );
        let text_response = text_reader.read(&text_request, &mut text_journal);
        let text_unfinished = text_journal.close_unfinished_as_failed()?;
        let spent = validate_spend(&text_journal.traces, &text_budget)?;
        pair_calls = pair_calls.saturating_add(spent.0);
        pair_cost = pair_cost.saturating_add(spent.1);
        text_calls = text_calls.saturating_add(spent.0);
        text_cost = text_cost.saturating_add(spent.1);
        text_spend.extend(text_journal.traces);
        enforce_budget(&plan.budget, pair_calls, pair_cost)?;
        let text_response = text_response
            .map_err(|error| anyhow::anyhow!("case '{}' text reader: {error}", case.case_id))?;
        anyhow::ensure!(
            !text_unfinished,
            "text reader returned unfinished reservation"
        );
        validate_bound_reader_response(&text_request, &text_response)?;

        let blob_budget = RemainingBudget {
            provider_calls: plan.budget.max_provider_calls.saturating_sub(pair_calls),
            cost_micro_usd: plan.budget.max_cost_micro_usd.saturating_sub(pair_cost),
        };
        let (blob_request, blob_inputs, blob_rendered) = resolve_bound_reader_request(
            case,
            corpus,
            material,
            &retrieval.retrieved,
            &frozen.fingerprint,
            ReaderMode::SourceBlob,
            blob_budget.clone(),
        )?;
        let mut blob_journal = SpendJournal::new(
            &plan.run_instance_id,
            &plan.effective_config_sha256,
            &format!("reader:blob:{}", case.case_id),
            plan.spend_ledger_path.as_deref(),
            blob_budget.clone(),
            plan.budget.max_step.permits_provider_calls(),
        );
        let blob_response = blob_reader.read(&blob_request, &mut blob_journal);
        let blob_unfinished = blob_journal.close_unfinished_as_failed()?;
        let spent = validate_spend(&blob_journal.traces, &blob_budget)?;
        pair_calls = pair_calls.saturating_add(spent.0);
        pair_cost = pair_cost.saturating_add(spent.1);
        blob_calls = blob_calls.saturating_add(spent.0);
        blob_cost = blob_cost.saturating_add(spent.1);
        blob_spend.extend(blob_journal.traces);
        enforce_budget(&plan.budget, pair_calls, pair_cost)?;
        let blob_response = blob_response
            .map_err(|error| anyhow::anyhow!("case '{}' blob reader: {error}", case.case_id))?;
        anyhow::ensure!(
            !blob_unfinished,
            "blob reader returned unfinished reservation"
        );
        validate_bound_reader_response(&blob_request, &blob_response)?;

        let build_case = |mode: ReaderMode,
                          response: &BoundReaderResponse,
                          inputs: Vec<ReaderInputProof>,
                          rendered: Vec<Option<String>>|
         -> anyhow::Result<CaseResult> {
            let mut retrieved = retrieval.retrieved.clone();
            for (item, rendered_text) in retrieved.iter_mut().zip(rendered) {
                item.rendered_text = rendered_text;
            }
            let execution = AdapterExecutionProof {
                effective_lane: retrieval.execution.effective_lane,
                effective_reader_mode: mode,
                text_branch_candidates: retrieval.execution.text_branch_candidates.clone(),
                native_branch_candidates: retrieval.execution.native_branch_candidates.clone(),
                collapse_clusters: retrieval.execution.collapse_clusters.clone(),
                oracle_regions_applied: retrieval.execution.oracle_regions_applied.clone(),
            };
            let request = RecallRequest {
                case_id: case.case_id.clone(),
                corpus_id: case.corpus_id.clone(),
                question: ContentSpec {
                    text: case.question.text.clone(),
                    media: Vec::new(),
                },
                corpus: corpus.clone(),
                lane: RecallLane::HybridCollapsed,
                reader_mode: mode,
                oracle_regions: Vec::new(),
                frozen_retrieval: Some(frozen.clone()),
                budget: RemainingBudget {
                    provider_calls: 0,
                    cost_micro_usd: 0,
                },
            };
            let recalled = RecallResponse {
                answer: response.answer.clone(),
                retrieved: retrieved.clone(),
                execution,
            };
            validate_response(case, &request, &recalled, &inputs)?;
            let correct = scorer.score(&case.gold.scoring, &case.gold.value, &response.answer)?;
            let mut proof = fired_proof(&request, &recalled, &inputs)?;
            proof.bound_reader_request_sha256 = Some(match mode {
                ReaderMode::TextProjection => text_request.request_sha256.clone(),
                ReaderMode::SourceBlob => blob_request.request_sha256.clone(),
            });
            proof.bound_reader_effective_request_sha256 =
                Some(response.effective_request_sha256.clone());
            proof.bound_reader_effective_input_fingerprint =
                Some(stable_json_hash(&response.effective_input_sha256)?);
            Ok(CaseResult {
                case_id: case.case_id.clone(),
                correct,
                answer: response.answer.clone(),
                requested_lane: RecallLane::HybridCollapsed,
                fired_proof: proof,
                retrieved,
                reader_inputs: inputs,
            })
        };
        text_cases.push(build_case(
            ReaderMode::TextProjection,
            &text_response,
            text_inputs,
            text_rendered,
        )?);
        blob_cases.push(build_case(
            ReaderMode::SourceBlob,
            &blob_response,
            blob_inputs,
            blob_rendered,
        )?);
    }

    let make_result = |mode: ReaderMode,
                       reader: ReaderDescriptor,
                       cases: Vec<CaseResult>,
                       reader_calls: u64,
                       reader_cost: u64,
                       reader_spend: Vec<SpendTrace>|
     -> anyhow::Result<MultimodalRunResult> {
        let mut arm = plan.arm.clone();
        arm.reader_mode = mode;
        let fixture_digest = stable_json_hash(fixture)?;
        let experiment_id = format!(
            "mmx-{}",
            &stable_json_hash(&serde_json::json!({
                "fixture_digest": fixture_digest,
                "arm": arm,
                "adapter": descriptor,
                "reader": reader,
                "scorer": scorer.scorer_id(),
                "hypothesis": plan.hypothesis,
                "budget": plan.budget,
                "effective_config_sha256": plan.effective_config_sha256,
                "git_sha": plan.git_sha,
            }))?[..16]
        );
        let media_ids: HashSet<_> = fixture
            .cases
            .iter()
            .filter(|case| case.media_dependent)
            .map(|case| case.case_id.as_str())
            .collect();
        let metrics = MultimodalMetrics {
            correct: cases.iter().filter(|case| case.correct).count(),
            total: cases.len(),
            media_dependent_correct: cases
                .iter()
                .filter(|case| case.correct && media_ids.contains(case.case_id.as_str()))
                .count(),
            media_dependent_total: media_ids.len(),
        };
        let mut spend = shared_spend.clone();
        spend.extend(reader_spend);
        Ok(MultimodalRunResult {
            schema: RESULT_SCHEMA.to_string(),
            run_id: format!(
                "{}-{}",
                plan.run_instance_id,
                match mode {
                    ReaderMode::TextProjection => "text",
                    ReaderMode::SourceBlob => "blob",
                }
            ),
            experiment_id,
            fixture_set_id: fixture.fixture_set_id.clone(),
            fixture_digest,
            arm,
            adapter: descriptor.clone(),
            reader: Some(reader),
            scorer_id: scorer.scorer_id().to_string(),
            hypothesis: plan.hypothesis.clone(),
            provenance: RunProvenance {
                git_sha: plan.git_sha.clone(),
                started_at: plan.started_at.clone(),
                provider_calls: shared_calls.saturating_add(reader_calls),
                cost_micro_usd: shared_cost.saturating_add(reader_cost),
                oracle_gold: false,
                leaderboard_eligible: false,
                budget: plan.budget.clone(),
                effective_config_sha256: plan.effective_config_sha256.clone(),
                spend,
            },
            metrics,
            cases,
        })
    };
    Ok((
        make_result(
            ReaderMode::TextProjection,
            text_reader_descriptor,
            text_cases,
            text_calls,
            text_cost,
            text_spend,
        )?,
        make_result(
            ReaderMode::SourceBlob,
            blob_reader_descriptor,
            blob_cases,
            blob_calls,
            blob_cost,
            blob_spend,
        )?,
    ))
}

fn run_experiment_internal<A: MultimodalRecallAdapter, S: MultimodalScorer>(
    fixture: &MultimodalFixtureSet,
    plan: &MultimodalRunPlan,
    adapter: &mut A,
    scorer: &S,
    frozen: Option<&BTreeMap<String, FrozenRetrieval>>,
    claim_instance: bool,
) -> anyhow::Result<MultimodalRunResult> {
    validate_fixture(fixture)?;
    validate_run_plan(plan)?;
    if claim_instance {
        claim_run_instance(plan)?;
    }

    let descriptor = adapter.descriptor();
    validate_descriptor(&descriptor)?;
    preflight_capabilities(fixture, &plan.arm, &descriptor.capabilities)?;

    let mut provider_calls = 0u64;
    let mut cost_micro_usd = 0u64;
    let mut spend = Vec::new();
    let mut captured_by_corpus = BTreeMap::new();
    let mut material_by_corpus = BTreeMap::new();
    for corpus in &fixture.corpora {
        let verified_items =
            verify_corpus_assets(corpus, &plan.asset_root, plan.max_import_asset_bytes)?;
        let verified_material = VerifiedCorpusMaterial::from_items(&verified_items);
        let remaining = RemainingBudget {
            provider_calls: plan
                .budget
                .max_provider_calls
                .saturating_sub(provider_calls),
            cost_micro_usd: plan
                .budget
                .max_cost_micro_usd
                .saturating_sub(cost_micro_usd),
        };
        let mut journal = SpendJournal::new(
            &plan.run_instance_id,
            &plan.effective_config_sha256,
            &format!("import: {}", corpus.corpus_id),
            plan.spend_ledger_path.as_deref(),
            remaining.clone(),
            plan.budget.max_step.permits_provider_calls(),
        );
        let captured = adapter.import_corpus(
            &CorpusImportRequest {
                corpus_id: corpus.corpus_id.clone(),
                items: verified_items,
            },
            &mut journal,
        );
        let had_unfinished = journal.close_unfinished_as_failed()?;
        let import_cost = validate_spend(&journal.traces, &remaining)?;
        provider_calls = provider_calls.saturating_add(import_cost.0);
        cost_micro_usd = cost_micro_usd.saturating_add(import_cost.1);
        spend.extend(journal.traces);
        enforce_budget(&plan.budget, provider_calls, cost_micro_usd)?;
        let captured = captured
            .map_err(|error| anyhow::anyhow!("import corpus '{}': {error}", corpus.corpus_id))?;
        anyhow::ensure!(
            !had_unfinished,
            "import corpus '{}' returned with an unfinished provider reservation",
            corpus.corpus_id
        );
        validate_captured_corpus(corpus, &captured)?;
        captured_by_corpus.insert(corpus.corpus_id.as_str(), captured);
        material_by_corpus.insert(corpus.corpus_id.as_str(), verified_material);
    }

    let fixture_digest = stable_json_hash(fixture)?;
    let run_identity = serde_json::json!({
        "fixture_digest": fixture_digest,
        "arm": plan.arm,
        "adapter": descriptor,
        "scorer": scorer.scorer_id(),
        "hypothesis": plan.hypothesis,
        "budget": plan.budget,
        "effective_config_sha256": plan.effective_config_sha256,
        "git_sha": plan.git_sha,
    });
    let experiment_id = format!("mmx-{}", &stable_json_hash(&run_identity)?[..16]);
    let run_id = plan.run_instance_id.clone();
    let mut results = Vec::with_capacity(fixture.cases.len());

    for case in &fixture.cases {
        let lane = plan.arm.requested_lane(case);
        let oracle_regions = if plan.arm.cell == ExperimentCell::DOracleLane {
            case.oracle_evidence.clone()
        } else {
            Vec::new()
        };
        let request = RecallRequest {
            case_id: case.case_id.clone(),
            corpus_id: case.corpus_id.clone(),
            // The recall-lane experiment treats question media as a separate future variable. A
            // text reader must never gain pixel access through the query side-channel.
            question: ContentSpec {
                text: case.question.text.clone(),
                media: Vec::new(),
            },
            corpus: captured_by_corpus
                .get(case.corpus_id.as_str())
                .cloned()
                .ok_or_else(|| {
                    anyhow::anyhow!("case '{}' references absent corpus", case.case_id)
                })?,
            lane,
            reader_mode: plan.arm.reader_mode,
            oracle_regions,
            frozen_retrieval: frozen.and_then(|items| items.get(&case.case_id)).cloned(),
            budget: RemainingBudget {
                provider_calls: plan
                    .budget
                    .max_provider_calls
                    .saturating_sub(provider_calls),
                cost_micro_usd: plan
                    .budget
                    .max_cost_micro_usd
                    .saturating_sub(cost_micro_usd),
            },
        };
        let mut journal = SpendJournal::new(
            &plan.run_instance_id,
            &plan.effective_config_sha256,
            &case.case_id,
            plan.spend_ledger_path.as_deref(),
            request.budget.clone(),
            plan.budget.max_step.permits_provider_calls(),
        );
        let material = material_by_corpus
            .get(case.corpus_id.as_str())
            .ok_or_else(|| anyhow::anyhow!("case '{}' has no verified material", case.case_id))?;
        let mut reader = ReaderJournal::new(request.reader_mode, &request.corpus, material);
        let response = adapter.recall(&request, &mut journal, &mut reader);
        let had_unfinished = journal.close_unfinished_as_failed()?;
        let response_cost = validate_spend(&journal.traces, &request.budget)?;
        provider_calls = provider_calls.saturating_add(response_cost.0);
        cost_micro_usd = cost_micro_usd.saturating_add(response_cost.1);
        spend.extend(journal.traces);
        enforce_budget(&plan.budget, provider_calls, cost_micro_usd)?;
        let response =
            response.map_err(|error| anyhow::anyhow!("case '{}': {error}", case.case_id))?;
        anyhow::ensure!(
            !had_unfinished,
            "case '{}' returned with an unfinished provider reservation",
            case.case_id
        );
        validate_response(case, &request, &response, &reader.reads)?;
        let correct = scorer
            .score(&case.gold.scoring, &case.gold.value, &response.answer)
            .with_context(|| format!("score case '{}'", case.case_id))?;
        results.push(CaseResult {
            case_id: case.case_id.clone(),
            correct,
            answer: response.answer.clone(),
            requested_lane: lane,
            fired_proof: fired_proof(&request, &response, &reader.reads)?,
            retrieved: response.retrieved.clone(),
            reader_inputs: reader.reads,
        });
    }

    let media_ids: HashSet<_> = fixture
        .cases
        .iter()
        .filter(|case| case.media_dependent)
        .map(|case| case.case_id.as_str())
        .collect();
    let metrics = MultimodalMetrics {
        correct: results.iter().filter(|result| result.correct).count(),
        total: results.len(),
        media_dependent_correct: results
            .iter()
            .filter(|result| result.correct && media_ids.contains(result.case_id.as_str()))
            .count(),
        media_dependent_total: media_ids.len(),
    };
    let oracle_gold = plan.arm.cell == ExperimentCell::DOracleLane;
    Ok(MultimodalRunResult {
        schema: RESULT_SCHEMA.to_string(),
        run_id,
        experiment_id,
        fixture_set_id: fixture.fixture_set_id.clone(),
        fixture_digest,
        arm: plan.arm.clone(),
        adapter: descriptor,
        reader: None,
        scorer_id: scorer.scorer_id().to_string(),
        hypothesis: plan.hypothesis.clone(),
        provenance: RunProvenance {
            git_sha: plan.git_sha.clone(),
            started_at: plan.started_at.clone(),
            provider_calls,
            cost_micro_usd,
            oracle_gold,
            // This pre-release apparatus is categorically non-rankable until it emits canonical
            // records that pass the repository's normal eligibility evaluator.
            leaderboard_eligible: false,
            budget: plan.budget.clone(),
            effective_config_sha256: plan.effective_config_sha256.clone(),
            spend,
        },
        metrics,
        cases: results,
    })
}

pub fn load_fixture_file(path: &Path) -> anyhow::Result<MultimodalFixtureSet> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("read multimodal fixture {}", path.display()))?;
    let fixture: MultimodalFixtureSet = serde_json::from_slice(&bytes)
        .with_context(|| format!("parse multimodal fixture {}", path.display()))?;
    validate_fixture(&fixture)
        .with_context(|| format!("validate multimodal fixture {}", path.display()))?;
    Ok(fixture)
}

fn verify_corpus_assets(
    corpus: &MultimodalCorpus,
    asset_root: &Path,
    max_asset_bytes: u64,
) -> anyhow::Result<Vec<VerifiedEvidenceImport>> {
    let mut verified = Vec::with_capacity(corpus.evidence.len());
    for item in &corpus.evidence {
        let relative = Path::new(&item.source_media.locator);
        ensure_safe_relative(relative, "media locator")?;
        let source_bytes = read_fixture_asset(asset_root, relative, max_asset_bytes)?;
        anyhow::ensure!(
            source_bytes.len() as u64 == item.source_media.size_bytes,
            "fixture media size mismatch for '{}'",
            item.source_media.asset_id
        );
        anyhow::ensure!(
            sha256(&source_bytes) == item.source_media.sha256,
            "fixture media digest mismatch for '{}'",
            item.source_media.asset_id
        );
        validate_importable_media(&item.source_media.media_type, &source_bytes)?;
        verified.push(VerifiedEvidenceImport {
            evidence: item.clone(),
            source_bytes,
        });
    }
    Ok(verified)
}

#[cfg(unix)]
fn read_fixture_asset(
    asset_root: &Path,
    relative: &Path,
    max_asset_bytes: u64,
) -> anyhow::Result<Vec<u8>> {
    use std::ffi::CString;
    use std::io::Read;
    use std::os::fd::{AsRawFd, FromRawFd};
    use std::os::unix::ffi::OsStrExt;
    use std::os::unix::fs::{MetadataExt, OpenOptionsExt};

    let root = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open(asset_root)
        .with_context(|| {
            format!(
                "open asset root without following links {}",
                asset_root.display()
            )
        })?;
    let components: Vec<_> = relative.components().collect();
    anyhow::ensure!(!components.is_empty(), "media locator is empty");
    let mut directory = root;
    let mut source = None;
    for (index, component) in components.iter().enumerate() {
        let Component::Normal(name) = component else {
            anyhow::bail!("media locator contains an unsafe path component");
        };
        let name = CString::new(name.as_bytes())
            .map_err(|_| anyhow::anyhow!("media locator contains NUL"))?;
        let final_component = index + 1 == components.len();
        let flags = libc::O_RDONLY
            | libc::O_NOFOLLOW
            | libc::O_CLOEXEC
            | if final_component {
                0
            } else {
                libc::O_DIRECTORY
            };
        // SAFETY: `directory` owns a valid directory fd, `name` is NUL-terminated, and the
        // returned descriptor is immediately wrapped in `File` for single ownership.
        let fd = unsafe { libc::openat(directory.as_raw_fd(), name.as_ptr(), flags) };
        if fd < 0 {
            return Err(std::io::Error::last_os_error()).with_context(|| {
                format!(
                    "open fixture component without following links {}",
                    relative.display()
                )
            });
        }
        // SAFETY: `openat` returned a new owned descriptor on the success path above.
        let opened = unsafe { std::fs::File::from_raw_fd(fd) };
        if final_component {
            source = Some(opened);
        } else {
            directory = opened;
        }
    }
    let source = source.ok_or_else(|| anyhow::anyhow!("fixture media was not opened"))?;
    let metadata = source
        .metadata()
        .with_context(|| format!("stat opened fixture media {}", relative.display()))?;
    anyhow::ensure!(
        metadata.file_type().is_file(),
        "fixture media must be a regular non-symlink file"
    );
    anyhow::ensure!(
        metadata.nlink() == 1,
        "fixture media must not be hard-linked"
    );
    anyhow::ensure!(
        metadata.len() <= max_asset_bytes,
        "fixture media exceeds max_import_asset_bytes"
    );
    let read_limit = max_asset_bytes
        .checked_add(1)
        .ok_or_else(|| anyhow::anyhow!("max_import_asset_bytes is too large"))?;
    let initial_capacity = usize::try_from(metadata.len().min(1024 * 1024))
        .map_err(|_| anyhow::anyhow!("fixture media size is not addressable"))?;
    let mut bytes = Vec::with_capacity(initial_capacity);
    source
        .take(read_limit)
        .read_to_end(&mut bytes)
        .with_context(|| format!("read opened fixture media {}", relative.display()))?;
    anyhow::ensure!(
        bytes.len() as u64 <= max_asset_bytes,
        "fixture media exceeds max_import_asset_bytes while streaming"
    );
    anyhow::ensure!(
        bytes.len() as u64 == metadata.len(),
        "fixture media changed while being read"
    );
    Ok(bytes)
}

#[cfg(not(unix))]
fn read_fixture_asset(
    _asset_root: &Path,
    _relative: &Path,
    _max_asset_bytes: u64,
) -> anyhow::Result<Vec<u8>> {
    anyhow::bail!("secure no-follow fixture import is unsupported on this platform")
}

fn validate_importable_media(media_type: &str, bytes: &[u8]) -> anyhow::Result<()> {
    let valid = match media_type {
        "application/pdf" => {
            bytes.starts_with(b"%PDF-")
                && bytes
                    .windows(b"%%EOF".len())
                    .any(|window| window == b"%%EOF")
        }
        "image/png" => bytes.starts_with(b"\x89PNG\r\n\x1a\n"),
        "image/jpeg" => bytes.starts_with(b"\xff\xd8\xff") && bytes.ends_with(b"\xff\xd9"),
        "image/gif" => bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a"),
        "image/webp" => bytes.len() >= 12 && bytes.starts_with(b"RIFF") && &bytes[8..12] == b"WEBP",
        "image/svg+xml" => std::str::from_utf8(bytes)
            .is_ok_and(|text| text.contains("<svg") && text.contains("</svg>")),
        "text/plain" | "text/csv" => std::str::from_utf8(bytes).is_ok(),
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" => {
            bytes.starts_with(b"PK\x03\x04")
        }
        _ => false,
    };
    anyhow::ensure!(
        valid,
        "fixture media bytes are not an importable '{media_type}' asset"
    );
    Ok(())
}

fn validate_captured_corpus(
    fixture: &MultimodalCorpus,
    captured: &CapturedCorpus,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        captured.corpus_id == fixture.corpus_id,
        "captured corpus id mismatch"
    );
    anyhow::ensure!(
        captured.artifacts.len() == fixture.evidence.len(),
        "captured corpus must map every fixture evidence item exactly once"
    );
    let fixture_by_id: BTreeMap<_, _> = fixture
        .evidence
        .iter()
        .map(|item| (item.evidence_id.as_str(), item))
        .collect();
    let mut seen = HashSet::new();
    let mut raw_bindings: BTreeMap<String, CapturedBlobRef> = BTreeMap::new();
    let mut projection_bindings: BTreeMap<String, CapturedBlobRef> = BTreeMap::new();
    let mut all_bindings: BTreeMap<String, CapturedBlobRef> = BTreeMap::new();
    for artifact in &captured.artifacts {
        anyhow::ensure!(
            seen.insert(artifact.evidence_id.as_str()),
            "duplicate captured evidence"
        );
        let source = fixture_by_id
            .get(artifact.evidence_id.as_str())
            .ok_or_else(|| anyhow::anyhow!("captured corpus invented evidence"))?;
        validate_binding_id(&artifact.binding_id)?;
        record_unique_binding(
            &mut raw_bindings,
            "raw",
            &artifact.binding_id,
            &artifact.blob,
        )?;
        record_unique_binding(
            &mut all_bindings,
            "raw or projection",
            &artifact.binding_id,
            &artifact.blob,
        )?;
        anyhow::ensure!(
            artifact.blob.id.algorithm == "sha256",
            "blob algorithm must be sha256"
        );
        validate_sha256("captured blob", &artifact.blob.id.value)?;
        anyhow::ensure!(
            artifact.blob.id.value == source.source_media.sha256
                && artifact.blob.size_bytes == source.source_media.size_bytes
                && artifact.blob.detected_media_type == source.source_media.media_type,
            "captured blob metadata does not match verified fixture bytes"
        );
        validate_id("region_id", &artifact.region.region_id)?;
        anyhow::ensure!(
            artifact.region == captured_region_ref(&artifact.binding_id, &source.region)?,
            "captured region does not match the deterministic product mapping"
        );
        if let Some(parent) = &artifact.region.parent_region_id {
            validate_id("parent_region_id", parent)?;
            anyhow::ensure!(
                parent != &artifact.region.region_id,
                "region cannot parent itself"
            );
        }
        validate_retrieval_scores(&artifact.retrieval)?;
        anyhow::ensure!(
            artifact.retrieval == CapturedRetrievalScores::default(),
            "captured corpus must not contain query-time retrieval scores"
        );
        match (
            &source.text_projection,
            &artifact.projection,
            &artifact.projection_output_blob,
        ) {
            (Some(text), Some(projection), Some(output_blob)) => {
                validate_sha256("projection execution", &projection.execution_id)?;
                validate_binding_id(&projection.source_binding_id)?;
                let output_binding_id =
                    projection.output_binding_id.as_deref().ok_or_else(|| {
                        anyhow::anyhow!("ready projection must bind its output bytes")
                    })?;
                validate_binding_id(output_binding_id)?;
                record_unique_binding(
                    &mut projection_bindings,
                    "projection output",
                    output_binding_id,
                    output_blob,
                )?;
                record_unique_binding(
                    &mut all_bindings,
                    "raw or projection",
                    output_binding_id,
                    output_blob,
                )?;
                if let Some(transformed_at) = &projection.transformed_at {
                    chrono::DateTime::parse_from_rfc3339(transformed_at)
                        .with_context(|| "projection transformed_at must be RFC3339")?;
                }
                let projection_output_sha256 = artifact
                    .projection_output_sha256
                    .as_deref()
                    .ok_or_else(|| anyhow::anyhow!("ready projection lacks output digest proof"))?;
                validate_sha256("projection output", projection_output_sha256)?;
                anyhow::ensure!(
                    projection.source_binding_id == artifact.binding_id
                        && projection.source_region_id == artifact.region.region_id
                        && projection_output_sha256 == sha256(text.as_bytes())
                        && output_blob.id.algorithm == "sha256"
                        && output_blob.id.value == projection_output_sha256
                        && output_blob.size_bytes == text.len() as u64
                        && output_blob.detected_media_type == "text/plain; charset=utf-8",
                    "projection provenance does not bind to verified source text"
                );
                anyhow::ensure!(
                    artifact.truth_tier == TruthTier::Raw,
                    "captured source artifact must retain raw truth tier"
                );
            }
            (None, None, None) => anyhow::ensure!(
                artifact.truth_tier == TruthTier::Raw,
                "raw-only artifact must use raw truth tier"
            ),
            _ => anyhow::bail!("captured projection presence does not match fixture"),
        }
    }
    Ok(())
}

fn record_unique_binding(
    bindings: &mut BTreeMap<String, CapturedBlobRef>,
    kind: &str,
    binding_id: &str,
    blob: &CapturedBlobRef,
) -> anyhow::Result<()> {
    if let Some(existing) = bindings.get(binding_id) {
        anyhow::ensure!(
            existing == blob,
            "{kind} binding '{binding_id}' maps to conflicting blob metadata"
        );
    } else {
        bindings.insert(binding_id.to_string(), blob.clone());
    }
    Ok(())
}

fn validate_binding_id(value: &str) -> anyhow::Result<()> {
    let digest = value
        .strip_prefix("binding-")
        .ok_or_else(|| anyhow::anyhow!("binding id must use product binding-<sha256> form"))?;
    validate_sha256("binding id", digest)
}

fn validate_retrieval_scores(scores: &CapturedRetrievalScores) -> anyhow::Result<()> {
    anyhow::ensure!(
        scores.fused.is_finite(),
        "fused retrieval score must be finite"
    );
    if let Some(score) = scores.text_projection {
        anyhow::ensure!(score.is_finite(), "text retrieval score must be finite");
    }
    for (profile, score) in &scores.native_profiles {
        non_empty("native retrieval profile", profile)?;
        anyhow::ensure!(score.is_finite(), "native retrieval score must be finite");
    }
    Ok(())
}

fn merge_retrieval_scores(target: &mut CapturedRetrievalScores, source: &CapturedRetrievalScores) {
    target.text_projection = match (target.text_projection, source.text_projection) {
        (Some(left), Some(right)) => Some(left.max(right)),
        (left @ Some(_), None) => left,
        (None, right) => right,
    };
    for (profile, score) in &source.native_profiles {
        target
            .native_profiles
            .entry(profile.clone())
            .and_modify(|current| *current = current.max(*score))
            .or_insert(*score);
    }
    target.fused = target.fused.max(source.fused);
}

fn captured_region_ref(
    binding_id: &str,
    source: &EvidenceRegion,
) -> anyhow::Result<CapturedRegionRef> {
    let stable_region_id = |suffix: &str| {
        format!(
            "region-{}",
            sha256(format!("{binding_id}\0{suffix}").as_bytes())
        )
    };
    let (region_id, parent_region_id, locator) = match source {
        EvidenceRegion::Screenshot {
            trajectory_id,
            state_index,
            locator,
            bounds: None,
        } => {
            let value = format!("{trajectory_id}/{state_index}/{locator}");
            (
                stable_region_id(&format!("screenshot\0{value}")),
                None,
                CapturedRegionLocator::Named {
                    scheme: "longmemeval_screenshot".to_string(),
                    value,
                },
            )
        }
        EvidenceRegion::Screenshot {
            trajectory_id,
            state_index,
            locator,
            bounds: Some(bounds),
        } => {
            let anchor = stable_region_id(&format!(
                "screenshot\0{trajectory_id}/{state_index}/{locator}"
            ));
            (
                stable_region_id(&format!(
                    "screenshot-rectangle\0{trajectory_id}/{state_index}/{locator}\0{}",
                    stable_json_hash(bounds)?
                )),
                Some(anchor.clone()),
                captured_rectangle(&anchor, bounds)?,
            )
        }
        EvidenceRegion::Page {
            document_id,
            page_number,
            bounds: None,
        } => {
            anyhow::ensure!(*page_number > 0, "fixture page numbers are one-based");
            (
                stable_region_id(&format!("page\0{document_id}\0{page_number}")),
                None,
                CapturedRegionLocator::Page {
                    index: page_number - 1,
                },
            )
        }
        EvidenceRegion::Page {
            document_id,
            page_number,
            bounds: Some(bounds),
        } => {
            anyhow::ensure!(*page_number > 0, "fixture page numbers are one-based");
            let anchor = stable_region_id(&format!("page\0{document_id}\0{page_number}"));
            (
                stable_region_id(&format!(
                    "page-rectangle\0{document_id}\0{page_number}\0{}",
                    stable_json_hash(bounds)?
                )),
                Some(anchor.clone()),
                captured_rectangle(&anchor, bounds)?,
            )
        }
        EvidenceRegion::Sheet {
            workbook_id,
            sheet_name,
            range,
        } => (
            stable_region_id(&format!("cell-range\0{workbook_id}\0{sheet_name}\0{range}")),
            None,
            CapturedRegionLocator::CellRange {
                sheet: sheet_name.clone(),
                a1: range.clone(),
            },
        ),
        EvidenceRegion::Cell {
            workbook_id,
            sheet_name,
            cell,
        } => (
            stable_region_id(&format!("cell-range\0{workbook_id}\0{sheet_name}\0{cell}")),
            None,
            CapturedRegionLocator::CellRange {
                sheet: sheet_name.clone(),
                a1: cell.clone(),
            },
        ),
    };
    Ok(CapturedRegionRef {
        region_id,
        parent_region_id,
        locator,
    })
}

fn captured_rectangle(
    anchor_region_id: &str,
    bounds: &NormalizedBounds,
) -> anyhow::Result<CapturedRegionLocator> {
    for value in [bounds.x, bounds.y, bounds.width, bounds.height] {
        anyhow::ensure!(value.is_finite(), "region bounds must be finite");
    }
    anyhow::ensure!(
        bounds.x >= 0.0
            && bounds.y >= 0.0
            && bounds.width >= 0.0
            && bounds.height >= 0.0
            && bounds.x + bounds.width <= 1.0
            && bounds.y + bounds.height <= 1.0,
        "normalized region bounds must remain inside the source"
    );
    Ok(CapturedRegionLocator::Rectangle {
        anchor_region_id: anchor_region_id.to_string(),
        coordinate_space: CapturedCoordinateSpace::NormalizedMillionths,
        x: (bounds.x as f64 * 1_000_000.0).round() as i64,
        y: (bounds.y as f64 * 1_000_000.0).round() as i64,
        width: (bounds.width as f64 * 1_000_000.0).round() as u64,
        height: (bounds.height as f64 * 1_000_000.0).round() as u64,
        rotation_millidegrees: 0,
    })
}

fn product_region_parent_id(region: &CapturedRegionRef) -> Option<&str> {
    region
        .parent_region_id
        .as_deref()
        .or(match &region.locator {
            CapturedRegionLocator::Rectangle {
                anchor_region_id, ..
            } => Some(anchor_region_id.as_str()),
            _ => None,
        })
}

fn product_regions_related(
    left: &CapturedRegionRef,
    right: &CapturedRegionRef,
    parent_by_region: &BTreeMap<String, String>,
    threshold: f64,
) -> bool {
    left.region_id == right.region_id
        || product_is_ancestor(&left.region_id, &right.region_id, parent_by_region)
        || product_is_ancestor(&right.region_id, &left.region_id, parent_by_region)
        || product_locators_overlap(&left.locator, &right.locator, threshold)
}

fn product_is_ancestor(
    ancestor: &str,
    descendant: &str,
    parent_by_region: &BTreeMap<String, String>,
) -> bool {
    let mut current = descendant;
    for _ in 0..=parent_by_region.len() {
        let Some(parent) = parent_by_region.get(current) else {
            return false;
        };
        if parent == ancestor {
            return true;
        }
        if parent == current {
            return false;
        }
        current = parent;
    }
    false
}

fn product_locators_overlap(
    left: &CapturedRegionLocator,
    right: &CapturedRegionLocator,
    threshold: f64,
) -> bool {
    match (left, right) {
        (CapturedRegionLocator::Whole, _) | (_, CapturedRegionLocator::Whole) => true,
        (
            CapturedRegionLocator::Page { index: left },
            CapturedRegionLocator::Page { index: right },
        ) => left == right,
        (CapturedRegionLocator::Sheet { name }, CapturedRegionLocator::CellRange { sheet, .. })
        | (CapturedRegionLocator::CellRange { sheet, .. }, CapturedRegionLocator::Sheet { name }) => {
            name == sheet
        }
        (
            CapturedRegionLocator::Sheet { name: left },
            CapturedRegionLocator::Sheet { name: right },
        ) => left == right,
        (
            CapturedRegionLocator::CellRange {
                sheet: left_sheet,
                a1: left,
            },
            CapturedRegionLocator::CellRange {
                sheet: right_sheet,
                a1: right,
            },
        ) => {
            left_sheet == right_sheet
                && product_a1_range(left)
                    .zip(product_a1_range(right))
                    .is_some_and(|(left, right)| product_grid_ranges_overlap(left, right))
        }
        (
            CapturedRegionLocator::Rectangle {
                anchor_region_id: left_anchor,
                coordinate_space: left_space,
                x: left_x,
                y: left_y,
                width: left_width,
                height: left_height,
                ..
            },
            CapturedRegionLocator::Rectangle {
                anchor_region_id: right_anchor,
                coordinate_space: right_space,
                x: right_x,
                y: right_y,
                width: right_width,
                height: right_height,
                ..
            },
        ) if left_anchor == right_anchor && left_space == right_space => {
            product_rectangle_iou(
                (*left_x, *left_y, *left_width, *left_height),
                (*right_x, *right_y, *right_width, *right_height),
            ) >= threshold
        }
        (
            CapturedRegionLocator::TimeRange {
                start_ms: left_start,
                end_ms: left_end,
            },
            CapturedRegionLocator::TimeRange {
                start_ms: right_start,
                end_ms: right_end,
            },
        ) => {
            product_interval_overlap_ratio(*left_start, *left_end, *right_start, *right_end)
                >= threshold
        }
        (
            CapturedRegionLocator::ByteRange {
                start: left_start,
                end_exclusive: left_end,
            },
            CapturedRegionLocator::ByteRange {
                start: right_start,
                end_exclusive: right_end,
            },
        ) => {
            product_interval_overlap_ratio(*left_start, *left_end, *right_start, *right_end)
                >= threshold
        }
        (
            CapturedRegionLocator::Named {
                scheme: left_scheme,
                value: left_value,
            },
            CapturedRegionLocator::Named {
                scheme: right_scheme,
                value: right_value,
            },
        ) => left_scheme == right_scheme && left_value == right_value,
        _ => false,
    }
}

fn product_rectangle_iou(left: (i64, i64, u64, u64), right: (i64, i64, u64, u64)) -> f64 {
    let (left_x, left_y, left_width, left_height) = left;
    let (right_x, right_y, right_width, right_height) = right;
    let left_end_x = i128::from(left_x) + i128::from(left_width);
    let left_end_y = i128::from(left_y) + i128::from(left_height);
    let right_end_x = i128::from(right_x) + i128::from(right_width);
    let right_end_y = i128::from(right_y) + i128::from(right_height);
    let width = (left_end_x.min(right_end_x) - i128::from(left_x.max(right_x))).max(0) as u128;
    let height = (left_end_y.min(right_end_y) - i128::from(left_y.max(right_y))).max(0) as u128;
    let intersection = width.saturating_mul(height);
    let left_area = u128::from(left_width).saturating_mul(u128::from(left_height));
    let right_area = u128::from(right_width).saturating_mul(u128::from(right_height));
    let union = left_area
        .saturating_add(right_area)
        .saturating_sub(intersection);
    if union == 0 {
        0.0
    } else {
        intersection as f64 / union as f64
    }
}

fn product_interval_overlap_ratio(
    left_start: u64,
    left_end: u64,
    right_start: u64,
    right_end: u64,
) -> f64 {
    if left_end <= left_start || right_end <= right_start {
        return 0.0;
    }
    let intersection = left_end
        .min(right_end)
        .saturating_sub(left_start.max(right_start));
    let shorter = (left_end - left_start).min(right_end - right_start);
    intersection as f64 / shorter as f64
}

type ProductGridRange = ((u32, u32), (u32, u32));

fn product_a1_range(value: &str) -> Option<ProductGridRange> {
    let mut cells = value.split(':');
    let start = product_a1_cell(cells.next()?)?;
    let end = match cells.next() {
        Some(value) => product_a1_cell(value)?,
        None => start,
    };
    if cells.next().is_some() {
        return None;
    }
    Some((
        (start.0.min(end.0), start.1.min(end.1)),
        (start.0.max(end.0), start.1.max(end.1)),
    ))
}

fn product_a1_cell(value: &str) -> Option<(u32, u32)> {
    let value = value.trim().replace('$', "");
    let split = value.find(|character: char| character.is_ascii_digit())?;
    let (column, row) = value.split_at(split);
    if column.is_empty()
        || row.is_empty()
        || !column.bytes().all(|byte| byte.is_ascii_alphabetic())
        || !row.bytes().all(|byte| byte.is_ascii_digit())
    {
        return None;
    }
    let column = column.bytes().try_fold(0u32, |value, byte| {
        value
            .checked_mul(26)?
            .checked_add(u32::from(byte.to_ascii_uppercase() - b'A' + 1))
    })?;
    let row = row.parse::<u32>().ok()?;
    (column > 0 && row > 0).then_some((column, row))
}

fn product_grid_ranges_overlap(left: ProductGridRange, right: ProductGridRange) -> bool {
    left.0.0 <= right.1.0 && right.0.0 <= left.1.0 && left.0.1 <= right.1.1 && right.0.1 <= left.1.1
}

fn validate_spend(
    traces: &[SpendTrace],
    remaining: &RemainingBudget,
) -> anyhow::Result<(u64, u64)> {
    let mut by_call: BTreeMap<&str, Vec<&SpendTrace>> = BTreeMap::new();
    for trace in traces {
        validate_id("call_id", &trace.call_id)?;
        non_empty("provider", &trace.provider)?;
        non_empty("model", &trace.model)?;
        non_empty("pricing table version", &trace.pricing_table_version)?;
        by_call.entry(&trace.call_id).or_default().push(trace);
    }
    let mut cost = 0u64;
    let mut reserved_cost = 0u64;
    for (call_id, events) in &by_call {
        anyhow::ensure!(
            events.len() == 2,
            "provider call '{call_id}' needs reserve and terminal events"
        );
        anyhow::ensure!(
            events[0].status == SpendStatus::Reserved,
            "provider reservation must precede dispatch"
        );
        anyhow::ensure!(
            matches!(
                events[1].status,
                SpendStatus::Succeeded | SpendStatus::Failed
            ),
            "provider call needs terminal spend event"
        );
        anyhow::ensure!(
            events[0].provider == events[1].provider && events[0].model == events[1].model,
            "provider spend identity changed after reservation"
        );
        anyhow::ensure!(
            events[1].cost_micro_usd <= events[0].cost_micro_usd,
            "provider terminal cost exceeded its pre-dispatch reservation"
        );
        reserved_cost = reserved_cost.saturating_add(events[0].cost_micro_usd);
        cost = cost.saturating_add(events[1].cost_micro_usd);
    }
    let calls = by_call.len() as u64;
    anyhow::ensure!(
        calls <= remaining.provider_calls,
        "adapter dispatched beyond reserved call budget"
    );
    anyhow::ensure!(
        reserved_cost <= remaining.cost_micro_usd,
        "adapter reserved beyond remaining cost budget"
    );
    anyhow::ensure!(
        cost <= remaining.cost_micro_usd,
        "adapter dispatched beyond reserved cost budget"
    );
    Ok((calls, cost))
}

enum SpendTransition {
    Reserve(SpendTrace),
    Finish(SpendTrace),
}

struct SpendLedgerLock {
    path: PathBuf,
    _file: std::fs::File,
}

impl Drop for SpendLedgerLock {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

fn spend_sibling_path(path: &Path, suffix: &str) -> PathBuf {
    let mut value = path.as_os_str().to_os_string();
    value.push(suffix);
    PathBuf::from(value)
}

fn lock_spend_ledger(path: &Path, run_instance_id: &str) -> anyhow::Result<SpendLedgerLock> {
    let lock_path = spend_sibling_path(path, ".lock");
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&lock_path)
        .with_context(|| {
            format!(
                "spend ledger is locked by another process at {}",
                lock_path.display()
            )
        })?;
    file.write_all(run_instance_id.as_bytes())?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    Ok(SpendLedgerLock {
        path: lock_path,
        _file: file,
    })
}

fn empty_spend_ledger(run_instance_id: &str, effective_config_sha256: &str) -> SpendLedgerPayload {
    SpendLedgerPayload {
        run_instance_id: run_instance_id.to_string(),
        effective_config_sha256: effective_config_sha256.to_string(),
        generation: 0,
        operations: BTreeMap::new(),
    }
}

fn read_spend_ledger(path: &Path) -> anyhow::Result<Option<SpendLedgerPayload>> {
    let metadata = match std::fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(error).with_context(|| format!("stat spend ledger {}", path.display()));
        }
    };
    anyhow::ensure!(
        metadata.file_type().is_file() && !metadata.file_type().is_symlink(),
        "spend ledger must be a regular non-symlink file"
    );
    anyhow::ensure!(
        metadata.len() <= 16 * 1024 * 1024,
        "spend ledger exceeds safety limit"
    );
    let bytes =
        std::fs::read(path).with_context(|| format!("read spend ledger {}", path.display()))?;
    let envelope: SpendLedgerEnvelope = serde_json::from_slice(&bytes)
        .with_context(|| format!("decode spend ledger {}", path.display()))?;
    anyhow::ensure!(
        envelope.schema == SPEND_LEDGER_SCHEMA,
        "unsupported spend ledger schema"
    );
    anyhow::ensure!(
        envelope.payload_sha256 == stable_json_hash(&envelope.payload)?,
        "spend ledger checksum mismatch"
    );
    Ok(Some(envelope.payload))
}

fn validate_spend_ledger_owner(
    payload: &SpendLedgerPayload,
    run_instance_id: &str,
    effective_config_sha256: &str,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        payload.run_instance_id == run_instance_id
            && payload.effective_config_sha256 == effective_config_sha256,
        "spend ledger is owned by another run/configuration"
    );
    Ok(())
}

fn write_spend_ledger_atomic(
    path: &Path,
    payload: &SpendLedgerPayload,
    injected_failure: Option<SpendPersistFailurePoint>,
) -> anyhow::Result<()> {
    #[cfg(not(test))]
    let _ = injected_failure;
    let parent = path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("spend ledger path has no parent"))?;
    std::fs::create_dir_all(parent)
        .with_context(|| format!("create spend ledger directory {}", parent.display()))?;
    let envelope = SpendLedgerEnvelope {
        schema: SPEND_LEDGER_SCHEMA.to_string(),
        payload_sha256: stable_json_hash(payload)?,
        payload: payload.clone(),
    };
    let temp_path = spend_sibling_path(
        path,
        &format!(".tmp.{}.{}", std::process::id(), payload.generation),
    );
    struct TempGuard(PathBuf);
    impl Drop for TempGuard {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }
    let _guard = TempGuard(temp_path.clone());
    use std::io::Write;
    let mut temp = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temp_path)
        .with_context(|| format!("create spend ledger temp file {}", temp_path.display()))?;
    serde_json::to_writer(&mut temp, &envelope)?;
    temp.write_all(b"\n")?;
    temp.sync_all()?;
    #[cfg(test)]
    if injected_failure == Some(SpendPersistFailurePoint::BeforeRename) {
        anyhow::bail!("injected spend ledger failure before atomic rename");
    }
    std::fs::rename(&temp_path, path)
        .with_context(|| format!("replace spend ledger {}", path.display()))?;
    #[cfg(test)]
    if injected_failure == Some(SpendPersistFailurePoint::AfterRenameBeforeDirectoryFsync) {
        anyhow::bail!("injected spend ledger failure after rename before directory fsync");
    }
    std::fs::File::open(parent)
        .with_context(|| format!("open spend ledger directory {}", parent.display()))?
        .sync_all()
        .with_context(|| format!("fsync spend ledger directory {}", parent.display()))?;
    Ok(())
}

fn update_spend_ledger(
    path: Option<&Path>,
    run_instance_id: &str,
    effective_config_sha256: &str,
    operation_id: &str,
    case_id: &str,
    transition: SpendTransition,
    injected_failure: Option<SpendPersistFailurePoint>,
) -> anyhow::Result<PersistedSpendOperation> {
    let path = path.ok_or_else(|| anyhow::anyhow!("provider spend has no durable ledger"))?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create spend ledger directory {}", parent.display()))?;
    }
    let _lock = lock_spend_ledger(path, run_instance_id)?;
    let mut payload = read_spend_ledger(path)?
        .unwrap_or_else(|| empty_spend_ledger(run_instance_id, effective_config_sha256));
    validate_spend_ledger_owner(&payload, run_instance_id, effective_config_sha256)?;
    match transition {
        SpendTransition::Reserve(reservation) => {
            anyhow::ensure!(
                reservation.status == SpendStatus::Reserved,
                "reservation transition has non-reserved status"
            );
            if let Some(existing) = payload.operations.get(operation_id) {
                anyhow::ensure!(
                    existing.operation_id == operation_id
                        && existing.case_id == case_id
                        && existing.reservation == reservation,
                    "spend reservation replay changed immutable operation fields"
                );
                return Ok(existing.clone());
            }
            payload.operations.insert(
                operation_id.to_string(),
                PersistedSpendOperation {
                    operation_id: operation_id.to_string(),
                    case_id: case_id.to_string(),
                    reservation,
                    terminal: None,
                },
            );
        }
        SpendTransition::Finish(terminal) => {
            anyhow::ensure!(
                matches!(
                    terminal.status,
                    SpendStatus::Succeeded | SpendStatus::Failed
                ),
                "terminal transition has reserved status"
            );
            let existing = payload
                .operations
                .get_mut(operation_id)
                .ok_or_else(|| anyhow::anyhow!("terminal transition has no durable reservation"))?;
            anyhow::ensure!(
                existing.case_id == case_id
                    && existing.reservation.call_id == terminal.call_id
                    && existing.reservation.provider == terminal.provider
                    && existing.reservation.model == terminal.model
                    && existing.reservation.pricing_table_version == terminal.pricing_table_version,
                "terminal transition changed immutable reservation identity"
            );
            if let Some(authoritative) = &existing.terminal {
                anyhow::ensure!(
                    authoritative == &terminal,
                    "spend operation already has a different authoritative terminal"
                );
                return Ok(existing.clone());
            }
            existing.terminal = Some(terminal);
        }
    }
    payload.generation = payload.generation.saturating_add(1);
    write_spend_ledger_atomic(path, &payload, injected_failure)?;
    payload
        .operations
        .get(operation_id)
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("persisted spend operation disappeared"))
}

fn read_spend_operation(
    path: Option<&Path>,
    run_instance_id: &str,
    effective_config_sha256: &str,
    operation_id: &str,
) -> anyhow::Result<Option<PersistedSpendOperation>> {
    let path = path.ok_or_else(|| anyhow::anyhow!("provider spend has no durable ledger"))?;
    let Some(payload) = read_spend_ledger(path)? else {
        return Ok(None);
    };
    validate_spend_ledger_owner(&payload, run_instance_id, effective_config_sha256)?;
    Ok(payload.operations.get(operation_id).cloned())
}

/// Builds a media-dependent LongMemEval-v2 subset from explicit human-reviewed annotations.
///
/// The loader verifies every screenshot against the selected question's haystack and state. It
/// hashes source bytes into `MediaRef`; it does not OCR, transcribe, embed, or call a provider.
/// Official evaluator strings remain `ScoringRule::External` so a missing evaluator fails closed.
pub fn load_longmemeval_v2_image_subset(
    dataset_root: &Path,
    annotations_path: &Path,
) -> anyhow::Result<MultimodalFixtureSet> {
    let annotation_bytes = std::fs::read(annotations_path)
        .with_context(|| format!("read annotations {}", annotations_path.display()))?;
    let annotations: LongMemEvalV2ImageAnnotations = serde_json::from_slice(&annotation_bytes)
        .with_context(|| format!("parse annotations {}", annotations_path.display()))?;
    anyhow::ensure!(
        annotations.schema == ANNOTATION_SCHEMA,
        "unsupported annotation schema '{}'; expected '{ANNOTATION_SCHEMA}'",
        annotations.schema
    );
    validate_id("fixture_set_id", &annotations.fixture_set_id)?;
    non_empty("dataset version", &annotations.dataset_version)?;
    anyhow::ensure!(
        !annotations.annotations.is_empty(),
        "image annotation set is empty"
    );
    let haystack_rel = Path::new("haystacks").join(&annotations.haystack_file);
    ensure_safe_relative(&haystack_rel, "haystack file")?;
    let questions_path = dataset_root.join("questions.jsonl");
    let trajectories_path = dataset_root.join("trajectories.jsonl");
    let haystack_path = dataset_root.join(&haystack_rel);
    let question_bytes = std::fs::read(&questions_path)
        .with_context(|| format!("read {}", questions_path.display()))?;
    let trajectory_bytes = std::fs::read(&trajectories_path)
        .with_context(|| format!("read {}", trajectories_path.display()))?;
    let haystack_bytes = std::fs::read(&haystack_path)
        .with_context(|| format!("read {}", haystack_path.display()))?;
    let questions = read_jsonl_map::<LmeV2Question>(&question_bytes, "question", |row| &row.id)?;
    let trajectories =
        read_jsonl_map::<LmeV2Trajectory>(&trajectory_bytes, "trajectory", |row| &row.id)?;
    let haystacks: BTreeMap<String, Vec<String>> = serde_json::from_slice(&haystack_bytes)
        .with_context(|| format!("parse {}", haystack_path.display()))?;

    let mut seen = HashSet::new();
    let mut cases = Vec::with_capacity(annotations.annotations.len());
    let mut corpora_by_id: BTreeMap<String, MultimodalCorpus> = BTreeMap::new();
    for annotation in &annotations.annotations {
        validate_id("question_id", &annotation.question_id)?;
        anyhow::ensure!(
            seen.insert(annotation.question_id.as_str()),
            "duplicate image annotation for question '{}'",
            annotation.question_id
        );
        anyhow::ensure!(
            annotation.image_dependent,
            "annotation '{}' must explicitly set image_dependent=true",
            annotation.question_id
        );
        non_empty("annotation rationale", &annotation.rationale)?;
        non_empty("annotation reviewer", &annotation.reviewed_by)?;
        anyhow::ensure!(
            matches!(
                annotation.oracle_lane,
                RecallLane::Native | RecallLane::HybridCollapsed
            ),
            "image-dependent annotation '{}' cannot use a text-only oracle lane",
            annotation.question_id
        );
        anyhow::ensure!(
            !annotation.evidence.is_empty(),
            "annotation '{}' has no screenshot evidence",
            annotation.question_id
        );
        let question = questions.get(&annotation.question_id).ok_or_else(|| {
            anyhow::anyhow!(
                "annotation references missing question '{}'",
                annotation.question_id
            )
        })?;
        non_empty("question text", &question.question)?;
        non_empty("question answer", &question.answer)?;
        non_empty("question evaluator", &question.eval_function)?;
        let haystack_trajectory_ids = haystacks
            .get(&question.id)
            .ok_or_else(|| anyhow::anyhow!("haystack has no question '{}'", question.id))?;
        let allowed_trajectories: HashSet<_> =
            haystack_trajectory_ids.iter().map(String::as_str).collect();
        let mut question_media = Vec::new();
        if let Some(locator) = question.image.as_deref() {
            question_media.push(media_ref_from_dataset(
                dataset_root,
                &format!("question:{}", question.id),
                locator,
            )?);
        }
        let mut corpus_evidence = Vec::new();
        for trajectory_id in haystack_trajectory_ids {
            let trajectory = trajectories.get(trajectory_id).ok_or_else(|| {
                anyhow::anyhow!("haystack references missing trajectory '{trajectory_id}'")
            })?;
            anyhow::ensure!(
                trajectory.domain == question.domain,
                "question '{}' domain '{}' does not match trajectory '{}' domain '{}'",
                question.id,
                question.domain,
                trajectory.id,
                trajectory.domain
            );
            for state in &trajectory.states {
                let index = state_index(state).ok_or_else(|| {
                    anyhow::anyhow!("trajectory '{}' has invalid state_index", trajectory.id)
                })?;
                corpus_evidence.push(EvidenceItem {
                    evidence_id: format!("trajectory:{}:state:{index}", trajectory.id),
                    source_media: media_ref_from_dataset(
                        dataset_root,
                        &format!("trajectory:{}:{index}", trajectory.id),
                        &state.screenshot,
                    )?,
                    region: EvidenceRegion::Screenshot {
                        trajectory_id: trajectory.id.clone(),
                        state_index: index,
                        locator: state.screenshot.clone(),
                        bounds: None,
                    },
                    text_projection: state_text_projection(state),
                });
            }
        }
        corpus_evidence.sort_by(|left, right| left.evidence_id.cmp(&right.evidence_id));
        let corpus_id = format!("question:{}", question.id);
        let corpus = MultimodalCorpus {
            corpus_id: corpus_id.clone(),
            evidence: corpus_evidence,
        };
        anyhow::ensure!(
            corpora_by_id.insert(corpus_id.clone(), corpus).is_none(),
            "duplicate corpus for question '{}'",
            question.id
        );
        let mut oracle_evidence = Vec::with_capacity(annotation.evidence.len());
        for labeled in &annotation.evidence {
            validate_id("trajectory_id", &labeled.trajectory_id)?;
            anyhow::ensure!(
                allowed_trajectories.contains(labeled.trajectory_id.as_str()),
                "question '{}' annotation references trajectory '{}' outside its haystack",
                question.id,
                labeled.trajectory_id
            );
            let trajectory = trajectories.get(&labeled.trajectory_id).ok_or_else(|| {
                anyhow::anyhow!(
                    "annotation references missing trajectory '{}'",
                    labeled.trajectory_id
                )
            })?;
            anyhow::ensure!(
                trajectory.domain == question.domain,
                "question '{}' domain '{}' does not match trajectory '{}' domain '{}'",
                question.id,
                question.domain,
                trajectory.id,
                trajectory.domain
            );
            let state = trajectory
                .states
                .iter()
                .find(|state| state_index(state).is_some_and(|value| value == labeled.state_index))
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "trajectory '{}' has no state_index {}",
                        trajectory.id,
                        labeled.state_index
                    )
                })?;
            let region = EvidenceRegion::Screenshot {
                trajectory_id: trajectory.id.clone(),
                state_index: labeled.state_index,
                locator: state.screenshot.clone(),
                bounds: labeled.bounds.clone(),
            };
            oracle_evidence.push(region.clone());
        }
        cases.push(MultimodalCase {
            case_id: question.id.clone(),
            origin: FixtureOrigin::LongMemEvalV2,
            corpus_id,
            question: ContentSpec {
                text: question.question.clone(),
                media: question_media,
            },
            gold: GoldAnswer {
                value: question.answer.clone(),
                scoring: ScoringRule::External {
                    evaluator: question.eval_function.clone(),
                },
            },
            media_dependent: true,
            media_dependence_review: Some(MediaDependenceReview {
                reviewed_by: annotation.reviewed_by.clone(),
                rationale: annotation.rationale.clone(),
            }),
            oracle_lane: annotation.oracle_lane,
            oracle_evidence,
        });
    }

    let source_digest = sha256_parts(&[
        ("questions.jsonl", &question_bytes),
        ("trajectories.jsonl", &trajectory_bytes),
        (annotations.haystack_file.as_str(), &haystack_bytes),
        ("annotations.json", &annotation_bytes),
    ]);
    let fixture = MultimodalFixtureSet {
        schema: FIXTURE_SCHEMA.to_string(),
        fixture_set_id: annotations.fixture_set_id,
        split: FixtureSplit::HeldOut,
        source: DatasetProvenance {
            dataset_id: "longmemeval-v2-image-dependent".to_string(),
            version: annotations.dataset_version,
            source_digest,
            source_uri: Some(
                "https://huggingface.co/datasets/xiaowu0162/LongMemEval-v2".to_string(),
            ),
            official_equivalent: false,
        },
        corpora: corpora_by_id.into_values().collect(),
        cases,
    };
    validate_fixture(&fixture)?;
    Ok(fixture)
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct LmeV2Question {
    id: String,
    domain: String,
    #[serde(rename = "environment")]
    _environment: String,
    #[serde(rename = "question_type")]
    _question_type: String,
    question: String,
    image: Option<String>,
    answer: String,
    eval_function: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct LmeV2Trajectory {
    id: String,
    domain: String,
    #[serde(rename = "environment")]
    _environment: String,
    #[serde(rename = "goal")]
    _goal: String,
    #[serde(rename = "outcome")]
    _outcome: String,
    #[serde(rename = "start_url")]
    _start_url: String,
    states: Vec<LmeV2State>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct LmeV2State {
    #[serde(rename = "step")]
    _step: serde_json::Value,
    state_index: serde_json::Value,
    url: String,
    action: Option<String>,
    thought: Option<String>,
    accessibility_tree: String,
    screenshot: String,
}

fn state_text_projection(state: &LmeV2State) -> Option<String> {
    let mut parts = Vec::new();
    if !state.url.trim().is_empty() {
        parts.push(format!("url: {}", state.url.trim()));
    }
    if let Some(thought) = state
        .thought
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        parts.push(format!("thought: {thought}"));
    }
    if let Some(action) = state
        .action
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        parts.push(format!("action: {action}"));
    }
    if !state.accessibility_tree.trim().is_empty() {
        parts.push(format!("observation:\n{}", state.accessibility_tree.trim()));
    }
    (!parts.is_empty()).then(|| parts.join("\n"))
}

fn state_index(state: &LmeV2State) -> Option<u32> {
    match &state.state_index {
        serde_json::Value::Number(value) => value.as_u64().and_then(|value| value.try_into().ok()),
        serde_json::Value::String(value) => value.parse().ok(),
        _ => None,
    }
}

fn media_ref_from_dataset(
    dataset_root: &Path,
    asset_id: &str,
    locator: &str,
) -> anyhow::Result<MediaRef> {
    let relative = Path::new(locator);
    ensure_safe_relative(relative, "media locator")?;
    let canonical_root = std::fs::canonicalize(dataset_root)
        .with_context(|| format!("canonicalize dataset root {}", dataset_root.display()))?;
    let unresolved = dataset_root.join(relative);
    let canonical_media = std::fs::canonicalize(&unresolved)
        .with_context(|| format!("canonicalize media {}", unresolved.display()))?;
    anyhow::ensure!(
        canonical_media.starts_with(&canonical_root),
        "media locator '{locator}' resolves outside the dataset root"
    );
    let extension = relative
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    let media_type = match extension.as_str() {
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "webp" => "image/webp",
        other => {
            anyhow::bail!("unsupported LongMemEval-v2 media extension '{other}' for '{locator}'")
        }
    };
    Ok(MediaRef {
        asset_id: asset_id.to_string(),
        locator: locator.to_string(),
        sha256: sha256_file(&canonical_media)?,
        media_type: media_type.to_string(),
        size_bytes: std::fs::metadata(&canonical_media)
            .with_context(|| format!("stat media {}", canonical_media.display()))?
            .len(),
    })
}

fn sha256_file(path: &Path) -> anyhow::Result<String> {
    use std::io::Read;

    let mut file = std::fs::File::open(path)
        .with_context(|| format!("open media for hashing {}", path.display()))?;
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 1024 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .with_context(|| format!("hash media {}", path.display()))?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn read_jsonl_map<'a, T: Deserialize<'a>>(
    bytes: &'a [u8],
    kind: &str,
    id: impl Fn(&T) -> &String,
) -> anyhow::Result<BTreeMap<String, T>> {
    let text = std::str::from_utf8(bytes).with_context(|| format!("{kind} JSONL is not UTF-8"))?;
    let mut rows = BTreeMap::new();
    for (index, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let row: T = serde_json::from_str(line)
            .with_context(|| format!("parse {kind} JSONL line {}", index + 1))?;
        let row_id = id(&row).clone();
        anyhow::ensure!(
            rows.insert(row_id.clone(), row).is_none(),
            "duplicate {kind} id '{row_id}'"
        );
    }
    anyhow::ensure!(!rows.is_empty(), "{kind} JSONL is empty");
    Ok(rows)
}

fn ensure_safe_relative(path: &Path, name: &str) -> anyhow::Result<()> {
    anyhow::ensure!(
        !path.as_os_str().is_empty() && !path.is_absolute(),
        "{name} must be a non-empty relative path"
    );
    anyhow::ensure!(
        path.components()
            .all(|component| matches!(component, Component::Normal(_))),
        "{name} contains an unsafe path component"
    );
    Ok(())
}

fn sha256_parts(parts: &[(&str, &[u8])]) -> String {
    let mut digest = Sha256::new();
    for (name, bytes) in parts {
        digest.update(name.as_bytes());
        digest.update([0]);
        digest.update(bytes);
        digest.update([0]);
    }
    format!("{:x}", digest.finalize())
}

pub fn validate_fixture(fixture: &MultimodalFixtureSet) -> anyhow::Result<()> {
    anyhow::ensure!(
        fixture.schema == FIXTURE_SCHEMA,
        "unsupported fixture schema '{}'; expected '{FIXTURE_SCHEMA}'",
        fixture.schema
    );
    validate_id("fixture_set_id", &fixture.fixture_set_id)?;
    validate_id("dataset_id", &fixture.source.dataset_id)?;
    non_empty("dataset version", &fixture.source.version)?;
    validate_sha256("source digest", &fixture.source.source_digest)?;
    anyhow::ensure!(!fixture.corpora.is_empty(), "fixture set has no corpora");
    anyhow::ensure!(!fixture.cases.is_empty(), "fixture set has no cases");
    let mut case_ids = HashSet::new();
    let mut corpus_ids = HashSet::new();
    let mut fixture_media = BTreeMap::new();
    for corpus in &fixture.corpora {
        validate_id("corpus_id", &corpus.corpus_id)?;
        anyhow::ensure!(
            corpus_ids.insert(corpus.corpus_id.as_str()),
            "duplicate corpus id '{}'",
            corpus.corpus_id
        );
        anyhow::ensure!(
            !corpus.evidence.is_empty(),
            "corpus '{}' has no evidence",
            corpus.corpus_id
        );
        let mut evidence_ids = HashSet::new();
        for item in &corpus.evidence {
            validate_evidence_item(item)?;
            anyhow::ensure!(
                evidence_ids.insert(item.evidence_id.as_str()),
                "duplicate evidence id '{}' in corpus '{}'",
                item.evidence_id,
                corpus.corpus_id
            );
            if let Some(previous) =
                fixture_media.insert(item.source_media.asset_id.as_str(), &item.source_media)
            {
                anyhow::ensure!(
                    previous == &item.source_media,
                    "asset id '{}' is ambiguous",
                    item.source_media.asset_id
                );
            }
        }
    }
    for case in &fixture.cases {
        let corpus = fixture
            .corpora
            .iter()
            .find(|corpus| corpus.corpus_id == case.corpus_id)
            .ok_or_else(|| anyhow::anyhow!("case '{}' references missing corpus", case.case_id))?;
        validate_case(case, corpus)?;
        anyhow::ensure!(
            case_ids.insert(case.case_id.as_str()),
            "duplicate case id '{}'",
            case.case_id
        );
        for media in case.question.media.iter() {
            if let Some(previous) = fixture_media.insert(media.asset_id.as_str(), media) {
                anyhow::ensure!(
                    previous == media,
                    "fixture asset id '{}' resolves to different media refs across cases",
                    media.asset_id
                );
            }
        }
    }
    Ok(())
}

fn validate_case(case: &MultimodalCase, corpus: &MultimodalCorpus) -> anyhow::Result<()> {
    validate_id("case_id", &case.case_id)?;
    validate_id("corpus_id", &case.corpus_id)?;
    non_empty("question text", &case.question.text)?;
    non_empty("gold answer", &case.gold.value)?;
    match (&case.media_dependence_review, case.media_dependent) {
        (Some(review), true) => {
            non_empty("media-dependence reviewer", &review.reviewed_by)?;
            non_empty("media-dependence rationale", &review.rationale)?;
        }
        (None, true) => anyhow::bail!(
            "case '{}' is media-dependent but has no human review",
            case.case_id
        ),
        (Some(_), false) => anyhow::bail!(
            "case '{}' has a media-dependence review but media_dependent=false",
            case.case_id
        ),
        (None, false) => {}
    }
    anyhow::ensure!(
        !case.oracle_evidence.is_empty(),
        "case '{}' has no oracle evidence",
        case.case_id
    );
    for media in &case.question.media {
        validate_media(media)?;
    }
    for region in &case.oracle_evidence {
        validate_region(region)?;
        anyhow::ensure!(
            corpus
                .evidence
                .iter()
                .any(|item| region_is_within_source(region, &item.region)),
            "case '{}' oracle region is absent from evidence",
            case.case_id
        );
    }
    Ok(())
}

fn validate_evidence_item(item: &EvidenceItem) -> anyhow::Result<()> {
    validate_id("evidence_id", &item.evidence_id)?;
    validate_media(&item.source_media)?;
    validate_region(&item.region)?;
    if let Some(text) = &item.text_projection {
        non_empty("text projection", text)?;
    }
    Ok(())
}

fn validate_media(media: &MediaRef) -> anyhow::Result<()> {
    validate_id("asset_id", &media.asset_id)?;
    non_empty("media locator", &media.locator)?;
    non_empty("media type", &media.media_type)?;
    validate_sha256("media sha256", &media.sha256)?;
    anyhow::ensure!(media.size_bytes > 0, "media size must be positive");
    anyhow::ensure!(
        !media.locator.contains("://"),
        "media locator is import-only and must not be adapter-addressed"
    );
    ensure_safe_relative(Path::new(&media.locator), "media locator")?;
    Ok(())
}

fn validate_region(region: &EvidenceRegion) -> anyhow::Result<()> {
    match region {
        EvidenceRegion::Screenshot {
            trajectory_id,
            locator,
            bounds,
            ..
        } => {
            validate_id("trajectory_id", trajectory_id)?;
            non_empty("screenshot locator", locator)?;
            if let Some(bounds) = bounds {
                validate_bounds(bounds)?;
            }
        }
        EvidenceRegion::Page {
            document_id,
            page_number,
            bounds,
        } => {
            validate_id("document_id", document_id)?;
            anyhow::ensure!(*page_number > 0, "page numbers are one-based");
            if let Some(bounds) = bounds {
                validate_bounds(bounds)?;
            }
        }
        EvidenceRegion::Sheet {
            workbook_id,
            sheet_name,
            range,
        } => {
            validate_id("workbook_id", workbook_id)?;
            non_empty("sheet name", sheet_name)?;
            validate_a1_range(range)?;
        }
        EvidenceRegion::Cell {
            workbook_id,
            sheet_name,
            cell,
        } => {
            validate_id("workbook_id", workbook_id)?;
            non_empty("sheet name", sheet_name)?;
            validate_a1_cell(cell)?;
        }
    }
    Ok(())
}

fn region_is_within_source(region: &EvidenceRegion, source: &EvidenceRegion) -> bool {
    match (region, source) {
        (
            EvidenceRegion::Screenshot {
                trajectory_id: left,
                state_index: left_index,
                ..
            },
            EvidenceRegion::Screenshot {
                trajectory_id: right,
                state_index: right_index,
                ..
            },
        ) => left == right && left_index == right_index,
        (
            EvidenceRegion::Page {
                document_id: left,
                page_number: left_page,
                ..
            },
            EvidenceRegion::Page {
                document_id: right,
                page_number: right_page,
                ..
            },
        ) => left == right && left_page == right_page,
        (
            EvidenceRegion::Cell {
                workbook_id: left,
                sheet_name: left_sheet,
                ..
            }
            | EvidenceRegion::Sheet {
                workbook_id: left,
                sheet_name: left_sheet,
                ..
            },
            EvidenceRegion::Sheet {
                workbook_id: right,
                sheet_name: right_sheet,
                ..
            }
            | EvidenceRegion::Cell {
                workbook_id: right,
                sheet_name: right_sheet,
                ..
            },
        ) => left == right && left_sheet == right_sheet,
        _ => false,
    }
}

fn validate_bounds(bounds: &NormalizedBounds) -> anyhow::Result<()> {
    for (name, value) in [
        ("x", bounds.x),
        ("y", bounds.y),
        ("width", bounds.width),
        ("height", bounds.height),
    ] {
        anyhow::ensure!(value.is_finite(), "bounds {name} must be finite");
    }
    anyhow::ensure!(
        bounds.x >= 0.0 && bounds.y >= 0.0,
        "bounds origin must be non-negative"
    );
    anyhow::ensure!(
        bounds.width > 0.0 && bounds.height > 0.0,
        "bounds dimensions must be positive"
    );
    anyhow::ensure!(
        bounds.x + bounds.width <= 1.0 && bounds.y + bounds.height <= 1.0,
        "bounds must fit within normalized media coordinates"
    );
    Ok(())
}

fn validate_a1_range(range: &str) -> anyhow::Result<()> {
    let (start, end) = range
        .split_once(':')
        .ok_or_else(|| anyhow::anyhow!("sheet range '{range}' must use A1:A1 form"))?;
    validate_a1_cell(start)?;
    validate_a1_cell(end)
}

fn validate_a1_cell(cell: &str) -> anyhow::Result<()> {
    let split = cell
        .bytes()
        .position(|byte| byte.is_ascii_digit())
        .unwrap_or(cell.len());
    let (column, row) = cell.split_at(split);
    anyhow::ensure!(
        !column.is_empty() && column.bytes().all(|byte| byte.is_ascii_uppercase()),
        "cell '{cell}' must use uppercase A1 notation"
    );
    anyhow::ensure!(
        !row.is_empty()
            && row.bytes().all(|byte| byte.is_ascii_digit())
            && row != "0"
            && !row.starts_with('0'),
        "cell '{cell}' must use a positive row number"
    );
    Ok(())
}

fn preflight_capabilities(
    fixture: &MultimodalFixtureSet,
    arm: &ExperimentArm,
    capabilities: &MultimodalCapabilities,
) -> anyhow::Result<()> {
    let mut missing = BTreeSet::new();
    if arm.reader_mode == ReaderMode::TextProjection && !capabilities.text_projection_reader {
        missing.insert("text_projection_reader");
    }
    if arm.reader_mode == ReaderMode::SourceBlob && !capabilities.source_blob_reader {
        missing.insert("source_blob_reader");
    }
    if arm.cell == ExperimentCell::DOracleLane && !capabilities.oracle_region_filter {
        missing.insert("oracle_region_filter");
    }
    for case in &fixture.cases {
        let corpus = fixture
            .corpora
            .iter()
            .find(|corpus| corpus.corpus_id == case.corpus_id)
            .ok_or_else(|| anyhow::anyhow!("case '{}' references missing corpus", case.case_id))?;
        match arm.requested_lane(case) {
            RecallLane::TextProjection if !capabilities.text_projection_recall => {
                missing.insert("text_projection_recall");
            }
            RecallLane::Native => {
                if corpus
                    .evidence
                    .iter()
                    .any(|item| matches!(item.region, EvidenceRegion::Screenshot { .. }))
                    && !capabilities.native_image_recall
                {
                    missing.insert("native_image_recall");
                }
                if corpus.evidence.iter().any(|item| {
                    matches!(
                        item.region,
                        EvidenceRegion::Page { .. }
                            | EvidenceRegion::Sheet { .. }
                            | EvidenceRegion::Cell { .. }
                    )
                }) && !capabilities.native_document_recall
                {
                    missing.insert("native_document_recall");
                }
            }
            RecallLane::HybridCollapsed => {
                if !capabilities.text_projection_recall {
                    missing.insert("text_projection_recall");
                }
                if !capabilities.hybrid_collapse {
                    missing.insert("hybrid_collapse");
                }
                if corpus
                    .evidence
                    .iter()
                    .any(|item| matches!(item.region, EvidenceRegion::Screenshot { .. }))
                    && !capabilities.native_image_recall
                {
                    missing.insert("native_image_recall");
                }
                if corpus.evidence.iter().any(|item| {
                    matches!(
                        item.region,
                        EvidenceRegion::Page { .. }
                            | EvidenceRegion::Sheet { .. }
                            | EvidenceRegion::Cell { .. }
                    )
                }) && !capabilities.native_document_recall
                {
                    missing.insert("native_document_recall");
                }
            }
            _ => {}
        }
    }
    anyhow::ensure!(
        missing.is_empty(),
        "adapter capability gap for {:?}: {}",
        arm.cell,
        missing.into_iter().collect::<Vec<_>>().join(", ")
    );
    Ok(())
}

fn validate_response(
    case: &MultimodalCase,
    request: &RecallRequest,
    response: &RecallResponse,
    reader_inputs: &[ReaderInputProof],
) -> anyhow::Result<()> {
    anyhow::ensure!(
        !response.answer.trim().is_empty(),
        "case '{}' adapter returned an empty answer",
        case.case_id
    );
    anyhow::ensure!(
        !response.retrieved.is_empty(),
        "case '{}' adapter returned no retrieved evidence",
        case.case_id
    );
    anyhow::ensure!(
        response.execution.effective_lane == request.lane,
        "case '{}' adapter effective lane {:?} did not match requested {:?}",
        case.case_id,
        response.execution.effective_lane,
        request.lane
    );
    anyhow::ensure!(
        response.execution.effective_reader_mode == request.reader_mode,
        "case '{}' adapter reader mode did not match request",
        case.case_id
    );
    let text = &response.execution.text_branch_candidates;
    let native = &response.execution.native_branch_candidates;
    match request.lane {
        RecallLane::TextProjection => anyhow::ensure!(
            !text.is_empty() && native.is_empty(),
            "case '{}' text control must expose text candidates and no native candidates",
            case.case_id
        ),
        RecallLane::Native => anyhow::ensure!(
            text.is_empty() && !native.is_empty(),
            "case '{}' native arm must expose native candidates and no text candidates",
            case.case_id
        ),
        RecallLane::HybridCollapsed => anyhow::ensure!(
            !text.is_empty() && !native.is_empty(),
            "case '{}' hybrid arm must expose both candidate branches",
            case.case_id
        ),
    }
    if request.oracle_regions.is_empty() {
        anyhow::ensure!(
            response.execution.oracle_regions_applied.is_empty(),
            "case '{}' adapter applied undeclared oracle regions",
            case.case_id
        );
    } else {
        anyhow::ensure!(
            response.execution.oracle_regions_applied == request.oracle_regions,
            "case '{}' oracle region proof did not match request",
            case.case_id
        );
    }
    let evidence_by_id: BTreeMap<_, _> = request
        .corpus
        .artifacts
        .iter()
        .map(|item| (item.evidence_id.as_str(), item))
        .collect();
    let mut candidates = BTreeMap::new();
    for candidate in text.iter().chain(native) {
        validate_id("candidate_id", &candidate.candidate_id)?;
        anyhow::ensure!(
            candidate.score.is_finite(),
            "candidate score must be finite"
        );
        let expected = evidence_by_id
            .get(candidate.evidence_id.as_str())
            .ok_or_else(|| anyhow::anyhow!("candidate invented evidence"))?;
        let expected_candidate = scored_artifact(
            expected,
            candidate.lane,
            candidate.profile_id.as_deref(),
            candidate.score,
        );
        anyhow::ensure!(
            candidate.artifact == expected_candidate,
            "candidate does not match the product branch pointer/truth/score mapping"
        );
        validate_retrieval_scores(&candidate.artifact.retrieval)?;
        match candidate.lane {
            RecallLane::TextProjection => {
                anyhow::ensure!(
                    text.contains(candidate) && candidate.profile_id.is_none(),
                    "text candidate is in the wrong branch or has a native profile"
                );
                anyhow::ensure!(
                    candidate.artifact.projection.is_some(),
                    "text candidate lacks projection provenance"
                );
                anyhow::ensure!(
                    candidate.artifact.retrieval.text_projection == Some(candidate.score)
                        && candidate.artifact.retrieval.native_profiles.is_empty()
                        && candidate.artifact.retrieval.fused == candidate.score,
                    "text candidate score is not mapped to product retrieval scores"
                );
            }
            RecallLane::Native => {
                anyhow::ensure!(
                    native.contains(candidate)
                        && candidate
                            .profile_id
                            .as_deref()
                            .is_some_and(|id| !id.trim().is_empty()),
                    "native candidate is in the wrong branch or lacks profile identity"
                );
                let profile = candidate.profile_id.as_deref().unwrap_or_default();
                anyhow::ensure!(
                    candidate.artifact.retrieval.text_projection.is_none()
                        && candidate.artifact.retrieval.native_profiles.len() == 1
                        && candidate.artifact.retrieval.native_profiles.get(profile)
                            == Some(&candidate.score)
                        && candidate.artifact.retrieval.fused == candidate.score,
                    "native candidate score is not mapped to product retrieval scores"
                );
            }
            RecallLane::HybridCollapsed => {
                anyhow::bail!("candidate lane must name its source branch")
            }
        }
        anyhow::ensure!(
            candidates
                .insert(candidate.candidate_id.as_str(), candidate)
                .is_none(),
            "duplicate candidate id"
        );
    }
    let expected_collapses = expected_product_collapses(text, native);
    let mut clustered = HashSet::new();
    for cluster in &response.execution.collapse_clusters {
        anyhow::ensure!(
            !cluster.member_candidate_ids.is_empty(),
            "empty collapse cluster"
        );
        anyhow::ensure!(
            cluster
                .member_candidate_ids
                .iter()
                .any(|id| id == &cluster.representative_candidate_id),
            "collapse representative is not a member"
        );
        candidates
            .get(cluster.representative_candidate_id.as_str())
            .ok_or_else(|| anyhow::anyhow!("collapse representative is unknown"))?;
        for member_id in &cluster.member_candidate_ids {
            anyhow::ensure!(
                clustered.insert(member_id.as_str()),
                "candidate appears in multiple collapse clusters"
            );
            candidates
                .get(member_id.as_str())
                .ok_or_else(|| anyhow::anyhow!("collapse member is unknown"))?;
        }
    }
    anyhow::ensure!(
        clustered.len() == candidates.len(),
        "every branch candidate must enter exactly one collapse cluster"
    );
    let actual_components: BTreeSet<Vec<String>> = response
        .execution
        .collapse_clusters
        .iter()
        .map(|cluster| {
            let mut members = cluster.member_candidate_ids.clone();
            members.sort();
            members
        })
        .collect();
    anyhow::ensure!(
        actual_components
            == expected_collapses
                .iter()
                .map(|collapse| collapse.member_candidate_ids.clone())
                .collect(),
        "collapse clusters do not match the product collapse algorithm"
    );
    let mut representatives = BTreeMap::new();
    for cluster in &response.execution.collapse_clusters {
        let representative = candidates[cluster.representative_candidate_id.as_str()];
        let mut members = cluster.member_candidate_ids.clone();
        members.sort();
        let mut collapsed = expected_collapses
            .iter()
            .find(|expected| expected.member_candidate_ids == members)
            .map(|expected| expected.artifact.clone())
            .ok_or_else(|| anyhow::anyhow!("collapse cluster has no product result"))?;
        collapsed.evidence_id = representative.evidence_id.clone();
        anyhow::ensure!(
            representatives
                .insert(representative.evidence_id.as_str(), collapsed)
                .is_none(),
            "multiple collapse representatives reuse one evidence id"
        );
    }
    if request.lane == RecallLane::HybridCollapsed {
        anyhow::ensure!(
            response
                .execution
                .collapse_clusters
                .iter()
                .any(|cluster| cluster.member_candidate_ids.len() > 1),
            "hybrid arm did not prove an actual collapse"
        );
    }
    let mut retrieved_ids = HashSet::new();
    for retrieved in &response.retrieved {
        let representative = representatives
            .get(retrieved.evidence_id.as_str())
            .ok_or_else(|| {
                anyhow::anyhow!("retrieved evidence is not a collapse representative")
            })?;
        anyhow::ensure!(
            retrieved_ids.insert(retrieved.evidence_id.as_str()),
            "duplicate retrieved evidence"
        );
        anyhow::ensure!(
            retrieved.artifact == *representative,
            "retrieved evidence does not match the harness-recomputed collapsed artifact"
        );
        if !request.oracle_regions.is_empty() {
            anyhow::ensure!(
                request.oracle_regions.iter().any(|region| {
                    captured_region_ref(artifact_source_binding_id(&retrieved.artifact), region)
                        .is_ok_and(|expected| {
                            let parent_by_region: BTreeMap<_, _> =
                                [&expected, &retrieved.artifact.region]
                                    .into_iter()
                                    .filter_map(|region| {
                                        product_region_parent_id(region).map(|parent| {
                                            (region.region_id.clone(), parent.to_string())
                                        })
                                    })
                                    .collect();
                            product_regions_related(
                                &expected,
                                &retrieved.artifact.region,
                                &parent_by_region,
                                0.5,
                            )
                        })
                }),
                "case '{}' oracle arm returned evidence outside the oracle regions",
                case.case_id
            );
        }
    }
    anyhow::ensure!(
        reader_inputs.len() == response.retrieved.len(),
        "reader proof must cover every retrieved representative"
    );
    for (retrieved, reader) in response.retrieved.iter().zip(reader_inputs) {
        anyhow::ensure!(
            reader.evidence_id == retrieved.evidence_id && reader.mode == request.reader_mode,
            "reader proof identity mismatch"
        );
        match request.reader_mode {
            ReaderMode::TextProjection => {
                let projection = retrieved.artifact.projection.as_ref().ok_or_else(|| {
                    anyhow::anyhow!("text reader representative lacks projection")
                })?;
                let text = retrieved.rendered_text.as_deref().ok_or_else(|| {
                    anyhow::anyhow!("text reader received no rendered projection")
                })?;
                anyhow::ensure!(
                    projection.output_binding_id.as_deref() == Some(reader.binding_id.as_str())
                        && retrieved.artifact.projection_output_sha256.as_deref()
                            == Some(reader.verified_sha256.as_str())
                        && reader.byte_len == text.len() as u64
                        && sha256(text.as_bytes()) == reader.verified_sha256,
                    "text reader input is not the registered projection output"
                );
            }
            ReaderMode::SourceBlob => anyhow::ensure!(
                reader.binding_id == retrieved.artifact.binding_id
                    && reader.verified_sha256 == retrieved.artifact.blob.id.value
                    && reader.byte_len == retrieved.artifact.blob.size_bytes
                    && retrieved.rendered_text.is_none(),
                "blob reader input is not the verified source binding"
            ),
        }
    }
    if let Some(frozen) = &request.frozen_retrieval {
        let actual = frozen_retrieval_identity(&response.retrieved);
        anyhow::ensure!(
            frozen.fingerprint == stable_json_hash(&frozen.retrieved)?
                && frozen.fingerprint == stable_json_hash(&actual)?
                && frozen.retrieved == actual,
            "reader-modality arm changed frozen retrieval"
        );
    }
    Ok(())
}

#[derive(Clone, Debug)]
struct ExpectedProductCollapse {
    member_candidate_ids: Vec<String>,
    artifact: CapturedArtifact,
}

fn expected_product_collapses(
    text: &[BranchCandidate],
    native: &[BranchCandidate],
) -> Vec<ExpectedProductCollapse> {
    let ordered: Vec<_> = text.iter().chain(native).collect();
    let parent_by_region: BTreeMap<_, _> = ordered
        .iter()
        .filter_map(|candidate| {
            product_region_parent_id(&candidate.artifact.region).map(|parent| {
                (
                    candidate.artifact.region.region_id.clone(),
                    parent.to_string(),
                )
            })
        })
        .collect();
    let mut collapsed: Vec<ExpectedProductCollapse> = Vec::new();
    for candidate in ordered {
        if let Some(existing) = collapsed.iter_mut().find(|existing| {
            artifact_source_binding_id(&existing.artifact)
                == artifact_source_binding_id(&candidate.artifact)
                && product_regions_related(
                    &existing.artifact.region,
                    &candidate.artifact.region,
                    &parent_by_region,
                    0.5,
                )
        }) {
            existing
                .member_candidate_ids
                .push(candidate.candidate_id.clone());
            product_merge_artifacts(
                &mut existing.artifact,
                candidate.artifact.clone(),
                &parent_by_region,
            );
        } else {
            collapsed.push(ExpectedProductCollapse {
                member_candidate_ids: vec![candidate.candidate_id.clone()],
                artifact: candidate.artifact.clone(),
            });
        }
    }
    for collapse in &mut collapsed {
        collapse.member_candidate_ids.sort();
    }
    collapsed
}

fn artifact_source_binding_id(artifact: &CapturedArtifact) -> &str {
    artifact
        .projection
        .as_ref()
        .map(|projection| projection.source_binding_id.as_str())
        .unwrap_or(&artifact.binding_id)
}

fn product_merge_artifacts(
    existing: &mut CapturedArtifact,
    candidate: CapturedArtifact,
    parent_by_region: &BTreeMap<String, String>,
) {
    merge_retrieval_scores(&mut existing.retrieval, &candidate.retrieval);
    let projection = existing
        .projection
        .clone()
        .or_else(|| candidate.projection.clone());
    let projection_output_sha256 = existing
        .projection_output_sha256
        .clone()
        .or_else(|| candidate.projection_output_sha256.clone());
    let projection_output_blob = existing
        .projection_output_blob
        .clone()
        .or_else(|| candidate.projection_output_blob.clone());
    let candidate_is_more_authoritative =
        truth_tier_priority(candidate.truth_tier) > truth_tier_priority(existing.truth_tier);
    let existing_is_ancestor = product_is_ancestor(
        &existing.region.region_id,
        &candidate.region.region_id,
        parent_by_region,
    );
    if candidate_is_more_authoritative {
        let retrieval = existing.retrieval.clone();
        *existing = candidate;
        existing.retrieval = retrieval;
        existing.projection = projection;
        existing.projection_output_sha256 = projection_output_sha256;
        existing.projection_output_blob = projection_output_blob;
    } else if existing_is_ancestor {
        existing.region = candidate.region.clone();
        existing.projection = projection;
        existing.projection_output_sha256 = projection_output_sha256;
        existing.projection_output_blob = projection_output_blob;
    } else if existing.projection.is_none() {
        existing.projection = projection;
        existing.projection_output_sha256 = projection_output_sha256;
        existing.projection_output_blob = projection_output_blob;
    }
}

fn truth_tier_priority(tier: TruthTier) -> u8 {
    match tier {
        TruthTier::Raw => 3,
        TruthTier::DeterministicProjection => 2,
        TruthTier::ModelProjection => 1,
    }
}

#[cfg(test)]
fn retrieval_fingerprint(retrieved: &[RetrievedEvidence]) -> anyhow::Result<String> {
    stable_json_hash(&frozen_retrieval_identity(retrieved))
}

fn frozen_retrieval_identity(retrieved: &[RetrievedEvidence]) -> Vec<FrozenRetrievedEvidence> {
    retrieved
        .iter()
        .map(|item| FrozenRetrievedEvidence {
            evidence_id: item.evidence_id.clone(),
            lane: item.lane,
            artifact: item.artifact.clone(),
        })
        .collect()
}

fn fired_proof(
    request: &RecallRequest,
    response: &RecallResponse,
    reader_inputs: &[ReaderInputProof],
) -> anyhow::Result<FiredProof> {
    let mut regions: Vec<_> = response
        .retrieved
        .iter()
        .map(|item| stable_json_hash(&item.artifact.region))
        .collect::<anyhow::Result<_>>()?;
    regions.sort();
    Ok(FiredProof {
        request_sha256: stable_json_hash(request)?,
        response_sha256: stable_json_hash(response)?,
        effective_lane: response.execution.effective_lane,
        effective_reader_mode: response.execution.effective_reader_mode,
        text_branch_fingerprint: stable_json_hash(&response.execution.text_branch_candidates)?,
        native_branch_fingerprint: stable_json_hash(&response.execution.native_branch_candidates)?,
        collapse_fingerprint: stable_json_hash(&response.execution.collapse_clusters)?,
        oracle_region_count: response.execution.oracle_regions_applied.len(),
        reader_input_fingerprint: stable_json_hash(reader_inputs)?,
        bound_reader_request_sha256: None,
        bound_reader_effective_request_sha256: None,
        bound_reader_effective_input_fingerprint: None,
        retrieved_region_fingerprint: stable_json_hash(&regions)?,
    })
}

fn enforce_budget(
    budget: &ExecutionBudget,
    provider_calls: u64,
    cost_micro_usd: u64,
) -> anyhow::Result<()> {
    if !budget.max_step.permits_provider_calls() {
        anyhow::ensure!(
            provider_calls == 0 && cost_micro_usd == 0,
            "offline cost-ladder step observed {provider_calls} provider calls and {cost_micro_usd} micro-USD"
        );
    }
    anyhow::ensure!(
        provider_calls <= budget.max_provider_calls,
        "provider-call budget exceeded: {provider_calls} > {}",
        budget.max_provider_calls
    );
    anyhow::ensure!(
        cost_micro_usd <= budget.max_cost_micro_usd,
        "cost budget exceeded: {cost_micro_usd} > {} micro-USD",
        budget.max_cost_micro_usd
    );
    Ok(())
}

fn git_object_bytes(product_root: &Path, revision: &str, path: &str) -> anyhow::Result<Vec<u8>> {
    let output = std::process::Command::new("git")
        .arg("-C")
        .arg(product_root)
        .arg("show")
        .arg(format!("{revision}:{path}"))
        .output()
        .with_context(|| format!("run git show for product object '{path}'"))?;
    anyhow::ensure!(
        output.status.success(),
        "git show failed for product object '{}': {}",
        path,
        String::from_utf8_lossy(&output.stderr).trim()
    );
    Ok(output.stdout)
}

/// Verify the product-owned multimodal recall contract from an exact local git checkout.
///
/// No worktree bytes are trusted: the gate reads the artifact and every declared source file from
/// the pinned commit's git objects, verifies their SHA-256 values, then replays the product-emitted
/// ordered collapse outputs through the bench implementation.
pub fn verify_pinned_product_contract(product_root: &Path) -> anyhow::Result<()> {
    anyhow::ensure!(
        PINNED_PRODUCT_GIT_SHA.len() == 40
            && PINNED_PRODUCT_GIT_SHA
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit()),
        "bench product contract commit pin is not finalized"
    );
    let metadata = std::fs::symlink_metadata(product_root)
        .with_context(|| format!("stat product root {}", product_root.display()))?;
    anyhow::ensure!(
        metadata.file_type().is_dir() && !metadata.file_type().is_symlink(),
        "product root must be a real non-symlink directory"
    );
    let head = std::process::Command::new("git")
        .arg("-C")
        .arg(product_root)
        .args(["rev-parse", "HEAD"])
        .output()
        .with_context(|| "resolve product HEAD")?;
    anyhow::ensure!(
        head.status.success(),
        "resolve product HEAD failed: {}",
        String::from_utf8_lossy(&head.stderr).trim()
    );
    anyhow::ensure!(
        String::from_utf8_lossy(&head.stdout).trim() == PINNED_PRODUCT_GIT_SHA,
        "product HEAD does not match pinned contract commit"
    );
    let artifact = git_object_bytes(product_root, PINNED_PRODUCT_GIT_SHA, PRODUCT_CONTRACT_PATH)?;
    anyhow::ensure!(
        sha256(&artifact) == PINNED_PRODUCT_CONTRACT_SHA256,
        "product-owned multimodal contract artifact hash mismatch"
    );
    let contract: ProductMultimodalContract = serde_json::from_slice(&artifact)
        .with_context(|| "decode product-owned multimodal contract")?;
    anyhow::ensure!(
        contract.contract_id == PRODUCT_CONTRACT_ID && contract.contract_version == 1,
        "unexpected product multimodal contract identity"
    );
    anyhow::ensure!(
        contract.schema.envelope == "multimodal_recall_contract.v1"
            && contract.schema.artifact_evidence_wire_type
                == "symbiotic_memory::recall::ArtifactEvidence"
            && contract.schema.collapse_function
                == "symbiotic_memory::recall::collapse_artifact_evidence(default_threshold=0.5)"
            && contract.schema.source_hash_algorithm == "sha256"
            && contract.schema.default_overlap_threshold_millionths == 500_000
            && contract.schema.vector_output_semantics == "ordered_actual_product_output",
        "unexpected product multimodal contract schema"
    );
    let expected_source_paths: BTreeSet<_> = [
        "src/content/transform.rs",
        "src/content/types.rs",
        "src/recall/types.rs",
    ]
    .into_iter()
    .collect();
    anyhow::ensure!(
        contract
            .source_files_sha256
            .keys()
            .map(String::as_str)
            .collect::<BTreeSet<_>>()
            == expected_source_paths,
        "product contract source file set changed"
    );
    for (path, expected_sha256) in &contract.source_files_sha256 {
        validate_sha256("product source object", expected_sha256)?;
        let bytes = git_object_bytes(product_root, PINNED_PRODUCT_GIT_SHA, path)?;
        anyhow::ensure!(
            sha256(&bytes) == *expected_sha256,
            "product source object hash mismatch for '{path}'"
        );
    }
    anyhow::ensure!(
        !contract.wire_specimens.is_empty(),
        "product contract has no real wire specimens"
    );
    for specimen in &contract.wire_specimens {
        validate_retrieval_scores(&specimen.retrieval)?;
        non_empty("product specimen binding", &specimen.binding_id)?;
        non_empty("product specimen region", &specimen.region.region_id)?;
    }
    let expected_case_ids: BTreeSet<_> = [
        "whole_contains_page",
        "transitive_parent_ancestry",
        "sheet_contains_cell_range",
        "overlapping_a1_ranges",
        "rectangle_iou_at_or_above_default_threshold",
        "rectangle_iou_below_default_threshold",
        "time_byte_and_named_non_overlap",
        "text_native_score_merge",
    ]
    .into_iter()
    .collect();
    anyhow::ensure!(
        contract
            .collapse_vectors
            .iter()
            .map(|vector| vector.case_id.as_str())
            .collect::<BTreeSet<_>>()
            == expected_case_ids,
        "product contract collapse vector set changed"
    );
    for vector in contract.collapse_vectors {
        let candidates: Vec<_> = vector
            .input
            .into_iter()
            .enumerate()
            .map(|(index, product)| {
                let evidence_id = format!("{}-{index}", vector.case_id);
                BranchCandidate {
                    candidate_id: format!("candidate-{index}"),
                    evidence_id: evidence_id.clone(),
                    lane: RecallLane::TextProjection,
                    profile_id: None,
                    score: product.retrieval.fused,
                    artifact: CapturedArtifact {
                        evidence_id,
                        projection_output_sha256: None,
                        projection_output_blob: None,
                        product,
                    },
                }
            })
            .collect();
        let actual: Vec<_> = expected_product_collapses(&candidates, &[])
            .into_iter()
            .map(|collapsed| collapsed.artifact.product)
            .collect();
        anyhow::ensure!(
            actual == vector.actual_collapsed_output,
            "bench collapse drifted from product vector '{}'",
            vector.case_id
        );
    }
    Ok(())
}

fn validate_descriptor(descriptor: &AdapterDescriptor) -> anyhow::Result<()> {
    validate_id("adapter_id", &descriptor.adapter_id)?;
    non_empty("adapter version", &descriptor.adapter_version)
}

fn normalize_text(value: &str) -> String {
    value
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

fn stable_json_hash(value: &(impl Serialize + ?Sized)) -> anyhow::Result<String> {
    let bytes = serde_json::to_vec(value)?;
    Ok(sha256(&bytes))
}

fn sha256(bytes: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(bytes);
    format!("{:x}", digest.finalize())
}

fn validate_sha256(name: &str, value: &str) -> anyhow::Result<()> {
    anyhow::ensure!(
        value.len() == 64
            && value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f')),
        "{name} must be a lowercase SHA-256 hex digest"
    );
    Ok(())
}

fn validate_id(name: &str, value: &str) -> anyhow::Result<()> {
    anyhow::ensure!(
        !value.is_empty()
            && value
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric()
                    || matches!(byte, b'-' | b'_' | b'.' | b':')),
        "{name} contains unsafe characters"
    );
    Ok(())
}

fn non_empty(name: &str, value: &str) -> anyhow::Result<()> {
    anyhow::ensure!(!value.trim().is_empty(), "{name} must be non-empty");
    Ok(())
}

fn scored_artifact(
    artifact: &CapturedArtifact,
    lane: RecallLane,
    profile_id: Option<&str>,
    score: f32,
) -> CapturedArtifact {
    let mut artifact = artifact.clone();
    match lane {
        RecallLane::TextProjection => {
            let projection = artifact
                .projection
                .as_ref()
                .expect("validated text candidate has projection provenance");
            artifact.binding_id = projection
                .output_binding_id
                .clone()
                .expect("validated text candidate has projection output binding");
            artifact.blob = artifact
                .projection_output_blob
                .clone()
                .expect("validated text candidate has projection output blob");
            artifact.truth_tier = TruthTier::DeterministicProjection;
        }
        RecallLane::Native => {
            artifact.truth_tier = TruthTier::Raw;
        }
        RecallLane::HybridCollapsed => {}
    }
    artifact.retrieval = match lane {
        RecallLane::TextProjection => CapturedRetrievalScores {
            text_projection: Some(score),
            native_profiles: BTreeMap::new(),
            fused: score,
        },
        RecallLane::Native => CapturedRetrievalScores {
            text_projection: None,
            native_profiles: profile_id
                .map(|profile| BTreeMap::from([(profile.to_string(), score)]))
                .unwrap_or_default(),
            fused: score,
        },
        RecallLane::HybridCollapsed => CapturedRetrievalScores::default(),
    };
    artifact
}

/// No-network cell-A control. It performs deterministic lexical retrieval over the supplied text
/// projections and returns the top projection verbatim. It never inspects gold answers or oracle
/// evidence and deliberately advertises no native capability.
#[derive(Clone, Debug)]
pub struct TextProjectionBaseline {
    descriptor: AdapterDescriptor,
    projections: BTreeMap<String, String>,
}

impl Default for TextProjectionBaseline {
    fn default() -> Self {
        Self {
            descriptor: AdapterDescriptor {
                adapter_id: "membench-text-projection-control".to_string(),
                adapter_version: "1".to_string(),
                capabilities: MultimodalCapabilities {
                    text_projection_recall: true,
                    text_projection_reader: true,
                    ..MultimodalCapabilities::default()
                },
            },
            projections: BTreeMap::new(),
        }
    }
}

impl MultimodalRecallAdapter for TextProjectionBaseline {
    fn descriptor(&self) -> AdapterDescriptor {
        self.descriptor.clone()
    }

    fn import_corpus(
        &mut self,
        request: &CorpusImportRequest,
        _spend: &mut SpendJournal,
    ) -> Result<CapturedCorpus, AdapterFailure> {
        let mut artifacts = Vec::with_capacity(request.items.len());
        for item in &request.items {
            let evidence = &item.evidence;
            let source_binding_id = format!(
                "binding-{}",
                sha256(
                    format!(
                        "{}\0{}\0source",
                        request.corpus_id, evidence.source_media.asset_id
                    )
                    .as_bytes()
                )
            );
            let region = captured_region_ref(&source_binding_id, &evidence.region)
                .map_err(|error| AdapterFailure::Failed(error.to_string()))?;
            let projection = evidence.text_projection.as_ref().map(|text| {
                self.projections
                    .insert(evidence.evidence_id.clone(), text.clone());
                CapturedProjectionRef {
                    execution_id: sha256(
                        format!("{}\0projection-execution", evidence.evidence_id).as_bytes(),
                    ),
                    source_binding_id: source_binding_id.clone(),
                    source_region_id: region.region_id.clone(),
                    output_binding_id: Some(format!(
                        "binding-{}",
                        sha256(format!("{}\0projection-output", evidence.evidence_id).as_bytes())
                    )),
                    transformed_at: None,
                }
            });
            let projection_output_sha256 = evidence
                .text_projection
                .as_ref()
                .map(|text| sha256(text.as_bytes()));
            let projection_output_blob =
                evidence
                    .text_projection
                    .as_ref()
                    .map(|text| CapturedBlobRef {
                        id: CapturedContentDigest {
                            algorithm: "sha256".to_string(),
                            value: sha256(text.as_bytes()),
                        },
                        size_bytes: text.len() as u64,
                        detected_media_type: "text/plain; charset=utf-8".to_string(),
                    });
            artifacts.push(CapturedArtifact {
                evidence_id: evidence.evidence_id.clone(),
                projection_output_sha256,
                projection_output_blob,
                product: CapturedArtifactEvidence {
                    binding_id: source_binding_id,
                    blob: CapturedBlobRef {
                        id: CapturedContentDigest {
                            algorithm: "sha256".to_string(),
                            value: evidence.source_media.sha256.clone(),
                        },
                        size_bytes: evidence.source_media.size_bytes,
                        detected_media_type: evidence.source_media.media_type.clone(),
                    },
                    region,
                    truth_tier: TruthTier::Raw,
                    projection,
                    retrieval: CapturedRetrievalScores::default(),
                },
            });
        }
        Ok(CapturedCorpus {
            corpus_id: request.corpus_id.clone(),
            artifacts,
        })
    }

    fn recall(
        &mut self,
        request: &RecallRequest,
        _spend: &mut SpendJournal,
        reader: &mut ReaderJournal,
    ) -> Result<RecallResponse, AdapterFailure> {
        (|| {
            if request.lane != RecallLane::TextProjection {
                return Err(AdapterFailure::CapabilityUnavailable(format!(
                    "lane {:?}",
                    request.lane
                )));
            }
            if request.reader_mode != ReaderMode::TextProjection {
                return Err(AdapterFailure::CapabilityUnavailable(
                    "source-blob reader".to_string(),
                ));
            }
            if !request.oracle_regions.is_empty() {
                return Err(AdapterFailure::CapabilityUnavailable(
                    "oracle region filter".to_string(),
                ));
            }
            let query = lexical_terms(&request.question.text);
            let best = request
                .corpus
                .artifacts
                .iter()
                .filter_map(|item| {
                    self.projections
                        .get(&item.evidence_id)
                        .map(|text| (item, text))
                })
                .map(|(item, text)| {
                    let terms = lexical_terms(text);
                    let overlap = query.intersection(&terms).count();
                    let union = query.union(&terms).count().max(1);
                    (overlap, union, item.evidence_id.as_str(), item, text)
                })
                .max_by(|left, right| {
                    ((left.0 as u128) * (right.1 as u128))
                        .cmp(&((right.0 as u128) * (left.1 as u128)))
                        .then_with(|| right.2.cmp(left.2))
                })
                .ok_or_else(|| {
                    AdapterFailure::Failed("no text projections available".to_string())
                })?;
            let best_id = best.3.evidence_id.clone();
            let text_branch_candidates: Vec<_> = request
                .corpus
                .artifacts
                .iter()
                .filter(|item| item.projection.is_some())
                .enumerate()
                .map(|(index, artifact)| {
                    let score = if artifact.evidence_id == best_id {
                        1.0
                    } else {
                        0.0
                    };
                    BranchCandidate {
                        candidate_id: format!("text-{index}"),
                        evidence_id: artifact.evidence_id.clone(),
                        lane: RecallLane::TextProjection,
                        profile_id: None,
                        score,
                        artifact: scored_artifact(
                            artifact,
                            RecallLane::TextProjection,
                            None,
                            score,
                        ),
                    }
                })
                .collect();
            let best_candidate = text_branch_candidates
                .iter()
                .find(|candidate| candidate.evidence_id == best_id)
                .ok_or_else(|| AdapterFailure::Failed("best candidate missing".to_string()))?;
            let product_collapses = expected_product_collapses(&text_branch_candidates, &[]);
            let selected_collapse = product_collapses
                .iter()
                .find(|collapse| {
                    collapse
                        .member_candidate_ids
                        .contains(&best_candidate.candidate_id)
                })
                .ok_or_else(|| AdapterFailure::Failed("best collapse missing".to_string()))?;
            let representative_id = &selected_collapse.member_candidate_ids[0];
            let representative = text_branch_candidates
                .iter()
                .find(|candidate| &candidate.candidate_id == representative_id)
                .ok_or_else(|| AdapterFailure::Failed("representative missing".to_string()))?;
            let best_id = representative.evidence_id.clone();
            let mut selected_artifact = selected_collapse.artifact.clone();
            selected_artifact.evidence_id = best_id.clone();
            let projection = selected_artifact
                .projection
                .as_ref()
                .ok_or_else(|| AdapterFailure::Failed("projection missing".to_string()))?;
            let projection_binding = projection
                .output_binding_id
                .clone()
                .ok_or_else(|| AdapterFailure::Failed("projection missing".to_string()))?;
            let rendered_text =
                String::from_utf8(reader.read_text_projection(&best_id, &projection_binding)?)
                    .map_err(|error| {
                        AdapterFailure::Failed(format!("projection is not UTF-8: {error}"))
                    })?;
            Ok(RecallResponse {
                answer: rendered_text.clone(),
                retrieved: vec![RetrievedEvidence {
                    evidence_id: best_id.clone(),
                    lane: RecallLane::TextProjection,
                    artifact: selected_artifact,
                    rendered_text: Some(rendered_text),
                }],
                execution: AdapterExecutionProof {
                    effective_lane: RecallLane::TextProjection,
                    effective_reader_mode: ReaderMode::TextProjection,
                    text_branch_candidates,
                    native_branch_candidates: Vec::new(),
                    collapse_clusters: product_collapses
                        .iter()
                        .map(|collapse| CollapseCluster {
                            representative_candidate_id: collapse.member_candidate_ids[0].clone(),
                            member_candidate_ids: collapse.member_candidate_ids.clone(),
                        })
                        .collect(),
                    oracle_regions_applied: Vec::new(),
                },
            })
        })()
    }
}

fn lexical_terms(value: &str) -> BTreeSet<String> {
    value
        .split(|character: char| !character.is_alphanumeric())
        .filter(|term| !term.is_empty())
        .map(str::to_lowercase)
        .collect()
}

fn contains_word_bounded(haystack: &str, needle: &str) -> bool {
    haystack.match_indices(needle).any(|(index, _)| {
        let before = haystack[..index].chars().next_back();
        let after = haystack[index + needle.len()..].chars().next();
        !before.is_some_and(char::is_alphanumeric) && !after.is_some_and(char::is_alphanumeric)
    })
}

#[cfg(test)]
mod repair_tests {
    use super::*;
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_RUN: AtomicU64 = AtomicU64::new(1);

    fn fixture() -> MultimodalFixtureSet {
        load_fixture_file(
            &Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("fixtures/multimodal/v1/heldout-recall.json"),
        )
        .unwrap()
    }

    fn plan(arm: ExperimentArm) -> MultimodalRunPlan {
        let run_number = NEXT_RUN.fetch_add(1, Ordering::Relaxed);
        MultimodalRunPlan {
            arm,
            budget: ExecutionBudget::offline(),
            hypothesis: PreregisteredHypothesis {
                statement: "Native evidence changes media-dependent recall.".to_string(),
                mechanism: "The native lane preserves source structure.".to_string(),
                decision_gate: "Promote only after separated replicated ranges.".to_string(),
            },
            started_at: "2026-08-22T00:00:00Z".to_string(),
            run_instance_id: format!("repair-test-run-{run_number}"),
            git_sha: "5b7c198b3880992a31140022c99b7514034acd53".to_string(),
            effective_config_sha256: sha256(b"repair-test-config"),
            asset_root: Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/multimodal/v1"),
            spend_ledger_path: None,
            run_registry_root: std::env::temp_dir().join(format!(
                "membench-multimodal-run-registry-{}-{run_number}",
                std::process::id()
            )),
            max_import_asset_bytes: 16 * 1024 * 1024,
        }
    }

    #[derive(Default)]
    struct SpyAdapter {
        inner: TextProjectionBaseline,
        imported_distractor: bool,
        observed_oracle: bool,
        observed_question_media: bool,
        corrupt_binding: bool,
        conflicting_raw_binding: bool,
        conflicting_projection_binding: bool,
        invalid_scores: bool,
        invalid_spend: bool,
        fail_after_spend: bool,
        ledger_visible_before_failed_dispatch: bool,
    }

    impl MultimodalRecallAdapter for SpyAdapter {
        fn descriptor(&self) -> AdapterDescriptor {
            self.inner.descriptor()
        }

        fn import_corpus(
            &mut self,
            request: &CorpusImportRequest,
            spend: &mut SpendJournal,
        ) -> Result<CapturedCorpus, AdapterFailure> {
            self.imported_distractor |= request
                .items
                .iter()
                .any(|item| item.evidence.evidence_id.contains("distractor"));
            let mut captured = self.inner.import_corpus(request, spend)?;
            if self.corrupt_binding {
                captured.artifacts[0].binding_id =
                    format!("cas://sha256/{}", captured.artifacts[0].blob.id.value);
            }
            if self.conflicting_raw_binding && captured.artifacts.len() > 1 {
                captured.artifacts[1].binding_id = captured.artifacts[0].binding_id.clone();
            }
            if self.conflicting_projection_binding && captured.artifacts.len() > 1 {
                let first = captured.artifacts[0]
                    .projection
                    .as_ref()
                    .and_then(|projection| projection.output_binding_id.clone());
                if let (Some(binding_id), Some(second)) =
                    (first, captured.artifacts[1].projection.as_mut())
                {
                    second.output_binding_id = Some(binding_id);
                }
            }
            Ok(captured)
        }

        fn recall(
            &mut self,
            request: &RecallRequest,
            spend: &mut SpendJournal,
            reader: &mut ReaderJournal,
        ) -> Result<RecallResponse, AdapterFailure> {
            self.observed_oracle |= !request.oracle_regions.is_empty();
            self.observed_question_media |= !request.question.media.is_empty();
            if self.invalid_spend {
                spend.reserve(
                    "over-budget-call",
                    "provider",
                    "model",
                    request.budget.cost_micro_usd.saturating_add(1),
                    "test",
                )?;
            }
            if self.fail_after_spend {
                let _reservation = spend.reserve("failed-call", "provider", "model", 1, "test")?;
                self.ledger_visible_before_failed_dispatch = spend
                    .ledger_path
                    .as_ref()
                    .and_then(|path| fs::read_to_string(path).ok())
                    .is_some_and(|ledger| ledger.contains("reserved"));
                return Err(AdapterFailure::Failed("provider failed".to_string()));
            }
            let mut response = self.inner.recall(request, spend, reader)?;
            if self.invalid_scores {
                response.execution.text_branch_candidates[0]
                    .artifact
                    .retrieval
                    .text_projection = None;
            }
            Ok(response)
        }
    }

    #[derive(Default)]
    struct HybridAdapter {
        inner: TextProjectionBaseline,
        tamper_blob_retrieval: bool,
        invalid_collapse: bool,
        skip_reader: bool,
        return_cached_reader_text_from_retrieval: bool,
    }

    impl MultimodalRecallAdapter for HybridAdapter {
        fn descriptor(&self) -> AdapterDescriptor {
            AdapterDescriptor {
                adapter_id: "hybrid-proof-adapter".to_string(),
                adapter_version: "1".to_string(),
                capabilities: MultimodalCapabilities {
                    text_projection_recall: true,
                    native_image_recall: true,
                    native_document_recall: true,
                    hybrid_collapse: true,
                    text_projection_reader: true,
                    source_blob_reader: true,
                    oracle_region_filter: true,
                },
            }
        }

        fn import_corpus(
            &mut self,
            request: &CorpusImportRequest,
            spend: &mut SpendJournal,
        ) -> Result<CapturedCorpus, AdapterFailure> {
            self.inner.import_corpus(request, spend)
        }

        fn recall(
            &mut self,
            request: &RecallRequest,
            _spend: &mut SpendJournal,
            reader: &mut ReaderJournal,
        ) -> Result<RecallResponse, AdapterFailure> {
            (|| {
                let artifacts: Vec<_> = request
                    .corpus
                    .artifacts
                    .iter()
                    .filter(|artifact| artifact.projection.is_some())
                    .cloned()
                    .collect();
                let mut text = Vec::new();
                let mut native = Vec::new();
                for (index, artifact) in artifacts.iter().enumerate() {
                    if request.lane != RecallLane::Native {
                        let score = 1.0 - index as f32 / 100.0;
                        text.push(BranchCandidate {
                            candidate_id: format!("text-{index}"),
                            evidence_id: artifact.evidence_id.clone(),
                            lane: RecallLane::TextProjection,
                            profile_id: None,
                            score,
                            artifact: scored_artifact(
                                artifact,
                                RecallLane::TextProjection,
                                None,
                                score,
                            ),
                        });
                    }
                    if request.lane != RecallLane::TextProjection {
                        let score = 1.0 - index as f32 / 100.0;
                        native.push(BranchCandidate {
                            candidate_id: format!("native-{index}"),
                            evidence_id: artifact.evidence_id.clone(),
                            lane: RecallLane::Native,
                            profile_id: Some("native-v1".to_string()),
                            score,
                            artifact: scored_artifact(
                                artifact,
                                RecallLane::Native,
                                Some("native-v1"),
                                score,
                            ),
                        });
                    }
                }
                let product_collapses = expected_product_collapses(&text, &native);
                let mut clusters: Vec<_> = product_collapses
                    .iter()
                    .map(|collapse| CollapseCluster {
                        representative_candidate_id: collapse.member_candidate_ids[0].clone(),
                        member_candidate_ids: collapse.member_candidate_ids.clone(),
                    })
                    .collect();
                if self.invalid_collapse && request.lane == RecallLane::HybridCollapsed {
                    let split = clusters[0]
                        .member_candidate_ids
                        .pop()
                        .ok_or_else(|| AdapterFailure::Failed("no collapse member".to_string()))?;
                    clusters.push(CollapseCluster {
                        representative_candidate_id: split.clone(),
                        member_candidate_ids: vec![split],
                    });
                }
                let selected_index = usize::from(
                    self.tamper_blob_retrieval
                        && request.reader_mode == ReaderMode::SourceBlob
                        && product_collapses.len() > 1,
                );
                let representative_id = clusters
                    .get(selected_index)
                    .map(|cluster| cluster.representative_candidate_id.as_str())
                    .ok_or_else(|| AdapterFailure::Failed("no artifact".to_string()))?;
                let representative = text
                    .iter()
                    .chain(&native)
                    .find(|candidate| candidate.candidate_id == representative_id)
                    .ok_or_else(|| AdapterFailure::Failed("no representative".to_string()))?;
                let mut artifact = product_collapses[selected_index].artifact.clone();
                artifact.evidence_id = representative.evidence_id.clone();
                let projection = artifact
                    .projection
                    .as_ref()
                    .ok_or_else(|| AdapterFailure::Failed("no projection".to_string()))?;
                let rendered_text = if self.skip_reader {
                    match request.reader_mode {
                        ReaderMode::TextProjection => Some("adapter bypassed resolver".to_string()),
                        ReaderMode::SourceBlob => None,
                    }
                } else {
                    match request.reader_mode {
                        ReaderMode::TextProjection => {
                            let binding_id =
                                projection.output_binding_id.as_deref().ok_or_else(|| {
                                    AdapterFailure::Failed(
                                        "projection output binding missing".to_string(),
                                    )
                                })?;
                            Some(
                                String::from_utf8(
                                    reader
                                        .read_text_projection(&artifact.evidence_id, binding_id)?,
                                )
                                .map_err(|error| {
                                    AdapterFailure::Failed(format!(
                                        "projection is not UTF-8: {error}"
                                    ))
                                })?,
                            )
                        }
                        ReaderMode::SourceBlob => {
                            let source_bytes = reader
                                .read_source_blob(&artifact.evidence_id, &artifact.binding_id)?;
                            if source_bytes.is_empty() {
                                return Err(AdapterFailure::Failed(
                                    "source blob reader received empty bytes".to_string(),
                                ));
                            }
                            None
                        }
                    }
                };
                Ok(RecallResponse {
                    answer: rendered_text
                        .clone()
                        .unwrap_or_else(|| "source blob reader answer".to_string()),
                    retrieved: vec![RetrievedEvidence {
                        evidence_id: artifact.evidence_id.clone(),
                        lane: if request.lane == RecallLane::Native {
                            RecallLane::Native
                        } else {
                            RecallLane::TextProjection
                        },
                        artifact,
                        rendered_text,
                    }],
                    execution: AdapterExecutionProof {
                        effective_lane: request.lane,
                        effective_reader_mode: request.reader_mode,
                        text_branch_candidates: text,
                        native_branch_candidates: native,
                        collapse_clusters: clusters,
                        oracle_regions_applied: request.oracle_regions.clone(),
                    },
                })
            })()
        }

        fn retrieve(
            &mut self,
            request: &RetrievalRequest,
            _spend: &mut SpendJournal,
        ) -> Result<RetrievalResponse, AdapterFailure> {
            (|| {
                if request.lane != RecallLane::HybridCollapsed {
                    return Err(AdapterFailure::Failed(
                        "cell E retrieval must be hybrid".to_string(),
                    ));
                }
                let artifacts: Vec<_> = request
                    .corpus
                    .artifacts
                    .iter()
                    .filter(|artifact| artifact.projection.is_some())
                    .cloned()
                    .collect();
                let mut text = Vec::new();
                let mut native = Vec::new();
                for (index, artifact) in artifacts.iter().enumerate() {
                    let score = 1.0 - index as f32 / 100.0;
                    text.push(BranchCandidate {
                        candidate_id: format!("text-{index}"),
                        evidence_id: artifact.evidence_id.clone(),
                        lane: RecallLane::TextProjection,
                        profile_id: None,
                        score,
                        artifact: scored_artifact(
                            artifact,
                            RecallLane::TextProjection,
                            None,
                            score,
                        ),
                    });
                    native.push(BranchCandidate {
                        candidate_id: format!("native-{index}"),
                        evidence_id: artifact.evidence_id.clone(),
                        lane: RecallLane::Native,
                        profile_id: Some("native-v1".to_string()),
                        score,
                        artifact: scored_artifact(
                            artifact,
                            RecallLane::Native,
                            Some("native-v1"),
                            score,
                        ),
                    });
                }
                let product_collapses = expected_product_collapses(&text, &native);
                let mut clusters: Vec<_> = product_collapses
                    .iter()
                    .map(|collapse| CollapseCluster {
                        representative_candidate_id: collapse.member_candidate_ids[0].clone(),
                        member_candidate_ids: collapse.member_candidate_ids.clone(),
                    })
                    .collect();
                if self.invalid_collapse {
                    let split = clusters[0]
                        .member_candidate_ids
                        .pop()
                        .ok_or_else(|| AdapterFailure::Failed("no collapse member".to_string()))?;
                    clusters.push(CollapseCluster {
                        representative_candidate_id: split.clone(),
                        member_candidate_ids: vec![split],
                    });
                }
                let representative_id = clusters
                    .first()
                    .map(|cluster| cluster.representative_candidate_id.as_str())
                    .ok_or_else(|| AdapterFailure::Failed("no artifact".to_string()))?;
                let representative = text
                    .iter()
                    .chain(&native)
                    .find(|candidate| candidate.candidate_id == representative_id)
                    .ok_or_else(|| AdapterFailure::Failed("no representative".to_string()))?;
                let mut artifact = product_collapses[0].artifact.clone();
                artifact.evidence_id = representative.evidence_id.clone();
                Ok(RetrievalResponse {
                    retrieved: vec![RetrievedEvidence {
                        evidence_id: artifact.evidence_id.clone(),
                        lane: RecallLane::TextProjection,
                        artifact,
                        rendered_text: self
                            .return_cached_reader_text_from_retrieval
                            .then(|| "cached import text".to_string()),
                    }],
                    execution: RetrievalExecutionProof {
                        effective_lane: RecallLane::HybridCollapsed,
                        text_branch_candidates: text,
                        native_branch_candidates: native,
                        collapse_clusters: clusters,
                        oracle_regions_applied: Vec::new(),
                    },
                })
            })()
        }
    }

    struct BoundEchoReader {
        mode: ReaderMode,
        tamper_request_hash: bool,
        calls: usize,
        observed_input_counts: Vec<usize>,
    }

    impl BoundEchoReader {
        fn new(mode: ReaderMode) -> Self {
            Self {
                mode,
                tamper_request_hash: false,
                calls: 0,
                observed_input_counts: Vec::new(),
            }
        }
    }

    impl MultimodalReaderAdapter for BoundEchoReader {
        fn descriptor(&self) -> ReaderDescriptor {
            ReaderDescriptor {
                reader_id: match self.mode {
                    ReaderMode::TextProjection => "bound-text-reader",
                    ReaderMode::SourceBlob => "bound-blob-reader",
                }
                .to_string(),
                reader_version: "1".to_string(),
                mode: self.mode,
            }
        }

        fn read(
            &mut self,
            request: &BoundReaderRequest,
            _spend: &mut SpendJournal,
        ) -> Result<BoundReaderResponse, AdapterFailure> {
            self.calls += 1;
            self.observed_input_counts
                .push(request.invocation.inputs.len());
            let answer = match self.mode {
                ReaderMode::TextProjection => request
                    .invocation
                    .inputs
                    .first()
                    .and_then(|input| String::from_utf8(input.bytes.clone()).ok())
                    .unwrap_or_else(|| "no text".to_string()),
                ReaderMode::SourceBlob => "source blob reader answer".to_string(),
            };
            Ok(BoundReaderResponse {
                answer,
                effective_request_sha256: if self.tamper_request_hash {
                    sha256(b"wrong request")
                } else {
                    request.request_sha256.clone()
                },
                effective_input_sha256: request
                    .invocation
                    .inputs
                    .iter()
                    .map(|input| sha256(&input.bytes))
                    .collect(),
            })
        }
    }

    #[test]
    fn checked_fixture_has_real_assets_and_distractors() {
        let fixture = fixture();
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/multimodal/v1");
        let source_assets: Vec<_> = [
            "assets/distractor.txt",
            "assets/forecast.csv",
            "assets/incident.svg",
            "assets/invoice.pdf",
        ]
        .into_iter()
        .map(|name| (name, fs::read(root.join(name)).unwrap()))
        .collect();
        let source_parts: Vec<_> = source_assets
            .iter()
            .map(|(name, bytes)| (*name, bytes.as_slice()))
            .collect();
        assert_eq!(fixture.source.source_digest, sha256_parts(&source_parts));
        assert!(
            fixture
                .corpora
                .iter()
                .all(|corpus| corpus.evidence.len() >= 2)
        );
        for corpus in &fixture.corpora {
            verify_corpus_assets(corpus, &root, 16 * 1024 * 1024).unwrap();
        }
    }

    #[test]
    fn per_asset_import_cap_fails_closed() {
        let checked = fixture();
        let error = verify_corpus_assets(
            &checked.corpora[0],
            &Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/multimodal/v1"),
            1,
        )
        .unwrap_err();
        assert!(error.to_string().contains("max_import_asset_bytes"));
    }

    #[cfg(unix)]
    #[test]
    fn fixture_import_rejects_symlinked_path_components() {
        use std::os::unix::fs::symlink;

        let root = tempfile::tempdir().unwrap();
        symlink(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/multimodal/v1/assets"),
            root.path().join("assets"),
        )
        .unwrap();
        let checked = fixture();
        let error =
            verify_corpus_assets(&checked.corpora[0], root.path(), 16 * 1024 * 1024).unwrap_err();
        assert!(error.to_string().contains("following links"));
    }

    #[test]
    fn official_loader_keeps_unlabelled_haystack_states_as_distractors() {
        let root = tempfile::tempdir().unwrap();
        fs::create_dir_all(root.path().join("haystacks")).unwrap();
        fs::create_dir_all(root.path().join("screenshots/t1")).unwrap();
        fs::create_dir_all(root.path().join("screenshots/t2")).unwrap();
        fs::write(root.path().join("screenshots/t1/0.png"), b"png-zero").unwrap();
        fs::write(root.path().join("screenshots/t1/1.png"), b"png-one").unwrap();
        fs::write(root.path().join("screenshots/t2/0.png"), b"png-two").unwrap();
        fs::write(
            root.path().join("questions.jsonl"),
            serde_json::json!({"id":"q1","domain":"web","environment":"web","question_type":"visual","question":"which?","image":null,"answer":"one","eval_function":"norm_phrase_set_match|lower=true"}).to_string()
                + "\n"
                + &serde_json::json!({"id":"q2","domain":"web","environment":"web","question_type":"visual","question":"second?","image":null,"answer":"two","eval_function":"norm_phrase_set_match|lower=true"}).to_string()
                + "\n",
        ).unwrap();
        fs::write(
            root.path().join("trajectories.jsonl"),
            serde_json::json!({"id":"t1","domain":"web","environment":"web","goal":"inspect","outcome":"success","start_url":"https://example.test","states":[
                {"step":0,"state_index":0,"url":"https://example.test","action":null,"thought":null,"accessibility_tree":"gold state","screenshot":"screenshots/t1/0.png"},
                {"step":1,"state_index":1,"url":"https://example.test","action":null,"thought":null,"accessibility_tree":"distractor state","screenshot":"screenshots/t1/1.png"}
            ]}).to_string()
                + "\n"
                + &serde_json::json!({"id":"t2","domain":"web","environment":"web","goal":"inspect","outcome":"success","start_url":"https://second.test","states":[
                    {"step":0,"state_index":0,"url":"https://second.test","action":null,"thought":null,"accessibility_tree":"second gold","screenshot":"screenshots/t2/0.png"}
                ]}).to_string()
                + "\n",
        ).unwrap();
        fs::write(
            root.path().join("haystacks/small.json"),
            b"{\"q1\":[\"t1\"],\"q2\":[\"t2\"]}",
        )
        .unwrap();
        let annotations = root.path().join("annotations.json");
        fs::write(
            &annotations,
            serde_json::to_vec(&serde_json::json!({"schema":ANNOTATION_SCHEMA,"fixture_set_id":"loader-regression","dataset_version":"test","haystack_file":"small.json","annotations":[
                {"question_id":"q1","image_dependent":true,"reviewed_by":"reviewer","rationale":"pixels","oracle_lane":"native","evidence":[{"trajectory_id":"t1","state_index":0}]},
                {"question_id":"q2","image_dependent":true,"reviewed_by":"reviewer","rationale":"pixels","oracle_lane":"native","evidence":[{"trajectory_id":"t2","state_index":0}]}
            ]})).unwrap(),
        ).unwrap();
        let loaded = load_longmemeval_v2_image_subset(root.path(), &annotations).unwrap();
        assert_eq!(loaded.corpora.len(), 2);
        assert_eq!(loaded.corpora[0].evidence.len(), 2);
        assert_eq!(loaded.corpora[1].evidence.len(), 1);
        assert_eq!(loaded.cases[0].oracle_evidence.len(), 1);
        assert_ne!(loaded.cases[0].corpus_id, loaded.cases[1].corpus_id);
    }

    #[test]
    fn non_oracle_control_imports_full_corpus_and_strips_question_media() {
        let mut fixture = fixture();
        fixture.cases[0]
            .question
            .media
            .push(fixture.corpora[0].evidence[0].source_media.clone());
        let mut adapter = SpyAdapter::default();
        let result = run_experiment(
            &fixture,
            &plan(ExperimentArm::text_control()),
            &mut adapter,
            &DeterministicScorer,
        )
        .unwrap();
        assert!(adapter.imported_distractor);
        assert!(!adapter.observed_oracle);
        assert!(!adapter.observed_question_media);
        assert!(!result.provenance.leaderboard_eligible);
        let encoded = serde_json::to_vec(&result).unwrap();
        let decoded: MultimodalRunResult = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(decoded, result);
    }

    #[test]
    fn fixture_bytes_are_verified_before_adapter_import() {
        let root = tempfile::tempdir().unwrap();
        fs::create_dir(root.path().join("assets")).unwrap();
        for name in [
            "incident.svg",
            "invoice.pdf",
            "forecast.csv",
            "distractor.txt",
        ] {
            fs::copy(
                Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("fixtures/multimodal/v1/assets")
                    .join(name),
                root.path().join("assets").join(name),
            )
            .unwrap();
        }
        fs::write(root.path().join("assets/incident.svg"), b"tampered").unwrap();
        let mut run_plan = plan(ExperimentArm::text_control());
        run_plan.asset_root = root.path().to_path_buf();
        let error = run_experiment(
            &fixture(),
            &run_plan,
            &mut SpyAdapter::default(),
            &DeterministicScorer,
        )
        .unwrap_err();
        assert!(error.to_string().contains("mismatch"));
    }

    #[test]
    fn digest_locator_cannot_masquerade_as_product_binding() {
        let mut adapter = SpyAdapter {
            corrupt_binding: true,
            ..SpyAdapter::default()
        };
        let error = run_experiment(
            &fixture(),
            &plan(ExperimentArm::text_control()),
            &mut adapter,
            &DeterministicScorer,
        )
        .unwrap_err();
        assert!(error.to_string().contains("binding id"));
    }

    #[test]
    fn corpus_rejects_conflicting_raw_binding_to_blob_mapping() {
        let mut adapter = SpyAdapter {
            conflicting_raw_binding: true,
            ..SpyAdapter::default()
        };
        let error = run_experiment(
            &fixture(),
            &plan(ExperimentArm::text_control()),
            &mut adapter,
            &DeterministicScorer,
        )
        .unwrap_err();
        assert!(error.to_string().contains("conflicting blob metadata"));
    }

    #[test]
    fn corpus_rejects_conflicting_projection_binding_to_blob_mapping() {
        let mut adapter = SpyAdapter {
            conflicting_projection_binding: true,
            ..SpyAdapter::default()
        };
        let error = run_experiment(
            &fixture(),
            &plan(ExperimentArm::text_control()),
            &mut adapter,
            &DeterministicScorer,
        )
        .unwrap_err();
        assert!(error.to_string().contains("conflicting blob metadata"));
    }

    #[cfg(unix)]
    #[test]
    fn fixture_import_rejects_hard_linked_source_descriptor() {
        let root = tempfile::tempdir().unwrap();
        fs::create_dir(root.path().join("assets")).unwrap();
        for name in [
            "incident.svg",
            "invoice.pdf",
            "forecast.csv",
            "distractor.txt",
        ] {
            fs::copy(
                Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("fixtures/multimodal/v1/assets")
                    .join(name),
                root.path().join("assets").join(name),
            )
            .unwrap();
        }
        fs::hard_link(
            root.path().join("assets/incident.svg"),
            root.path().join("assets/incident-alias.svg"),
        )
        .unwrap();
        let error =
            verify_corpus_assets(&fixture().corpora[0], root.path(), 16 * 1024 * 1024).unwrap_err();
        assert!(error.to_string().contains("hard-linked"));
    }

    #[test]
    fn provider_dispatch_cannot_reserve_beyond_budget() {
        let root = tempfile::tempdir().unwrap();
        let mut run_plan = plan(ExperimentArm::text_control());
        run_plan.budget = ExecutionBudget {
            max_step: CostLadderStep::ProviderPilot,
            max_provider_calls: 1,
            max_cost_micro_usd: 10,
        };
        run_plan.spend_ledger_path = Some(root.path().join("spend.jsonl"));
        let mut adapter = SpyAdapter {
            invalid_spend: true,
            ..SpyAdapter::default()
        };
        let error =
            run_experiment(&fixture(), &run_plan, &mut adapter, &DeterministicScorer).unwrap_err();
        assert!(error.to_string().contains("exceeds remaining budget"));
    }

    #[test]
    fn offline_native_rejects_nonzero_budget_before_adapter_work() {
        let mut run_plan = plan(ExperimentArm::text_control());
        run_plan.budget = ExecutionBudget {
            max_step: CostLadderStep::OfflineNative,
            max_provider_calls: 1,
            max_cost_micro_usd: 10,
        };
        let mut adapter = SpyAdapter::default();
        let error =
            run_experiment(&fixture(), &run_plan, &mut adapter, &DeterministicScorer).unwrap_err();
        assert!(error.to_string().contains("categorically require zero"));
        assert!(!adapter.imported_distractor);
        assert!(
            !run_plan
                .run_registry_root
                .join(format!("{}.json", run_plan.run_instance_id))
                .exists()
        );
    }

    #[test]
    fn offline_spend_journal_rejects_dispatch_even_with_numeric_capacity() {
        let mut journal = SpendJournal::new(
            "offline-journal",
            &sha256(b"offline-config"),
            "case",
            None,
            RemainingBudget {
                provider_calls: u64::MAX,
                cost_micro_usd: u64::MAX,
            },
            false,
        );
        let error = journal
            .reserve("forbidden", "provider", "model", 1, "test")
            .unwrap_err();
        assert!(error.to_string().contains("categorically prohibits"));
        assert!(journal.traces.is_empty());
        assert!(journal.open.is_empty());
    }

    #[test]
    fn spend_ledger_rejects_a_second_run_owner() {
        let root = tempfile::tempdir().unwrap();
        let ledger = root.path().join("spend.json");
        let config = sha256(b"shared-ledger-config");
        let budget = RemainingBudget {
            provider_calls: 1,
            cost_micro_usd: 10,
        };
        let mut first = SpendJournal::new(
            "ledger-owner-one",
            &config,
            "case",
            Some(&ledger),
            budget.clone(),
            true,
        );
        first
            .reserve("call", "provider", "model", 10, "test")
            .unwrap();
        let mut second = SpendJournal::new(
            "ledger-owner-two",
            &config,
            "case",
            Some(&ledger),
            budget,
            true,
        );
        let error = second
            .reserve("call", "provider", "model", 10, "test")
            .unwrap_err();
        assert!(error.to_string().contains("owned by another run"));
    }

    #[test]
    fn spend_ledger_lock_prevents_interleaved_transition() {
        let root = tempfile::tempdir().unwrap();
        let ledger = root.path().join("spend.json");
        let mut journal = SpendJournal::new(
            "locked-ledger",
            &sha256(b"locked-ledger-config"),
            "case",
            Some(&ledger),
            RemainingBudget {
                provider_calls: 1,
                cost_micro_usd: 10,
            },
            true,
        );
        let reservation = journal
            .reserve("call", "provider", "model", 10, "test")
            .unwrap();
        let lock = spend_sibling_path(&ledger, ".lock");
        fs::write(&lock, b"other process\n").unwrap();
        let error = journal.finish(reservation, true, 1, 1, 1).unwrap_err();
        assert!(error.to_string().contains("locked by another process"));
        assert!(journal.open.contains_key("call"));
        fs::remove_file(lock).unwrap();
        assert!(journal.close_unfinished_as_failed().unwrap());
    }

    #[test]
    fn spend_operation_replay_is_idempotent_and_rejects_contradictory_terminal() {
        let root = tempfile::tempdir().unwrap();
        let ledger = root.path().join("spend.json");
        let config = sha256(b"replay-ledger-config");
        let budget = RemainingBudget {
            provider_calls: 1,
            cost_micro_usd: 10,
        };
        let mut crashed = SpendJournal::new(
            "replay-ledger",
            &config,
            "case",
            Some(&ledger),
            budget.clone(),
            true,
        );
        crashed
            .reserve("call", "provider", "model", 10, "test")
            .unwrap();
        drop(crashed);

        let mut replay = SpendJournal::new(
            "replay-ledger",
            &config,
            "case",
            Some(&ledger),
            budget.clone(),
            true,
        );
        let reservation = replay
            .reserve("call", "provider", "model", 10, "test")
            .unwrap();
        replay.finish(reservation, true, 2, 1, 3).unwrap();

        let mut contradiction = SpendJournal::new(
            "replay-ledger",
            &config,
            "case",
            Some(&ledger),
            budget,
            true,
        );
        let reservation = contradiction
            .reserve("call", "provider", "model", 10, "test")
            .unwrap();
        let error = contradiction
            .finish(reservation, false, 0, 0, 10)
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("different authoritative terminal")
        );
        assert!(contradiction.close_unfinished_as_failed().unwrap());

        let payload = read_spend_ledger(&ledger).unwrap().unwrap();
        assert_eq!(payload.operations.len(), 1);
        let terminal = payload
            .operations
            .values()
            .next()
            .unwrap()
            .terminal
            .as_ref()
            .unwrap();
        assert_eq!(terminal.status, SpendStatus::Succeeded);
        assert_eq!(terminal.cost_micro_usd, 3);
    }

    #[test]
    fn branch_score_must_map_to_product_retrieval_scores() {
        let mut adapter = SpyAdapter {
            invalid_scores: true,
            ..SpyAdapter::default()
        };
        let error = run_experiment(
            &fixture(),
            &plan(ExperimentArm::text_control()),
            &mut adapter,
            &DeterministicScorer,
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("product branch pointer/truth/score")
        );
    }

    #[test]
    fn failed_provider_attempt_is_fsynced_before_error_returns() {
        let root = tempfile::tempdir().unwrap();
        let ledger = root.path().join("spend.jsonl");
        let mut run_plan = plan(ExperimentArm::text_control());
        run_plan.budget = ExecutionBudget {
            max_step: CostLadderStep::ProviderPilot,
            max_provider_calls: 1,
            max_cost_micro_usd: 10,
        };
        run_plan.spend_ledger_path = Some(ledger.clone());
        let mut adapter = SpyAdapter {
            fail_after_spend: true,
            ..SpyAdapter::default()
        };
        let error =
            run_experiment(&fixture(), &run_plan, &mut adapter, &DeterministicScorer).unwrap_err();
        assert!(error.to_string().contains("provider failed"));
        assert!(adapter.ledger_visible_before_failed_dispatch);
        let payload = read_spend_ledger(&ledger).unwrap().unwrap();
        assert_eq!(payload.operations.len(), 1);
        assert_eq!(payload.run_instance_id, run_plan.run_instance_id);
        assert_eq!(
            payload.effective_config_sha256,
            run_plan.effective_config_sha256
        );
        let operation = payload.operations.values().next().unwrap();
        assert_eq!(
            operation.terminal.as_ref().unwrap().status,
            SpendStatus::Failed
        );
    }

    #[test]
    fn atomic_ledger_reconciles_pre_and_post_rename_failures_without_two_terminals() {
        for (index, failure) in [
            SpendPersistFailurePoint::BeforeRename,
            SpendPersistFailurePoint::AfterRenameBeforeDirectoryFsync,
        ]
        .into_iter()
        .enumerate()
        {
            let root = tempfile::tempdir().unwrap();
            let ledger = root.path().join("spend.jsonl");
            let mut journal = SpendJournal::new(
                &format!("durable-terminal-{index}"),
                &sha256(b"durable-terminal-config"),
                "case",
                Some(&ledger),
                RemainingBudget {
                    provider_calls: 1,
                    cost_micro_usd: 100,
                },
                true,
            );
            let reservation = journal
                .reserve("call", "provider", "model", 100, "test")
                .unwrap();
            journal.injected_persist_failure = Some(failure);
            let error = journal.finish(reservation, true, 10, 5, 7).unwrap_err();
            assert!(error.to_string().contains("injected spend ledger"));
            assert!(journal.open.contains_key("call"));
            assert_eq!(journal.traces.len(), 1);

            assert!(journal.close_unfinished_as_failed().unwrap());
            assert!(journal.open.is_empty());
            let payload = read_spend_ledger(&ledger).unwrap().unwrap();
            assert_eq!(payload.operations.len(), 1);
            let operation = payload.operations.values().next().unwrap();
            let terminal = operation.terminal.as_ref().unwrap();
            match failure {
                SpendPersistFailurePoint::BeforeRename => {
                    assert_eq!(terminal.status, SpendStatus::Failed);
                    assert_eq!(terminal.cost_micro_usd, 100);
                }
                SpendPersistFailurePoint::AfterRenameBeforeDirectoryFsync => {
                    assert_eq!(terminal.status, SpendStatus::Succeeded);
                    assert_eq!(terminal.cost_micro_usd, 7);
                }
            }
            assert_eq!(journal.traces.len(), 2);
        }
    }

    #[test]
    fn duplicate_run_instance_fails_before_second_adapter_work() {
        let run_plan = plan(ExperimentArm::text_control());
        let mut first = SpyAdapter::default();
        run_experiment(&fixture(), &run_plan, &mut first, &DeterministicScorer).unwrap();
        assert!(first.imported_distractor);

        let mut second = SpyAdapter::default();
        let error =
            run_experiment(&fixture(), &run_plan, &mut second, &DeterministicScorer).unwrap_err();
        assert!(error.to_string().contains("already claimed"));
        assert!(!second.imported_distractor);
    }

    #[test]
    fn cell_e_reuses_byte_identical_retrieval_for_both_readers() {
        let mut adapter = HybridAdapter::default();
        let mut text_reader = BoundEchoReader::new(ReaderMode::TextProjection);
        let mut blob_reader = BoundEchoReader::new(ReaderMode::SourceBlob);
        let pair = run_reader_modality_pair(
            &fixture(),
            &plan(ExperimentArm::reader_modality(ReaderMode::TextProjection)),
            &mut adapter,
            &mut text_reader,
            &mut blob_reader,
            &DeterministicScorer,
        )
        .unwrap();
        for (text, blob) in pair.0.cases.iter().zip(&pair.1.cases) {
            assert_eq!(
                retrieval_fingerprint(&text.retrieved).unwrap(),
                retrieval_fingerprint(&blob.retrieved).unwrap()
            );
            assert!(
                text.reader_inputs
                    .iter()
                    .all(|read| read.mode == ReaderMode::TextProjection)
            );
            assert!(
                blob.reader_inputs
                    .iter()
                    .all(|read| read.mode == ReaderMode::SourceBlob)
            );
        }
        assert_eq!(text_reader.calls, fixture().cases.len());
        assert_eq!(blob_reader.calls, fixture().cases.len());
        assert!(
            text_reader
                .observed_input_counts
                .iter()
                .all(|count| *count == 1)
        );
        assert!(
            blob_reader
                .observed_input_counts
                .iter()
                .all(|count| *count == 1)
        );
    }

    #[test]
    fn adapter_cannot_self_attest_reader_work_without_resolver_reads() {
        let mut adapter = HybridAdapter {
            skip_reader: true,
            ..HybridAdapter::default()
        };
        let error = run_experiment(
            &fixture(),
            &plan(ExperimentArm::hybrid()),
            &mut adapter,
            &DeterministicScorer,
        )
        .unwrap_err();
        assert!(error.to_string().contains("reader proof"));
    }

    #[test]
    fn oracle_regions_map_to_typed_product_regions_without_exposing_gold_answers() {
        let mut adapter = HybridAdapter::default();
        let result = run_experiment(
            &fixture(),
            &plan(ExperimentArm::oracle()),
            &mut adapter,
            &DeterministicScorer,
        )
        .unwrap();
        assert!(result.provenance.oracle_gold);
        assert!(!result.provenance.leaderboard_eligible);
        assert!(
            result
                .cases
                .iter()
                .all(|case| case.fired_proof.oracle_region_count > 0)
        );
    }

    #[test]
    fn cell_e_rejects_reader_effective_request_hash_mismatch() {
        let mut adapter = HybridAdapter::default();
        let mut text_reader = BoundEchoReader::new(ReaderMode::TextProjection);
        let mut blob_reader = BoundEchoReader::new(ReaderMode::SourceBlob);
        blob_reader.tamper_request_hash = true;
        let error = run_reader_modality_pair(
            &fixture(),
            &plan(ExperimentArm::reader_modality(ReaderMode::TextProjection)),
            &mut adapter,
            &mut text_reader,
            &mut blob_reader,
            &DeterministicScorer,
        )
        .unwrap_err();
        assert!(error.to_string().contains("effective request hash"));
    }

    #[test]
    fn cell_e_retrieval_cannot_smuggle_cached_projection_text_to_reader() {
        let mut adapter = HybridAdapter {
            return_cached_reader_text_from_retrieval: true,
            ..HybridAdapter::default()
        };
        let mut text_reader = BoundEchoReader::new(ReaderMode::TextProjection);
        let mut blob_reader = BoundEchoReader::new(ReaderMode::SourceBlob);
        let error = run_reader_modality_pair(
            &fixture(),
            &plan(ExperimentArm::reader_modality(ReaderMode::TextProjection)),
            &mut adapter,
            &mut text_reader,
            &mut blob_reader,
            &DeterministicScorer,
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("attempted to return reader material")
        );
        assert_eq!(text_reader.calls, 0);
        assert_eq!(blob_reader.calls, 0);
    }

    #[test]
    fn self_reported_collapse_without_members_is_rejected() {
        let mut adapter = HybridAdapter {
            invalid_collapse: true,
            ..HybridAdapter::default()
        };
        let mut run_plan = plan(ExperimentArm::hybrid());
        run_plan.run_instance_id = "collapse-proof".to_string();
        let error =
            run_experiment(&fixture(), &run_plan, &mut adapter, &DeterministicScorer).unwrap_err();
        assert!(error.to_string().contains("collapse"));
    }
}
