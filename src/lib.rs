use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

pub mod artifacts;
pub mod cohort;
pub mod compare;
pub mod cost;
pub mod eligibility;
pub mod jsonutil;
pub mod leaderboard;
pub mod leaderboard_export;
pub mod live;
pub mod multimodal;
pub mod proto;
pub mod registry;
pub mod runner;
pub mod step_analytics;
pub mod trials;

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct BenchRunMetadata {
    pub capabilities: BenchTraceCapabilities,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub models: Vec<BenchModel>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub trace_artifacts: BTreeMap<String, String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pricing_table_version: Option<String>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct BenchTraceCapabilities {
    pub supported: BenchSupportedCapabilities,
    pub observed: BenchObservedCapabilities,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct BenchSupportedCapabilities {
    pub reset: bool,
    pub durable_state: bool,
    pub ingest: bool,
    pub flush: bool,
    pub retrieve: bool,
    pub answer: bool,
    pub provider_injection: bool,
    pub embedding_injection: bool,
    pub raw_context: bool,
    pub score_explain: bool,
    pub retry_trace: bool,
    pub token_usage: bool,
    pub cache_usage: bool,
    pub cost_usage: bool,
    pub queue_events: bool,
    pub state_export: bool,
    pub native_stage_trace: bool,
    pub wrapped_api_trace: bool,
    pub provider_trace: bool,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct BenchObservedCapabilities {
    pub ingest_input: bool,
    pub ingest_output: bool,
    pub model_calls: bool,
    pub embedding_calls: bool,
    pub retrieval_queries: bool,
    pub retrieval_candidates: bool,
    pub retrieval_scores: bool,
    pub raw_context: bool,
    pub answer_prompt: bool,
    pub answer_output: bool,
    pub errors: bool,
    pub retries: bool,
    pub token_usage: bool,
    pub cache_usage: bool,
    pub timing: bool,
    pub cost: bool,
    pub scoring_verdict: bool,
    pub native_stage_trace: bool,
    pub wrapped_api_trace: bool,
    pub provider_trace: bool,
    pub memory_stage_events: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct BenchModel {
    pub label: String,
    pub operation: String,
    pub operator: String,
    pub model: String,
    pub queue_id: String,
    pub role_binding: String,
    pub max_in_flight: usize,
    pub lease_seconds: u64,
    pub retry_attempts: u32,
    pub logical_retry_attempts: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeout_seconds: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub requests_per_minute: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_units_per_minute: Option<u64>,
    pub response_cache_enabled: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct TokenUsage {
    pub input_tokens: Option<u64>,
    pub cached_input_tokens: Option<u64>,
    pub cache_miss_input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct TokenPricing {
    pub input_token_micro_usd: Option<u64>,
    pub cached_input_token_micro_usd: Option<u64>,
    pub output_token_micro_usd: Option<u64>,
}

impl TokenPricing {
    pub fn estimate_micro_usd(&self, usage: &TokenUsage) -> Option<u64> {
        let input_tokens = usage.input_tokens.unwrap_or(0);
        let cached_input_tokens = usage.cached_input_tokens.unwrap_or(0);
        let uncached_input_tokens = input_tokens.saturating_sub(cached_input_tokens);
        let output_tokens = usage.output_tokens.unwrap_or(0);

        let mut total = 0u64;
        let mut observed_price = false;
        if let Some(price) = self.input_token_micro_usd {
            total = total.saturating_add(uncached_input_tokens.saturating_mul(price));
            observed_price = true;
        }
        if let Some(price) = self.cached_input_token_micro_usd {
            total = total.saturating_add(cached_input_tokens.saturating_mul(price));
            observed_price = true;
        }
        if let Some(price) = self.output_token_micro_usd {
            total = total.saturating_add(output_tokens.saturating_mul(price));
            observed_price = true;
        }
        observed_price.then_some(total)
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BenchEventStatus {
    #[serde(alias = "pending")]
    Queued,
    Running,
    Succeeded,
    Failed,
    Dead,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct BenchQueueEvent {
    pub queue_id: String,
    pub item_id: String,
    #[serde(alias = "kind")]
    pub operation: String,
    pub status: BenchEventStatus,
    pub attempt: usize,
    pub timestamp: DateTime<Utc>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_hash: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<TokenUsage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost_micro_usd: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryTraceEventKind {
    RunStarted,
    RunSucceeded,
    RunFailed,
    OperationStarted,
    OperationSucceeded,
    OperationFailed,
    BranchStarted,
    BranchJoined,
    BatchStarted,
    BatchSucceeded,
    BatchFailed,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryTraceOperation {
    Capture,
    Ingest,
    Distill,
    WriteArchive,
    EmbedRaw,
    EmbedFacts,
    Index,
    Retrieve,
    Answer,
    Score,
    Flush,
    StateExport,
    AdapterCall,
    ModelCall,
    EmbeddingCall,
    VectorSearch,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryInstrumentation {
    NativeStage,
    WrappedApi,
    Provider,
    Imported,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct MemoryTraceEvent {
    pub schema_version: u32,
    pub trace_id: String,
    pub parent_trace_id: Option<String>,
    pub source_system: String,
    pub instrumentation: MemoryInstrumentation,
    pub run_id: String,
    pub question_id: Option<String>,
    pub source_id: Option<String>,
    pub operation: MemoryTraceOperation,
    pub stage: Option<String>,
    pub event: MemoryTraceEventKind,
    pub attempt: u32,
    pub timestamp: DateTime<Utc>,
    pub started_at: Option<DateTime<Utc>>,
    pub finished_at: Option<DateTime<Utc>>,
    pub duration_ms: Option<i64>,
    pub input_hash: Option<String>,
    pub output_hash: Option<String>,
    pub item_count: Option<u64>,
    pub model_trace_ids: Vec<String>,
    pub queue_item_ids: Vec<String>,
    pub metrics: serde_json::Value,
    pub error_class: Option<String>,
    pub error: Option<String>,
}

impl MemoryTraceEvent {
    pub fn new(
        source_system: impl Into<String>,
        run_id: impl Into<String>,
        instrumentation: MemoryInstrumentation,
        operation: MemoryTraceOperation,
        event: MemoryTraceEventKind,
    ) -> Self {
        Self {
            schema_version: 1,
            trace_id: uuid_like_trace_id(),
            parent_trace_id: None,
            source_system: source_system.into(),
            instrumentation,
            run_id: run_id.into(),
            question_id: None,
            source_id: None,
            operation,
            stage: None,
            event,
            attempt: 1,
            timestamp: Utc::now(),
            started_at: None,
            finished_at: None,
            duration_ms: None,
            input_hash: None,
            output_hash: None,
            item_count: None,
            model_trace_ids: Vec::new(),
            queue_item_ids: Vec::new(),
            metrics: serde_json::json!({}),
            error_class: None,
            error: None,
        }
    }
}

pub fn read_memory_trace_jsonl(
    path: impl AsRef<std::path::Path>,
) -> anyhow::Result<Vec<MemoryTraceEvent>> {
    if !path.as_ref().is_file() {
        return Ok(Vec::new());
    }
    let raw = std::fs::read_to_string(path)?;
    raw.lines()
        .enumerate()
        .filter(|(_, line)| !line.trim().is_empty())
        .map(|(idx, line)| {
            serde_json::from_str(line)
                .map_err(|err| anyhow::anyhow!("invalid memory trace on line {}: {err}", idx + 1))
        })
        .collect()
}

pub fn append_memory_trace_jsonl(
    path: impl AsRef<std::path::Path>,
    event: &MemoryTraceEvent,
) -> anyhow::Result<()> {
    use std::io::Write;

    if let Some(parent) = path.as_ref().parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    writeln!(file, "{}", serde_json::to_string(event)?)?;
    file.flush()?;
    Ok(())
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct BenchTimingSummary {
    pub queue_id: String,
    pub item_id: String,
    pub operation: String,
    pub attempts: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wait_ms: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run_ms: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_ms: Option<i64>,
    pub final_status: Option<BenchEventStatus>,
}

pub fn summarize_queue_timing(events: &[BenchQueueEvent]) -> Vec<BenchTimingSummary> {
    let mut grouped: BTreeMap<(&str, &str), Vec<&BenchQueueEvent>> = BTreeMap::new();
    for event in events {
        grouped
            .entry((&event.queue_id, &event.item_id))
            .or_default()
            .push(event);
    }

    grouped
        .into_values()
        .map(|mut group| {
            group.sort_by_key(|event| event.timestamp);
            let first = group.first().expect("group is non-empty");
            let queued_at = group
                .iter()
                .find(|event| event.status == BenchEventStatus::Queued)
                .map(|event| event.timestamp);
            let first_running_at = group
                .iter()
                .find(|event| event.status == BenchEventStatus::Running)
                .map(|event| event.timestamp);
            let terminal = group.iter().rev().find(|event| {
                matches!(
                    event.status,
                    BenchEventStatus::Succeeded | BenchEventStatus::Failed | BenchEventStatus::Dead
                )
            });
            let terminal_at = terminal.map(|event| event.timestamp);
            let wait_ms = duration_ms(queued_at, first_running_at);
            let run_ms = duration_ms(first_running_at, terminal_at);
            let total_ms = duration_ms(queued_at, terminal_at);

            BenchTimingSummary {
                queue_id: first.queue_id.clone(),
                item_id: first.item_id.clone(),
                operation: first.operation.clone(),
                attempts: group
                    .iter()
                    .map(|event| event.attempt)
                    .max()
                    .unwrap_or_default(),
                wait_ms,
                run_ms,
                total_ms,
                final_status: terminal.map(|event| event.status),
            }
        })
        .collect()
}

fn duration_ms(start: Option<DateTime<Utc>>, end: Option<DateTime<Utc>>) -> Option<i64> {
    Some(end?.signed_duration_since(start?).num_milliseconds())
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct MemoryQuestionTrace {
    pub schema_version: u32,
    pub question_id: String,
    pub question_hash: String,
    pub source_system: String,
    pub run_metadata: BenchRunMetadata,
    pub payload: serde_json::Value,
}

impl MemoryQuestionTrace {
    pub fn new(
        question_id: impl Into<String>,
        source_system: impl Into<String>,
        run_metadata: BenchRunMetadata,
        payload: serde_json::Value,
    ) -> Self {
        let question_id = question_id.into();
        let question_hash = stable_hash(question_id.as_bytes());
        Self {
            schema_version: 1,
            question_id,
            question_hash,
            source_system: source_system.into(),
            run_metadata,
            payload,
        }
    }
}

pub fn stable_hash(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn uuid_like_trace_id() -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let now = Utc::now();
    let counter = COUNTER.fetch_add(1, Ordering::Relaxed);
    stable_hash(
        format!(
            "{}:{}:{counter}",
            now.timestamp_nanos_opt().unwrap_or_default(),
            now
        )
        .as_bytes(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cost_is_derived_from_tokens_and_pricing() {
        let usage = TokenUsage {
            input_tokens: Some(10),
            cached_input_tokens: Some(4),
            cache_miss_input_tokens: Some(6),
            output_tokens: Some(2),
        };
        let pricing = TokenPricing {
            input_token_micro_usd: Some(2),
            cached_input_token_micro_usd: Some(1),
            output_token_micro_usd: Some(5),
        };

        assert_eq!(pricing.estimate_micro_usd(&usage), Some(26));
    }

    #[test]
    fn memory_trace_jsonl_round_trips_generic_event() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("memory-traces.jsonl");
        let mut event = MemoryTraceEvent::new(
            "mem0",
            "run-a",
            MemoryInstrumentation::WrappedApi,
            MemoryTraceOperation::AdapterCall,
            MemoryTraceEventKind::OperationSucceeded,
        );
        event.question_id = Some("q1".to_string());
        event.metrics = serde_json::json!({"retrieved": 5});

        append_memory_trace_jsonl(&path, &event).unwrap();
        let records = read_memory_trace_jsonl(&path).unwrap();

        assert_eq!(records.len(), 1);
        assert_eq!(records[0].source_system, "mem0");
        assert_eq!(
            records[0].instrumentation,
            MemoryInstrumentation::WrappedApi
        );
        assert_eq!(records[0].metrics["retrieved"], serde_json::json!(5));
    }

    #[test]
    fn queue_timing_is_derived_from_event_timestamps() {
        let base = DateTime::parse_from_rfc3339("2026-06-16T00:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let event = |status, offset_ms| BenchQueueEvent {
            queue_id: "chat:model".to_string(),
            item_id: "req-1".to_string(),
            operation: "chat".to_string(),
            status,
            attempt: if status == BenchEventStatus::Queued {
                0
            } else {
                1
            },
            timestamp: base + chrono::Duration::milliseconds(offset_ms),
            model: Some("model".to_string()),
            input_hash: Some("abc".to_string()),
            usage: None,
            cost_micro_usd: None,
            error: None,
        };
        let summary = summarize_queue_timing(&[
            event(BenchEventStatus::Queued, 0),
            event(BenchEventStatus::Running, 25),
            event(BenchEventStatus::Succeeded, 125),
        ]);

        assert_eq!(summary.len(), 1);
        assert_eq!(summary[0].wait_ms, Some(25));
        assert_eq!(summary[0].run_ms, Some(100));
        assert_eq!(summary[0].total_ms, Some(125));
        assert_eq!(summary[0].final_status, Some(BenchEventStatus::Succeeded));
    }

    #[test]
    fn queue_event_accepts_foundation_trace_schema() {
        let event: BenchQueueEvent = serde_json::from_value(serde_json::json!({
            "trace_id": "trace-1",
            "item_id": "req-1",
            "queue_id": "embedding:gemini:gemini-embedding-2",
            "kind": "embedding",
            "status": "pending",
            "attempt": 0,
            "timestamp": "2026-06-16T00:00:00Z",
            "metadata": {}
        }))
        .unwrap();

        assert_eq!(event.operation, "embedding");
        assert_eq!(event.status, BenchEventStatus::Queued);
    }
}
