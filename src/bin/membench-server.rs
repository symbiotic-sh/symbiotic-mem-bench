//! Dashboard backend for the membench benchmark explorer.
//!
//! A small axum service over the same `runs/` and `records/` files the CLI
//! reads. It exposes the normalized index, run detail, the merged question
//! browser, artifact paging, run comparison, the leaderboard, and a
//! command-preview runner. Static SPA assets are served from `--dist`.
//!
//! Run ids are repo-relative run-folder paths (which contain slashes), so
//! run-scoped endpoints take the id as an `?id=` query parameter rather than a
//! path segment.

use axum::{
    Json, Router,
    extract::{Query, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
};
use chrono::{DateTime, Utc};
use clap::Parser;
use membench::{
    BenchQueueEvent, artifacts, compare, cost, leaderboard, live, registry, runner,
    summarize_queue_timing,
};
use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::{BTreeMap, HashMap};
use std::net::SocketAddr;
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::SystemTime;
use tower_http::cors::CorsLayer;
use tower_http::services::{ServeDir, ServeFile};

#[derive(Parser)]
#[command(name = "membench-server")]
#[command(about = "Dashboard backend for the membench benchmark explorer")]
struct Cli {
    /// Port to bind.
    #[arg(long, default_value_t = 8787)]
    port: u16,
    /// Repository root; run ids are made relative to it. Defaults to the
    /// extracted release-bundle root, then to this source checkout.
    #[arg(long)]
    repo_root: Option<PathBuf>,
    /// Registry roots to scan. Defaults to `<repo>/runs` and `<repo>/records`.
    #[arg(long)]
    root: Vec<PathBuf>,
    /// Built SPA directory to serve.
    #[arg(long)]
    dist: Option<PathBuf>,
}

struct AppState {
    repo_root: PathBuf,
    roots: Vec<PathBuf>,
    /// Mtime-keyed cache of the registry index, trial markers, and derived
    /// summaries. Refreshed only when a `benchmark-report.json` path appears,
    /// disappears, or changes mtime (or a `trials.jsonl` does). The dashboard
    /// polls frequently; without this every list/leaderboard/detail request
    /// re-walked `runs/` + `records/` and re-parsed every report.
    registry_cache: RwLock<Option<Arc<RegistrySnapshot>>>,
    /// Short-TTL cache for `live::live_detail`, keyed on the mtimes of the
    /// trace files it reads. Live polling (2s) is faster than the files grow
    /// meaningfully, so a brief TTL collapses repeated multi-MB parses.
    live_cache: RwLock<HashMap<String, LiveCacheEntry>>,
}

/// Cached projection of the whole registry.
struct RegistrySnapshot {
    records: Vec<registry::RunRecord>,
    summaries: Vec<registry::RunSummary>,
    summary_by_id: HashMap<String, registry::RunSummary>,
    by_id: HashMap<String, registry::RunRecord>,
    /// Sorted `(path, mtime)` fingerprints used to invalidate this snapshot.
    report_sig: Vec<(PathBuf, Option<SystemTime>)>,
    trial_sig: Vec<(PathBuf, Option<SystemTime>)>,
}

/// One cached `live_detail` result.
struct LiveCacheEntry {
    detail: Arc<live::LiveDetail>,
    built_at: SystemTime,
    /// Trace-file mtimes at build time; a change invalidates immediately.
    signature: Vec<(PathBuf, Option<SystemTime>)>,
}

/// Live detail TTL: serve cached within this window (subject to mtime change).
const LIVE_CACHE_TTL: std::time::Duration = std::time::Duration::from_millis(2500);

impl AppState {
    /// Sorted `(path, mtime)` fingerprint of every `benchmark-report.json`
    /// beneath the registry roots. Cheap (pruned walk + one `stat` per report),
    /// and changes the moment a run finalizes or a report is rewritten.
    fn report_signature(&self) -> Vec<(PathBuf, Option<SystemTime>)> {
        let mut paths = Vec::new();
        for root in &self.roots {
            registry::collect_report_paths(root, &mut paths);
        }
        paths.sort();
        paths
            .into_iter()
            .map(|path| {
                let mtime = std::fs::metadata(&path)
                    .and_then(|meta| meta.modified())
                    .ok();
                (path, mtime)
            })
            .collect()
    }

    /// Sorted `(path, mtime)` fingerprint of every `trials.jsonl` ledger.
    fn trial_signature(&self) -> Vec<(PathBuf, Option<SystemTime>)> {
        let mut paths = Vec::new();
        registry::collect_trial_paths(&self.repo_root.join("runs").join("analysis"), &mut paths);
        paths.sort();
        paths
            .into_iter()
            .map(|path| {
                let mtime = std::fs::metadata(&path)
                    .and_then(|meta| meta.modified())
                    .ok();
                (path, mtime)
            })
            .collect()
    }

    /// Return the current registry snapshot, refreshing it if the on-disk
    /// fingerprints have changed. Does blocking I/O — call from
    /// `spawn_blocking`. Uses double-checked locking so concurrent requests
    /// only rebuild once.
    fn registry_snapshot(&self) -> Arc<RegistrySnapshot> {
        let report_sig = self.report_signature();
        let trial_sig = self.trial_signature();
        {
            let cache = self.registry_cache.read().unwrap();
            if let Some(snapshot) = cache.as_ref()
                && snapshot.report_sig == report_sig
                && snapshot.trial_sig == trial_sig
            {
                return Arc::clone(snapshot);
            }
        }
        let mut cache = self.registry_cache.write().unwrap();
        if let Some(snapshot) = cache.as_ref()
            && snapshot.report_sig == report_sig
            && snapshot.trial_sig == trial_sig
        {
            return Arc::clone(snapshot);
        }
        let records = registry::scan_registry(&self.roots, &self.repo_root);
        let trial_index = registry::scan_trial_markers(&self.repo_root);
        let summaries: Vec<registry::RunSummary> = records
            .iter()
            .map(|record| registry::summarize_with_trials(record, &trial_index))
            .collect();
        let summary_by_id = summaries
            .iter()
            .map(|summary| (summary.run_id.clone(), summary.clone()))
            .collect();
        let by_id = records
            .iter()
            .map(|record| (record.run_id.clone(), record.clone()))
            .collect();
        let snapshot = Arc::new(RegistrySnapshot {
            records,
            summaries,
            summary_by_id,
            by_id,
            report_sig,
            trial_sig,
        });
        *cache = Some(Arc::clone(&snapshot));
        snapshot
    }

    /// Look up a single run record by id (one clone, not a full scan).
    fn find(&self, run_id: &str) -> Option<registry::RunRecord> {
        self.registry_snapshot().by_id.get(run_id).cloned()
    }

    fn pending(&self) -> Vec<registry::PendingRun> {
        registry::scan_pending(&self.roots, &self.repo_root)
    }

    /// Cached `live_detail`. Keyed on the mtimes of the trace files; expires
    /// after `LIVE_CACHE_TTL` even if mtimes are unchanged (so a growing file
    /// whose mtime second rounds the same still refreshes periodically).
    fn live_detail(&self, run_id: &str) -> Option<Arc<live::LiveDetail>> {
        let run_root = self.repo_root.join(run_id);
        if !run_root.is_dir() {
            return None;
        }
        let signature = live_detail_signature(&run_root);
        let now = SystemTime::now();
        {
            let cache = self.live_cache.read().unwrap();
            if let Some(entry) = cache.get(run_id)
                && entry.signature == signature
                && now
                    .duration_since(entry.built_at)
                    .map(|d| d < LIVE_CACHE_TTL)
                    .unwrap_or(false)
            {
                return Some(Arc::clone(&entry.detail));
            }
        }
        let detail = Arc::new(live::live_detail(&run_root));
        let mut cache = self.live_cache.write().unwrap();
        cache.insert(
            run_id.to_string(),
            LiveCacheEntry {
                detail: Arc::clone(&detail),
                built_at: now,
                signature,
            },
        );
        Some(detail)
    }
}

/// Fingerprint the trace files `live_detail` reads so a cache entry invalidates
/// the moment any of them is touched.
fn live_detail_signature(run_root: &Path) -> Vec<(PathBuf, Option<SystemTime>)> {
    let mut paths: Vec<PathBuf> = vec![
        run_root.join("provider-queue/model-queue-traces.jsonl"),
        run_root.join("artifacts/model-traces.jsonl"),
        run_root.join("raw/model-traces.jsonl"),
        run_root.join("artifacts/memory-traces.jsonl"),
        run_root.join("raw/memory-traces.jsonl"),
        run_root.join("traces/memory-events.jsonl"),
        run_root.join("run-params.json"),
    ];
    paths.sort();
    paths
        .into_iter()
        .map(|path| {
            let mtime = std::fs::metadata(&path)
                .and_then(|meta| meta.modified())
                .ok();
            (path, mtime)
        })
        .collect()
}

type Shared = Arc<AppState>;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let repo_root = resolve_repo_root(
        cli.repo_root.clone(),
        std::env::current_exe().ok().as_deref(),
    );
    let roots = if cli.root.is_empty() {
        vec![repo_root.join("runs"), repo_root.join("records")]
    } else {
        cli.root.clone()
    };
    let dist = cli.dist.unwrap_or_else(|| repo_root.join("dashboard/dist"));

    let state: Shared = Arc::new(AppState {
        repo_root: repo_root.clone(),
        roots,
        registry_cache: RwLock::new(None),
        live_cache: RwLock::new(HashMap::new()),
    });

    let api = Router::new()
        .route("/health", get(health))
        .route("/runs", get(list_runs))
        .route("/pending", get(pending_handler))
        .route("/leaderboard", get(leaderboard_handler))
        .route("/run", get(run_detail))
        .route("/run/live", get(run_live))
        .route("/run/questions", get(run_questions))
        .route("/run/question-debug", get(run_question_debug))
        .route("/run/artifact", get(run_artifact))
        .route("/run/traces", get(run_traces))
        .route("/compare", get(compare_handler))
        .route("/runner/schema", get(runner_schema))
        .route("/runner/plan", post(runner_plan))
        .with_state(state.clone());

    let mut app = Router::new()
        .nest("/api", api)
        .layer(CorsLayer::permissive());

    // Serve the built SPA when present; index.html backs unknown asset paths.
    if dist.is_dir() {
        let index = dist.join("index.html");
        app = app.fallback_service(ServeDir::new(&dist).fallback(ServeFile::new(index)));
        eprintln!("serving SPA from {}", dist.display());
    } else {
        eprintln!(
            "note: SPA dir {} not found; serving API only (run `npm run build` in dashboard/)",
            dist.display()
        );
    }

    let addr = SocketAddr::from(([127, 0, 0, 1], cli.port));
    let listener = tokio::net::TcpListener::bind(addr).await?;
    eprintln!(
        "membench-server v{} (git {}, bin {}) listening on http://{addr}",
        env!("CARGO_PKG_VERSION"),
        option_env!("GIT_SHA").unwrap_or("unknown"),
        BINARY_SHA.as_str(),
    );
    eprintln!("registry roots: {:?}", state.roots);
    axum::serve(listener, app).await?;
    Ok(())
}

fn resolve_repo_root(explicit: Option<PathBuf>, executable: Option<&Path>) -> PathBuf {
    if let Some(root) = explicit {
        return root;
    }
    if let Some(root) = executable.and_then(Path::parent)
        && root.join("PROVENANCE.json").is_file()
        && root.join("dashboard/dist/index.html").is_file()
        && root.join("records").is_dir()
    {
        return root.to_path_buf();
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn err(status: StatusCode, message: impl Into<String>) -> (StatusCode, Json<Value>) {
    (status, Json(json!({ "error": message.into() })))
}

async fn health() -> impl IntoResponse {
    Json(json!({
        "ok": true,
        "service": "membench-server",
        "version": env!("CARGO_PKG_VERSION"),
        "git_sha": option_env!("GIT_SHA").unwrap_or("unknown"),
        "binary_sha": BINARY_SHA.as_str(),
    }))
}

/// Short content hash of this server's own executable, computed once at startup.
/// Changes on every rebuild that changes the binary — no commit or version bump
/// required — so it identifies the exact binary the dashboard is talking to.
fn compute_binary_sha() -> String {
    let Some(path) = std::env::current_exe().ok() else {
        return "unknown".to_string();
    };
    let Ok(bytes) = std::fs::read(&path) else {
        return "unknown".to_string();
    };
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    let digest = hasher.finalize();
    digest
        .iter()
        .take(6)
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

static BINARY_SHA: std::sync::LazyLock<String> = std::sync::LazyLock::new(compute_binary_sha);

#[derive(Deserialize)]
struct RunsQuery {
    system: Option<String>,
    benchmark: Option<String>,
    origin: Option<String>,
}

async fn list_runs(
    State(state): State<Shared>,
    Query(query): Query<RunsQuery>,
) -> impl IntoResponse {
    let summaries = tokio::task::spawn_blocking(move || {
        let snapshot = state.registry_snapshot();
        snapshot
            .summaries
            .iter()
            .filter(|summary| {
                query
                    .system
                    .as_ref()
                    .is_none_or(|system| &summary.system == system)
                    && query
                        .benchmark
                        .as_ref()
                        .is_none_or(|benchmark| &summary.benchmark == benchmark)
                    && query
                        .origin
                        .as_ref()
                        .is_none_or(|origin| &summary.origin == origin)
            })
            .cloned()
            .collect::<Vec<_>>()
    })
    .await
    .unwrap_or_default();
    Json(json!({ "runs": summaries }))
}

async fn pending_handler(State(state): State<Shared>) -> impl IntoResponse {
    let pending = tokio::task::spawn_blocking(move || state.pending())
        .await
        .unwrap_or_default();
    Json(json!({ "pending": pending }))
}

async fn run_live(
    State(state): State<Shared>,
    Query(query): Query<IdQuery>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let id = query.id.clone();
    let out = tokio::task::spawn_blocking(move || {
        let pending = if let Some(pending) =
            state.pending().into_iter().find(|run| run.run_id == id)
        {
            pending
        } else {
            let snapshot = state.registry_snapshot();
            let record = snapshot
                .by_id
                .get(&id)
                .cloned()
                .ok_or_else(|| err(StatusCode::NOT_FOUND, "run not found"))?;
            let summary = snapshot
                .summary_by_id
                .get(&id)
                .cloned()
                .ok_or_else(|| err(StatusCode::NOT_FOUND, "run not found"))?;
            registry::PendingRun {
                run_id: summary.run_id.clone(),
                origin: summary.origin,
                system: summary.system,
                benchmark: summary.benchmark,
                limit: summary.limit,
                run_name: summary.run_name,
                config_label: summary.config_label,
                settings_label: summary.settings_label,
                status: "complete".to_string(),
                started_ms: None,
                updated_ms: record.modified_ms,
                age_secs: None,
                hypotheses: registry::count_nonempty_lines_opt(
                    &record.run_root.join("artifacts/hypotheses.jsonl"),
                )
                .or_else(|| {
                    registry::count_nonempty_lines_opt(
                        &record.run_root.join("raw/hypotheses.jsonl"),
                    )
                })
                .unwrap_or_default(),
                ingested: count_dir_entries(&record.run_root.join("vaults")).unwrap_or_default(),
                oracle_gold: summary.oracle_gold,
            }
        };
        let detail = state
            .live_detail(&pending.run_id)
            .ok_or_else(|| err(StatusCode::NOT_FOUND, "run not found"))?;
        Ok::<_, (StatusCode, Json<Value>)>((pending, detail))
    })
    .await
    .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))??;
    let (pending, detail) = out;
    Ok(Json(json!({ "pending": pending, "detail": *detail })))
}

fn count_dir_entries(path: &Path) -> Option<u64> {
    Some(std::fs::read_dir(path).ok()?.flatten().count() as u64)
}

#[derive(Deserialize)]
struct LeaderboardQuery {
    benchmark: Option<String>,
    limit: Option<u64>,
}

/// Live counterpart of the static `membench.leaderboard.v1` export. Both call
/// `leaderboard::build_view`, so a record ranked here is ranked there — the
/// live board cannot be more permissive than the published one.
async fn leaderboard_handler(
    State(state): State<Shared>,
    Query(query): Query<LeaderboardQuery>,
) -> impl IntoResponse {
    let view = tokio::task::spawn_blocking(move || {
        let snapshot = state.registry_snapshot();
        let mut view = leaderboard::build_view(snapshot.summaries.clone());
        if let Some(benchmark) = &query.benchmark {
            view.cohorts.retain(|cohort| &cohort.benchmark == benchmark);
            view.unranked.retain(|row| &row.benchmark == benchmark);
        }
        if let Some(limit) = query.limit {
            view.cohorts.retain(|cohort| cohort.limit == Some(limit));
            view.unranked.retain(|row| row.limit == Some(limit));
        }
        view
    })
    .await
    .map(|view| json!({ "cohorts": view.cohorts, "unranked": view.unranked }))
    .unwrap_or_else(|_| json!({ "cohorts": [], "unranked": [] }));
    Json(view)
}

#[derive(Deserialize)]
struct IdQuery {
    id: String,
}

async fn run_detail(
    State(state): State<Shared>,
    Query(query): Query<IdQuery>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let id = query.id.clone();
    let payload = tokio::task::spawn_blocking(move || {
        let snapshot = state.registry_snapshot();
        let record = snapshot
            .by_id
            .get(&id)
            .cloned()
            .ok_or_else(|| err(StatusCode::NOT_FOUND, "run not found"))?;
        let summary = snapshot
            .summary_by_id
            .get(&id)
            .cloned()
            .ok_or_else(|| err(StatusCode::NOT_FOUND, "run not found"))?;
        // Roll up model traces once and reuse it for both the cohort fields and
        // the top-level cost payload (previously parsed twice per request).
        let cost_rollup = cost::rollup_model_traces(&record.run_root);
        let cohort_fields = registry::compute_cohort_fields_with_rollup(
            &record.run_root,
            &record.params,
            cost_rollup.as_ref(),
        );
        Ok::<_, (StatusCode, Json<Value>)>(json!({
            "summary": summary,
            "report": record.report,
            "params": record.params,
            "cohort": cohort_fields,
            "cost": cost_rollup,
        }))
    })
    .await
    .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))??;
    Ok(Json(payload))
}

async fn run_questions(
    State(state): State<Shared>,
    Query(query): Query<IdQuery>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let id = query.id.clone();
    let rows = tokio::task::spawn_blocking(move || {
        let record = state
            .find(&id)
            .ok_or_else(|| err(StatusCode::NOT_FOUND, "run not found"))?;
        Ok::<_, (StatusCode, Json<Value>)>(artifacts::question_rows(&record.run_root))
    })
    .await
    .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))??;
    Ok(Json(json!({ "total": rows.len(), "questions": rows })))
}

#[derive(Deserialize)]
struct QuestionDebugQuery {
    id: String,
    path: String,
}

async fn run_question_debug(
    State(state): State<Shared>,
    Query(query): Query<QuestionDebugQuery>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let id = query.id.clone();
    let artifact = query.path.clone();
    let payload = tokio::task::spawn_blocking(move || {
        let record = state
            .find(&id)
            .ok_or_else(|| err(StatusCode::NOT_FOUND, "run not found"))?;
        let debug_path = question_debug_path(&record.run_root, &artifact)?;
        let raw = std::fs::read_to_string(&debug_path)
            .map_err(|_| err(StatusCode::NOT_FOUND, "question debug not present"))?;
        let value: Value = serde_json::from_str(&raw)
            .map_err(|error| err(StatusCode::INTERNAL_SERVER_ERROR, error.to_string()))?;
        Ok::<_, (StatusCode, Json<Value>)>(json!({ "path": artifact, "json": value }))
    })
    .await
    .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))??;
    Ok(Json(payload))
}

fn question_debug_path(
    run_root: &Path,
    artifact: &str,
) -> Result<PathBuf, (StatusCode, Json<Value>)> {
    let relative = Path::new(artifact);
    if relative.is_absolute()
        || relative
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(err(
            StatusCode::BAD_REQUEST,
            "question debug path must be run-relative",
        ));
    }
    if !relative.ends_with("question-debug.json") {
        return Err(err(
            StatusCode::BAD_REQUEST,
            "question debug path must point to question-debug.json",
        ));
    }

    let candidate = run_root.join(relative);
    let root = run_root
        .canonicalize()
        .map_err(|_| err(StatusCode::NOT_FOUND, "run root not found"))?;
    let candidate = candidate
        .canonicalize()
        .map_err(|_| err(StatusCode::NOT_FOUND, "question debug not present"))?;
    if !candidate.starts_with(root) {
        return Err(err(
            StatusCode::BAD_REQUEST,
            "question debug path escaped run root",
        ));
    }
    Ok(candidate)
}

#[derive(Deserialize)]
struct ArtifactQuery {
    id: String,
    kind: String,
    offset: Option<usize>,
    limit: Option<usize>,
}

/// Map an artifact kind to its file name and whether it is line-delimited.
fn artifact_file(kind: &str) -> Option<(&'static str, bool)> {
    Some(match kind {
        "hypotheses" => ("hypotheses.jsonl", true),
        "verdicts" => ("verdicts.jsonl", true),
        "partial_verdicts" => ("partial-verdicts.jsonl", true),
        "provenance" => ("provenance.jsonl", true),
        "memory_traces" => ("memory-traces.jsonl", true),
        "model_traces" => ("model-traces.jsonl", true),
        "step_analytics" => ("step-analytics.json", false),
        "scored" => ("scored.json", false),
        "score_summary" => ("score-summary.json", false),
        "gold_eval" => ("gold-eval.json", false),
        _ => return None,
    })
}

async fn run_artifact(
    State(state): State<Shared>,
    Query(query): Query<ArtifactQuery>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let id = query.id.clone();
    let payload = tokio::task::spawn_blocking(move || {
        let record = state
            .find(&id)
            .ok_or_else(|| err(StatusCode::NOT_FOUND, "run not found"))?;
        let (file, is_jsonl) = artifact_file(&query.kind)
            .ok_or_else(|| err(StatusCode::BAD_REQUEST, "unknown artifact kind"))?;
        let path = record.run_root.join("artifacts").join(file);

        if !is_jsonl {
            let raw = std::fs::read_to_string(&path)
                .map_err(|_| err(StatusCode::NOT_FOUND, "artifact not present"))?;
            let value: Value = serde_json::from_str(&raw)
                .map_err(|error| err(StatusCode::INTERNAL_SERVER_ERROR, error.to_string()))?;
            return Ok::<_, (StatusCode, Json<Value>)>(json!({
                "kind": query.kind,
                "json": value,
            }));
        }

        let offset = query.offset.unwrap_or(0);
        let limit = query.limit.unwrap_or(200).min(2000);
        let (total, rows) = read_jsonl_values(&path, offset, limit);
        Ok(json!({
            "kind": query.kind,
            "total": total,
            "offset": offset,
            "limit": limit,
            "rows": rows,
        }))
    })
    .await
    .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))??;
    Ok(Json(payload))
}

fn read_jsonl_values(path: &Path, offset: usize, limit: usize) -> (usize, Vec<Value>) {
    let Ok(raw) = std::fs::read_to_string(path) else {
        return (0, Vec::new());
    };
    // Iterate lazily so we never materialize a `Vec<&str>` of every line just
    // to slice a page out of it (the JSONL files can be hundreds of thousands
    // of lines). `total` counts non-empty lines to match historical semantics.
    let mut total = 0usize;
    let mut rows = Vec::new();
    let mut in_window = 0usize;
    for line in raw.lines() {
        if line.trim().is_empty() {
            continue;
        }
        if total < offset {
            total += 1;
            continue;
        }
        total += 1;
        in_window += 1;
        if in_window <= limit
            && let Ok(value) = serde_json::from_str::<Value>(line)
        {
            rows.push(value);
        }
    }
    (total, rows)
}

/// Cap on memory-trace rows returned at once.
const TRACE_ROW_CAP: usize = 4000;
const TRACE_EVENT_CAP: usize = 12000;

async fn run_traces(
    State(state): State<Shared>,
    Query(query): Query<IdQuery>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let id = query.id.clone();
    let payload = tokio::task::spawn_blocking(move || {
        let record = state
            .find(&id)
            .ok_or_else(|| err(StatusCode::NOT_FOUND, "run not found"))?;
        let root = &record.run_root;

        let (memory_total, memory_rows) = read_jsonl_values(
            &root.join("artifacts").join("memory-traces.jsonl"),
            0,
            TRACE_ROW_CAP,
        );
        let memory_stage_timing = summarize_memory_stage_timing(&memory_rows);
        let model_rollup = cost::rollup_model_traces(root);

        // Provider/model queue timing, when provider-backed runs emit queue
        // JSONL. Computed once and reused for both the `queue_timing` payload
        // and the unified trace-event rows (previously parsed twice).
        let queue_events = read_queue_events(root).unwrap_or_default();
        let queue_timing_rows = summarize_queue_timing(&queue_events);
        let queue_timing = if queue_events.is_empty() {
            None
        } else {
            serde_json::to_value(&queue_timing_rows).ok()
        };
        let trace_waterfall = summarize_trace_waterfall(&memory_rows, &queue_events);
        let dependency_waterfall = summarize_dependency_waterfall(&memory_rows);
        let trace_events = summarize_trace_events(&memory_rows, &queue_events, &queue_timing_rows);
        let workflow_queue = read_workflow_queue(root);

        Ok::<_, (StatusCode, Json<Value>)>(json!({
            "memory_traces": {
                "total": memory_total,
                "truncated": memory_total > memory_rows.len(),
                "rows": memory_rows,
            },
            "memory_stage_timing": memory_stage_timing,
            "model_rollup": model_rollup,
            "queue_timing": queue_timing,
            "trace_waterfall": trace_waterfall,
            "dependency_waterfall": dependency_waterfall,
            "trace_events": trace_events,
            "workflow_queue": workflow_queue,
        }))
    })
    .await
    .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))??;
    Ok(Json(payload))
}

fn read_queue_events(run_root: &Path) -> Option<Vec<BenchQueueEvent>> {
    let path = run_root
        .join("provider-queue")
        .join("model-queue-traces.jsonl");
    let raw = std::fs::read_to_string(&path).ok()?;
    let mut events = Vec::new();
    for line in raw.lines() {
        if line.trim().is_empty() {
            continue;
        }
        if let Ok(event) = serde_json::from_str::<BenchQueueEvent>(line) {
            events.push(event);
        }
    }
    if events.is_empty() {
        return None;
    }
    Some(events)
}

fn summarize_trace_events(
    memory_rows: &[Value],
    queue_events: &[BenchQueueEvent],
    queue_timing: &[membench::BenchTimingSummary],
) -> Value {
    let queue_timing: HashMap<(String, String), &membench::BenchTimingSummary> = queue_timing
        .iter()
        .map(|row| ((row.queue_id.clone(), row.item_id.clone()), row))
        .collect();

    let mut rows = Vec::new();
    for trace in memory_rows {
        let Some(timestamp) = trace.get("timestamp").and_then(Value::as_str) else {
            continue;
        };
        let Some(operation) = memory_operation_for_trace(trace) else {
            continue;
        };
        let event = trace
            .get("event")
            .and_then(Value::as_str)
            .unwrap_or("?")
            .to_string();
        let (item_count, item_unit) = memory_item_count_for_trace(trace).unwrap_or((0, "items"));
        rows.push(json!({
            "timestamp": timestamp,
            "kind": "memory",
            "operation": operation,
            "lane": trace.get("stage").and_then(Value::as_str).unwrap_or("memory"),
            "event": event,
            "status": trace_status(&event),
            "attempt": trace.get("attempt").and_then(Value::as_u64).unwrap_or(0),
            "duration_ms": trace.get("duration_ms").and_then(Value::as_i64),
            "wait_ms": Value::Null,
            "run_ms": Value::Null,
            "total_ms": Value::Null,
            "item_count": item_count,
            "item_unit": item_unit,
            "source": trace_source(trace),
            "error": trace.get("error").and_then(Value::as_str),
        }));
    }

    for event in queue_events {
        let timing = queue_timing.get(&(event.queue_id.clone(), event.item_id.clone()));
        rows.push(json!({
            "timestamp": event.timestamp.to_rfc3339(),
            "kind": "provider",
            "operation": event.operation,
            "lane": event.queue_id,
            "event": queue_status_name(event.status),
            "status": queue_status_name(event.status),
            "attempt": event.attempt,
            "duration_ms": Value::Null,
            "wait_ms": timing.and_then(|row| row.wait_ms),
            "run_ms": timing.and_then(|row| row.run_ms),
            "total_ms": timing.and_then(|row| row.total_ms),
            "item_count": 0,
            "item_unit": "items",
            "source": event.item_id,
            "error": event.error,
        }));
    }

    rows.sort_by(|a, b| {
        let a_ts = a.get("timestamp").and_then(Value::as_str).unwrap_or("");
        let b_ts = b.get("timestamp").and_then(Value::as_str).unwrap_or("");
        a_ts.cmp(b_ts)
    });
    let total = rows.len();
    let truncated = total > TRACE_EVENT_CAP;
    if truncated {
        rows.truncate(TRACE_EVENT_CAP);
    }
    json!({
        "total": total,
        "truncated": truncated,
        "rows": rows,
    })
}

fn trace_status(event: &str) -> &'static str {
    if event.ends_with("failed") {
        "failed"
    } else if event.ends_with("succeeded") || event == "branch_joined" {
        "succeeded"
    } else {
        "running"
    }
}

#[derive(Clone)]
struct WaterfallBlock {
    lane: String,
    lane_kind: &'static str,
    lane_order: usize,
    kind: &'static str,
    start_at: DateTime<Utc>,
    end_at: DateTime<Utc>,
    label: String,
    status: String,
    source: String,
    item_count: u64,
    item_unit: String,
}

fn summarize_trace_waterfall(memory_rows: &[Value], queue_events: &[BenchQueueEvent]) -> Value {
    const WATERFALL_BLOCK_CAP: usize = 8000;

    let mut blocks = Vec::new();
    blocks.extend(memory_waterfall_blocks(memory_rows));
    blocks.extend(provider_waterfall_blocks(queue_events));
    blocks.retain(|block| block.end_at >= block.start_at);
    blocks.sort_by_key(|block| (block.lane_order, block.start_at, block.end_at));

    let Some(timeline_start) = blocks.iter().map(|block| block.start_at).min() else {
        return json!({
            "timeline_start": null,
            "timeline_end": null,
            "duration_ms": 0,
            "block_count": 0,
            "truncated": false,
            "lanes": [],
        });
    };
    let timeline_end = blocks
        .iter()
        .map(|block| block.end_at)
        .max()
        .unwrap_or(timeline_start);
    let duration_ms = (timeline_end - timeline_start).num_milliseconds().max(1);
    let truncated = blocks.len() > WATERFALL_BLOCK_CAP;
    if truncated {
        blocks.truncate(WATERFALL_BLOCK_CAP);
    }

    let mut lanes: BTreeMap<(usize, String, &'static str), Vec<Value>> = BTreeMap::new();
    for block in blocks {
        let start_ms = (block.start_at - timeline_start).num_milliseconds().max(0);
        let end_ms = (block.end_at - timeline_start)
            .num_milliseconds()
            .max(start_ms);
        lanes
            .entry((block.lane_order, block.lane.clone(), block.lane_kind))
            .or_default()
            .push(json!({
                "kind": block.kind,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "duration_ms": end_ms - start_ms,
                "label": block.label,
                "status": block.status,
                "source": block.source,
                "item_count": block.item_count,
                "item_unit": block.item_unit,
            }));
    }

    let lanes: Vec<Value> = lanes
        .into_iter()
        .map(|((_order, name, kind), blocks)| {
            json!({
                "name": name,
                "kind": kind,
                "blocks": blocks,
            })
        })
        .collect();

    json!({
        "timeline_start": timeline_start.to_rfc3339(),
        "timeline_end": timeline_end.to_rfc3339(),
        "duration_ms": duration_ms,
        "block_count": lanes
            .iter()
            .map(|lane| lane.get("blocks").and_then(Value::as_array).map(Vec::len).unwrap_or(0))
            .sum::<usize>(),
        "truncated": truncated,
        "lanes": lanes,
    })
}

#[derive(Default)]
struct SourcePhaseTimes {
    first_at: Option<DateTime<Utc>>,
    last_at: Option<DateTime<Utc>>,
    pre_capture_start: Option<DateTime<Utc>>,
    pre_capture_end: Option<DateTime<Utc>>,
    capture_start: Option<DateTime<Utc>>,
    capture_end: Option<DateTime<Utc>>,
    raw_start: Option<DateTime<Utc>>,
    raw_last_batch: Option<DateTime<Utc>>,
    raw_items: u64,
    distill_start: Option<DateTime<Utc>>,
    distill_join: Option<DateTime<Utc>>,
    distill_items: u64,
    write_start: Option<DateTime<Utc>>,
    write_end: Option<DateTime<Utc>>,
    index_start: Option<DateTime<Utc>>,
    index_end: Option<DateTime<Utc>>,
    consolidate_start: Option<DateTime<Utc>>,
    consolidate_end: Option<DateTime<Utc>>,
    answer_start: Option<DateTime<Utc>>,
    answer_end: Option<DateTime<Utc>>,
}

fn summarize_dependency_waterfall(memory_rows: &[Value]) -> Value {
    let mut sources: BTreeMap<String, SourcePhaseTimes> = BTreeMap::new();

    for trace in memory_rows {
        let Some(timestamp) = timestamp_for_trace(trace) else {
            continue;
        };
        let Some(operation) = memory_operation_for_trace(trace) else {
            continue;
        };
        let event = trace.get("event").and_then(Value::as_str).unwrap_or("");
        let source = trace_source(trace);
        if source == "run" {
            continue;
        }
        let entry = sources.entry(source).or_default();
        let trace_start = datetime_field(trace, "started_at").unwrap_or(timestamp);
        let trace_end = datetime_field(trace, "finished_at").unwrap_or(timestamp);
        entry.first_at = Some(
            entry
                .first_at
                .map_or(trace_start, |prev| prev.min(trace_start)),
        );
        entry.last_at = Some(entry.last_at.map_or(trace_end, |prev| prev.max(trace_end)));

        match (operation.as_str(), event) {
            ("pre_capture_setup", "operation_succeeded" | "operation_failed") => {
                entry.pre_capture_start =
                    Some(datetime_field(trace, "started_at").unwrap_or(timestamp));
                entry.pre_capture_end =
                    Some(datetime_field(trace, "finished_at").unwrap_or(timestamp));
            }
            ("capture", "operation_started") => {
                entry.capture_start = Some(datetime_field(trace, "started_at").unwrap_or(timestamp))
            }
            ("capture", "operation_succeeded" | "operation_failed") => {
                entry.capture_end = Some(datetime_field(trace, "finished_at").unwrap_or(timestamp))
            }
            ("embed_raw", "branch_started") => {
                entry.raw_start = Some(datetime_field(trace, "started_at").unwrap_or(timestamp))
            }
            ("embed_raw", "batch_succeeded" | "batch_failed") => {
                entry.raw_last_batch = Some(
                    entry
                        .raw_last_batch
                        .map_or(timestamp, |prev| prev.max(timestamp)),
                );
                if let Some((count, _unit)) = memory_item_count_for_trace(trace) {
                    entry.raw_items = entry.raw_items.saturating_add(count);
                }
            }
            ("distill", "branch_started") => {
                entry.distill_start = Some(datetime_field(trace, "started_at").unwrap_or(timestamp))
            }
            ("distill", "batch_succeeded" | "batch_failed") => {
                if let Some((count, _unit)) = memory_item_count_for_trace(trace) {
                    entry.distill_items = entry.distill_items.saturating_add(count);
                }
            }
            ("distill", "branch_joined") => {
                entry.distill_join = Some(datetime_field(trace, "finished_at").unwrap_or(timestamp))
            }
            ("write_archive", "operation_started") => {
                entry.write_start = Some(datetime_field(trace, "started_at").unwrap_or(timestamp))
            }
            ("write_archive", "operation_succeeded" | "operation_failed") => {
                entry.write_end = Some(datetime_field(trace, "finished_at").unwrap_or(timestamp))
            }
            ("index", "operation_started") => {
                entry.index_start = Some(datetime_field(trace, "started_at").unwrap_or(timestamp))
            }
            ("index", "operation_succeeded" | "operation_failed") => {
                entry.index_end = Some(datetime_field(trace, "finished_at").unwrap_or(timestamp))
            }
            ("consolidate", "operation_started") => {
                entry.consolidate_start =
                    Some(datetime_field(trace, "started_at").unwrap_or(timestamp))
            }
            ("consolidate", "operation_succeeded" | "operation_failed") => {
                entry.consolidate_end =
                    Some(datetime_field(trace, "finished_at").unwrap_or(timestamp))
            }
            (
                "query_plan" | "embed_query" | "fact_search" | "raw_search" | "support_check"
                | "answer_context" | "answer",
                "operation_started",
            ) => {
                entry.answer_start = Some(
                    entry
                        .answer_start
                        .map_or(timestamp, |prev| prev.min(timestamp)),
                );
            }
            (
                "query_plan" | "embed_query" | "fact_search" | "raw_search" | "support_check"
                | "answer_context" | "answer",
                "operation_succeeded" | "operation_failed",
            ) => {
                entry.answer_end = Some(
                    entry
                        .answer_end
                        .map_or(timestamp, |prev| prev.max(timestamp)),
                );
            }
            _ => {}
        }
    }

    let Some(timeline_start) = sources.values().filter_map(|source| source.first_at).min() else {
        return json!({
            "timeline_start": null,
            "timeline_end": null,
            "duration_ms": 0,
            "lanes": [],
        });
    };
    let timeline_end = sources
        .values()
        .filter_map(|source| source.last_at)
        .max()
        .unwrap_or(timeline_start);
    let duration_ms = (timeline_end - timeline_start).num_milliseconds().max(1);

    let mut lanes = Vec::new();
    for (source, phase) in sources {
        let mut blocks = Vec::new();
        push_dependency_block(
            &mut blocks,
            timeline_start,
            "setup",
            "vault setup",
            phase.pre_capture_start,
            phase.pre_capture_end,
            0,
            "items",
        );
        push_dependency_block(
            &mut blocks,
            timeline_start,
            "capture",
            "capture",
            phase.capture_start,
            phase.capture_end,
            0,
            "items",
        );

        let raw_done = phase.raw_last_batch;
        let distill_done = phase.distill_join;
        let parallel_start = [phase.raw_start, phase.distill_start]
            .into_iter()
            .flatten()
            .min();
        let first_done = [raw_done, distill_done].into_iter().flatten().min();
        if let (Some(start), Some(end)) = (parallel_start, first_done) {
            push_dependency_block_abs(
                &mut blocks,
                timeline_start,
                "parallel",
                "raw + distill",
                start,
                end,
                phase.raw_items.saturating_add(phase.distill_items),
                "items",
            );
        }
        if let (Some(raw_done), Some(distill_done)) = (raw_done, distill_done) {
            if raw_done < distill_done {
                push_dependency_block_abs(
                    &mut blocks,
                    timeline_start,
                    "blocked_distill",
                    "archive/index waiting on distill",
                    raw_done,
                    distill_done,
                    phase.distill_items,
                    "facts",
                );
            } else if distill_done < raw_done {
                push_dependency_block_abs(
                    &mut blocks,
                    timeline_start,
                    "blocked_raw",
                    "archive/index waiting on raw embeds",
                    distill_done,
                    raw_done,
                    phase.raw_items,
                    "turns",
                );
            }
        }

        push_dependency_block(
            &mut blocks,
            timeline_start,
            "archive",
            "write archive",
            phase.write_start,
            phase.write_end,
            phase.distill_items,
            "facts",
        );
        push_dependency_block(
            &mut blocks,
            timeline_start,
            "index",
            "index",
            phase.index_start,
            phase.index_end,
            phase.distill_items,
            "records",
        );
        push_dependency_block(
            &mut blocks,
            timeline_start,
            "consolidate",
            "briefs",
            phase.consolidate_start,
            phase.consolidate_end,
            0,
            "briefs",
        );
        push_dependency_block(
            &mut blocks,
            timeline_start,
            "answer",
            "answer path",
            phase.answer_start,
            phase.answer_end,
            0,
            "items",
        );

        blocks.sort_by_key(|block| block.get("start_ms").and_then(Value::as_i64).unwrap_or(0));
        let total_wait_ms: i64 = blocks
            .iter()
            .filter(|block| {
                block
                    .get("kind")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .starts_with("blocked_")
            })
            .filter_map(|block| block.get("duration_ms").and_then(Value::as_i64))
            .sum();
        let setup_ms: i64 = blocks
            .iter()
            .filter(|block| block.get("kind").and_then(Value::as_str) == Some("setup"))
            .filter_map(|block| block.get("duration_ms").and_then(Value::as_i64))
            .sum();
        lanes.push(json!({
            "source": source,
            "wait_ms": total_wait_ms,
            "setup_ms": setup_ms,
            "blocks": blocks,
        }));
    }
    lanes.sort_by(|a, b| {
        b.get("wait_ms")
            .and_then(Value::as_i64)
            .cmp(&a.get("wait_ms").and_then(Value::as_i64))
            .then_with(|| {
                b.get("setup_ms")
                    .and_then(Value::as_i64)
                    .cmp(&a.get("setup_ms").and_then(Value::as_i64))
            })
    });

    json!({
        "timeline_start": timeline_start.to_rfc3339(),
        "timeline_end": timeline_end.to_rfc3339(),
        "duration_ms": duration_ms,
        "lanes": lanes,
    })
}

#[allow(clippy::too_many_arguments)]
fn push_dependency_block(
    blocks: &mut Vec<Value>,
    timeline_start: DateTime<Utc>,
    kind: &'static str,
    label: &'static str,
    start: Option<DateTime<Utc>>,
    end: Option<DateTime<Utc>>,
    item_count: u64,
    item_unit: &'static str,
) {
    if let (Some(start), Some(end)) = (start, end) {
        push_dependency_block_abs(
            blocks,
            timeline_start,
            kind,
            label,
            start,
            end,
            item_count,
            item_unit,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn push_dependency_block_abs(
    blocks: &mut Vec<Value>,
    timeline_start: DateTime<Utc>,
    kind: &'static str,
    label: &'static str,
    start: DateTime<Utc>,
    end: DateTime<Utc>,
    item_count: u64,
    item_unit: &'static str,
) {
    let start_ms = (start - timeline_start).num_milliseconds().max(0);
    let end_ms = (end - timeline_start).num_milliseconds().max(start_ms);
    blocks.push(json!({
        "kind": kind,
        "label": label,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "duration_ms": end_ms - start_ms,
        "item_count": item_count,
        "item_unit": item_unit,
    }));
}

fn memory_waterfall_blocks(rows: &[Value]) -> Vec<WaterfallBlock> {
    let mut has_batches = HashMap::<String, bool>::new();
    for trace in rows {
        let Some(operation) = memory_operation_for_trace(trace) else {
            continue;
        };
        let event = trace.get("event").and_then(Value::as_str).unwrap_or("");
        if matches!(event, "batch_succeeded" | "batch_failed") {
            has_batches.insert(operation, true);
        }
    }

    let mut traces: Vec<&Value> = rows.iter().collect();
    traces.sort_by_key(|trace| timestamp_for_trace(trace));
    let mut anchors: HashMap<(String, String), DateTime<Utc>> = HashMap::new();
    let mut blocks = Vec::new();

    for trace in traces {
        let Some(operation) = memory_operation_for_trace(trace) else {
            continue;
        };
        let event = trace.get("event").and_then(Value::as_str).unwrap_or("");
        let source = trace_source(trace);
        let anchor_key = (operation.clone(), source.clone());
        let timestamp = timestamp_for_trace(trace);
        if matches!(
            event,
            "operation_started" | "branch_started" | "batch_started"
        ) {
            if let Some(start) = datetime_field(trace, "started_at").or(timestamp) {
                anchors.insert(anchor_key, start);
            }
            continue;
        }

        let operation_has_batches = has_batches.get(&operation).copied().unwrap_or(false);
        if operation_has_batches
            && matches!(
                event,
                "operation_succeeded" | "operation_failed" | "branch_joined"
            )
        {
            continue;
        }
        if !matches!(
            event,
            "operation_succeeded" | "operation_failed" | "batch_succeeded" | "batch_failed"
        ) {
            continue;
        }

        let Some(end) = datetime_field(trace, "finished_at").or(timestamp) else {
            continue;
        };
        let start = datetime_field(trace, "started_at")
            .or_else(|| {
                trace
                    .get("duration_ms")
                    .and_then(Value::as_i64)
                    .map(|duration| end - chrono::Duration::milliseconds(duration.max(0)))
            })
            .or_else(|| anchors.get(&anchor_key).copied())
            .unwrap_or(end);
        if matches!(event, "batch_succeeded" | "batch_failed") {
            anchors.insert(anchor_key, end);
        }
        let (item_count, item_unit) = memory_item_count_for_trace(trace).unwrap_or((0, "items"));
        blocks.push(WaterfallBlock {
            lane: operation.clone(),
            lane_kind: "memory",
            lane_order: stage_order(&operation),
            kind: if event.ends_with("failed") {
                "memory_failed"
            } else {
                "memory_work"
            },
            start_at: start,
            end_at: end,
            label: event.replace("operation_", "").replace("batch_", ""),
            status: if event.ends_with("failed") {
                "failed"
            } else {
                "succeeded"
            }
            .to_string(),
            source,
            item_count,
            item_unit: item_unit.to_string(),
        });
    }

    blocks
}

fn provider_waterfall_blocks(events: &[BenchQueueEvent]) -> Vec<WaterfallBlock> {
    let mut grouped: BTreeMap<(&str, &str), Vec<&BenchQueueEvent>> = BTreeMap::new();
    for event in events {
        grouped
            .entry((&event.queue_id, &event.item_id))
            .or_default()
            .push(event);
    }

    let mut blocks = Vec::new();
    for mut group in grouped.into_values() {
        group.sort_by_key(|event| event.timestamp);
        let Some(first) = group.first() else {
            continue;
        };
        let queued_at = group
            .iter()
            .find(|event| event.status == membench::BenchEventStatus::Queued)
            .map(|event| event.timestamp);
        let running_at = group
            .iter()
            .find(|event| event.status == membench::BenchEventStatus::Running)
            .map(|event| event.timestamp);
        let terminal = group.iter().rev().find(|event| {
            matches!(
                event.status,
                membench::BenchEventStatus::Succeeded
                    | membench::BenchEventStatus::Failed
                    | membench::BenchEventStatus::Dead
            )
        });
        let terminal_at = terminal.map(|event| event.timestamp);
        let status = terminal
            .map(|event| queue_status_name(event.status).to_string())
            .unwrap_or_else(|| "running".to_string());
        let attempts = group
            .iter()
            .map(|event| event.attempt)
            .max()
            .unwrap_or_default();
        let lane = format!("{}:{}", first.operation, short_queue_id(&first.queue_id));
        let lane_order = 1000 + if first.operation == "embedding" { 1 } else { 0 };

        if let (Some(start), Some(end)) = (queued_at, running_at) {
            blocks.push(WaterfallBlock {
                lane: lane.clone(),
                lane_kind: "provider",
                lane_order,
                kind: "provider_wait",
                start_at: start,
                end_at: end,
                label: format!("wait a{attempts}"),
                status: status.clone(),
                source: first.item_id.clone(),
                item_count: 0,
                item_unit: "items".to_string(),
            });
        }
        if let (Some(start), Some(end)) = (running_at, terminal_at) {
            blocks.push(WaterfallBlock {
                lane,
                lane_kind: "provider",
                lane_order,
                kind: if matches!(
                    terminal.map(|event| event.status),
                    Some(membench::BenchEventStatus::Failed | membench::BenchEventStatus::Dead)
                ) {
                    "provider_failed"
                } else {
                    "provider_run"
                },
                start_at: start,
                end_at: end,
                label: format!("run a{attempts}"),
                status,
                source: first.item_id.clone(),
                item_count: 0,
                item_unit: "items".to_string(),
            });
        }
    }

    blocks
}

fn datetime_field(trace: &Value, key: &str) -> Option<DateTime<Utc>> {
    trace
        .get(key)
        .and_then(Value::as_str)
        .and_then(|raw| DateTime::parse_from_rfc3339(raw).ok())
        .map(|dt| dt.with_timezone(&Utc))
}

fn trace_source(trace: &Value) -> String {
    trace
        .get("source_id")
        .or_else(|| trace.get("run_id"))
        .or_else(|| trace.get("question_id"))
        .and_then(Value::as_str)
        .unwrap_or("run")
        .to_string()
}

fn short_queue_id(id: &str) -> String {
    id.strip_prefix("chat:")
        .or_else(|| id.strip_prefix("embedding:"))
        .unwrap_or(id)
        .to_string()
}

fn queue_status_name(status: membench::BenchEventStatus) -> &'static str {
    match status {
        membench::BenchEventStatus::Queued => "queued",
        membench::BenchEventStatus::Running => "running",
        membench::BenchEventStatus::Succeeded => "succeeded",
        membench::BenchEventStatus::Failed => "failed",
        membench::BenchEventStatus::Dead => "dead",
    }
}

#[derive(Default)]
struct MemoryStageTimingAcc {
    events: u64,
    batch_events: u64,
    intermediate_failed: u64,
    failed: u64,
    batch_item_count: u64,
    batch_item_unit: String,
    terminal_item_count: u64,
    terminal_item_unit: String,
    cadence_ms: Vec<u64>,
    numeric_metrics: BTreeMap<String, Vec<f64>>,
}

fn summarize_memory_stage_timing(rows: &[Value]) -> Vec<Value> {
    let mut stages: BTreeMap<String, MemoryStageTimingAcc> = BTreeMap::new();
    let mut anchors: HashMap<(String, String), DateTime<Utc>> = HashMap::new();

    for trace in rows {
        let Some(operation) = memory_operation_for_trace(trace) else {
            continue;
        };
        let event = trace.get("event").and_then(Value::as_str).unwrap_or("");
        let source = trace
            .get("source_id")
            .or_else(|| trace.get("run_id"))
            .or_else(|| trace.get("question_id"))
            .and_then(Value::as_str)
            .unwrap_or("run")
            .to_string();
        let timestamp = timestamp_for_trace(trace);
        let entry = stages.entry(operation.clone()).or_default();
        entry.events += 1;

        if matches!(event, "operation_failed" | "batch_failed") {
            entry.failed += 1;
        }
        if event == "batch_failed" {
            entry.intermediate_failed += 1;
        }
        if let Some((count, unit)) = memory_item_count_for_trace(trace) {
            if matches!(event, "batch_succeeded" | "batch_failed") {
                entry.batch_item_count += count;
                entry.batch_item_unit = unit.to_string();
            } else {
                entry.terminal_item_count += count;
                entry.terminal_item_unit = unit.to_string();
            }
        }
        collect_numeric_metrics(trace, &mut entry.numeric_metrics);

        let anchor_key = (operation.clone(), source);
        if matches!(
            event,
            "operation_started" | "branch_started" | "batch_started"
        ) {
            if let Some(timestamp) = timestamp {
                anchors.insert(anchor_key, timestamp);
            }
            continue;
        }

        if matches!(event, "batch_succeeded" | "batch_failed") {
            entry.batch_events += 1;
            if let Some(duration_ms) = trace.get("duration_ms").and_then(Value::as_u64) {
                entry.cadence_ms.push(duration_ms);
            } else if let Some(timestamp) = timestamp {
                if let Some(previous) = anchors.get(&anchor_key) {
                    entry
                        .cadence_ms
                        .push((timestamp - *previous).num_milliseconds().max(0) as u64);
                }
                anchors.insert(anchor_key, timestamp);
            }
        } else if matches!(operation.as_str(), "pre_capture_setup" | "pre_recall_setup")
            && matches!(event, "operation_succeeded" | "operation_failed")
        {
            if let Some(duration_ms) = trace.get("duration_ms").and_then(Value::as_u64) {
                entry.cadence_ms.push(duration_ms);
            } else if let Some(timestamp) = timestamp
                && let Some(previous) = anchors.get(&anchor_key)
            {
                entry
                    .cadence_ms
                    .push((timestamp - *previous).num_milliseconds().max(0) as u64);
            }
        }
    }

    let mut rows: Vec<Value> = stages
        .into_iter()
        .map(|(operation, acc)| {
            let (item_count, item_unit) = if acc.batch_item_count > 0 {
                (acc.batch_item_count, acc.batch_item_unit.as_str())
            } else {
                (acc.terminal_item_count, acc.terminal_item_unit.as_str())
            };
            // Sort the cadence sample once and index into it for all four
            // percentiles (previously each percentile re-cloned + re-sorted).
            let mut cadence = acc.cadence_ms;
            cadence.sort_unstable();
            json!({
                "operation": operation,
                "events": acc.events,
                "batch_events": acc.batch_events,
                "intermediate_failed": acc.intermediate_failed,
                "failed": acc.failed,
                "item_count": item_count,
                "item_unit": if item_unit.is_empty() { "items" } else { item_unit },
                "work_ms_p50": percentile_sorted_u64(&cadence, 50.0),
                "work_ms_p80": percentile_sorted_u64(&cadence, 80.0),
                "work_ms_p95": percentile_sorted_u64(&cadence, 95.0),
                "work_ms_p98": percentile_sorted_u64(&cadence, 98.0),
                "numeric_metrics": summarize_numeric_metrics(acc.numeric_metrics),
            })
        })
        .collect();
    rows.sort_by_key(|row| {
        let operation = row.get("operation").and_then(Value::as_str).unwrap_or("");
        stage_order(operation)
    });
    rows
}

fn collect_numeric_metrics(trace: &Value, out: &mut BTreeMap<String, Vec<f64>>) {
    let Some(metrics) = trace.get("metrics").and_then(Value::as_object) else {
        return;
    };
    for (key, value) in metrics {
        if let Some(number) = value.as_f64() {
            out.entry(key.clone()).or_default().push(number);
        }
    }
}

fn summarize_numeric_metrics(metrics: BTreeMap<String, Vec<f64>>) -> Value {
    Value::Object(
        metrics
            .into_iter()
            .filter_map(|(key, mut values)| {
                if values.is_empty() {
                    return None;
                }
                values.sort_by(|a, b| a.total_cmp(b));
                Some((
                    key,
                    json!({
                        "count": values.len(),
                        "p50": percentile_sorted_f64(&values, 50.0),
                        "p80": percentile_sorted_f64(&values, 80.0),
                        "p95": percentile_sorted_f64(&values, 95.0),
                        "p98": percentile_sorted_f64(&values, 98.0),
                        "max": values.last().copied(),
                    }),
                ))
            })
            .collect(),
    )
}

fn memory_operation_for_trace(trace: &Value) -> Option<String> {
    let op = trace.get("operation").and_then(Value::as_str)?;
    if op == "adapter_call" {
        if let Some(stage @ ("pre_capture_setup" | "pre_recall_setup")) =
            trace.get("stage").and_then(Value::as_str)
        {
            return Some(stage.to_string());
        }
        return None;
    }
    let metrics = trace.get("metrics").unwrap_or(&Value::Null);
    if op == "embed_facts" && metrics.get("kind").and_then(Value::as_str) == Some("brief") {
        return Some("consolidate".to_string());
    }
    Some(op.to_string())
}

fn timestamp_for_trace(trace: &Value) -> Option<DateTime<Utc>> {
    trace
        .get("timestamp")
        .and_then(Value::as_str)
        .and_then(|raw| DateTime::parse_from_rfc3339(raw).ok())
        .map(|dt| dt.with_timezone(&Utc))
}

fn memory_item_count_for_trace(trace: &Value) -> Option<(u64, &'static str)> {
    let metrics = trace.get("metrics").unwrap_or(&Value::Null);
    let keys = [
        ("turn_count", "turns"),
        ("raw_turn_count", "turns"),
        ("total_turn_count", "turns"),
        ("fact_count", "facts"),
        ("base_fact_count", "facts"),
        ("total_fact_count", "facts"),
        ("record_count", "records"),
        ("brief_count", "briefs"),
        ("extractive_brief_count", "briefs"),
        ("context_item_count", "ctx"),
        ("evidence_id_count", "ctx"),
    ];
    for (key, unit) in keys {
        if let Some(count) = metrics.get(key).and_then(Value::as_u64) {
            if metrics.get("kind").and_then(Value::as_str) == Some("brief") && key == "fact_count" {
                return Some((count, "briefs"));
            }
            return Some((count, unit));
        }
    }
    trace
        .get("item_count")
        .and_then(Value::as_u64)
        .map(|count| (count, "items"))
}

/// Percentile over a sample already sorted ascending, so callers that need
/// several percentiles of the same sample can sort once and index repeatedly.
fn percentile_sorted_u64(values: &[u64], percentile: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let rank = ((percentile / 100.0) * (values.len().saturating_sub(1)) as f64).round() as usize;
    values.get(rank).map(|value| *value as f64)
}

fn percentile_sorted_f64(values: &[f64], percentile: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let rank = ((percentile / 100.0) * (values.len().saturating_sub(1)) as f64).round() as usize;
    values.get(rank).copied()
}

fn stage_order(operation: &str) -> usize {
    [
        "pre_capture_setup",
        "capture",
        "ingest",
        "distill",
        "write_archive",
        "embed_raw",
        "embed_facts",
        "index",
        "consolidate",
        "pre_recall_setup",
        "query_plan",
        "embed_query",
        "fact_search",
        "raw_search",
        "support_check",
        "answer_context",
        "retrieve",
        "answer",
        "score",
    ]
    .iter()
    .position(|known| *known == operation)
    .unwrap_or(usize::MAX)
}

fn read_workflow_queue(run_root: &Path) -> Option<Value> {
    let workflow_root = run_root.join("workflow");
    if !workflow_root.is_dir() {
        return None;
    }

    let mut queue_paths = Vec::new();
    let direct = workflow_root.join("queue.sqlite");
    if direct.is_file() {
        queue_paths.push(direct);
    }
    if let Ok(entries) = std::fs::read_dir(&workflow_root) {
        for entry in entries.flatten() {
            let path = entry.path().join("queue.sqlite");
            if path.is_file() {
                queue_paths.push(path);
            }
        }
    }

    let databases: Vec<Value> = queue_paths
        .iter()
        .filter_map(|path| read_workflow_queue_db(run_root, path))
        .collect();
    if databases.is_empty() {
        None
    } else {
        Some(json!({ "databases": databases }))
    }
}

fn read_workflow_queue_db(run_root: &Path, path: &Path) -> Option<Value> {
    let conn = rusqlite::Connection::open_with_flags(
        path,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .ok()?;
    let items_by_status = query_count_map(
        &conn,
        "select status, count(*) from queue_items group by status",
    );
    let events_by_status = query_count_map(
        &conn,
        "select status, count(*) from queue_events group by status",
    );
    let queues = query_count_map(
        &conn,
        "select queue_id, count(*) from queue_items group by queue_id",
    );
    let total_items = items_by_status.values().sum::<u64>();
    let total_events = scalar_count(&conn, "select count(*) from queue_events").unwrap_or(0);
    let retried_items =
        scalar_count(&conn, "select count(*) from queue_items where attempt > 1").unwrap_or(0);
    let max_attempt =
        scalar_count(&conn, "select coalesce(max(attempt), 0) from queue_items").unwrap_or(0);
    let recent_errors = recent_workflow_errors(&conn);
    let recent_events = recent_workflow_events(&conn);
    Some(json!({
        "path": path.strip_prefix(run_root).unwrap_or(path).display().to_string(),
        "total_items": total_items,
        "total_events": total_events,
        "items_by_status": items_by_status,
        "events_by_status": events_by_status,
        "queues": queues,
        "retried_items": retried_items,
        "max_attempt": max_attempt,
        "recent_errors": recent_errors,
        "recent_events": recent_events,
    }))
}

fn query_count_map(conn: &rusqlite::Connection, sql: &str) -> BTreeMap<String, u64> {
    let mut out = BTreeMap::new();
    let Ok(mut stmt) = conn.prepare(sql) else {
        return out;
    };
    let Ok(rows) = stmt.query_map([], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
    }) else {
        return out;
    };
    for row in rows.flatten() {
        out.insert(row.0, row.1.max(0) as u64);
    }
    out
}

fn scalar_count(conn: &rusqlite::Connection, sql: &str) -> Option<u64> {
    conn.query_row(sql, [], |row| row.get::<_, i64>(0))
        .ok()
        .map(|count| count.max(0) as u64)
}

fn recent_workflow_errors(conn: &rusqlite::Connection) -> Vec<Value> {
    let Ok(mut stmt) = conn.prepare(
        "select item_id, queue_id, kind, status, attempt, last_error \
         from queue_items \
         where last_error is not null and last_error != '' \
           and status in ('failed', 'dead') \
         order by updated_at desc \
         limit 20",
    ) else {
        return Vec::new();
    };
    let Ok(rows) = stmt.query_map([], |row| {
        Ok(json!({
            "item_id": row.get::<_, String>(0)?,
            "queue_id": row.get::<_, String>(1)?,
            "kind": row.get::<_, String>(2)?,
            "status": row.get::<_, String>(3)?,
            "attempt": row.get::<_, i64>(4)?.max(0) as u64,
            "error": row.get::<_, String>(5)?,
        }))
    }) else {
        return Vec::new();
    };
    rows.flatten().collect()
}

fn recent_workflow_events(conn: &rusqlite::Connection) -> Vec<Value> {
    let Ok(mut stmt) = conn.prepare(
        "select item_id, queue_id, kind, status, attempt, timestamp, error \
         from queue_events \
         order by event_id desc \
         limit 40",
    ) else {
        return Vec::new();
    };
    let Ok(rows) = stmt.query_map([], |row| {
        Ok(json!({
            "item_id": row.get::<_, String>(0)?,
            "queue_id": row.get::<_, String>(1)?,
            "kind": row.get::<_, String>(2)?,
            "status": row.get::<_, String>(3)?,
            "attempt": row.get::<_, i64>(4)?.max(0) as u64,
            "timestamp": row.get::<_, String>(5)?,
            "error": row.get::<_, Option<String>>(6)?,
        }))
    }) else {
        return Vec::new();
    };
    rows.flatten().collect()
}

#[derive(Deserialize)]
struct CompareQuery {
    base: String,
    cand: String,
}

async fn compare_handler(
    State(state): State<Shared>,
    Query(query): Query<CompareQuery>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let (base_id, cand_id) = (query.base.clone(), query.cand.clone());
    let payload = tokio::task::spawn_blocking(move || {
        // One registry snapshot resolves both ids (previously two full scans).
        let snapshot = state.registry_snapshot();
        let base_record = snapshot
            .by_id
            .get(&base_id)
            .cloned()
            .ok_or_else(|| err(StatusCode::NOT_FOUND, "base run not found"))?;
        let cand_record = snapshot
            .by_id
            .get(&cand_id)
            .cloned()
            .ok_or_else(|| err(StatusCode::NOT_FOUND, "candidate run not found"))?;
        let base_summary = snapshot
            .summary_by_id
            .get(&base_id)
            .cloned()
            .ok_or_else(|| err(StatusCode::NOT_FOUND, "base run not found"))?;
        let cand_summary = snapshot
            .summary_by_id
            .get(&cand_id)
            .cloned()
            .ok_or_else(|| err(StatusCode::NOT_FOUND, "candidate run not found"))?;
        let result = compare::compare_runs(&base_record.run_root, &cand_record.run_root);
        Ok::<_, (StatusCode, Json<Value>)>(json!({
            "base": base_summary,
            "candidate": cand_summary,
            "result": result,
        }))
    })
    .await
    .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))??;
    Ok(Json(payload))
}

async fn runner_schema(State(state): State<Shared>) -> impl IntoResponse {
    let body = tokio::task::spawn_blocking(move || {
        // Enrich enum options with values observed in existing runs so the form
        // reflects the configurations actually used here.
        let snapshot = state.registry_snapshot();
        let records = &snapshot.records;
        let mut observed: HashMap<String, Vec<String>> = HashMap::new();
        for record in records {
            for key in [
                "distiller",
                "embedder",
                "store",
                "query_planner",
                "scorer",
                "dataset",
                "oracle",
                "memory_config",
                "symem_bin",
            ] {
                if let Some(value) = record.params.get(key).and_then(Value::as_str)
                    && !value.is_empty()
                {
                    let entry = observed.entry(key.to_string()).or_default();
                    if !entry.contains(&value.to_string()) {
                        entry.push(value.to_string());
                    }
                }
            }
        }

        let mut fields: Vec<Value> = runner::symem_param_schema()
            .into_iter()
            .map(|field| {
                let mut value = serde_json::to_value(&field).unwrap_or(Value::Null);
                if let Some(seen) = observed.get(&field.name) {
                    value["observed"] = json!(seen);
                }
                value
            })
            .collect();
        fields.sort_by(|a, b| {
            a["group"]
                .as_str()
                .unwrap_or("")
                .cmp(b["group"].as_str().unwrap_or(""))
        });

        json!({
            "system": "symbiotic-memory",
            "benchmark": "long-mem-eval",
            "fields": fields,
        })
    })
    .await
    .unwrap_or_else(
        |_| json!({ "system": "symbiotic-memory", "benchmark": "long-mem-eval", "fields": [] }),
    );
    Json(body)
}

async fn runner_plan(State(state): State<Shared>, Json(params): Json<Value>) -> impl IntoResponse {
    let repo_root = state.repo_root.clone();
    let plan = tokio::task::spawn_blocking(move || runner::plan_from_params(&params, &repo_root))
        .await
        .ok();
    Json(
        plan.map(|p| serde_json::to_value(p.preview()).unwrap_or(Value::Null))
            .unwrap_or(Value::Null),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn release_bundle_resolves_repo_root_beside_the_binary() {
        let temp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(temp.path().join("dashboard/dist")).unwrap();
        std::fs::create_dir_all(temp.path().join("records")).unwrap();
        std::fs::write(
            temp.path().join("dashboard/dist/index.html"),
            "<!doctype html>",
        )
        .unwrap();
        std::fs::write(temp.path().join("PROVENANCE.json"), "{}\n").unwrap();
        let executable = temp.path().join("membench-server");

        assert_eq!(resolve_repo_root(None, Some(&executable)), temp.path());
        assert_eq!(
            resolve_repo_root(Some(PathBuf::from("/explicit")), Some(&executable)),
            PathBuf::from("/explicit")
        );
    }

    #[test]
    fn question_debug_path_accepts_run_local_debug_bundle() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp
            .path()
            .join("vaults/q1/debug/hypotheses/q1/question-debug.json");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, "{}\n").unwrap();

        let resolved = question_debug_path(
            temp.path(),
            "vaults/q1/debug/hypotheses/q1/question-debug.json",
        )
        .unwrap();
        assert_eq!(resolved, path.canonicalize().unwrap());
    }

    #[test]
    fn question_debug_path_rejects_escape_paths() {
        let temp = tempfile::tempdir().unwrap();
        assert!(question_debug_path(temp.path(), "../question-debug.json").is_err());
        assert!(question_debug_path(temp.path(), "/tmp/question-debug.json").is_err());
        assert!(question_debug_path(temp.path(), "artifacts/scored.json").is_err());
    }

    #[test]
    fn workflow_queue_summary_reads_sqlite_state() {
        let temp = tempfile::tempdir().unwrap();
        let queue_dir = temp.path().join("workflow").join("longmemeval");
        std::fs::create_dir_all(&queue_dir).unwrap();
        let db_path = queue_dir.join("queue.sqlite");
        let conn = rusqlite::Connection::open(&db_path).unwrap();
        conn.execute_batch(
            r#"
            create table queue_items (
                item_id text primary key,
                queue_id text not null,
                kind text not null,
                payload_json text not null,
                status text not null,
                attempt integer not null,
                max_attempts integer not null,
                run_after text not null,
                lease_owner text,
                lease_until text,
                idempotency_key text,
                last_error text,
                created_at text not null,
                updated_at text not null
            );
            create table queue_events (
                event_id integer primary key autoincrement,
                item_id text not null,
                queue_id text not null,
                kind text not null,
                status text not null,
                attempt integer not null,
                timestamp text not null,
                error text
            );
            insert into queue_items values (
                'item-1', 'workflow:longmemeval', 'longmemeval.row', '{}',
                'succeeded', 2, 4, '2026-01-01T00:00:00Z', null, null,
                null, '', '2026-01-01T00:00:00Z', '2026-01-01T00:00:02Z'
            );
            insert into queue_events
                (item_id, queue_id, kind, status, attempt, timestamp, error)
            values
                ('item-1', 'workflow:longmemeval', 'longmemeval.row', 'pending', 0, '2026-01-01T00:00:00Z', null),
                ('item-1', 'workflow:longmemeval', 'longmemeval.row', 'running', 1, '2026-01-01T00:00:01Z', null),
                ('item-1', 'workflow:longmemeval', 'longmemeval.row', 'succeeded', 2, '2026-01-01T00:00:02Z', null);
            "#,
        )
        .unwrap();

        let summary = read_workflow_queue(temp.path()).unwrap();
        let db = &summary["databases"][0];
        assert_eq!(db["total_items"], json!(1));
        assert_eq!(db["total_events"], json!(3));
        assert_eq!(db["items_by_status"]["succeeded"], json!(1));
        assert_eq!(db["retried_items"], json!(1));
        assert_eq!(db["max_attempt"], json!(2));
    }

    #[test]
    fn adapter_setup_stages_are_first_class_trace_operations() {
        let rows = vec![
            json!({
                "timestamp": "2026-01-01T00:00:00.900Z",
                "operation": "adapter_call",
                "stage": "pre_capture_setup",
                "event": "operation_succeeded",
                "source_id": "q1",
                "started_at": "2026-01-01T00:00:00.000Z",
                "finished_at": "2026-01-01T00:00:00.900Z",
                "duration_ms": 900,
                "metrics": {
                    "store_open_ms": 700,
                    "zvec_cache_ms": 120,
                    "load_existing_ms": 40
                }
            }),
            json!({
                "timestamp": "2026-01-01T00:00:01.000Z",
                "operation": "capture",
                "event": "operation_started",
                "source_id": "q1",
                "started_at": "2026-01-01T00:00:01.000Z"
            }),
            json!({
                "timestamp": "2026-01-01T00:00:01.010Z",
                "operation": "capture",
                "event": "operation_succeeded",
                "source_id": "q1",
                "finished_at": "2026-01-01T00:00:01.010Z",
                "duration_ms": 10
            }),
            json!({
                "timestamp": "2026-01-01T00:00:02.050Z",
                "operation": "adapter_call",
                "stage": "pre_recall_setup",
                "event": "operation_succeeded",
                "source_id": "q1",
                "started_at": "2026-01-01T00:00:02.000Z",
                "finished_at": "2026-01-01T00:00:02.050Z",
                "duration_ms": 50,
                "metrics": {
                    "ensure_recall_index_ms": 45,
                    "fact_count": 7,
                    "turn_count": 11
                }
            }),
        ];

        assert_eq!(
            memory_operation_for_trace(&rows[0]).as_deref(),
            Some("pre_capture_setup")
        );
        assert_eq!(
            memory_operation_for_trace(&rows[3]).as_deref(),
            Some("pre_recall_setup")
        );

        let timing = summarize_memory_stage_timing(&rows);
        assert_eq!(timing[0]["operation"], json!("pre_capture_setup"));
        assert_eq!(timing[0]["work_ms_p98"], json!(900.0));
        assert_eq!(
            timing[0]["numeric_metrics"]["store_open_ms"]["p98"],
            json!(700.0)
        );

        let dependency = summarize_dependency_waterfall(&rows);
        let blocks = dependency["lanes"][0]["blocks"].as_array().unwrap();
        assert_eq!(blocks[0]["kind"], json!("setup"));
        assert_eq!(blocks[0]["duration_ms"], json!(900));
        assert_eq!(blocks[1]["kind"], json!("capture"));
    }

    #[test]
    fn legacy_adapter_call_is_hidden_from_dashboard_summaries() {
        let rows = vec![json!({
            "timestamp": "2026-01-01T00:00:00.100Z",
            "operation": "adapter_call",
            "stage": "legacy_probe",
            "event": "operation_succeeded",
            "source_id": "q1",
            "started_at": "2026-01-01T00:00:00.000Z",
            "finished_at": "2026-01-01T00:00:00.100Z",
            "duration_ms": 100,
            "metrics": {"store_open_ms": 99}
        })];

        assert_eq!(memory_operation_for_trace(&rows[0]), None);
        assert!(summarize_memory_stage_timing(&rows).is_empty());
        assert!(
            summarize_dependency_waterfall(&rows)["lanes"]
                .as_array()
                .unwrap()
                .is_empty()
        );
        assert!(
            summarize_trace_events(&rows, &[], &[])["rows"]
                .as_array()
                .unwrap()
                .is_empty()
        );
    }
}
