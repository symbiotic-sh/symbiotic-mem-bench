//! Benchmark loader and grading plug-in surface.
//!
//! The v0.1 LongMemEval path remains the default `PerQuestion` implementation. LongMemEval-v2 uses
//! the same normalized record at the adapter seam, but declares `SharedCorpus` so its domain corpus
//! is ingested once and queried by many questions.

use std::collections::{HashMap, HashSet};
use std::io::BufRead;
use std::path::Path;

use anyhow::Context;
use serde::Deserialize;
use serde_json::Value;

use crate::symbiotic_memory_adapter::{
    LongMemEvalMessage, LongMemEvalRecord, SharedCorpusQuestion, load_longmemeval,
};

pub enum DatasetSource {
    Path {
        rel_path: &'static str,
        download_url: Option<&'static str>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HaystackScope {
    PerQuestion,
    SharedCorpus,
}

#[derive(Clone, Debug)]
pub struct GradeTarget {
    pub question_type: Option<String>,
    pub gold_answer: Option<Value>,
    pub eval_function: Option<String>,
}

pub trait BenchmarkLoader: Send + Sync {
    fn id(&self) -> &'static str;
    fn manifest_tag(&self) -> &'static str;
    fn default_dataset(&self) -> DatasetSource;
    fn load(&self, path: &Path, limit: Option<usize>) -> anyhow::Result<Vec<LongMemEvalRecord>>;
    fn grade_targets(&self, path: &Path) -> anyhow::Result<HashMap<String, GradeTarget>>;

    fn haystack_scope(&self) -> HaystackScope {
        HaystackScope::PerQuestion
    }

    fn shared_questions(
        &self,
        _path: &Path,
        _limit: Option<usize>,
    ) -> anyhow::Result<Vec<SharedCorpusQuestion>> {
        anyhow::bail!("benchmark '{}' does not support shared corpora", self.id())
    }

    fn corpus_record(&self, _path: &Path, _corpus_key: &str) -> anyhow::Result<LongMemEvalRecord> {
        anyhow::bail!("benchmark '{}' does not support shared corpora", self.id())
    }
}

pub fn loader_for(id: &str) -> Option<Box<dyn BenchmarkLoader>> {
    match id {
        "long-mem-eval" => Some(Box::new(LongMemEvalV1)),
        crate::eligibility::LONGMEMEVAL_V2_TEXT_ID => Some(Box::new(LongMemEvalV2Text)),
        _ => None,
    }
}

pub struct LongMemEvalV1;

impl BenchmarkLoader for LongMemEvalV1 {
    fn id(&self) -> &'static str {
        "long-mem-eval"
    }

    fn manifest_tag(&self) -> &'static str {
        "longmemeval-v1"
    }

    fn default_dataset(&self) -> DatasetSource {
        DatasetSource::Path {
            rel_path: "runs/inputs/longmemeval-cleaned/longmemeval_s_cleaned.json",
            download_url: Some(
                "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json",
            ),
        }
    }

    fn load(&self, path: &Path, limit: Option<usize>) -> anyhow::Result<Vec<LongMemEvalRecord>> {
        load_longmemeval(path, limit)
    }

    fn grade_targets(&self, path: &Path) -> anyhow::Result<HashMap<String, GradeTarget>> {
        Ok(load_longmemeval(path, None)?
            .into_iter()
            .map(|row| {
                (
                    row.question_id,
                    GradeTarget {
                        question_type: row.question_type,
                        gold_answer: row.answer,
                        eval_function: None,
                    },
                )
            })
            .collect())
    }
}

/// Experimental, non-official text projection of LongMemEval-V2.
///
/// The official benchmark is multimodal. This loader deliberately excludes every question with a
/// query image and retains screenshot *locators* only; it must never be reported as an official
/// LongMemEval-V2 score.
pub struct LongMemEvalV2Text;

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct V2Question {
    id: String,
    domain: String,
    environment: String,
    question_type: String,
    question: String,
    image: Option<String>,
    answer: Value,
    eval_function: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct V2Trajectory {
    id: String,
    domain: String,
    environment: String,
    goal: String,
    outcome: String,
    start_url: String,
    states: Vec<V2State>,
}

#[derive(Debug, Deserialize)]
struct V2TrajectoryId {
    id: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct V2State {
    step: Value,
    state_index: Value,
    url: String,
    action: Option<String>,
    thought: Option<String>,
    accessibility_tree: String,
    screenshot: String,
}

impl LongMemEvalV2Text {
    fn haystack_file() -> anyhow::Result<&'static str> {
        match std::env::var("MEMBENCH_V2_HAYSTACK").ok().as_deref() {
            None | Some("") | Some("small") => Ok("lme_v2_small.json"),
            Some("medium") => Ok("lme_v2_medium.json"),
            Some(other) => {
                anyhow::bail!("MEMBENCH_V2_HAYSTACK must be 'small' or 'medium', got '{other}'")
            }
        }
    }

    fn read_all_questions(dir: &Path) -> anyhow::Result<Vec<V2Question>> {
        let path = dir.join("questions.jsonl");
        let file = std::fs::File::open(&path)
            .with_context(|| format!("open v2 questions {}", path.display()))?;
        let mut questions = Vec::new();
        let mut ids = HashSet::new();
        for (index, line) in std::io::BufReader::new(file).lines().enumerate() {
            let line =
                line.with_context(|| format!("read {} line {}", path.display(), index + 1))?;
            if line.trim().is_empty() {
                continue;
            }
            let question: V2Question = serde_json::from_str(&line)
                .with_context(|| format!("parse {} line {}", path.display(), index + 1))?;
            validate_question(&question)
                .with_context(|| format!("validate {} line {}", path.display(), index + 1))?;
            if !ids.insert(question.id.clone()) {
                anyhow::bail!("duplicate v2 question id '{}'", question.id);
            }
            questions.push(question);
        }
        anyhow::ensure!(!questions.is_empty(), "v2 dataset has no questions");
        Ok(questions)
    }

    fn read_questions(dir: &Path, limit: Option<usize>) -> anyhow::Result<Vec<V2Question>> {
        let mut text_questions: Vec<_> = Self::read_all_questions(dir)?
            .into_iter()
            .filter(|question| question.image.is_none())
            .collect();
        if let Some(limit) = limit {
            text_questions.truncate(limit);
        }
        anyhow::ensure!(
            !text_questions.is_empty(),
            "v2 text projection selected no text-only questions"
        );
        Ok(text_questions)
    }

    fn read_haystack_file(
        dir: &Path,
        file_name: &str,
    ) -> anyhow::Result<HashMap<String, Vec<String>>> {
        let path = dir.join("haystacks").join(file_name);
        let raw = std::fs::read_to_string(&path)
            .with_context(|| format!("read v2 haystack map {}", path.display()))?;
        let map: HashMap<String, Vec<String>> = serde_json::from_str(&raw)
            .with_context(|| format!("parse v2 haystack map {}", path.display()))?;
        for (question_id, ids) in &map {
            anyhow::ensure!(
                !ids.is_empty(),
                "v2 haystack for question '{question_id}' is empty"
            );
            let unique: HashSet<_> = ids.iter().collect();
            anyhow::ensure!(
                unique.len() == ids.len(),
                "v2 haystack for question '{question_id}' contains duplicate trajectory ids"
            );
            for id in ids {
                anyhow::ensure!(
                    non_empty(Some(id)).is_some(),
                    "v2 haystack for question '{question_id}' contains an empty trajectory id"
                );
            }
        }
        let question_ids: HashSet<_> = Self::read_all_questions(dir)?
            .into_iter()
            .map(|question| question.id)
            .collect();
        let haystack_ids: HashSet<_> = map.keys().cloned().collect();
        anyhow::ensure!(
            question_ids == haystack_ids,
            "v2 haystack question keys do not exactly match questions.jsonl \
             (questions={} haystacks={})",
            question_ids.len(),
            haystack_ids.len()
        );
        Ok(map)
    }

    fn read_haystack_map(dir: &Path) -> anyhow::Result<HashMap<String, Vec<String>>> {
        Self::read_haystack_file(dir, Self::haystack_file()?)
    }

    fn read_trajectories(
        dir: &Path,
        needed: &HashSet<String>,
    ) -> anyhow::Result<HashMap<String, V2Trajectory>> {
        let path = dir.join("trajectories.jsonl");
        let file = std::fs::File::open(&path)
            .with_context(|| format!("open v2 trajectories {}", path.display()))?;
        let mut trajectories = HashMap::new();
        for (index, line) in std::io::BufReader::new(file).lines().enumerate() {
            let line =
                line.with_context(|| format!("read {} line {}", path.display(), index + 1))?;
            if line.trim().is_empty() {
                continue;
            }
            let header: V2TrajectoryId = serde_json::from_str(&line).with_context(|| {
                format!("parse trajectory id at {}:{}", path.display(), index + 1)
            })?;
            if !needed.contains(&header.id) {
                continue;
            }
            let trajectory: V2Trajectory = serde_json::from_str(&line).with_context(|| {
                format!(
                    "parse trajectory '{}' at {}:{}",
                    header.id,
                    path.display(),
                    index + 1
                )
            })?;
            validate_trajectory(&trajectory).with_context(|| {
                format!(
                    "validate trajectory '{}' at {}:{}",
                    header.id,
                    path.display(),
                    index + 1
                )
            })?;
            if trajectories
                .insert(trajectory.id.clone(), trajectory)
                .is_some()
            {
                anyhow::bail!("duplicate v2 trajectory id '{}'", header.id);
            }
        }

        let mut missing: Vec<_> = needed
            .iter()
            .filter(|id| !trajectories.contains_key(*id))
            .cloned()
            .collect();
        missing.sort();
        if !missing.is_empty() {
            anyhow::bail!(
                "v2 haystack references missing trajectories: {}",
                missing.join(", ")
            );
        }
        Ok(trajectories)
    }

    fn read_all_trajectories(dir: &Path) -> anyhow::Result<HashMap<String, V2Trajectory>> {
        let path = dir.join("trajectories.jsonl");
        let file = std::fs::File::open(&path)
            .with_context(|| format!("open v2 trajectories {}", path.display()))?;
        let mut trajectories = HashMap::new();
        for (index, line) in std::io::BufReader::new(file).lines().enumerate() {
            let line =
                line.with_context(|| format!("read {} line {}", path.display(), index + 1))?;
            if line.trim().is_empty() {
                continue;
            }
            let trajectory: V2Trajectory = serde_json::from_str(&line)
                .with_context(|| format!("parse {} line {}", path.display(), index + 1))?;
            validate_trajectory(&trajectory)
                .with_context(|| format!("validate {} line {}", path.display(), index + 1))?;
            let id = trajectory.id.clone();
            anyhow::ensure!(
                trajectories.insert(id.clone(), trajectory).is_none(),
                "duplicate v2 trajectory id '{id}'"
            );
        }
        Ok(trajectories)
    }

    fn state_text(state: &V2State) -> String {
        let mut parts = Vec::new();
        let step = value_text(&state.step);
        let mut header = String::new();
        if !step.is_empty() {
            header.push_str(&format!("[step {step}]"));
        }
        if let Some(url) = non_empty(Some(&state.url)) {
            if !header.is_empty() {
                header.push(' ');
            }
            header.push_str(url);
        }
        if !header.is_empty() {
            parts.push(header);
        }
        if let Some(thought) = non_empty(state.thought.as_deref()) {
            parts.push(format!("thought: {thought}"));
        }
        if let Some(action) = non_empty(state.action.as_deref()) {
            parts.push(format!("action: {action}"));
        }
        if let Some(tree) = non_empty(Some(&state.accessibility_tree)) {
            parts.push(format!("observation:\n{tree}"));
        }
        parts.push(format!("screenshot_locator: {}", state.screenshot));
        parts.join("\n")
    }

    fn build_session(
        trajectory: &V2Trajectory,
        max_states: Option<usize>,
    ) -> Vec<LongMemEvalMessage> {
        let mut session = Vec::new();
        session.push(LongMemEvalMessage {
            role: "trajectory".to_string(),
            content: format!(
                "domain: {}\nenvironment: {}\noutcome: {}\nstart_url: {}\ngoal: {}",
                trajectory.domain,
                trajectory.environment,
                trajectory.outcome,
                trajectory.start_url,
                trajectory.goal
            ),
            has_answer: false,
        });
        for state in trajectory
            .states
            .iter()
            .take(max_states.unwrap_or(usize::MAX))
        {
            let content = Self::state_text(state);
            if !content.is_empty() {
                session.push(LongMemEvalMessage {
                    role: "observation".to_string(),
                    content,
                    has_answer: false,
                });
            }
        }
        session
    }

    fn trajectory_ids_for_questions(
        questions: &[V2Question],
        haystacks: &HashMap<String, Vec<String>>,
        max_trajectories: Option<usize>,
    ) -> anyhow::Result<Vec<String>> {
        let mut ids = Vec::new();
        let mut seen = HashSet::new();
        for question in questions {
            let question_ids = haystacks.get(&question.id).ok_or_else(|| {
                anyhow::anyhow!(
                    "v2 haystack map has no entry for question '{}'",
                    question.id
                )
            })?;
            for id in question_ids
                .iter()
                .take(max_trajectories.unwrap_or(usize::MAX))
            {
                if seen.insert(id.clone()) {
                    ids.push(id.clone());
                }
            }
        }
        Ok(ids)
    }

    fn shared_trajectory_ids(
        questions: &[V2Question],
        haystacks: &HashMap<String, Vec<String>>,
        max_trajectories: Option<usize>,
    ) -> anyhow::Result<Vec<String>> {
        if Self::haystack_file()? != "lme_v2_small.json" {
            anyhow::bail!(
                "LongMemEval-v2 medium haystacks are per-question scoped and are not supported by \
                 the shared-corpus runner"
            );
        }
        let first = questions
            .first()
            .ok_or_else(|| anyhow::anyhow!("shared corpus has no questions"))?;
        let first_ids = haystacks.get(&first.id).ok_or_else(|| {
            anyhow::anyhow!("v2 haystack map has no entry for question '{}'", first.id)
        })?;
        for question in &questions[1..] {
            let ids = haystacks.get(&question.id).ok_or_else(|| {
                anyhow::anyhow!(
                    "v2 haystack map has no entry for question '{}'",
                    question.id
                )
            })?;
            if ids != first_ids {
                anyhow::bail!(
                    "v2 small corpus '{}' is not shared consistently: question '{}' differs from '{}'",
                    question.domain,
                    question.id,
                    first.id
                );
            }
        }
        Ok(first_ids
            .iter()
            .take(max_trajectories.unwrap_or(usize::MAX))
            .cloned()
            .collect())
    }

    fn record_from_trajectory_ids(
        question_id: String,
        question_type: Option<String>,
        question: String,
        answer: Option<Value>,
        trajectory_ids: Vec<String>,
        trajectories: &HashMap<String, V2Trajectory>,
        max_states: Option<usize>,
    ) -> LongMemEvalRecord {
        let mut session_ids = Vec::new();
        let mut sessions = Vec::new();
        for id in trajectory_ids {
            let trajectory = trajectories
                .get(&id)
                .expect("trajectory completeness checked before record construction");
            let session = Self::build_session(trajectory, max_states);
            if session.is_empty() {
                continue;
            }
            session_ids.push(id);
            sessions.push(session);
        }
        LongMemEvalRecord {
            question_id,
            question_type,
            question,
            question_date: None,
            answer,
            answer_session_ids: Vec::new(),
            // The v2 release has no trajectory timestamps. A stable sentinel prevents the source
            // hash from changing across reconstruction/resume.
            haystack_dates: vec!["1970/01/01 00:00".to_string(); sessions.len()],
            haystack_session_ids: session_ids,
            haystack_sessions: sessions,
        }
    }
}

pub fn longmemeval_v2_text_projection_metadata(path: &Path) -> anyhow::Result<Value> {
    let questions = LongMemEvalV2Text::read_all_questions(path)?;
    let mut excluded_ids: Vec<_> = questions
        .iter()
        .filter(|question| question.image.is_some())
        .map(|question| question.id.clone())
        .collect();
    excluded_ids.sort();
    Ok(serde_json::json!({
        "total_questions": questions.len(),
        "included_text_questions": questions.len() - excluded_ids.len(),
        "excluded_query_image_questions": excluded_ids.len(),
        "excluded_question_ids_fingerprint": crate::stable_hash(excluded_ids.join(",").as_bytes()),
    }))
}

/// Validate the complete released dataset before any paid/provider-backed execution.
pub fn validate_longmemeval_v2_text_release(path: &Path) -> anyhow::Result<()> {
    anyhow::ensure!(
        LongMemEvalV2Text::haystack_file()? == "lme_v2_small.json",
        "LongMemEval-V2 Medium uses question-specific corpora and is unsupported by the current \
         shared-corpus text projection; use the upstream multimodal harness"
    );
    let questions = LongMemEvalV2Text::read_all_questions(path)?;
    let image_count = questions
        .iter()
        .filter(|question| question.image.is_some())
        .count();
    anyhow::ensure!(
        questions.len() == 451 && image_count == 29,
        "LongMemEval-V2 release shape mismatch: expected 451 questions with 29 query images, \
         found {} questions with {} query images",
        questions.len(),
        image_count
    );
    let questions_by_id: HashMap<_, _> = questions
        .iter()
        .map(|question| (question.id.as_str(), question))
        .collect();
    let mut needed = HashSet::new();
    for file_name in ["lme_v2_small.json", "lme_v2_medium.json"] {
        let haystacks = LongMemEvalV2Text::read_haystack_file(path, file_name)?;
        for (question_id, trajectory_ids) in haystacks {
            let question = questions_by_id
                .get(question_id.as_str())
                .expect("exact haystack/question key equality checked by loader");
            for trajectory_id in trajectory_ids {
                needed.insert((trajectory_id, question.domain.clone()));
            }
        }
    }
    let trajectories = LongMemEvalV2Text::read_all_trajectories(path)?;
    anyhow::ensure!(
        trajectories.len() == 1_870,
        "LongMemEval-V2 release shape mismatch: expected 1870 trajectories, found {}",
        trajectories.len()
    );
    for (trajectory_id, expected_domain) in needed {
        let trajectory = trajectories
            .get(&trajectory_id)
            .expect("trajectory reference completeness checked by loader");
        anyhow::ensure!(
            trajectory.domain == expected_domain,
            "question haystack in domain '{expected_domain}' references trajectory '{}' in domain '{}'",
            trajectory.id,
            trajectory.domain
        );
    }
    Ok(())
}

impl BenchmarkLoader for LongMemEvalV2Text {
    fn id(&self) -> &'static str {
        crate::eligibility::LONGMEMEVAL_V2_TEXT_ID
    }

    fn manifest_tag(&self) -> &'static str {
        "longmemeval-v2-text-projection-v1"
    }

    fn default_dataset(&self) -> DatasetSource {
        DatasetSource::Path {
            rel_path: "runs/inputs/longmemeval-v2",
            download_url: None,
        }
    }

    fn load(&self, path: &Path, limit: Option<usize>) -> anyhow::Result<Vec<LongMemEvalRecord>> {
        let questions = Self::read_questions(path, limit)?;
        let haystacks = Self::read_haystack_map(path)?;
        let max_trajectories = env_cap("MEMBENCH_V2_MAX_TRAJ")?;
        let max_states = env_cap("MEMBENCH_V2_MAX_STATES")?;
        warn_if_capped(max_trajectories, max_states);
        let needed_ids =
            Self::trajectory_ids_for_questions(&questions, &haystacks, max_trajectories)?;
        let needed: HashSet<_> = needed_ids.into_iter().collect();
        let trajectories = Self::read_trajectories(path, &needed)?;

        questions
            .into_iter()
            .map(|question| {
                let ids = haystacks
                    .get(&question.id)
                    .expect("haystack completeness checked above")
                    .iter()
                    .take(max_trajectories.unwrap_or(usize::MAX))
                    .cloned()
                    .collect();
                Ok(Self::record_from_trajectory_ids(
                    question.id,
                    Some(question.question_type),
                    question.question,
                    Some(question.answer),
                    ids,
                    &trajectories,
                    max_states,
                ))
            })
            .collect()
    }

    fn grade_targets(&self, path: &Path) -> anyhow::Result<HashMap<String, GradeTarget>> {
        Ok(Self::read_questions(path, None)?
            .into_iter()
            .map(|question| {
                (
                    question.id,
                    GradeTarget {
                        question_type: Some(question.question_type),
                        gold_answer: Some(question.answer),
                        eval_function: Some(question.eval_function),
                    },
                )
            })
            .collect())
    }

    fn haystack_scope(&self) -> HaystackScope {
        HaystackScope::SharedCorpus
    }

    fn shared_questions(
        &self,
        path: &Path,
        limit: Option<usize>,
    ) -> anyhow::Result<Vec<SharedCorpusQuestion>> {
        Self::read_questions(path, limit)?
            .into_iter()
            .map(|question| {
                Ok(SharedCorpusQuestion {
                    id: question.id,
                    question: question.question,
                    question_type: Some(question.question_type),
                    reference_date: None,
                    corpus_key: question.domain,
                })
            })
            .collect()
    }

    fn corpus_record(&self, path: &Path, corpus_key: &str) -> anyhow::Result<LongMemEvalRecord> {
        let questions: Vec<_> = Self::read_questions(path, None)?
            .into_iter()
            .filter(|question| question.domain == corpus_key)
            .collect();
        if questions.is_empty() {
            anyhow::bail!("v2 dataset has no questions for corpus '{corpus_key}'");
        }
        let haystacks = Self::read_haystack_map(path)?;
        let max_trajectories = env_cap("MEMBENCH_V2_MAX_TRAJ")?;
        let max_states = env_cap("MEMBENCH_V2_MAX_STATES")?;
        warn_if_capped(max_trajectories, max_states);
        let ids = Self::shared_trajectory_ids(&questions, &haystacks, max_trajectories)?;
        let needed: HashSet<_> = ids.iter().cloned().collect();
        let trajectories = Self::read_trajectories(path, &needed)?;
        for trajectory in trajectories.values() {
            anyhow::ensure!(
                trajectory.domain == corpus_key,
                "v2 corpus '{corpus_key}' references trajectory '{}' from domain '{}'",
                trajectory.id,
                trajectory.domain
            );
        }
        Ok(Self::record_from_trajectory_ids(
            format!("corpus:{corpus_key}"),
            None,
            String::new(),
            None,
            ids,
            &trajectories,
            max_states,
        ))
    }
}

fn non_empty(value: Option<&str>) -> Option<&str> {
    value.map(str::trim).filter(|value| !value.is_empty())
}

fn validate_question(question: &V2Question) -> anyhow::Result<()> {
    for (name, value) in [
        ("id", question.id.as_str()),
        ("environment", question.environment.as_str()),
        ("question_type", question.question_type.as_str()),
        ("question", question.question.as_str()),
        ("eval_function", question.eval_function.as_str()),
    ] {
        anyhow::ensure!(
            !value.trim().is_empty(),
            "question {name} must be non-empty"
        );
    }
    anyhow::ensure!(
        matches!(question.domain.as_str(), "web" | "enterprise"),
        "question '{}' has invalid domain '{}'",
        question.id,
        question.domain
    );
    anyhow::ensure!(
        question
            .answer
            .as_str()
            .is_some_and(|value| !value.trim().is_empty()),
        "question '{}' answer must be a non-empty string",
        question.id
    );
    if let Some(image) = &question.image {
        anyhow::ensure!(
            !image.trim().is_empty(),
            "question '{}' image locator must be non-empty or null",
            question.id
        );
    }
    // Parse now so malformed/unknown/duplicate evaluator options fail dataset loading rather than
    // after an answer has been paid for.
    parse_grader_spec(&question.eval_function)?;
    Ok(())
}

fn validate_trajectory(trajectory: &V2Trajectory) -> anyhow::Result<()> {
    for (name, value) in [
        ("id", trajectory.id.as_str()),
        ("environment", trajectory.environment.as_str()),
        ("goal", trajectory.goal.as_str()),
        ("outcome", trajectory.outcome.as_str()),
        ("start_url", trajectory.start_url.as_str()),
    ] {
        anyhow::ensure!(
            !value.trim().is_empty(),
            "trajectory {name} must be non-empty"
        );
    }
    anyhow::ensure!(
        matches!(trajectory.domain.as_str(), "web" | "enterprise"),
        "trajectory '{}' has invalid domain '{}'",
        trajectory.id,
        trajectory.domain
    );
    anyhow::ensure!(
        matches!(trajectory.outcome.as_str(), "success" | "failure"),
        "trajectory '{}' has invalid outcome '{}'",
        trajectory.id,
        trajectory.outcome
    );
    anyhow::ensure!(
        !trajectory.states.is_empty(),
        "trajectory '{}' has no states",
        trajectory.id
    );
    let mut indices = HashSet::new();
    for state in &trajectory.states {
        anyhow::ensure!(
            !state.url.trim().is_empty()
                && !state.accessibility_tree.trim().is_empty()
                && !state.screenshot.trim().is_empty(),
            "trajectory '{}' state fields url/accessibility_tree/screenshot must be non-empty",
            trajectory.id
        );
        let index = value_text(&state.state_index);
        anyhow::ensure!(
            indices.insert(index),
            "trajectory '{}' contains duplicate state_index",
            trajectory.id
        );
    }
    Ok(())
}

fn value_text(value: &Value) -> String {
    match value {
        Value::String(value) => value.clone(),
        other => other.to_string(),
    }
}

fn env_cap(name: &str) -> anyhow::Result<Option<usize>> {
    match std::env::var(name) {
        Ok(value) => parse_cap_value(name, &value).map(Some),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(std::env::VarError::NotUnicode(_)) => {
            anyhow::bail!("{name} must be valid UTF-8")
        }
    }
}

fn parse_cap_value(name: &str, value: &str) -> anyhow::Result<usize> {
    let parsed = value
        .trim()
        .parse::<usize>()
        .with_context(|| format!("{name} must be a positive integer, got '{value}'"))?;
    anyhow::ensure!(parsed > 0, "{name} must be greater than zero");
    Ok(parsed)
}

fn warn_if_capped(max_trajectories: Option<usize>, max_states: Option<usize>) {
    if max_trajectories.is_some() || max_states.is_some() {
        eprintln!(
            "[longmemeval-v2] corpus capped for pipeline smoke only; max_traj={max_trajectories:?} max_states={max_states:?}"
        );
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GradeOutcome {
    Deterministic(bool),
    Unsupported(JudgeKind),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum JudgeKind {
    Abstention,
    Gotchas,
    Generic,
}

/// LongMemEval-v2 scorer plug-in for the provider-neutral multimodal apparatus.
///
/// Deterministic official evaluators run locally. Evaluators that require the official LLM judge
/// remain explicit errors here; the caller must supply the benchmark's judge-backed scorer rather
/// than silently substituting a different rubric.
#[derive(Clone, Copy, Debug, Default)]
pub struct LongMemEvalV2MultimodalScorer;

impl crate::multimodal::MultimodalScorer for LongMemEvalV2MultimodalScorer {
    fn scorer_id(&self) -> &str {
        "longmemeval-v2-official-local-v1"
    }

    fn score(
        &self,
        rule: &crate::multimodal::ScoringRule,
        gold: &str,
        answer: &str,
    ) -> anyhow::Result<bool> {
        match rule {
            crate::multimodal::ScoringRule::External { evaluator } => {
                match grade_v2(evaluator, gold, answer)? {
                    GradeOutcome::Deterministic(correct) => Ok(correct),
                    GradeOutcome::Unsupported(judge) => anyhow::bail!(
                        "LongMemEval-v2 evaluator '{}' requires the official '{}' judge capability",
                        evaluator,
                        judge.id()
                    ),
                }
            }
            _ => crate::multimodal::MultimodalScorer::score(
                &crate::multimodal::DeterministicScorer,
                rule,
                gold,
                answer,
            ),
        }
    }
}

impl JudgeKind {
    pub fn id(&self) -> &'static str {
        match self {
            Self::Abstention => "llm_abstention_checker",
            Self::Gotchas => "llm_gotchas_checker",
            Self::Generic => "unknown_eval_function",
        }
    }
}

pub fn grade_v2(eval_function: &str, gold: &str, hypothesis: &str) -> anyhow::Result<GradeOutcome> {
    let (head, options) = parse_grader_spec(eval_function)?;
    let answer = extract_boxed(hypothesis);
    Ok(match head.as_str() {
        "norm_phrase_set_match" => {
            GradeOutcome::Deterministic(phrase_set_match(gold, &answer, &options, false))
        }
        "norm_phrase_set_match_ordered" => {
            GradeOutcome::Deterministic(phrase_set_match(gold, &answer, &options, true))
        }
        "mc_choice_match" => GradeOutcome::Deterministic(mc_choice_match(gold, &answer, &options)),
        "mc_choice_set_match" => {
            GradeOutcome::Deterministic(mc_choice_set_match(gold, &answer, &options))
        }
        "llm_abstention_checker" => GradeOutcome::Unsupported(JudgeKind::Abstention),
        "llm_gotchas_checker" => GradeOutcome::Unsupported(JudgeKind::Gotchas),
        _ => GradeOutcome::Unsupported(JudgeKind::Generic),
    })
}

#[derive(Debug)]
struct GraderOptions {
    lower: bool,
    normalize_hyphen: bool,
    strip_punct: bool,
    separators: String,
    strip_chars: String,
    require_non_empty: bool,
}

impl GraderOptions {
    fn defaults() -> Self {
        Self {
            lower: true,
            normalize_hyphen: true,
            strip_punct: true,
            separators: ",;".to_string(),
            strip_chars: ".".to_string(),
            require_non_empty: true,
        }
    }
}

fn parse_grader_spec(spec: &str) -> anyhow::Result<(String, GraderOptions)> {
    anyhow::ensure!(
        !spec.trim().is_empty(),
        "eval function spec must be non-empty"
    );
    let mut parts = spec.split('|');
    let head = parts.next().unwrap_or_default().trim().to_string();
    anyhow::ensure!(
        !head.is_empty(),
        "eval function spec is missing its function name"
    );
    let supported = [
        "norm_phrase_set_match",
        "norm_phrase_set_match_ordered",
        "mc_choice_match",
        "mc_choice_set_match",
        "llm_abstention_checker",
        "llm_gotchas_checker",
    ];
    anyhow::ensure!(
        supported.contains(&head.as_str()),
        "unknown LongMemEval-v2 eval function '{head}'"
    );
    let mut options = GraderOptions::defaults();
    let mut seen = HashSet::new();
    for raw in parts {
        let part = raw.trim();
        anyhow::ensure!(!part.is_empty(), "empty eval function option in '{spec}'");
        let (key, value) = part
            .split_once('=')
            .ok_or_else(|| anyhow::anyhow!("invalid eval function option '{part}'"))?;
        let key = key.trim();
        let value = value.trim();
        anyhow::ensure!(!key.is_empty(), "invalid eval function option '{part}'");
        anyhow::ensure!(seen.insert(key), "duplicate eval function option '{key}'");
        let allowed = match head.as_str() {
            "norm_phrase_set_match" | "norm_phrase_set_match_ordered" => &[
                "lower",
                "normalize_hyphen",
                "strip_punct",
                "separators",
                "require_non_empty",
            ][..],
            "mc_choice_match" => &["strip_chars", "require_non_empty"][..],
            "mc_choice_set_match" | "llm_abstention_checker" | "llm_gotchas_checker" => {
                &["require_non_empty"][..]
            }
            _ => unreachable!("supported head checked above"),
        };
        anyhow::ensure!(
            allowed.contains(&key),
            "eval function '{head}' does not accept option '{key}'"
        );
        match key {
            "lower" => options.lower = parse_bool_option(key, value)?,
            "normalize_hyphen" => options.normalize_hyphen = parse_bool_option(key, value)?,
            "strip_punct" => options.strip_punct = parse_bool_option(key, value)?,
            "separators" => {
                anyhow::ensure!(!value.is_empty(), "separators must be non-empty");
                options.separators = value.to_string();
            }
            "strip_chars" => options.strip_chars = value.to_string(),
            "require_non_empty" => {
                options.require_non_empty = parse_bool_option(key, value)?;
            }
            _ => anyhow::bail!("unknown eval function option '{key}'"),
        }
    }
    Ok((head, options))
}

fn parse_bool_option(key: &str, value: &str) -> anyhow::Result<bool> {
    match value.to_ascii_lowercase().as_str() {
        "true" => Ok(true),
        "false" => Ok(false),
        _ => anyhow::bail!("eval function option '{key}' must be true or false, got '{value}'"),
    }
}

pub fn extract_boxed(text: &str) -> String {
    let marker = r"\boxed{";
    let Some(start) = text.rfind(marker) else {
        return text.trim().to_string();
    };
    let rest = &text[start + marker.len()..];
    let mut depth = 1usize;
    for (index, character) in rest.char_indices() {
        match character {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return rest[..index].trim().to_string();
                }
            }
            _ => {}
        }
    }
    text.trim().to_string()
}

fn normalize(value: &str, options: &GraderOptions) -> String {
    let mut normalized = if options.normalize_hyphen {
        value.replace(['-', '_'], " ")
    } else {
        value.to_string()
    };
    // The official helper always treats comma/semicolon as spaces before punctuation stripping.
    normalized = normalized.replace([',', ';'], " ");
    if options.strip_punct {
        let punctuation =
            regex::Regex::new(r"[^\w\s]").expect("static normalization regex is valid");
        normalized = punctuation.replace_all(&normalized, "").into_owned();
    }
    if options.lower {
        normalized = normalized.to_lowercase();
    }
    normalized.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn phrases(value: &str, options: &GraderOptions) -> Vec<String> {
    value
        .split(|character| options.separators.contains(character))
        .map(|phrase| normalize(phrase, options))
        .filter(|phrase| !phrase.is_empty())
        .collect()
}

fn phrase_set_match(gold: &str, answer: &str, options: &GraderOptions, ordered: bool) -> bool {
    let gold = phrases(gold, options);
    let normalized_answer = normalize(answer, options);
    if options.require_non_empty && (normalized_answer.is_empty() || gold.is_empty()) {
        return false;
    }
    if ordered {
        let mut offset = 0usize;
        for phrase in gold {
            let Some(relative) = word_bounded_find(&normalized_answer[offset..], &phrase) else {
                return false;
            };
            offset += relative + phrase.len();
        }
        return true;
    }
    gold.iter()
        .collect::<HashSet<_>>()
        .into_iter()
        .all(|phrase| word_bounded_find(&normalized_answer, phrase).is_some())
}

fn word_bounded_find(haystack: &str, needle: &str) -> Option<usize> {
    haystack.match_indices(needle).find_map(|(index, _)| {
        let before = haystack[..index].chars().next_back();
        let after = haystack[index + needle.len()..].chars().next();
        (!before.is_some_and(|character| character.is_alphanumeric() || character == '_')
            && !after.is_some_and(|character| character.is_alphanumeric() || character == '_'))
        .then_some(index)
    })
}

fn mc_choice_match(gold: &str, answer: &str, options: &GraderOptions) -> bool {
    let expected = gold.trim().to_uppercase();
    let choice_words =
        regex::Regex::new(r"(?i)\b(choice|option)\b").expect("static choice regex is valid");
    let candidate: String = choice_words
        .replace_all(answer, "")
        .chars()
        .filter(|character| !options.strip_chars.contains(*character))
        .collect::<String>()
        .trim()
        .to_uppercase();
    if options.require_non_empty && (expected.is_empty() || candidate.is_empty()) {
        return false;
    }
    candidate == expected
}

fn mc_choice_set_match(gold: &str, answer: &str, options: &GraderOptions) -> bool {
    fn letters(value: &str) -> HashSet<char> {
        const FILLER: &[&str] = &[
            "AND", "ANSWER", "ANSWERS", "CHOICE", "CHOICES", "FINAL", "LETTER", "LETTERS",
            "OPTION", "OPTIONS",
        ];
        value
            .to_uppercase()
            .split(|character: char| !character.is_ascii_alphabetic())
            .filter(|chunk| !chunk.is_empty() && !FILLER.contains(chunk))
            .flat_map(str::chars)
            .collect()
    }
    let expected = letters(gold);
    let candidate = letters(answer);
    if options.require_non_empty && (expected.is_empty() || candidate.is_empty()) {
        return false;
    }
    expected == candidate
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boxed_extraction_uses_last_complete_box_and_handles_nesting() {
        assert_eq!(
            extract_boxed(r"first \boxed{a} then \boxed{b {c}}"),
            "b {c}"
        );
        assert_eq!(extract_boxed("plain"), "plain");
    }

    #[test]
    fn invalid_caps_fail_closed() {
        assert_eq!(parse_cap_value("CAP", "2").unwrap(), 2);
        for invalid in ["", "0", "-1", "not-a-number"] {
            assert!(parse_cap_value("CAP", invalid).is_err(), "{invalid}");
        }
    }

    #[test]
    fn unknown_grader_is_typed_as_unsupported() {
        assert_eq!(
            grade_v2("future_checker", "gold", "answer")
                .unwrap_err()
                .to_string(),
            "unknown LongMemEval-v2 eval function 'future_checker'"
        );
    }
}
