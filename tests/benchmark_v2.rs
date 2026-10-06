#![cfg(feature = "symbiotic-memory-adapter")]

use std::fs;

use membench::benchmark::{
    BenchmarkLoader, GradeOutcome, HaystackScope, JudgeKind, LongMemEvalV2MultimodalScorer,
    LongMemEvalV2Text, grade_v2, loader_for,
};
use membench::eligibility::NON_PROMOTABLE_BENCHMARKS;
use membench::multimodal::{MultimodalScorer, ScoringRule};
use serde_json::json;

fn write_fixture() -> tempfile::TempDir {
    let root = tempfile::tempdir().unwrap();
    fs::create_dir(root.path().join("haystacks")).unwrap();
    fs::write(
        root.path().join("questions.jsonl"),
        [
            json!({
                "id": "web-1",
                "domain": "web",
                "environment": "webarena",
                "question_type": "list",
                "question": "Which portals?",
                "image": null,
                "answer": "Alpha, Beta",
                "eval_function": "norm_phrase_set_match|lower=true|normalize_hyphen=true|strip_punct=true|separators=,;|require_non_empty=true"
            })
            .to_string(),
            json!({
                "id": "enterprise-1",
                "domain": "enterprise",
                "environment": "workarena",
                "question_type": "choice",
                "question": "Which option?",
                "image": null,
                "answer": "B",
                "eval_function": "mc_choice_match|require_non_empty=true"
            })
            .to_string(),
            json!({
                "id": "web-2",
                "domain": "web",
                "environment": "webarena",
                "question_type": "list",
                "question": "Which mobile portal?",
                "image": null,
                "answer": "Mobile Portal",
                "eval_function": "norm_phrase_set_match|lower=true|require_non_empty=true"
            })
            .to_string(),
        ]
        .join("\n")
            + "\n",
    )
    .unwrap();
    fs::write(
        root.path().join("haystacks/lme_v2_small.json"),
        serde_json::to_string_pretty(&json!({
            "web-1": ["web-t1", "web-t2"],
            "enterprise-1": ["enterprise-t1"],
            "web-2": ["web-t1", "web-t2"]
        }))
        .unwrap(),
    )
    .unwrap();
    fs::write(
        root.path().join("trajectories.jsonl"),
        [
            json!({
                "id": "web-t1",
                "domain": "web",
                "environment": "webarena",
                "goal": "open incidents",
                "outcome": "success",
                "start_url": "https://example.test/incidents",
                "states": [{
                    "step": 1,
                    "state_index": 0,
                    "url": "https://example.test/incidents",
                    "thought": "inspect the list",
                    "action": "click Open",
                    "accessibility_tree": "heading Incident Portal",
                    "screenshot": "screenshots/web-t1/0.png"
                }]
            })
            .to_string(),
            json!({
                "id": "web-t2",
                "domain": "web",
                "environment": "webarena",
                "goal": "open mobile portal",
                "outcome": "failure",
                "start_url": "https://example.test/mobile",
                "states": [{"step": 2, "state_index": 0, "url": "https://example.test/mobile", "action": null, "thought": null, "accessibility_tree": "heading Mobile Portal", "screenshot": "screenshots/web-t2/0.png"}]
            })
            .to_string(),
            json!({
                "id": "enterprise-t1",
                "domain": "enterprise",
                "environment": "workarena",
                "goal": "choose B",
                "outcome": "success",
                "start_url": "https://example.test/choice",
                "states": [{"step": 1, "state_index": 0, "url": "https://example.test/choice", "action": null, "thought": null, "accessibility_tree": "radio B selected", "screenshot": "screenshots/enterprise-t1/0.png"}]
            })
            .to_string(),
        ]
        .join("\n")
            + "\n",
    )
    .unwrap();
    root
}

#[test]
fn registry_preserves_v1_and_adds_v2() {
    assert_eq!(loader_for("long-mem-eval").unwrap().id(), "long-mem-eval");
    assert!(loader_for("longmemeval-v2").is_none());
    let v2 = loader_for("longmemeval-v2-text").unwrap();
    assert_eq!(v2.id(), "longmemeval-v2-text");
    assert!(NON_PROMOTABLE_BENCHMARKS.contains(&v2.id()));
    assert_eq!(v2.haystack_scope(), HaystackScope::SharedCorpus);
    assert!(loader_for("unknown").is_none());
}

#[test]
fn v2_loader_projects_shared_domain_corpus_into_text_turns() {
    let fixture = write_fixture();
    let loader = LongMemEvalV2Text;
    let questions = loader.shared_questions(fixture.path(), Some(1)).unwrap();
    assert_eq!(questions.len(), 1);
    assert_eq!(questions[0].id, "web-1");
    assert_eq!(questions[0].corpus_key, "web");

    let corpus = loader.corpus_record(fixture.path(), "web").unwrap();
    assert_eq!(corpus.question_id, "corpus:web");
    assert_eq!(corpus.haystack_session_ids, ["web-t1", "web-t2"]);
    assert_eq!(corpus.haystack_sessions.len(), 2);
    let first = &corpus.haystack_sessions[0];
    assert_eq!(first[0].role, "trajectory");
    assert!(first[0].content.contains("outcome: success"));
    assert!(first[0].content.contains("environment: webarena"));
    assert!(first[1].content.contains("heading Incident Portal"));
    assert!(first[1].content.contains("action: click Open"));
    assert!(
        first[1]
            .content
            .contains("screenshot_locator: screenshots/web-t1/0.png")
    );
    assert_eq!(corpus.haystack_dates, ["1970/01/01 00:00"; 2]);
}

#[test]
fn v2_loader_fails_closed_when_haystack_trajectory_is_missing() {
    let fixture = write_fixture();
    let path = fixture.path().join("trajectories.jsonl");
    let raw = fs::read_to_string(&path).unwrap();
    fs::write(
        &path,
        raw.lines()
            .filter(|line| !line.contains("\"web-t2\""))
            .collect::<Vec<_>>()
            .join("\n")
            + "\n",
    )
    .unwrap();

    let err = LongMemEvalV2Text
        .corpus_record(fixture.path(), "web")
        .unwrap_err();
    assert!(err.to_string().contains("web-t2"));
}

#[test]
fn v2_loader_rejects_a_non_shared_small_domain_haystack() {
    let fixture = write_fixture();
    fs::write(
        fixture.path().join("haystacks/lme_v2_small.json"),
        serde_json::to_string_pretty(&json!({
            "web-1": ["web-t1", "web-t2"],
            "enterprise-1": ["enterprise-t1"],
            "web-2": ["web-t2"]
        }))
        .unwrap(),
    )
    .unwrap();

    let err = LongMemEvalV2Text
        .corpus_record(fixture.path(), "web")
        .unwrap_err();
    assert!(err.to_string().contains("not shared consistently"));
}

#[test]
fn v2_text_projection_excludes_query_image_questions() {
    let fixture = write_fixture();
    let path = fixture.path().join("questions.jsonl");
    let mut raw = fs::read_to_string(&path).unwrap();
    raw.push_str(
        &(json!({
            "id": "image-1",
            "domain": "web",
            "environment": "webarena",
            "question_type": "errors-gotchas",
            "question": "What is shown?",
            "image": "question_screenshots/image-1.png",
            "answer": "a warning",
            "eval_function": "llm_gotchas_checker|require_non_empty=true"
        })
        .to_string()
            + "\n"),
    );
    fs::write(path, raw).unwrap();
    let questions = LongMemEvalV2Text
        .shared_questions(fixture.path(), None)
        .unwrap();
    assert_eq!(questions.len(), 3);
    assert!(questions.iter().all(|question| question.id != "image-1"));
}

#[test]
fn v2_deterministic_graders_and_judge_routing_are_typed() {
    let set =
        "norm_phrase_set_match|lower=true|strip_punct=true|separators=,;|require_non_empty=true";
    assert_eq!(
        grade_v2(set, "Alpha, Beta", r"\boxed{beta; alpha}").unwrap(),
        GradeOutcome::Deterministic(true)
    );
    assert_eq!(
        grade_v2(
            "norm_phrase_set_match_ordered|lower=true|separators=;",
            "alpha; beta",
            r"\boxed{beta; alpha}",
        )
        .unwrap(),
        GradeOutcome::Deterministic(false)
    );
    assert_eq!(
        grade_v2(
            "norm_phrase_set_match|lower=true|normalize_hyphen=true|strip_punct=true|separators=,;|require_non_empty=true",
            "alpha-beta",
            r"\boxed{extra words alpha beta and more}",
        )
        .unwrap(),
        GradeOutcome::Deterministic(true)
    );
    assert_eq!(
        grade_v2(
            "norm_phrase_set_match_ordered|lower=true|normalize_hyphen=true|strip_punct=true|separators=,;|require_non_empty=true",
            "alpha; gamma",
            r"\boxed{alpha with beta before gamma}",
        )
        .unwrap(),
        GradeOutcome::Deterministic(true)
    );
    assert_eq!(
        grade_v2(
            "norm_phrase_set_match|lower=true|normalize_hyphen=true|strip_punct=true|separators=;|require_non_empty=true",
            "foobar",
            r"\boxed{foo.bar}",
        )
        .unwrap(),
        GradeOutcome::Deterministic(true)
    );
    assert_eq!(
        grade_v2(
            "mc_choice_match|require_non_empty=true",
            "B",
            r"\boxed{Option B}"
        )
        .unwrap(),
        GradeOutcome::Deterministic(true)
    );
    assert_eq!(
        grade_v2(
            "mc_choice_match|require_non_empty=true",
            "B",
            r"\boxed{Option.B}"
        )
        .unwrap(),
        GradeOutcome::Deterministic(true)
    );
    assert_eq!(
        grade_v2("mc_choice_set_match", "A,B,F", r"\boxed{F, A, B}").unwrap(),
        GradeOutcome::Deterministic(true)
    );
    assert_eq!(
        grade_v2(
            "mc_choice_set_match",
            "A,B",
            r"\boxed{Final answer: A and B}"
        )
        .unwrap(),
        GradeOutcome::Deterministic(true)
    );
    assert_eq!(
        grade_v2("mc_choice_set_match", "A,B,F", r"\boxed{A, B}").unwrap(),
        GradeOutcome::Deterministic(false)
    );
    assert_eq!(
        grade_v2("llm_gotchas_checker", "expected", "candidate").unwrap(),
        GradeOutcome::Unsupported(JudgeKind::Gotchas)
    );
    assert_eq!(
        grade_v2("future_checker", "expected", "candidate")
            .unwrap_err()
            .to_string(),
        "unknown LongMemEval-v2 eval function 'future_checker'"
    );
}

#[test]
fn v2_multimodal_scorer_uses_official_deterministic_rules_and_fails_on_missing_judge() {
    let scorer = LongMemEvalV2MultimodalScorer;
    assert!(
        scorer
            .score(
                &ScoringRule::External {
                    evaluator: "mc_choice_match|require_non_empty=true".to_string(),
                },
                "B",
                r"\boxed{Option B}",
            )
            .unwrap()
    );
    let error = scorer
        .score(
            &ScoringRule::External {
                evaluator: "llm_gotchas_checker|require_non_empty=true".to_string(),
            },
            "expected",
            "candidate",
        )
        .unwrap_err();
    assert!(error.to_string().contains("requires the official"));
}

#[test]
fn v2_eval_specs_reject_unknown_duplicate_and_malformed_options() {
    for spec in [
        "norm_phrase_set_match|mystery=true",
        "norm_phrase_set_match|lower=true|lower=false",
        "norm_phrase_set_match|lower",
        "norm_phrase_set_match|lower=perhaps",
    ] {
        assert!(grade_v2(spec, "a", "a").is_err(), "{spec}");
    }
}

#[test]
fn v2_source_projection_is_hash_stable() {
    let fixture = write_fixture();
    let record = LongMemEvalV2Text
        .corpus_record(fixture.path(), "web")
        .unwrap();
    let first = membench::symbiotic_memory_adapter::longmemeval_to_source(&record);
    let second = membench::symbiotic_memory_adapter::longmemeval_to_source(&record);
    assert_eq!(
        serde_json::to_value(first).unwrap(),
        serde_json::to_value(second).unwrap()
    );
}
