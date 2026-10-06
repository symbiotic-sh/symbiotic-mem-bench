# Agent Instructions

Base rules: [House Rules](https://github.com/jak-pan/house-rules), its `AGENTS.md` (layout and naming in its `STRUCTURE.md`). Read and follow them first. This file holds only this repository's own rules. An override may tighten or loosen a House Rules rule, and it names the rule it changes.

Reviews: Warden, our review service, reviews pull requests on request: comment `/warden review` on the pull request.

This repository owns neutral benchmark orchestration, scoring, records, and dashboard artifacts.
Memory implementation behavior belongs in the system under test. Do not restore benchmark
subcommands in `symem`.

## Project Boundaries

- The public root package builds the core, server, and leaderboard without private dependencies.
  The native `membench` CLI is the separate, non-workspace
  `adapters/symbiotic-memory/Cargo.toml` package. It resolves exact Git revisions; sibling code
  changes affect it only with a deliberate co-development override (see
  [docs/environment.md](docs/environment.md)).
- Run benchmark orchestration through that adapter CLI, not Python scoring scripts or manual
  score entry. No gold-string matching, per-question special cases, or answer-key-driven tuning.
- Do not commit `runs/`, `.debug-session/`, `target/`, provider queues, or local datasets.
  Never commit raw prompts or provider payloads; retain them only in ignored local debug storage.
  Keep native state out of tracked records.
  Public records use repo-relative paths and omit raw prompts and provider payloads. Missing
  artifacts stay explicitly missing in `artifact_manifest`.
- GitHub Actions and release bundles run on Linux, including the private adapter's zvec gate.
- Use `CARGO_TARGET_DIR=/tmp/symbiotic-mem-bench-target` for publication validation so it does
  not recreate `target/` in the checkout. Paid benchmarks use release builds.
- Model/provider defaults belong to the pinned Memory code/config and foundation model catalog.
  Harness profiles describe benchmark policy or an explicitly named tuning arm; do not mirror
  production model defaults into profiles.

## Validate

CI (`.github/workflows/ci.yml`) owns the full core and server-feature suites. Local checks
use the same debug profile; replace `<filter>` with a test/module name for the changed area
(for example, `registry::tests`). See House Rules §Verification for full-suite exceptions.

```bash
CARGO_TARGET_DIR=/tmp/symbiotic-mem-bench-target cargo fmt -- --check
CARGO_TARGET_DIR=/tmp/symbiotic-mem-bench-target cargo clippy --all-targets --features server -- -D warnings
CARGO_TARGET_DIR=/tmp/symbiotic-mem-bench-target cargo test <filter>
CARGO_TARGET_DIR=/tmp/symbiotic-mem-bench-target cargo test --features server <filter>
```

If you need to run from another repository, use:

```bash
CARGO_TARGET_DIR=/tmp/symbiotic-mem-bench-target \
  cargo test --manifest-path ../symbiotic-mem-bench/Cargo.toml <filter>
```

## Benchmark Workflow

Use [skills/membench/SKILL.md](skills/membench/SKILL.md) as the workflow entry point and
[its command reference](skills/membench/references/membench-commands.md) for native runs,
imports, inspection, promotion, and trial derivation. Environment variables and credential
file loading are described in [docs/environment.md](docs/environment.md).

- Native launches are paid and scored by default. Only explicit `--smoke` selects deterministic
  no-network providers and no scorer; it deletes successful temporary runs unless kept explicitly.
- Fresh runs reset their selected run root and re-ingest. `--resume` continues interrupted work;
  `--answer-only` reuses ingested state. Pass `--run-root` only for a deliberate named run.
- The persistent store is `--store zvec`. `sqlite` and `zvec-hybrid` are retired and rejected;
  old run roots need fresh ingestion. `memory` supports only simple unscored slices.
- Answer-only reruns use `workflow_max_in_flight=64` by default; ingest uses its configured window.
  Override either with `MEMBENCH_WORKFLOW_MAX_IN_FLIGHT`. Provider concurrency is still enforced
  by model queue id.
- Paid runs serialize through `runs/.locks/paid-provider-run.lock/`. Inspect `owner.json` and
  confirm the recorded process is dead before removing a stranded lock.

## Run Registry

Scratch: `runs/{system}/{benchmark}/{limit}/{run_name}/`.
Curated: `records/{system}/{benchmark}/{limit}/{run_name}/`.
Complete runs contain `run-params.json`, `benchmark-report.json`, and `artifacts/`.
Native scratch state additionally includes `raw/`, `vaults/`, `workflow/`, and `provider-queue/`.
See [docs/run-registry.md](docs/run-registry.md) for the artifact contract and
[docs/schemas.md](docs/schemas.md) for fields.

Use neutral run names describing the condition, such as `baseline-clean` or
`candidate-answer-thinking`. Trial ledgers are derived from run artifacts through `membench trials derive`; do not hand-edit them when source runs exist. Focused sub-25Q trials diagnose
one failure class; stratified 25–50Q trials support broader diagnosis. Only full benchmark
runs promoted to records can support benchmark claims. The dashboard marks ledger-referenced
runs `TRIAL`; the ranking gate is in [docs/longmemeval-methodology.md](docs/longmemeval-methodology.md).

## Async Adapter Contract

Preserve the system under test's native pipeline:

- no benchmark-owned duplicate ingestion, embedding, or recall;
- no artificial phase barrier between capture, distill, embedding, indexing, answer, and score;
- stage outputs and traces written incrementally, before acknowledging durable work;
- provider caps controlled by model queue id, not benchmark stage.

## Trace Interpretation

Provider/model trace facts:

- Native Symbiotic Memory provider calls are emitted under
  `provider-queue/model-queue-traces.jsonl`.
- New completed native runs also export that file as `artifacts/model-traces.jsonl`.
- Older runs may show `model_traces` missing in their manifest but still have provider traces under
  `provider-queue/`; the dashboard live/detail paths read that fallback directly.
- In the live monitor, provider queue counts are latest state per queue item, not cumulative event
  counts. `running` should drop after terminal events.
- Live per-queue `rpm`, `maxrpm`, and `peak` are observed from tailed provider events. They are
  useful for diagnosing pressure and async flow, but are not a replacement for full-run cost or
  provider billing summaries.
- Completed runs show queue `avg`/`avgrpm` summaries derived from provider event timing instead of
  emphasizing current active counts, which should naturally be zero after completion.
- The live activity stream should show memory-stage events and provider events interleaved. If stages
  appear wave-like in the bars, inspect activity before concluding the executor is sequential.
- The stage label `setup` maps to adapter `pre_capture_setup`: vault directory/manifest/hash,
  zvec cache validation, store open, and existing-state load before the memory pipeline emits
  `capture`.
- The stage label `recall setup` maps to adapter `pre_recall_setup`: post-ingest count loading and
  recall-index readiness before the answer/recall path starts.
- The stage label `briefs` maps to the trace operation `consolidate`, which is the source-backed
  brief pass.
- The stage label `prompt plan` maps to the trace operation `query_plan`. It is emitted from the
  memory engine's recall debug result; the benchmark must not run an extra planner call for display.
  Inspect `recall.query_planner_call` in the per-question debug bundle for the raw planner prompt
  and response; memory traces keep hashes/pointers.

Imported runs can be artifact-only. Their `artifact_manifest.native_state_available` must be `false`,
and their `artifact_manifest.missing` list must make absent traces or state folders explicit.

## Publication Hygiene

Use [Validate](#validate) for local checks and
[scripts/check-publication-hygiene.sh](scripts/check-publication-hygiene.sh) for the tracked-tree
scan. Inspect runs with `membench explore`; promotion defaults to portable artifacts without
native state. [RELEASING.md](RELEASING.md) owns the release checklist and private-adapter gate.

## Historical LongMemEval Campaign Decisions

Goal: 95% on LongMemEval-S 500Q with a clean, generic pipeline. The historical campaign log is
[PUSH-TO-95.md](PUSH-TO-95.md); its old env spellings, scratch commands, and running statuses
are historical, not launch instructions. Current commands come from the skill reference.

The forensic correction to that log's earlier “89.1% best stack” claim is retained here:
rerank-ON with Cohere `cohere/rerank-4-fast`, 100 candidates, the 8229-character answer prompt,
20 facts/10 raw turns, briefs, and DeepSeek Flash high reasoning measured about 88.3–88.5%.
These are campaign conditions, not the shipped defaults. The reranker was the demonstrated
lever (+3.34pp); low-confidence filtering at 0.7 removed no facts, and dedup removed only
0.43% before the 20-fact cap backfilled. The claimed +0.8pp stack had no demonstrated mechanism.
Conditional Chain-of-Note gains also collapsed under replication.

- Answerer variance was about ±1.5pp (std ~0.8); one run or N=2 deltas below ~2pp are inconclusive.
  Verify that a lever changed evidence/prompts, compare its deterministic subset, and use the
  untouched subset as a noise check. Inspect prompts, reasoning, flips, and raw source evidence.
- Rerank-ON retrieval had gold evidence in 51/52 missed questions' candidate pools. Counting/list
  questions were the weak class (82.4% versus 90.1% otherwise). Prompt-content changes did not
  improve that subset; cleaner evidence remained a hypothesis.
- Flash high reasoning was the settled reader; max was unstable (90.8/86.8), Pro worse (87%).
  Short/surgical/Chain-of-Note/split prompts, wider rerank, and smaller top-k were noise or worse.
  Deterministic count-in-code was shelved by operator decision.
- Removed levers (`SYMEM_EXCLUDE_BRIEFS`, `SYMEM_LEDGER_RETRIEVAL`, `SYMEM_DEDUP_EVIDENCE`,
  `SYMEM_DETERMINISTIC_COUNT`, `SYMEM_RERANK_RESERVE`) are historical only. Re-add only as typed
  `[experimental]` fields with referee evidence. Use the typed engine config and harness variables
  documented in `docs/environment.md` for new work.

The old campaign vaults and custom prompt directories are local prerequisites, absent in a
clean checkout. Historical measurements do not establish the current pipeline's score.
