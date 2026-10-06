# v2 Multimodal Recall Experiment — Lanes, Model Slots, Ladder

> Implementation note (2026-08-22): the provider-neutral A–E contract, offline control, fixture
> schema, oracle evidence regions, cost gates, and fired proofs now live in
> [`../multimodal-recall-apparatus.md`](../multimodal-recall-apparatus.md). Model/provider names in
> this historical design are hypotheses, not current availability or pricing claims. Native arms
> remain capability-gated until a product adapter implements the public seam.

**Scope:** the concrete, bench-first prove-out of multimodal recall on **LongMemEval-v2**. This doc
rides on [`04-benchmark-multimodal-core.md`](04-benchmark-multimodal-core.md) — 04 is the *plumbing*
(the `Content=[ContentPart]` spine, `MediaRef`+sha256 media store, `BenchmarkManifest`, the
capability gate, and the "text-degenerate-first, don't flatten at the symbiotic-memory boundary"
sequencing). **This doc (06) is the *experiment* that rides that plumbing:** which recall lanes we
build, how we measure whether the native lane earns its place, the exact model slots, and the ladder.

Do not re-litigate 04 here. Where 06 needs a substrate primitive (`Content`, `MediaRef`, media store,
capability gate), it references 04.

**Locked upstream decisions this depends on:**
- Content spine is text-degenerate-first; media is additive (04 §4).
- Media stored as pointer (`MediaRef{locator, sha256, format}`), base64 only at call-time (04 §4).
- Multimodal ingest/answer/rerank is a **coordinated symbiotic-memory change** — text lane needs none (04 §6 DON'T).

---

## 1. Target & method (locked)

- **Bench-first, on LongMemEval-v2.** Prove the mechanism on v2's images before hardening it in the
  product engine for the apps. v2 validates **text + image** only — there is no audio/video in v2, so
  the pdf/video extractors are real product scope but **unprovable on this bench.** Don't let "prove on
  v2" imply v2 validates video.
- **Control-first.** Build the text-projection lane (the control) before the native lane (the
  treatment). A native-lane win only counts against a genuinely strong text baseline.
- **Isolate on the multimodal subset.** The headline metric is accuracy on the **image-dependent
  question subset**, lane vs lane — never the aggregate (the text-heavy majority washes out the signal).
- **Oracle ceiling before learned routing** (the [[oracle-ceiling-first-method]] move): fix the best
  lane per question with gold labels (cell D) to get the ceiling any router chases, *then* measure a
  router against it.

## 2. v2 ground truth (verified from the HF `SCHEMA.md`)

**`questions.jsonl`** — `{id, domain: web|enterprise, environment, question_type, question,
image: path|null, answer, eval_function}`. `image` is an optional screenshot path (multimodal Qs).
`eval_function` is a per-question evaluator spec (not a single global judge prompt).

**`trajectories.jsonl`** — `{id, domain, environment, goal, outcome: success|failure, start_url,
states: [...] }`. Each **state** = `{state_index, step, url, action|null, thought|null,
accessibility_tree, screenshot: "screenshots/<trajectory_id>/<step>.png"}`.

**Linking:** haystack files `lme_v2_small.json` / `lme_v2_medium.json` map `question_id → [trajectory_id…]`.
Screenshots ship **by local path** in `.tar.gz` bundles (`question_screenshots/`,
`trajectory_screenshots/`) — extractor front door is a **local byte-reader**, not a URL fetcher, not inline.

**The gift — `accessibility_tree`.** Every web state already carries a **text** observation (the a11y
tree) plus `thought`/`action` text. So for the **web** domain the text-projection lane is *largely
free* — no OCR needed; the a11y tree *is* the text projection. The screenshot is precisely what the
native lane adds on top. The interesting image-dependent subset is where the answer lives in the
**rendered pixels** (a chart trend, a visual state) that the a11y tree doesn't capture. (Enterprise
domain may have weaker/absent a11y text — confirm during the loader build.)

**Loader is a new record type**, not a tweak to `LongMemEvalRecord` — v2's trajectory/state shape and
`question.image` differ from v1's `haystack_sessions` chat turns. Under 04 this is a new
`BenchmarkLoader` registered in the manifest (04 §5).

## 3. The lanes and the cell matrix (locked)

A **lane** is a recall path from stored evidence to reader-consumable context.

| Lane | Stored unit → recall | Reader gets |
|---|---|---|
| **Text projection** | a11y-tree / thought / action text (+ VLM transcription of screenshots where the a11y tree is thin) → text embed → rerank | text |
| **Native image** | screenshot blob → image embed (Nemotron Embed VL) → VL rerank | text projection of the hit **or** the blob (see cell E) |

**Cells to run (the experiment):**

| Cell | Recall from | Question it answers |
|---|---|---|
| **A · text-only** | text projection | control / baseline — how far does extract-to-text get? |
| **B · native-only** | native image | does image retrieval stand alone? |
| **C · combined + collapse** | both, deduped | the real system |
| **D · oracle-lane** | gold modality label picks the lane per question | **the ceiling** C and any router chase |
| **E · reader modality** | (on C) reader reads *transcription* vs reads the *blob* | does letting the reader see the pixels beat a good transcription of them? |

**Collapse rule (cell C).** Same shape as today's collapse-to-raw-turn, but the key is
**underlying-blob-region identity** (the `MediaRef.sha256` from 04). When a text-projection segment and
a native hit derive from the same blob region, they are **one** piece of evidence: occupy one context
slot, keep the **text projection as the readable representative** for a text reader, and retain the
native hit only as the **locator + blob pointer** (this is the "recall the picture" case — native
locates, blob displays, transcription reads). Collapse is *collapse-to-representative*, not drop.

## 4. Model slots (locked; free unless noted)

All verified on OpenRouter July 2026. Native lane runs through the *same OpenRouter surface already used
for the qwen text embedder* — OpenRouter's embeddings API accepts `image_url` multimodal input.

| Slot | Model | Notes |
|---|---|---|
| Text embed (existing) | qwen3-8b @1024 | unchanged; the text-lane spine |
| **Native image embed** | `nvidia/llama-nemotron-embed-vl-1b-v2:free` | **$0**, 2048-dim, embeds image/text/combined; tuned for "images containing text, tables, charts, infographics" = v2 screenshots |
| **Native rerank** | existing **VL reranker** | already VL → native lane's rank slot is free (no new model to stand up) |
| **Transcription** (text projection of image-only / chart-dense states) | `nvidia/nemotron-nano-12b-v2-vl:free` | **$0**; VLM *transcribe+describe*, not glyph OCR — see §5 |
| Reader (cell E variable) | text reader (existing) **vs** a VL reader over the blob | E isolates transcription-read vs pixel-read |
| Cheap OCR fallback (optional) | Tesseract / PaddleOCR | deterministic, for text-dense enterprise states if VLM cost/latency bites |

## 5. OCR is a separate model (finding)

The embedder and reranker **embed and rank; neither emits text.** Nemotron Embed VL "understands" a
chart but returns a 2048-dim vector, not a string. Producing readable text from a screenshot is a
**third, distinct step** — needed unconditionally for the text-projection control lane (text embedder +
reader + collapse representative all need text), and needed on the native lane *unless the reader is VL*
(cell E's blob-read path).

**Use a VLM transcriber, not glyph OCR.** v2 screenshots are dashboards/tables/UIs where the answer
often lives in a chart trend or table relationship that glyph-OCR flattens to stray labels. A VLM
(`nemotron-nano-12b-v2-vl:free`) transcribes *and describes*, preserving that. Keep Tesseract/Paddle as
the cheap deterministic path for text-dense enterprise states only. **Consequence, stated honestly:** a
semantic VLM transcription makes the text control *strong*, so the native lane must beat a good
baseline — that's the experiment working, not a problem.

## 6. Runtime lane knob + router (locked direction)

- **Recall shape becomes a per-query request field, not a boot flag.** Raw-vs-distilled and
  lane-selection differ by question ("what did the chart show" wants native+blob; "my recurring
  preference" wants the distilled fact), so `{lanes, distill_level}` rides on the recall request
  alongside the existing `with_optional_forced_context` / `with_recall_tuning` machinery at the
  `answer_debug_with_reference_date` entry — not `SYMEM_*` process env. Cost of this (bench-discipline):
  we're now scoring a *policy*, not a config — which is exactly why cell D exists.
- **Build overlays generously at ingest; route only at read.** Raw is the moat; derived lanes are
  rebuildable. A router that gates *ingest* leaves holes you can't recall from; a router that chooses at
  *read* is free to redo. For a bounded v2 dataset, over-build all lanes once, route among what exists.
- **Decision placement:** the *caller* supplies intent (not lane names); a **read-time router** picks
  lane + distill level; and — bench-only — **question-generation supplies the labels** (the generator
  knows each question's gold modality, so it emits the router's train/eval labels for free).
- **Oracle router first (cell D, no model), learned router second.** A ~120M tool-selection classifier
  earns its place only if D beats best-fixed-C by a margin worth chasing; its job is then to close
  `D − C`, and its own error is the metric. **Do not train it before that gap is shown to exist.**

## 7. The ladder (sequencing)

| Phase | Work | Where | Engine change? | Yields |
|---|---|---|---|---|
| **0** | v2 loader (new `BenchmarkLoader`) + text-projection from a11y-tree/thought/action (+ VLM transcription of screenshots) into the existing text seam | **adapter only** | **none** | **cell A** (shippable number) |
| **1** | promote `RawArchiveReceipt` blob to addressable via `MediaRef.sha256` (media store, 04 §4) | engine, small | small | picture recall |
| **2** | typed media part on `SourceTurn` (04 `Content` spine) + native image lane + collapse-to-representative | engine (coordinated, 04 §6) | yes | **cells B, C** |
| **3** | recall-shape as request field → oracle-lane ceiling (D) → reader-modality cell (E) → 120M router | engine + harness | yes | **cells D, E**, router-vs-ceiling |

**Phase 0 needs zero symbiotic-memory change** — the whole content boundary is `longmemeval_to_source`
(`src/symbiotic_memory_adapter.rs:258-305`) copying a `String` into `SourceTurn.text`. A v2 loader that
emits a11y-tree/thought/action (+ transcription) text into that same seam gets cell A with everything
downstream untouched. This is the de-risked first brick and it forces the extractor interface into
existence where every later lane plugs in.

## 8. Seam anchors (for implementation)

- Content boundary (Phase 0 insertion): `longmemeval_to_source` — `src/symbiotic_memory_adapter.rs:258-305`.
- Blob archive to promote (Phase 1): `RawArchiveReceipt{media_type, digest, size_bytes}` —
  `../symbiotic-memory/src/types.rs:297`; `.with_archive_root(&vault_dir)` at the ingest call sites.
- Text seam that grows to a media part (Phase 2): `SourceTurn.text: String` —
  `../symbiotic-memory/src/types.rs:311-350` (04's `Content` lands here).
- Recall entry for the per-query lane field (Phase 3): `RecallEngine::answer_debug_with_reference_date`
  — `src/symbiotic_memory_adapter.rs:1639`; existing per-call shaping via `with_optional_forced_context`
  / `with_recall_tuning`.
- No harness-side engine trait — harness binds directly to engine `SourceDocument`/`SourceTurn`
  (path dep, per the until-v1 policy); envelope changes recompile straight through.

## 8b. Shared-corpus scaffolding (the real v2 run model) — LOCKED 2026-07-08

Building the loader surfaced the defining fact about v2 (from the HF `SCHEMA.md`): **"within each
domain, all questions share one 100-trajectory haystack."** So the *small* tier is **two shared
corpora** — `web` (100 trajectories) and `enterprise` (100 trajectories), union = 200 (verified). Every
web question answers against the same web corpus; every enterprise question against the same enterprise
corpus. (Medium tier is ~500 trajectories/question and *not* fully shared — it needs per-question
scoping, deferred.)

Two consequences that make a naive port wrong, and define the scaffolding to build **properly**:

1. **Ingest-once / answer-many is mandatory, not an optimization.** A v2 small haystack is
   **~3,358 states / ~80 MB / ~20 M tokens** of a11y text. The per-question-vault model (v1) would
   re-embed that shared corpus for *every* question → 20 M × 451 ≈ 9 B tokens. The correct model
   ingests each corpus **once** (~40 M tokens total for the 200-trajectory small pool) into a shared
   vault, then runs recall+answer per question against it. This is the same capability the
   document-memory product needs (a corpus many queries hit), so it is built as a **general
   benchmark capability**, not a v2 special-case.

2. **v2 is answer-graded, not evidence-graded.** The dataset ships **no gold-evidence labels** (no
   `has_answer`, no gold-trajectory marker); scoring is `eval_function(answer)` only. So the v1
   gold-coverage/rank machinery simply does not apply to v2 — `answer_session_ids=[]` /
   `has_answer=false` is correct, and there is no gold-marking work to do.

**The general model — `HaystackScope` on a benchmark:**

| Scope | Meaning | Benchmarks |
|---|---|---|
| `PerQuestion` | each question ingests its own haystack (current behavior) | LongMemEval-v1, LoCoMo-per-Q |
| `SharedCorpus` | a corpus keyed by `corpus_key` ingested once; questions recall against their corpus | LongMemEval-v2 (`corpus_key = domain`), document-memory tests |

`BenchmarkLoader` gains `haystack_scope()`. For `SharedCorpus` a question carries a **`corpus_key`**
(v2: its `domain`); the run groups questions by `corpus_key`, ingests each corpus once into a shared
vault, and answers each question against that vault. (Small tier needs no per-question subset scoping —
each question's haystack IS its domain's full 100-trajectory corpus; medium-tier subset scoping is a
later refinement via recall-time source filtering.)

**Run mode** rides membench's existing ingest-once/answer-many machinery: build the corpus vault(s)
once (like the golden source vaults), then run **answer-only** per question against the domain vault.
The corpus vault is keyed by `corpus_key` + source-hash, so it is reused across questions and across
runs. Mechanism anchors are being mapped (source-vault-root / answer-only / manifest source-hash reuse)
before wiring.

**Cost ladder:** the one-time small-corpus embed (~40 M tokens) is the price of a real v2 run,
amortized across all 451 questions. The `MEMBENCH_V2_MAX_TRAJ/_MAX_STATES` cap stays a **dev smoke
only** (it can drop distractors and distort the corpus) — never the real score.

## 9. Open items (confirm during Phase 0)

1. **Enterprise-domain a11y text**: is `accessibility_tree` present/rich for `enterprise` states, or do
   those need VLM transcription? Decides how much of cell A leans on the transcriber.
2. **v2 download**: tarball size + whether HF auth is needed (discrete, outward action — confirm before pulling).
3. **`eval_function` shape**: per-question evaluator spec — maps onto 04's `ScorerSpec`/`JudgeSpec`; read a few before wiring the grader.
4. **Image-dependent subset definition**: which v2 questions actually require the pixels vs are answerable from a11y text — this subset *is* the headline metric's denominator; derive it, don't guess.
