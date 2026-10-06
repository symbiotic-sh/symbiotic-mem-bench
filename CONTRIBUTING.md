# Contributing

Thanks for your interest in membench — the neutral benchmark harness for memory systems.

This project follows [House Rules](https://github.com/jak-pan/house-rules) for collaboration, Git, verification and security. Repository rules are in [AGENTS.md](AGENTS.md).

## Benchmark integrity

The harness exists to publish **verifiable** benchmark results. Every contribution is bound by
the same integrity rules the maintainers follow:

- No benchmark-targeted hacks: no gold-string matching, no per-question special-casing, no
  tuning that reads the answer key. Levers land behind gated flags, off by default.
- No Python scoring scripts and no manual score entry. Scores come from the Rust pipeline and
  its judged artifacts.
- Public records exclude native state, provider payloads, and local datasets. Real `.env*` files
  are ignored; only the `*.example` templates are tracked.
- Tracked records must not contain raw prompts or secrets. Missing artifacts are declared
  missing in the manifest, never synthesized.
- Diagnostic experiment runs are `TRIAL`-flagged, not benchmark claims. Only records passing
  the review gate in `docs/longmemeval-methodology.md` may be ranked.

`AGENTS.md` is the full operating rulebook; on conflict it wins.

## Development setup

- Rust: pinned by `rust-toolchain.toml` (rustup picks it up automatically).
- Node 22 for the dashboard (`dashboard/`).
- Use an external Cargo target dir so `target/` never lands in the repo:
  `CARGO_TARGET_DIR=/tmp/symbiotic-mem-bench-target`.

No provider API keys are needed for core development, leaderboard exports, or the dashboard.
The native explorer and `--smoke` need a credentialed adapter build; their execution uses no
provider calls. See the adapter boundary in `AGENTS.md`.

## Quality gates

Use the commands in [AGENTS.md](AGENTS.md#validate): formatting, server-feature lint, and
tests filtered to the changed area in CI's debug profile. CI owns the complete core/server
suites, dependency checks, and dashboard build (`.github/workflows/ci.yml`).

If you change the leaderboard export contract intentionally, regenerate the canary
(`canary/README.md`) and review the diff. If you change `records/`, regenerate the bundled
dashboard snapshot with `scripts/export-leaderboard-snapshot.sh`.

## Pull requests

- Neutral run names in any committed record (see [AGENTS.md](AGENTS.md#run-registry)).
- Explain benchmark-relevant changes with evidence (which runs, what artifacts) — score deltas
  under the run-to-run variance floor are noise, not results.

## License

By contributing you agree that your contributions are licensed under the Apache License 2.0
(`LICENSE`).
