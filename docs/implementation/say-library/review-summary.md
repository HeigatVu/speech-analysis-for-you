# OpenCode implementation

Implementation record for the SAY 0.2 library refactor
(`docs/superpowers/plans/2026-08-06-say-vietnamese-speech-library.md`,
spec `docs/superpowers/specs/2026-08-06-say-vietnamese-speech-library-design.md`).

## Tasks 1–11

| Task | Implementation commit | Fix round 1/5 | Controller cleanup | Agy | Suite (tests) |
|---|---|---|---|---|---|
| 1 packaging boundary | `3759c69` | — | — | spec/quality approved | 200 |
| 2 document model | `d2a5e15` | `e4d255d` | — | spec/quality approved | 247 |
| 3 CHAT interchange | `12aa46a` | `c049308` | — | spec/quality approved | 286 |
| 4 catalog/result contracts | `13f5b3a` | `a4a1edd` | — | spec/quality approved | 329 |
| 5 acoustic quality/timing | `86924af` | `f2e44a1` | — | spec/quality approved | 373 |
| 6 phonation/prosody | `535b55d` | `29d0fbb` | — | spec/quality approved | 393 |
| 7 resonance/spectrum/rhythm | `548a1fc` | `9d704e7` | — | spec/quality approved | 419 |
| 8 lexical/disfluency | `42d9e0c` | `ecaab9d` | — | spec/quality approved | 449 |
| 9 morphosyntax/conversation | `774a0c4` | `f70c150` | `b6a1b89` | approved | 483 |
| 10 pipeline/CLI | `0506fa0` | `9a46d13` | `724adb6` | approved | 552 |
| 11 legacy AD + notebook cleanup | `c9b3fa3` | — | review record `2332ef2` | approved, no findings | 571 |

Baseline: `81f5d17`, 194 tests, Ruff clean. Plan commit: `a0c594b`.

## Task 12 — documentation, packaging, and release verification

- **RED (test-first):** extended `tests/speech_features/test_packaging.py`
  (exact `setuptools>=68` build requirement, accurate description,
  Python-floor/Ruff alignment, operator/extras/marker-agnostic stdlib
  dependency-name parsing) and rewrote
  `tests/speech_features/test_documentation.py` (five user/release docs and
  cross-links, research-only boundary and prohibited claims, real public
  API/CLI quick starts, transcript JSON v2/CHAT/manual-review/migration
  coverage, all ten `STABLE_ERROR_CODES`, catalog parity with the live
  registry, documentation-only pediatric example). Result before docs:
  **38 failed, 9 passed** (expected RED).
- **GREEN:** after implementing `README.md`,
  `docs/feature-extraction.md`, `docs/transcript-formats.md`,
  `docs/feature-catalog-v1.md` (exactly 170 parseable rows matching live
  metadata), `docs/migration-0.2.md`,
  `docs/implementation/say-library/review-summary.md`, `pyproject.toml`
  (description, `target-version = "py310"`), and the deferred `uv.lock`
  update — focused documentation/packaging tests pass and the full
  `tests/speech_features` suite passes with Ruff clean.
- **Verification:** focused and full pytest, Ruff check and format check,
  `git diff --check`, catalog row count 170, forbidden runtime import scan.
- **Deferred to controller (sandbox boundary):** `uv lock --check`,
  `uv build --wheel`, and the isolated Python 3.10 wheel-install smoke
  (import `speech_features`, version 0.2.0, 170 definitions, no eager
  `sklearn` import, `say-features --help`, `say-features list-features
  --pack acoustic`) are **controller-pending** and must be run after this
  commit; the committed report and this summary mark them as pending.

## Deferred minors (SDD ledger — not fixed in Task 12)

- Task 1: Ruff `target-version` alignment (fixed in Task 12); exact
  setuptools build-requirement assertion (fixed in Task 12);
  operator-agnostic dependency-name parsing (fixed in Task 12).
- Task 2: reject malformed non-list `tokens` instead of coercing to empty;
  reconsider local `FORMATS` imports if the registry is revised.
- Task 3: reject duplicate `@Languages` headers explicitly; reject empty
  `@:`/`%:` header/tier keys explicitly.
- Task 4: validate each prerequisite as a non-empty string; validate
  `FeatureIssue` recording/speaker identifiers as non-empty strings.
- Task 5: validate `ExtractionConfig` numeric ranges; clarify/revisit the
  articulation-rate denominator during formula consolidation.
- Task 6: share precomputed Task 5 frames/VAD if profiling justifies it;
  pair a reachable zero-time-variance OLS NaN with a feature-specific issue.
- Task 7: guard one-bin entropy normalization only if a future public
  configuration permits `frame_size` below 3.

None of these deferred minors were fixed as part of Task 12.

# Agy review

(Ready for the final whole-branch reviewer/controller: this section will
record the whole-branch review verdict and any fix rounds.)

# Resolution

(Ready for the final whole-branch reviewer/controller: this section will
record the final acceptance run and the closing verdict.)
