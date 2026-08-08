# Library-Only Repository Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make SAY a clean library-only repository whose tracked source, package metadata, wheel contents, and documentation consistently describe the supported `speech_features` library.

**Architecture:** Preserve the existing document → extraction → feature-pack → `FeatureBundle` pipeline and its Python/CLI interfaces. Remove only repository-only preprocessing tooling that is already excluded from the wheel, keep the deprecated `speech_features.legacy.ad` compatibility boundary, and document automated transcription as a future external, human-validated input workflow.

**Tech Stack:** Python 3.10+, setuptools, uv, pytest, Ruff, NumPy, pandas, SciPy, optional scikit-learn through `legacy-ad`.

## Global Constraints

- OpenCode uses `deepseek-flash` at maximum effort as the implementer; Agy uses `gemini-3.6-flash-high` at high effort as the reviewer.
- Work only on branch `refactor/library-repository-cleanup` in `/tmp/say-library-refactor`.
- Use test-driven development: change the focused contract test first, record the expected RED result, then implement the minimum GREEN change.
- Keep distribution name `speech-analysis-for-you`, import namespace `speech_features`, CLI command `say-features`, package version `0.2.0`, and Python floor `>=3.10` unchanged.
- Keep the existing 170-feature catalog, public API, extraction outputs, CLI behavior, and `speech_features.legacy.ad` migration window unchanged.
- Add no runtime dependency and no publishing, release, credential, ASR, diarization, alignment, diagnostic, modelling, or pediatric implementation.
- Do not edit or delete user-owned `.serena/`, `data/`, or generated local output.
- Do not rewrite historical 0.2 plans, specifications, dispatch records, or review reports.
- Each task ends in a focused commit, then Agy review. OpenCode resolves every required finding before the next task starts.

---

### Task 1: Define the library package and license contract

**Files:**
- Create: `LICENSE`
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Test: `tests/speech_features/test_packaging.py`

**Interfaces:**
- Consumes: the existing PEP 621 project table and setuptools `src`-layout discovery.
- Produces: PEP 639 MIT metadata, a shipped license file, and exactly two optional extras: `legacy-ad` and `dev`.

- [ ] **Step 1: Change the packaging tests first**

Replace the build requirement assertion and optional-extra assertions, then add the license contract:

```python
def test_setuptools_build_backend_and_exact_requirement():
    assert PYPROJECT["build-system"]["build-backend"] == "setuptools.build_meta"
    assert PYPROJECT["build-system"]["requires"] == ["setuptools>=77"]


def test_mit_license_metadata_and_file():
    assert PROJECT["license"] == "MIT"
    assert PROJECT["license-files"] == ["LICENSE"]
    license_text = (PROJECT_ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "MIT License" in license_text
    assert "SAY contributors" in license_text


def test_optional_extras():
    extras = PROJECT["optional-dependencies"]
    assert set(extras) == {"legacy-ad", "dev"}
    assert _names(extras["legacy-ad"]) == ["scikit-learn"]
    assert "scikit-learn>=1.3" in extras["legacy-ad"]
    assert _names(extras["dev"]) == ["pytest", "ruff"]
```

- [ ] **Step 2: Run the focused test and record RED**

Run:

```bash
rtk proxy .venv/bin/python -m pytest tests/speech_features/test_packaging.py -q
```

Expected: FAIL because setuptools is still `>=68`, MIT metadata and `LICENSE` are absent, and `legacy-preprocessing` still exists.

- [ ] **Step 3: Implement the minimal project metadata**

Change the build and project tables in `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=77"]
build-backend = "setuptools.build_meta"

[project]
name = "speech-analysis-for-you"
version = "0.2.0"
description = "Research/descriptive Vietnamese speech feature library with a CLAN-compatible CHAT subset, label-free extraction, and an adult-neuro feature catalog."
readme = "README.md"
requires-python = ">=3.10"
license = "MIT"
license-files = ["LICENSE"]
```

Delete the complete `legacy-preprocessing` optional-dependency entry. Keep `legacy-ad` and `dev` unchanged.

Create `LICENSE` with the standard MIT text and this non-invented collective holder:

```text
MIT License

Copyright (c) 2026 SAY contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

- [ ] **Step 4: Regenerate the lockfile**

Run:

```bash
rtk proxy uv lock
```

Expected: `uv.lock` no longer resolves dependencies that existed only for `legacy-preprocessing`; do not hand-edit the lockfile.

- [ ] **Step 5: Run focused GREEN checks**

Run:

```bash
rtk proxy .venv/bin/python -m pytest tests/speech_features/test_packaging.py -q
rtk proxy .venv/bin/python -m ruff check tests/speech_features/test_packaging.py
rtk proxy .venv/bin/python -m ruff format --check tests/speech_features/test_packaging.py
rtk proxy uv lock --check
```

Expected: all commands pass.

- [ ] **Step 6: Commit Task 1**

```bash
rtk proxy git add LICENSE pyproject.toml uv.lock tests/speech_features/test_packaging.py
rtk proxy git commit -m "chore: define clean library package metadata"
```

- [ ] **Step 7: Agy reviews Task 1**

Agy checks PEP 639/setuptools validity, dependency removal, lockfile provenance, test quality, and absence of invented owner/URL metadata. OpenCode fixes required findings and Agy re-reviews until APPROVE.

---

### Task 2: Remove obsolete research tooling and normalize repository hygiene

**Files:**
- Modify: `.gitignore`
- Delete: `.DS_Store`
- Delete: `src/.DS_Store`
- Delete: `environment.yml`
- Delete: `src/config/main.yaml`
- Delete: `src/preprocessing/audio-preprocessing.py`
- Delete: `src/preprocessing/clapperboard.py`
- Delete: `src/utils/audio.py`
- Delete: `src/utils/file_io.py`
- Delete: `src/utils/visualization.py`
- Test: `tests/speech_features/test_packaging.py`

**Interfaces:**
- Consumes: Task 1's package discovery restriction and optional-extra contract.
- Produces: a tracked repository where `src/speech_features` is the only production source tree and local agent/data files stay untracked.

- [ ] **Step 1: Add the repository-boundary test first**

Add near the top of `tests/speech_features/test_packaging.py`:

```python
OBSOLETE_REPOSITORY_PATHS = (
    ".DS_Store",
    "src/.DS_Store",
    "environment.yml",
    "src/config/main.yaml",
    "src/preprocessing/audio-preprocessing.py",
    "src/preprocessing/clapperboard.py",
    "src/utils/audio.py",
    "src/utils/file_io.py",
    "src/utils/visualization.py",
)
```

Add the contract test:

```python
def test_obsolete_nonpackage_research_tools_are_absent():
    remaining = [path for path in OBSOLETE_REPOSITORY_PATHS if (PROJECT_ROOT / path).exists()]
    assert remaining == []
```

- [ ] **Step 2: Run the focused test and record RED**

Run:

```bash
rtk proxy .venv/bin/python -m pytest tests/speech_features/test_packaging.py::test_obsolete_nonpackage_research_tools_are_absent -q
```

Expected: FAIL listing all tracked obsolete paths.

- [ ] **Step 3: Delete only the approved obsolete files**

Run one explicit removal command with no glob:

```bash
rtk proxy git rm -- .DS_Store src/.DS_Store environment.yml src/config/main.yaml src/preprocessing/audio-preprocessing.py src/preprocessing/clapperboard.py src/utils/audio.py src/utils/file_io.py src/utils/visualization.py
```

Do not delete `.serena/`, `data/`, `output/`, or any other untracked path. The deleted tracked files remain recoverable from Git history.

- [ ] **Step 4: Replace `.gitignore` with the minimal repository policy**

```gitignore
# Python
__pycache__/
*.py[cod]
*.egg-info/
.pytest_cache/
.ruff_cache/
.coverage
coverage.xml
htmlcov/
build/
dist/
wheels/

# Virtual environments
.venv/
venv/

# Local data and generated outputs
data/*
output/
log/

# Editors, operating systems, and local agents
.DS_Store
.vscode/
.idea/
.serena/
.superpowers/
```

- [ ] **Step 5: Run focused GREEN and repository scans**

Run:

```bash
rtk proxy .venv/bin/python -m pytest tests/speech_features/test_packaging.py -q
rtk proxy .venv/bin/python -m ruff check tests/speech_features/test_packaging.py
rtk proxy .venv/bin/python -m ruff format --check tests/speech_features/test_packaging.py
rtk proxy rg -n "src\\.utils|from utils|import utils|src/config" src tests pyproject.toml README.md
rtk proxy git ls-files "*.ipynb" ".DS_Store" "src/.DS_Store" "src/preprocessing/**" "src/utils/**" "src/config/**"
```

Expected: tests/lint pass; both scans return no matches. Historical design/review documents are intentionally outside this scan.

- [ ] **Step 6: Commit Task 2**

```bash
rtk proxy git add .gitignore tests/speech_features/test_packaging.py
rtk proxy git commit -m "refactor: remove obsolete research tooling"
```

- [ ] **Step 7: Agy reviews Task 2**

Agy verifies every deletion is in the approved list, no supported import references a deleted module, `.gitignore` preserves local data, and no user-owned files were removed. OpenCode fixes required findings and Agy re-reviews until APPROVE.

---

### Task 3: Document the clean library and future transcript-validation boundary

**Files:**
- Modify: `README.md`
- Modify: `docs/migration-0.2.md`
- Test: `tests/speech_features/test_documentation.py`

**Interfaces:**
- Consumes: Tasks 1–2 package metadata and repository boundary.
- Produces: user-facing installation/migration guidance plus a roadmap contract for external automated transcription validated against human ground truth.

- [ ] **Step 1: Add semantic documentation tests first**

Add to `TestResearchOnlyBoundary`:

```python
def test_readme_documents_future_transcript_validation_boundary(self):
    _require(
        _doc("README"),
        "future",
        "automated transcription",
        "ground truth",
        "human review",
        "annotation provenance",
        subject="README future transcript validation",
    )
```

Add to `TestMigrationGuide`:

```python
def test_documents_removed_preprocessing_workflow(self):
    _require(
        _doc("migration-0.2"),
        "legacy-preprocessing",
        "removed",
        "Git history",
        subject="migration removed preprocessing workflow",
    )
```

- [ ] **Step 2: Run the focused tests and record RED**

Run:

```bash
rtk proxy .venv/bin/python -m pytest tests/speech_features/test_documentation.py::TestResearchOnlyBoundary::test_readme_documents_future_transcript_validation_boundary tests/speech_features/test_documentation.py::TestMigrationGuide::test_documents_removed_preprocessing_workflow -q
```

Expected: both tests fail because the README roadmap note is absent and the migration guide still says the preprocessing extra is retained.

- [ ] **Step 3: Add the README roadmap note**

Insert this section after `Architecture and extension boundary` and before `Development`:

```markdown
## Future transcript automation

SAY does not currently transcribe audio. A future companion may produce draft JSON v2
or CHAT transcripts, but automated transcription remains outside the core feature
extractor. It should be evaluated on held-out human ground-truth transcripts using at
least word and character error rates, plus speaker and timestamp accuracy when those
annotations are produced. Report results across relevant Vietnamese populations and
recording conditions, record annotation provenance, and require human review and
correction before SAY consumes the transcript.
```

- [ ] **Step 4: Correct the migration guide**

Replace the obsolete `legacy-preprocessing` retention bullet with:

```markdown
- The unshipped `legacy-preprocessing` Hydra/pydub workflow and optional extra were
  removed from the library repository. They remain recoverable from Git history but
  are not supported by SAY 0.2.
```

- [ ] **Step 5: Run focused GREEN checks**

Run:

```bash
rtk proxy .venv/bin/python -m pytest tests/speech_features/test_documentation.py -q
rtk proxy .venv/bin/python -m ruff check tests/speech_features/test_documentation.py
rtk proxy .venv/bin/python -m ruff format --check tests/speech_features/test_documentation.py
rtk proxy git diff --check
```

Expected: all commands pass and README makes no claim that SAY ships ASR.

- [ ] **Step 6: Commit Task 3**

```bash
rtk proxy git add README.md docs/migration-0.2.md tests/speech_features/test_documentation.py
rtk proxy git commit -m "docs: describe library and transcript validation roadmap"
```

- [ ] **Step 7: Agy reviews Task 3**

Agy checks research-only wording, absence of diagnostic/ASR claims, human-ground-truth validation, human review, provenance, Vietnamese population/recording-condition coverage, and migration accuracy. OpenCode fixes required findings and Agy re-reviews until APPROVE.

---

## Whole-branch verification

After every task is approved, the primary agent runs these commands from `/tmp/say-library-refactor` on the exact branch head:

```bash
rtk proxy .venv/bin/python -m pytest tests/speech_features -q
rtk proxy .venv/bin/python -m ruff check src/speech_features tests/speech_features
rtk proxy .venv/bin/python -m ruff format --check src/speech_features tests/speech_features
rtk proxy uv lock --check
rtk proxy uv build --wheel
rtk proxy git diff --check main...HEAD
rtk proxy git ls-files "*.ipynb" ".DS_Store" "src/.DS_Store" "src/preprocessing/**" "src/utils/**" "src/config/**"
```

Inspect the built wheel and prove it contains only `speech_features/**`, `speech_analysis_for_you-0.2.0.dist-info/**`, and the MIT license metadata. Install that wheel into a fresh Python 3.10 environment, run `import speech_features`, confirm `len(speech_features.list_features()) == 170`, and run `python -m speech_features.cli list-features`. Then install the wheel with `legacy-ad` and import `speech_features.legacy.ad`.

Finally, Agy reviews the complete branch diff from `5b5e0ca` through HEAD. Merge with `--ff-only` into the original `main` checkout only after the complete review is APPROVE and all verification passes. Preserve the isolated branch/worktree and any untracked `.serena/` data.
