# Library-Only Repository Cleanup Design

## Context

SAY 0.2 already has the core shape of an installable Python library: production code
lives under `src/speech_features`, tests live under `tests/speech_features`, the
distribution is built from `pyproject.toml`, and the same extraction pipeline powers
the Python API and the `say-features` CLI.

The repository still tracks an older Hydra/pydub preprocessing workflow under
`src/preprocessing`, helper scripts under `src/utils`, a project-specific YAML file,
and a CUDA-oriented Conda environment. Setuptools intentionally excludes these files
from the wheel, so retaining them makes the source repository look broader than the
library that users actually install. The README also declares the MIT license while
the referenced `LICENSE` file is missing.

## Goals

- Make `src/speech_features` the only production Python package in the repository.
- Remove obsolete, unshipped preprocessing scripts, configuration, and dependencies.
- Preserve the stable 0.2 Python API, CLI, feature catalog, extraction behavior, and
  documented legacy AD migration window.
- Make the source tree and built wheel tell the same story.
- Leave a conventional package that can be published to PyPI later without adding a
  publishing workflow now.
- Record the safe future boundary for automated transcription: machine output is a
  draft that must be benchmarked against held-out human ground truth and reviewed by a
  human before feature extraction.

## Non-goals

- Do not publish a package, create a release, tag a version, or add credentials.
- Do not rename the `speech-analysis-for-you` distribution, `speech_features` import,
  or `say-features` command.
- Do not add acoustic, linguistic, pediatric, ASR, diarization, alignment, diagnostic,
  or modelling behavior.
- Do not introduce a dynamic plug-in registry or new runtime dependency.
- Do not remove `speech_features.legacy.ad` or its compatibility wrappers before the
  documented 0.3 removal point.
- Do not rewrite historical implementation reports or the approved 0.2 design record.

## System design retained

```text
PCM WAV + reviewed JSON/CHAT transcript
                  |
                  v
      document parsing and validation
      (document.py, formats/)
                  |
                  v
         extraction orchestration
            (extraction.py)
                  |
          +-------+--------+
          |                |
          v                v
   acoustic features   linguistic features
   (features/acoustic) (features/linguistic)
          |                |
          +-------+--------+
                  v
       catalogued FeatureBundle
   (recordings, utterances, issues,
             provenance)
                  |
          +-------+--------+
          |                |
          v                v
       Python API      say-features CLI
```

The catalog remains a static built-in mapping. A future pediatric pack extends the
feature layer and catalog while reusing the same document, extraction, result, and CLI
boundaries.

## Repository boundary

The supported repository layout is:

```text
LICENSE
README.md
pyproject.toml
uv.lock
src/speech_features/
tests/speech_features/
docs/
```

Delete these obsolete tracked files:

- `.DS_Store`
- `src/.DS_Store`
- `environment.yml`
- `src/config/main.yaml`
- `src/preprocessing/audio-preprocessing.py`
- `src/preprocessing/clapperboard.py`
- `src/utils/audio.py`
- `src/utils/file_io.py`
- `src/utils/visualization.py`

The deleted scripts are recoverable from Git history. No supported module imports them,
and they are already absent from built wheels.

## Packaging and metadata

- Keep setuptools and the existing `src` layout.
- Keep package discovery restricted to `speech_features*`.
- Keep NumPy, pandas, and SciPy as the only core runtime dependencies.
- Keep the `legacy-ad` extra for scikit-learn-backed compatibility behavior.
- Remove the `legacy-preprocessing` extra and its Hydra, OmegaConf, pydub, matplotlib,
  and tqdm dependency declarations.
- Add the standard MIT license text in `LICENSE`.
- Declare the MIT license in project metadata and include `LICENSE` in built source and
  wheel metadata using supported setuptools/PEP 639 fields.
- Do not invent author names, repository URLs, or package indexes that the user has not
  supplied.
- Regenerate `uv.lock`; never hand-edit it.

## Repository hygiene

Normalize `.gitignore` without changing data-retention policy:

- remove duplicate virtual-environment and Python-cache entries;
- ignore `.serena/`, `.pytest_cache/`, `.ruff_cache/`, coverage files, build output,
  virtual environments, editor files, logs, and regenerated output;
- continue ignoring `data/*` without deleting local data.

No user-owned `.serena` or data files are deleted; only tracked obsolete files listed
in this design are removed.

## Documentation changes

- Update `README.md` only where repository setup or packaging metadata changes.
- Add a short README roadmap note that future transcription automation stays outside
  the core extractor, records annotation provenance, is evaluated against human ground
  truth, and requires human review before SAY consumes the transcript.
- Update `docs/migration-0.2.md` to state that the unshipped preprocessing workflow and
  its optional extra were removed from the library repository.
- Preserve user-facing extraction, transcript, and feature-catalog documentation.
- Preserve historical plans and implementation review records as history; they may
  describe decisions that were true when 0.2 was assembled.

## Tests and verification

Update packaging tests before production metadata so the expected test failure proves
the contract changed intentionally. The final branch must pass:

1. `pytest tests/speech_features -q`
2. `ruff check src/speech_features tests/speech_features`
3. `ruff format --check src/speech_features tests/speech_features`
4. `uv lock --check`
5. `uv build --wheel`
6. wheel-content inspection proving only `speech_features/**`, license metadata, and
   distribution metadata ship;
7. installation of the wheel into a fresh Python 3.10 environment with only core
   dependencies, followed by `import speech_features` and
   `python -m speech_features.cli list-features`;
8. installation with the `legacy-ad` extra and a compatibility import smoke test;
9. `git diff --check` and a tracked-file scan proving no notebook, `.DS_Store`, old
   preprocessing/configuration file, or top-level `preprocessing`/`utils` package
   remains.

## Agent collaboration

- OpenCode implements each plan task with tests first and records exact verification.
- Agy reviews each task for correctness, packaging boundaries, simplicity, and scope.
- OpenCode resolves required findings; Agy re-reviews until approval.
- The primary agent performs final whole-branch verification and integration only after
  every task is approved.

## Acceptance criteria

- `src/speech_features` is the only tracked production Python package.
- The obsolete preprocessing/configuration files and optional extra are absent.
- Core and `legacy-ad` installations behave as documented.
- The wheel contains no repository-only scripts, notebooks, or unrelated packages.
- The existing public API, CLI, 170-feature catalog, and extraction outputs remain
  behaviorally unchanged.
- The repository contains a valid MIT license and consistent package metadata.
- The README records the human-ground-truth and human-review requirements for any
  future automated transcription workflow without claiming that SAY currently ships
  ASR.
- All required tests, lint checks, builds, installation smoke tests, and reviews pass.
