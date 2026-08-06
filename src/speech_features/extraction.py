"""Label-free extraction pipeline and provenance (Task 10).

:func:`extract` runs the selected feature packs over an already loaded
:class:`~speech_features.document.SpeechDocument` and one audio file,
returning the table-oriented :class:`~speech_features.result.FeatureBundle`;
:func:`extract_batch` runs a manifest v2 through the same per-target path
with row/target isolation expressed as severity-``error`` issues.

No diagnosis, labels, tasks, demographics, or clinical values ever enter the
result tables or provenance. The legacy label-bearing
:class:`~speech_features.pipeline.BatchResult` is untouched.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import pandas as pd

from . import __version__
from .catalog import CATALOG_VERSION, PACK_LEVELS, PACKS, list_features
from .document import SpeechDocument, load_document
from .features.acoustic import _extract as _extract_acoustic_pack
from .features.linguistic import extract_adult_neuro_features
from .formats.json import encode_json
from .pipeline import _read_wav_with_width
from .result import (
    ExtractionError,
    FeatureBundle,
    FeatureIssue,
    InvalidConfigError,
    TargetSpeakerRequiredError,
    UnknownPackError,
)
from .schema import (
    ExtractionConfig,
    FeatureExtractionError,
    InvalidManifestError,
    sha256_file,
)

_ISSUE_COLUMNS = (
    "recording_id",
    "speaker_id",
    "utterance_id",
    "feature",
    "code",
    "severity",
    "message",
)
_RECORDING_ID_COLUMNS = ("recording_id", "speaker_id")
_UTTERANCE_ID_COLUMNS = ("recording_id", "speaker_id", "utterance_id", "start_s", "end_s")
_MANIFEST_ROW_KEYS = frozenset({"recording_id", "audio_path", "transcript_path", "target_speakers"})


def _sorted_issues(issues_df: pd.DataFrame) -> pd.DataFrame:
    """Sort issues by all seven columns with nulls ordered as empty strings,
    preserving the original ``None``/NaN values in the returned frame."""
    if issues_df.empty:
        return issues_df
    order = issues_df.fillna("").sort_values(by=list(_ISSUE_COLUMNS), kind="stable").index
    return issues_df.loc[order].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Selection and input validation
# ---------------------------------------------------------------------------
def _config(config) -> ExtractionConfig:
    if config is None:
        return ExtractionConfig()
    if not isinstance(config, ExtractionConfig):
        raise InvalidConfigError(f"config must be an ExtractionConfig, got {type(config).__name__}")
    return config


def _canonical_packs(packs) -> tuple[str, ...]:
    """Validate ``packs`` and return them in static catalog order.

    Accepts only list/tuple selections; every element must be a string, then
    duplicates are rejected, then unknown names raise ``UNKNOWN_PACK``.
    """
    if not isinstance(packs, (list, tuple)):
        raise InvalidConfigError(
            "packs must be a non-empty duplicate-free list or tuple of pack names"
        )
    if not packs:
        raise InvalidConfigError(
            "packs must be a non-empty duplicate-free list or tuple of pack names"
        )
    for name in packs:
        if not isinstance(name, str):
            raise InvalidConfigError(f"pack names must be strings, got {name!r}")
    if len(packs) != len(set(packs)):
        raise InvalidConfigError(
            "packs must be a non-empty duplicate-free list or tuple of pack names"
        )
    for name in packs:
        if name not in PACKS:
            raise UnknownPackError(f"unknown feature pack {name!r}; known packs: {sorted(PACKS)}")
    return tuple(name for name in PACKS if name in packs)


def _canonical_levels(levels) -> tuple[str, ...]:
    """Validate ``levels`` and return them in the canonical recording, utterance order."""
    if not isinstance(levels, (list, tuple)):
        raise InvalidConfigError(
            "levels must be a non-empty duplicate-free list or tuple of recording/utterance"
        )
    if not levels:
        raise InvalidConfigError(
            "levels must be a non-empty duplicate-free list or tuple of recording/utterance"
        )
    for level in levels:
        if not isinstance(level, str):
            raise InvalidConfigError(f"level names must be strings, got {level!r}")
    if len(levels) != len(set(levels)):
        raise InvalidConfigError(
            "levels must be a non-empty duplicate-free list or tuple of recording/utterance"
        )
    for level in levels:
        if level not in PACK_LEVELS:
            raise InvalidConfigError(
                f"unknown feature level {level!r}; allowed levels: {sorted(PACK_LEVELS)}"
            )
    return tuple(level for level in ("recording", "utterance") if level in levels)


def _resolve_speaker(document, target_speaker) -> str:
    """Resolve the target speaker id with the pack rules: an explicit id must
    exist; one documented speaker is inferred; several require an explicit
    target; an empty document resolves to the empty speaker id."""
    speaker_ids = {speaker.id for speaker in document.speakers}
    if target_speaker is not None:
        if not isinstance(target_speaker, str):
            raise InvalidConfigError(f"target speaker must be a string, got {target_speaker!r}")
        if target_speaker not in speaker_ids:
            raise InvalidConfigError(
                f"target speaker {target_speaker!r} is not a speaker of the speech document"
            )
        return target_speaker
    if len(speaker_ids) > 1:
        raise TargetSpeakerRequiredError(
            "speech document has multiple speakers; pass target_speaker to isolate one"
        )
    return next(iter(speaker_ids)) if speaker_ids else ""


def _canonical_document_sha256(document: SpeechDocument) -> str:
    """SHA-256 of the deterministic canonical JSON-v2 serialization."""
    return hashlib.sha256(encode_json(document).encode("utf-8")).hexdigest()


def _read_inputs(audio_path, document, packs, cfg):
    """Hash both inputs and load PCM audio once, only when acoustic is selected.

    Returns ``(audio, width, audio_sha256, transcript_sha256, hash_kind)``
    where ``hash_kind`` is ``"source"`` when the document records its source
    SHA-256 and ``"canonical_document"`` otherwise.
    """
    if document.source_sha256:
        transcript_sha256, hash_kind = document.source_sha256, "source"
    else:
        transcript_sha256, hash_kind = _canonical_document_sha256(document), "canonical_document"
    audio_sha256 = sha256_file(audio_path)
    if "acoustic" in packs:
        audio, width = _read_wav_with_width(audio_path, sample_rate=cfg.sample_rate)
    else:
        audio, width = None, None
    return audio, width, audio_sha256, transcript_sha256, hash_kind


# ---------------------------------------------------------------------------
# Single extraction
# ---------------------------------------------------------------------------
def _catalog_columns(packs, levels) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Selected recording and utterance feature keys, sorted lexicographically."""
    recording_keys = []
    utterance_keys = []
    for definition in list_features():
        if definition.pack in packs and definition.level in levels:
            (recording_keys if definition.level == "recording" else utterance_keys).append(
                definition.key
            )
    return tuple(sorted(recording_keys)), tuple(sorted(utterance_keys))


def _extract_bundle(
    audio,
    width,
    document: SpeechDocument,
    *,
    recording_id: str,
    target_speaker,
    packs,
    levels,
    config: ExtractionConfig,
    audio_sha256: str,
    transcript_sha256: str,
    transcript_hash_kind: str,
) -> FeatureBundle:
    """Run every selected pack exactly once for one resolved target and build
    the deterministic bundle tables plus single-extraction provenance.

    Unexpected internal failures become :class:`ExtractionError`; structured
    extraction errors keep their stable codes.
    """
    try:
        resolved = _resolve_speaker(document, target_speaker)
        pack_target = resolved or None  # empty id is inferred by the packs themselves

        recording_features: dict[str, float] = {}
        utterance_rows: list[dict] = []
        issues: list[FeatureIssue] = []
        speaker_id = resolved

        if "acoustic" in packs:
            features, pack_issues, pack_speaker = _extract_acoustic_pack(
                audio,
                config.sample_rate,
                document=document,
                target_speaker=pack_target,
                allow_unaligned=False,
                config=config,
                recording_id=recording_id,
                clipping_boundary=1.0 - 2.0 ** -(width * 8 - 1),
            )
            recording_features.update(features)
            issues.extend(pack_issues)
            speaker_id = pack_speaker

        if "adult_neuro" in packs:
            features, rows, pack_issues = extract_adult_neuro_features(
                document, target_speaker=pack_target, recording_id=recording_id
            )
            recording_features.update(features)
            utterance_rows.extend(rows)
            issues.extend(pack_issues)

        recording_keys, utterance_keys = _catalog_columns(packs, levels)

        if "recording" in levels:
            recordings = pd.DataFrame(
                [
                    [recording_id, speaker_id]
                    + [recording_features.get(key, float("nan")) for key in recording_keys]
                ],
                columns=_RECORDING_ID_COLUMNS + recording_keys,
            )
            recordings = recordings.sort_values(
                by=list(_RECORDING_ID_COLUMNS), kind="stable"
            ).reset_index(drop=True)
        else:
            recordings = pd.DataFrame(columns=_RECORDING_ID_COLUMNS)

        if "utterance" in levels and utterance_keys:
            rows = [
                [recording_id, speaker_id, row["utterance_id"], row["start_s"], row["end_s"]]
                + [row.get(key, float("nan")) for key in utterance_keys]
                for row in utterance_rows
            ]
            utterances = pd.DataFrame(
                rows,
                columns=_UTTERANCE_ID_COLUMNS + utterance_keys,
            )
            if len(utterances):
                utterances = utterances.sort_values(
                    by=list(_UTTERANCE_ID_COLUMNS), kind="stable"
                ).reset_index(drop=True)
        else:
            utterances = pd.DataFrame(columns=_UTTERANCE_ID_COLUMNS)

        level_of = {definition.key: definition.level for definition in list_features()}
        kept: list[FeatureIssue] = [
            issue
            for issue in issues
            if issue.feature is None or level_of.get(issue.feature) in levels
        ]
        for warning in document.warnings:
            code = str(getattr(warning, "code", "DOCUMENT_WARNING"))
            tier = getattr(warning, "tier", "")
            line = getattr(warning, "line", "")
            kept.append(
                FeatureIssue(
                    recording_id=recording_id,
                    speaker_id=speaker_id,
                    code=code,
                    severity="warning",
                    message=f"document warning {code}: tier {tier!r}, line {line!r}",
                )
            )

        issue_rows = [
            (i.recording_id, i.speaker_id, i.utterance_id, i.feature, i.code, i.severity, i.message)
            for i in kept
        ]
        issues_df = _sorted_issues(pd.DataFrame(issue_rows, columns=_ISSUE_COLUMNS))

        provenance = {
            "package_version": __version__,
            "catalog_version": CATALOG_VERSION,
            "packs": list(packs),
            "levels": list(levels),
            "config": dataclasses.asdict(config),
            "target_speakers": [speaker_id],
            "audio_sha256": audio_sha256,
            "transcript_sha256": transcript_sha256,
            "transcript_hash_kind": transcript_hash_kind,
            "annotation_sources": [
                {"layer": a.layer, "source": a.source, "confidence": a.confidence}
                for a in sorted(
                    document.annotations, key=lambda a: (a.layer, a.source, a.confidence)
                )
            ],
        }
        return FeatureBundle(
            recordings=recordings, utterances=utterances, issues=issues_df, provenance=provenance
        )
    except FeatureExtractionError:
        raise
    except Exception as exc:  # noqa: BLE001 - isolate any unexpected single failure
        raise ExtractionError(f"unexpected failure during extraction: {exc}") from exc


def extract(
    audio_path,
    document: SpeechDocument,
    *,
    target_speaker=None,
    packs=("acoustic", "adult_neuro"),
    levels=("recording", "utterance"),
    config=None,
) -> FeatureBundle:
    """Extract the selected packs for one target of an already loaded document.

    ``document`` must be a :class:`SpeechDocument` (anything else raises
    ``INVALID_CONFIG``); its ``document_id`` is the recording id. Audio input
    is hashed before extraction and PCM audio is loaded exactly once, only
    when the ``acoustic`` pack is selected.
    """
    if not isinstance(document, SpeechDocument):
        raise InvalidConfigError(
            f"extract requires a SpeechDocument, got {type(document).__name__}"
        )
    cfg = _config(config)
    canonical_packs = _canonical_packs(packs)
    canonical_levels = _canonical_levels(levels)
    audio, width, audio_sha256, transcript_sha256, hash_kind = _read_inputs(
        audio_path, document, canonical_packs, cfg
    )
    return _extract_bundle(
        audio,
        width,
        document,
        recording_id=document.document_id,
        target_speaker=target_speaker,
        packs=canonical_packs,
        levels=canonical_levels,
        config=cfg,
        audio_sha256=audio_sha256,
        transcript_sha256=transcript_sha256,
        transcript_hash_kind=hash_kind,
    )


# ---------------------------------------------------------------------------
# Manifest v2 and batch orchestration
# ---------------------------------------------------------------------------
def _load_manifest(path) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError) as exc:
        raise InvalidManifestError(f"cannot read manifest {path}: {exc}") from exc


def _validate_manifest_v2(manifest: dict, base_dir) -> list[dict]:
    """Validate manifest v2 and return rows with manifest-relative paths resolved.

    The row schema is exactly ``recording_id, audio_path, transcript_path,
    target_speakers``; every extra key is rejected. Relative asset paths are
    resolved against ``base_dir`` (the manifest file's parent).
    """
    if not isinstance(manifest, dict):
        raise InvalidManifestError(f"manifest must be a JSON object, got {type(manifest).__name__}")
    if manifest.get("version") != 2:
        raise InvalidManifestError("manifest must have version == 2")
    rows = manifest.get("rows")
    if not isinstance(rows, list) or not rows:
        raise InvalidManifestError("manifest must contain a non-empty 'rows' list")

    parsed = []
    seen_ids = set()
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise InvalidManifestError(f"row {i} must be an object, got {type(row).__name__}")
        extra = set(row) - _MANIFEST_ROW_KEYS
        if extra:
            raise InvalidManifestError(f"row {i} has invalid keys: {sorted(extra)}")
        for key in ("recording_id", "audio_path", "transcript_path"):
            value = row.get(key)
            if not isinstance(value, str) or not value:
                raise InvalidManifestError(f"row {i} {key} must be a non-empty string")
        recording_id = row["recording_id"]
        if recording_id in seen_ids:
            raise InvalidManifestError(f"duplicate recording_id: {recording_id!r}")
        seen_ids.add(recording_id)
        targets = row.get("target_speakers")
        if targets is not None:
            if not isinstance(targets, list) or not targets:
                raise InvalidManifestError(f"row {i} target_speakers must be a non-empty list")
            for target in targets:
                if not isinstance(target, str) or not target:
                    raise InvalidManifestError(
                        f"row {i} target_speakers entries must be non-empty strings"
                    )
            if len(targets) != len(set(targets)):
                raise InvalidManifestError(f"row {i} target_speakers must be unique")
        parsed.append(
            {
                "recording_id": recording_id,
                "audio_path": str(Path(base_dir, row["audio_path"])),
                "transcript_path": str(Path(base_dir, row["transcript_path"])),
                "target_speakers": list(targets) if targets is not None else None,
            }
        )
    return parsed


def _annotation_sources(document):
    return sorted(
        (
            {"layer": a.layer, "source": a.source, "confidence": a.confidence}
            for a in document.annotations
        ),
        key=lambda s: (s["layer"], s["source"], s["confidence"]),
    )


def _row_provenance_entry(target_speakers, transcript_sha256, audio_sha256, sources, error=None):
    """One deterministic per-recording provenance entry; every available input
    hash and annotation source is retained, and an error is recorded without
    inventing hashes for unavailable inputs."""
    entry = {"target_speakers": list(target_speakers)}
    if transcript_sha256 is not None:
        entry["transcript_sha256"] = transcript_sha256
    if audio_sha256 is not None:
        entry["audio_sha256"] = audio_sha256
    if sources is not None:
        entry["annotation_sources"] = sources
    if error is not None:
        entry["error_code"] = getattr(error, "code", "EXTRACTION_ERROR")
        entry["error_message"] = str(error)
    return entry


def extract_batch(
    manifest_path,
    *,
    packs=("acoustic", "adult_neuro"),
    config=None,
) -> FeatureBundle:
    """Run manifest v2 through per-target extraction with row/target isolation.

    Manifest structure, pack selection, and config are validated globally
    (invalid input raises and no extraction begins). Failures inside a
    structurally valid row are isolated: a failure before a target is known
    emits one feature-less severity-``error`` issue with the recording id and
    an empty speaker id; a target-specific failure emits one ``error`` issue
    with that target id. Inputs are hashed and acoustic audio is read at most
    once per row, shared by all declared targets. Both levels are always
    returned; all-failed batches keep the identifier plus selected feature
    columns and a writable provenance.
    """
    cfg = _config(config)
    canonical_packs = _canonical_packs(packs)
    levels = ("recording", "utterance")
    manifest_file = Path(manifest_path)
    rows = _validate_manifest_v2(_load_manifest(manifest_file), manifest_file.parent)
    manifest_sha256 = sha256_file(manifest_file)

    recording_keys, utterance_keys = _catalog_columns(canonical_packs, levels)
    recording_cols = ("recording_id", "speaker_id") + recording_keys
    utterance_cols = (
        "recording_id",
        "speaker_id",
        "utterance_id",
        "start_s",
        "end_s",
    ) + utterance_keys

    bundles: list[FeatureBundle] = []
    batch_issues: list[tuple] = []
    per_recording: dict[str, dict] = {}
    total = success = failure = 0

    for row in rows:
        recording_id = row["recording_id"]
        transcript_sha256 = None
        audio_sha256 = None
        document = None
        sources = None
        row_error = None

        try:
            transcript_sha256 = sha256_file(row["transcript_path"])
        except Exception as exc:  # noqa: BLE001 - isolated per-input hashing
            row_error = row_error or exc
        try:
            document = load_document(row["transcript_path"])
            sources = _annotation_sources(document)
        except Exception as exc:  # noqa: BLE001 - isolate loader failures
            row_error = row_error or exc
        try:
            audio_sha256 = sha256_file(row["audio_path"])
        except Exception as exc:  # noqa: BLE001 - isolated per-input hashing
            row_error = row_error or exc
        try:
            if "acoustic" in canonical_packs and audio_sha256 is not None:
                audio, width = _read_wav_with_width(row["audio_path"], sample_rate=cfg.sample_rate)
            else:
                audio, width = None, None
        except Exception as exc:  # noqa: BLE001 - isolate per-row audio read
            audio, width = None, None
            row_error = row_error or exc

        if row_error is not None:
            code = getattr(row_error, "code", "EXTRACTION_ERROR")
            batch_issues.append((recording_id, "", None, None, code, "error", str(row_error)))
            per_recording[recording_id] = _row_provenance_entry(
                row["target_speakers"] or [],
                transcript_sha256,
                audio_sha256,
                sources,
                error=row_error,
            )
            total += 1
            failure += 1
            continue

        declared = row["target_speakers"]
        if declared is not None:
            targets = list(declared)
        else:
            try:
                targets = [_resolve_speaker(document, None)]
            except TargetSpeakerRequiredError as exc:
                batch_issues.append((recording_id, "", None, None, exc.code, "error", str(exc)))
                per_recording[recording_id] = _row_provenance_entry(
                    [], transcript_sha256, audio_sha256, sources, error=exc
                )
                total += 1
                failure += 1
                continue

        for target in targets:
            total += 1
            try:
                bundle = _extract_bundle(
                    audio,
                    width,
                    document,
                    recording_id=recording_id,
                    target_speaker=target,
                    packs=canonical_packs,
                    levels=levels,
                    config=cfg,
                    audio_sha256=audio_sha256,
                    transcript_sha256=transcript_sha256,
                    transcript_hash_kind="file",
                )
                bundles.append(bundle)
                success += 1
            except FeatureExtractionError as exc:
                batch_issues.append((recording_id, target, None, None, exc.code, "error", str(exc)))
                failure += 1

        per_recording[recording_id] = _row_provenance_entry(
            targets, transcript_sha256, audio_sha256, sources
        )

    recordings = (
        pd.concat([bundle.recordings for bundle in bundles], ignore_index=True)
        if bundles
        else pd.DataFrame(columns=recording_cols)
    )
    recordings = recordings.sort_values(
        by=["recording_id", "speaker_id"], kind="stable"
    ).reset_index(drop=True)
    utterances = (
        pd.concat([bundle.utterances for bundle in bundles], ignore_index=True)
        if bundles
        else pd.DataFrame(columns=utterance_cols)
    )
    if len(utterances):
        utterances = utterances.sort_values(
            by=["recording_id", "speaker_id", "start_s", "end_s", "utterance_id"], kind="stable"
        ).reset_index(drop=True)

    for bundle in bundles:
        batch_issues.extend(bundle.issues.itertuples(index=False))
    issues_df = _sorted_issues(pd.DataFrame(batch_issues, columns=_ISSUE_COLUMNS))

    provenance = {
        "package_version": __version__,
        "catalog_version": CATALOG_VERSION,
        "packs": list(canonical_packs),
        "levels": list(levels),
        "config": dataclasses.asdict(cfg),
        "manifest_sha256": manifest_sha256,
        "recordings": per_recording,
        "counts": {"total": total, "success": success, "failure": failure},
    }
    return FeatureBundle(
        recordings=recordings, utterances=utterances, issues=issues_df, provenance=provenance
    )


__all__ = [
    "extract",
    "extract_batch",
]
