"""Study manifest loading for the opt-in preprocessing study.

Manifest rows are private inputs: error messages name row indexes and field
names only, never values, paths, or transcript content.
"""

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

from say_transcribe.denoise import DenoiserSpec

_REQUIRED_FIELDS = (
    "session_id",
    "participant_id",
    "audio_path",
    "channel_index",
    "sha256",
    "split",
    "reference_path",
    "asr_revision",
)
_SPLITS = ("dev", "held_out")
_DENOISER_ARMS = ("PF", "PD")
_DENOISER_FIELDS = ("python", "worker", "checkpoint")
_DENOISER_OPTIONAL_FIELDS = ("config",)
# Workers that take a pinned model configuration file; declaring (or omitting)
# it for the wrong arm would only surface later as a worker argparse failure.
_DENOISER_CONFIG_ARMS = ("PF",)
_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")


class ManifestError(Exception):
    """Manifest failed schema validation before any audio was touched."""

    code = "INVALID_ARGUMENT"

    def __init__(self, message: str) -> None:
        super().__init__(f"[{self.code}] {message}")
        self.message = message


@dataclass(frozen=True)
class ManifestRow:
    session_id: str
    participant_id: str
    audio_path: Path
    channel_index: int
    sha256: str
    split: str
    reference_path: Path
    asr_revision: str


def _require_identifier(row: dict[str, Any], index: int, field: str) -> str:
    value = row[field]
    if not isinstance(value, str) or not _ID_PATTERN.match(value):
        raise ManifestError(f"manifest row {index}: field '{field}' is not a valid identifier")
    return value


def _parse_row(row: Any, index: int) -> ManifestRow:
    if not isinstance(row, dict):
        raise ManifestError(f"manifest row {index}: row must be an object")
    missing = [field for field in _REQUIRED_FIELDS if field not in row]
    if missing:
        raise ManifestError(f"manifest row {index}: missing required field '{missing[0]}'")

    session_id = _require_identifier(row, index, "session_id")
    participant_id = _require_identifier(row, index, "participant_id")
    asr_revision = row["asr_revision"]
    if not isinstance(asr_revision, str) or not asr_revision.strip():
        raise ManifestError(f"manifest row {index}: field 'asr_revision' must be a non-empty string")

    channel_index = row["channel_index"]
    if isinstance(channel_index, bool) or not isinstance(channel_index, int) or channel_index < 0:
        raise ManifestError(f"manifest row {index}: field 'channel_index' must be a non-negative integer")

    sha256 = row["sha256"]
    if not isinstance(sha256, str) or not _SHA256_PATTERN.match(sha256):
        raise ManifestError(f"manifest row {index}: field 'sha256' must be 64 hexadecimal characters")

    split = row["split"]
    if split not in _SPLITS:
        raise ManifestError(f"manifest row {index}: field 'split' must be 'dev' or 'held_out'")

    audio_path = row["audio_path"]
    reference_path = row["reference_path"]
    for field, value in (("audio_path", audio_path), ("reference_path", reference_path)):
        if not isinstance(value, str) or not value.strip():
            raise ManifestError(f"manifest row {index}: field '{field}' must be a non-empty string")

    return ManifestRow(
        session_id=session_id,
        participant_id=participant_id,
        audio_path=Path(audio_path),
        channel_index=channel_index,
        sha256=sha256,
        split=split,
        reference_path=Path(reference_path),
        asr_revision=asr_revision,
    )


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError:
        raise ManifestError("manifest file could not be read") from None
    except json.JSONDecodeError:
        raise ManifestError("manifest is not valid JSON") from None


def load_manifest(path: Path) -> tuple[ManifestRow, ...]:
    """Load and validate a study manifest; raise ManifestError on any schema problem."""
    data = _read_json(path)

    if not isinstance(data, dict) or not isinstance(data.get("rows"), list):
        raise ManifestError("manifest must be an object with a 'rows' list")
    if not data["rows"]:
        raise ManifestError("manifest has no rows")

    rows = tuple(_parse_row(row, index) for index, row in enumerate(data["rows"]))
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if row.session_id in seen:
            raise ManifestError(f"manifest row {index}: duplicate session_id")
        seen.add(row.session_id)
    return rows


def load_denoiser_specs(path: Path) -> dict[str, DenoiserSpec]:
    """Load the optional top-level "denoisers" mapping: arm id -> isolated env spec.

    Keys are arm ids ("PF", "PD"); an absent key means that arm is not run.
    Values carry the isolated environment's interpreter, worker script, and
    local checkpoint. Messages name keys and fields only, never values.
    """
    data = _read_json(path)

    denoisers = data.get("denoisers", {})
    if not isinstance(denoisers, dict):
        raise ManifestError("manifest 'denoisers' must be an object")
    specs: dict[str, DenoiserSpec] = {}
    for arm, config in denoisers.items():
        if arm not in _DENOISER_ARMS:
            raise ManifestError(f"manifest denoisers: unsupported arm '{arm}'")
        if not isinstance(config, dict):
            raise ManifestError(f"manifest denoisers.{arm}: entry must be an object")
        missing = [field for field in _DENOISER_FIELDS if field not in config]
        if missing:
            raise ManifestError(f"manifest denoisers.{arm}: missing field '{missing[0]}'")
        if arm in _DENOISER_CONFIG_ARMS and "config" not in config:
            raise ManifestError(f"manifest denoisers.{arm}: missing field 'config'")
        if arm not in _DENOISER_CONFIG_ARMS and "config" in config:
            raise ManifestError(f"manifest denoisers.{arm}: field 'config' is not supported for this arm")
        paths: dict[str, Path] = {}
        for field in (*_DENOISER_FIELDS, *_DENOISER_OPTIONAL_FIELDS):
            if field not in config:
                continue
            value = config[field]
            if not isinstance(value, str) or not value.strip():
                raise ManifestError(f"manifest denoisers.{arm}: field '{field}' must be a non-empty string")
            paths[field] = Path(value)
        specs[arm] = DenoiserSpec(
            name=arm,
            python=paths["python"],
            worker=paths["worker"],
            checkpoint=paths["checkpoint"],
            config=paths.get("config"),
        )
    return specs
