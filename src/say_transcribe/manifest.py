"""Study manifest loading for the opt-in preprocessing study.

Manifest rows are private inputs: error messages name row indexes and field
names only, never values, paths, or transcript content.
"""

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

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


def load_manifest(path: Path) -> tuple[ManifestRow, ...]:
    """Load and validate a study manifest; raise ManifestError on any schema problem."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError:
        raise ManifestError("manifest file could not be read") from None
    except json.JSONDecodeError:
        raise ManifestError("manifest is not valid JSON") from None

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
