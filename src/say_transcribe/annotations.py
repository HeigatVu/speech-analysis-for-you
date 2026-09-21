from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Any, Callable, Literal

SHA256_REGEX = re.compile(r"^[0-9a-f]{64}$")
ALLOWED_ROLES = {"participant", "investigator"}


class AnnotationValidationError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message


def _require_keys(data: dict[str, Any], expected: set[str], error_path: str) -> None:
    if set(data.keys()) != expected:
        raise AnnotationValidationError("INVALID_ANNOTATIONS", error_path)


def _require_id(data: dict[str, Any], field: str, error_path: str) -> str:
    value = data[field]
    if not isinstance(value, str) or not value.strip():
        raise AnnotationValidationError("INVALID_ANNOTATIONS", error_path)
    return value


def _require_bounds(data: dict[str, Any], error_path: str) -> tuple[int, int]:
    start_ms, end_ms = data["start_ms"], data["end_ms"]
    if (
        isinstance(start_ms, bool)
        or isinstance(end_ms, bool)
        or not isinstance(start_ms, int)
        or not isinstance(end_ms, int)
    ):
        raise AnnotationValidationError("INVALID_ANNOTATIONS", error_path)
    if start_ms < 0 or start_ms >= end_ms:
        raise AnnotationValidationError("INVALID_ANNOTATIONS", error_path)
    return start_ms, end_ms


def _parse_ordered_children(
    raw_children: Any,
    path: str,
    id_field: str,
    validator: Callable[[dict[str, Any]], Any],
) -> tuple[Any, ...]:
    """Validate a list of child objects, rejecting non-dict entries, duplicate
    ids, and children whose start_ms precedes the previous child's end_ms."""
    if not isinstance(raw_children, list):
        raise AnnotationValidationError("INVALID_ANNOTATIONS", path)

    parsed: list[Any] = []
    seen_ids: set[str] = set()
    prev_end = -1
    for child_dict in raw_children:
        if not isinstance(child_dict, dict):
            raise AnnotationValidationError("INVALID_ANNOTATIONS", f"{path}[]")
        child = validator(child_dict)
        child_id = getattr(child, id_field)
        if child_id in seen_ids:
            raise AnnotationValidationError(
                "INVALID_ANNOTATIONS", f"{path}[].{id_field}"
            )
        seen_ids.add(child_id)
        if child.start_ms < prev_end:
            raise AnnotationValidationError("INVALID_ANNOTATIONS", f"{path}[].start_ms")
        prev_end = child.end_ms
        parsed.append(child)
    return tuple(parsed)


@dataclass(frozen=True)
class TurnAnnotation:
    turn_id: str
    speaker_role: Literal["participant", "investigator"]
    start_ms: int
    end_ms: int


@dataclass(frozen=True)
class TaskAnnotation:
    task_id: str
    start_ms: int
    end_ms: int
    turns: tuple[TurnAnnotation, ...]


@dataclass(frozen=True)
class SessionAnnotations:
    schema_version: str
    source_audio: str
    source_sha256: str
    channel_index: int
    sample_rate: int
    review_status: Literal["approved"]
    tasks: tuple[TaskAnnotation, ...]


def _validate_turn(
    data: dict[str, Any], task_start_ms: int, task_end_ms: int
) -> TurnAnnotation:
    _require_keys(
        data, {"turn_id", "speaker_role", "start_ms", "end_ms"}, "turns[].<keys>"
    )
    turn_id = _require_id(data, "turn_id", "turns[].turn_id")

    role = data["speaker_role"]
    if role not in ALLOWED_ROLES:
        raise AnnotationValidationError("INVALID_ANNOTATIONS", "turns[].speaker_role")

    start_ms, end_ms = _require_bounds(data, "turns[].start_ms/end_ms")
    if start_ms < task_start_ms or end_ms > task_end_ms:
        raise AnnotationValidationError(
            "INVALID_ANNOTATIONS", "turns[].start_ms/end_ms"
        )

    return TurnAnnotation(
        turn_id=turn_id,
        speaker_role=role,
        start_ms=start_ms,
        end_ms=end_ms,
    )


def _validate_task(data: dict[str, Any]) -> TaskAnnotation:
    _require_keys(data, {"task_id", "start_ms", "end_ms", "turns"}, "tasks[].<keys>")
    task_id = _require_id(data, "task_id", "tasks[].task_id")
    start_ms, end_ms = _require_bounds(data, "tasks[].start_ms/end_ms")

    turns = _parse_ordered_children(
        data.get("turns", []),
        "tasks[].turns",
        "turn_id",
        lambda turn_dict: _validate_turn(turn_dict, start_ms, end_ms),
    )

    return TaskAnnotation(
        task_id=task_id,
        start_ms=start_ms,
        end_ms=end_ms,
        turns=turns,
    )


def load_approved_annotations(
    data: dict[str, Any], audio_root: Path | None = None
) -> SessionAnnotations:
    _require_keys(
        data,
        {
            "schema_version",
            "source_audio",
            "source_sha256",
            "channel_index",
            "sample_rate",
            "review_status",
            "tasks",
        },
        "<keys>",
    )

    status = data.get("review_status")
    if status != "approved":
        raise AnnotationValidationError("UNAPPROVED_ANNOTATIONS", "review_status")

    source_audio = data.get("source_audio", "")
    if (
        not isinstance(source_audio, str)
        or PurePath(source_audio).is_absolute()
        or ".." in PurePath(source_audio).parts
    ):
        raise AnnotationValidationError(
            "INVALID_ANNOTATIONS",
            "source_audio must be a non-empty relative path without '..'",
        )

    source_sha256 = data.get("source_sha256", "")
    if not isinstance(source_sha256, str) or not SHA256_REGEX.match(source_sha256):
        raise AnnotationValidationError(
            "INVALID_ANNOTATIONS",
            "source_sha256 must be a 64-character lowercase hex string",
        )

    if audio_root is not None:
        digest = hashlib.sha256((audio_root / source_audio).read_bytes()).hexdigest()
        if digest != source_sha256:
            raise AnnotationValidationError("SOURCE_HASH_MISMATCH", "source_sha256")

    channel_index = data.get("channel_index")
    if not isinstance(channel_index, int) or channel_index < 0:
        raise AnnotationValidationError(
            "INVALID_ANNOTATIONS", "channel_index must be an integer >= 0"
        )

    sample_rate = data.get("sample_rate")
    if not isinstance(sample_rate, int) or sample_rate <= 0:
        raise AnnotationValidationError(
            "INVALID_ANNOTATIONS", "sample_rate must be a positive integer"
        )

    tasks = _parse_ordered_children(
        data.get("tasks", []), "tasks", "task_id", _validate_task
    )

    return SessionAnnotations(
        schema_version=data["schema_version"],
        source_audio=source_audio,
        source_sha256=source_sha256,
        channel_index=channel_index,
        sample_rate=sample_rate,
        review_status="approved",
        tasks=tasks,
    )
