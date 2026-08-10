"""Small helpers for reviewed motor-speech annotation layers."""

from __future__ import annotations

import math

import numpy as np

from ...document import DocumentToken


def layer_values(document, name: str, token_ids: set[str]) -> dict[str, str | float] | None:
    """Return a complete token slice of one layer, or ``None`` when unavailable."""
    layer = next((item for item in document.annotations if item.layer == name), None)
    if layer is None or any(token_id not in layer.values for token_id in token_ids):
        return None
    return {token_id: layer.values[token_id] for token_id in token_ids}


def target_tokens(document, speaker_id: str) -> tuple[DocumentToken, ...]:
    """Return target-speaker tokens in document order."""
    return tuple(
        token
        for utterance in document.utterances
        if utterance.speaker_id == speaker_id
        for token in utterance.tokens
    )


def finite_mean_sd(values) -> tuple[float, float]:
    """Return population mean/SD, rejecting empty or non-finite input."""
    array = np.asarray(values, dtype=float)
    if array.size == 0 or not np.all(np.isfinite(array)):
        return math.nan, math.nan
    return float(array.mean()), float(array.std(ddof=0))


def polygon_area(points: tuple[tuple[float, float], ...]) -> float:
    """Return the shoelace area of a polygon."""
    return (
        abs(sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1])))
        / 2.0
    )


def token_duration(token: DocumentToken) -> float | None:
    """Return a finite positive token duration when explicitly aligned."""
    if token.start_s is None or token.end_s is None:
        return None
    duration = token.end_s - token.start_s
    return duration if math.isfinite(duration) and duration > 0.0 else None


__all__ = ["finite_mean_sd", "layer_values", "polygon_area", "target_tokens", "token_duration"]
