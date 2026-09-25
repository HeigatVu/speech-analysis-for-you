"""Deterministic overlapping-chunk arithmetic for isolated-environment denoise workers.

Full-band denoisers (FullSubNet's LSTM in particular) cannot process a whole
session in one pass, so workers render fixed-length chunks and overlap-add them.
This module holds that arithmetic with numpy only: the workers import it from
their own directory, and the offline tests import it directly, so the chunk
boundaries, the unity-gain reconstruction, and the exact output length are
verified without any model in the loop.

Every chunk except a signal shorter than one chunk is exactly ``chunk_samples``
long. Weighting is a Hann window per chunk, normalized by the summed weights,
with the first chunk's ramp-up and the last chunk's ramp-down forced to 1.0
because no neighbour covers that audio. Overlap-add of a constant signal
therefore returns that same constant everywhere, first and last sample included.
"""

from collections.abc import Callable

import numpy as np


def chunk_bounds(n_samples: int, chunk_samples: int, hop_samples: int) -> list[tuple[int, int]]:
    """Half-open [start, stop) spans covering [0, n_samples) exactly once to the end.

    Full-length windows start every ``hop_samples`` samples; when the hop grid
    would leave a tail shorter than one hop, a final full-length window ending at
    ``n_samples`` covers it, so no chunk is ever a partial window.
    """
    if n_samples <= 0:
        return []
    if chunk_samples <= 0 or hop_samples <= 0:
        raise ValueError("chunk and hop must be positive")
    if hop_samples > chunk_samples:
        raise ValueError("hop must not exceed the chunk length")

    bounds: list[tuple[int, int]] = []
    start = 0
    while start + chunk_samples <= n_samples:
        bounds.append((start, start + chunk_samples))
        start += hop_samples

    if not bounds:
        bounds.append((0, n_samples))
    elif bounds[-1][1] < n_samples:
        bounds.append((n_samples - chunk_samples, n_samples))
    return bounds


def overlap_add(
    n_samples: int,
    chunk_samples: int,
    hop_samples: int,
    render: Callable[[int, int], np.ndarray],
) -> np.ndarray:
    """Overlap-add ``render(start, stop)`` chunk renders into exactly ``n_samples``.

    ``render`` must return exactly ``stop - start`` finite samples for the span it
    was given; anything else raises ValueError rather than being padded or
    trimmed silently.
    """
    bounds = chunk_bounds(n_samples, chunk_samples, hop_samples)
    if not bounds:
        raise ValueError("no chunks to render for an empty signal")

    window = np.hanning(chunk_samples)
    accumulated = np.zeros(n_samples, dtype=np.float64)
    weights = np.zeros(n_samples, dtype=np.float64)

    for index, (start, stop) in enumerate(bounds):
        piece = np.asarray(render(start, stop), dtype=np.float64)
        if piece.shape != (stop - start,):
            raise ValueError("chunk render returned the wrong number of samples")
        if not np.isfinite(piece).all():
            raise ValueError("chunk render returned non-finite samples")

        weight = window[: stop - start].copy()
        if index == 0:
            weight[: min(hop_samples, weight.size)] = 1.0
        if index == len(bounds) - 1:
            weight[max(weight.size - hop_samples, 0) :] = 1.0
        weight = np.maximum(weight, 1e-6)

        accumulated[start:stop] += piece * weight
        weights[start:stop] += weight

    return (accumulated / weights).astype(np.float32)
