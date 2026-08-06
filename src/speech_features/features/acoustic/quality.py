"""Recording quality measures (Task 5).

All four measures operate on the loaded mono float array and its sample rate:

- ``audio_duration_s`` = ``len(audio) / sample_rate``.
- ``audio_dc_offset`` = sample mean (normalized amplitude; 0 for silence).
- ``audio_clipping_ratio`` = fraction of samples whose absolute amplitude is
  at or beyond the normalized clipping boundary ``|x| >= 1.0``.
- ``audio_rms_dbfs`` = ``20*log10(rms)`` with ``rms = sqrt(mean(x**2))``.
  Digital silence (rms == 0) yields ``NaN``; the caller pairs it with a
  structured issue instead of reporting ``-inf`` or a fabricated value.
"""

from __future__ import annotations

import math

import numpy as np


def quality_features(audio, sample_rate: int) -> dict[str, float]:
    """Return the four recording-level quality features as floats."""
    duration_s = float(audio.size) / sample_rate
    rms = float(np.sqrt(np.mean(audio**2)))
    return {
        "audio_duration_s": duration_s,
        "audio_dc_offset": float(np.mean(audio)),
        "audio_clipping_ratio": float(np.count_nonzero(np.abs(audio) >= 1.0) / audio.size),
        "audio_rms_dbfs": 20.0 * math.log10(rms) if rms > 0.0 else math.nan,
    }
