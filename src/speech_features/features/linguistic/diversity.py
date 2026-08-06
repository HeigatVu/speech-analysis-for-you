"""Pure lexical diversity formulas over normalized form sequences (Task 8).

All functions take an ordered sequence of already-normalized (NFC + casefold)
forms and return a float; ``math.nan`` signals an undefined value, never a
fabricated zero. For a sequence of length ``N`` with ``V`` types and ``V1``
hapax types:

- ``ttr`` — ``V / N``.
- ``mattr_20`` — mean TTR of every contiguous length-20 window;
  ``N < 20`` is ``NaN``.
- ``mtld`` — the standard forward/reverse factor method with threshold
  ``0.72``: a factor closes when the running TTR is ``<= 0.72``; a trailing
  open factor contributes the partial factor ``(1 - ttr) / (1 - 0.72)``; the
  reported value is the mean of the forward and reverse MTLD. A zero factor
  denominator (no factor ever closes, e.g. a single-type or all-distinct
  sample) is ``NaN``.
- ``hdd_42`` — HD-D with sample size 42,
  ``sum_types(1 - C(N-f_i,42)/C(N,42)) / 42``, requiring ``N >= 42``; each
  ratio is the bounded iterative product ``product(k=0..41, (N-f_i-k)/(N-k))``
  (exactly zero when ``N - f_i < 42``); no factorials or large combinations
  are formed.
- ``hapax_ratio`` — ``V1 / N``.
- ``brunet_w`` — ``N ** (V ** -0.165)``.
- ``honore_r`` — ``100 * log(N) / (1 - V1 / V)``; a zero divisor
  (``V == V1``) is ``NaN``.
- ``entropy`` — normalized Shannon entropy
  ``-sum(p_i * log(p_i)) / log(V)``; a single-type non-empty sample is
  defined as ``0``.
"""

from __future__ import annotations

import math
from collections import Counter

_THRESHOLD = 0.72
_MATTR_WINDOW = 20
_HDD_SAMPLE = 42


def ttr(forms) -> float:
    n = len(forms)
    return len(Counter(forms)) / n if n else math.nan


def mattr_20(forms) -> float:
    n = len(forms)
    if n < _MATTR_WINDOW:
        return math.nan
    windows = n - _MATTR_WINDOW + 1
    return (
        sum(len(set(forms[i : i + _MATTR_WINDOW])) / _MATTR_WINDOW for i in range(windows))
        / windows
    )


def _mtld_directional(forms) -> float:
    seen = set()
    run = 0
    closed = 0
    ttr_value = 1.0
    for form in forms:
        seen.add(form)
        run += 1
        ttr_value = len(seen) / run
        if ttr_value <= _THRESHOLD:
            closed += 1
            seen = set()
            run = 0
    if run:
        return closed + (1.0 - ttr_value) / (1.0 - _THRESHOLD)
    return float(closed)


def mtld(forms) -> float:
    n = len(forms)
    if n == 0:
        return math.nan
    forward = _mtld_directional(forms)
    reverse = _mtld_directional(reversed(forms))
    if forward == 0.0 or reverse == 0.0:
        return math.nan
    return (n / forward + n / reverse) / 2.0


def hdd_42(forms) -> float:
    n = len(forms)
    if n < _HDD_SAMPLE:
        return math.nan
    total = 0.0
    for frequency in Counter(forms).values():
        ratio = 1.0
        for k in range(_HDD_SAMPLE):
            ratio *= (n - frequency - k) / (n - k)
        total += 1.0 - ratio
    return total / _HDD_SAMPLE


def hapax_ratio(forms) -> float:
    counts = list(Counter(forms).values())
    n = len(forms)
    if n == 0:
        return math.nan
    v1 = sum(1 for count in counts if count == 1)
    return v1 / n


def brunet_w(forms) -> float:
    n = len(forms)
    v = len(Counter(forms))
    if n == 0 or v == 0:
        return math.nan
    return n ** (v**-0.165)


def honore_r(forms) -> float:
    counts = list(Counter(forms).values())
    n = len(forms)
    v = len(counts)
    if n == 0 or v == 0:
        return math.nan
    v1 = sum(1 for count in counts if count == 1)
    if v1 == v:
        return math.nan
    return 100.0 * math.log(n) / (1.0 - v1 / v)


def entropy(forms) -> float:
    n = len(forms)
    if n == 0:
        return math.nan
    counts = Counter(forms)
    v = len(counts)
    if v == 1:
        return 0.0
    return -sum((count / n) * math.log(count / n) for count in counts.values()) / math.log(v)
