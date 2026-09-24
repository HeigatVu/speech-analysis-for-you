import json
from pathlib import Path
import re
from typing import Any, Sequence
import unicodedata

import numpy as np

from speech_features.formats.chat import InvalidChatError, decode_chat

_MAIN_TIER_LINE = re.compile(r"^\*[A-Z]{3}:\s*(.*)$")
_PUNCTUATION_ONLY = {".", "?", "!", ",", "...", "…"}


def levenshtein(seq1: Sequence[Any], seq2: Sequence[Any]) -> int:
    """Compute standard Levenshtein edit distance between two sequences."""
    n, m = len(seq1), len(seq2)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost,  # substitution
            )
    return dp[n][m]


def _extract_lenient_text_items(cha_text: str) -> dict[str, Any]:
    """Best-effort main-tier word/syllable/char scan for transcripts the strict
    CHAT parser rejects (untimed drafts, malformed headers). Text-similarity
    metrics only: intervals and morphosyntax tiers need the strict parser, so
    those come back empty rather than being guessed at.
    """
    syllables: list[str] = []
    words: list[str] = []
    chars: list[str] = []
    for line in cha_text.splitlines():
        match = _MAIN_TIER_LINE.match(line.strip())
        if not match:
            continue
        content = re.sub(r"[\x15•]\d+_\d+[\x15•]", "", match.group(1))
        for item in content.split():
            norm_word = unicodedata.normalize("NFC", item.strip())
            if not norm_word or norm_word in _PUNCTUATION_ONLY:
                continue
            words.append(norm_word)
            syllables.extend(norm_word.split("_"))
            chars.extend(norm_word.replace("_", ""))
    return {
        "syllables": syllables,
        "words": words,
        "chars": chars,
        "pos": [],
        "heads": [],
        "rels": [],
        "intervals": [],
    }


def extract_session_items(cha_text: str) -> dict[str, Any]:
    """Parse .cha transcript and extract evaluation items without logging private data.

    Falls back to a lenient main-tier scan for text-similarity metrics when the
    strict CHAT parser rejects the file; see `_extract_lenient_text_items`.
    """
    try:
        doc = decode_chat(cha_text)
    except InvalidChatError:
        return _extract_lenient_text_items(cha_text)

    syllables: list[str] = []
    words: list[str] = []
    chars: list[str] = []
    pos_tags: list[str] = []
    heads: list[str] = []
    rels: list[str] = []
    speaker_intervals: list[tuple[int, int, str]] = []  # (start_ms, end_ms, speaker)

    layers = {a.layer: a.values for a in doc.annotations}
    mor_dict = layers.get("mor", {})
    gra_dict = layers.get("gra", {})

    for u in doc.utterances:
        start_ms = int(round(u.start_s * 1000.0))
        end_ms = int(round(u.end_s * 1000.0))
        speaker_intervals.append((start_ms, end_ms, u.speaker_id))

        for t in u.tokens:
            norm_word = unicodedata.normalize("NFC", t.text.strip())
            if not norm_word or norm_word in {".", "?", "!", ",", "...", "…"}:
                continue
            words.append(norm_word)
            # Syllables from word
            syls = norm_word.split("_")
            syllables.extend(syls)
            chars.extend(list(norm_word.replace("_", "")))

            # Morphosyntax
            if t.id in mor_dict:
                mor_val = mor_dict[t.id]
                pos_part = mor_val.split("|")[0].lower() if "|" in mor_val else mor_val.lower()
                pos_tags.append(pos_part)

            if t.id in gra_dict:
                gra_val = gra_dict[t.id]
                parts = gra_val.split("|")
                if len(parts) >= 3:
                    heads.append(parts[1])
                    rels.append(parts[2].upper())

    return {
        "syllables": syllables,
        "words": words,
        "chars": chars,
        "pos": pos_tags,
        "heads": heads,
        "rels": rels,
        "intervals": speaker_intervals,
    }


def compute_der(
    ref_intervals: list[tuple[int, int, str]],
    hyp_intervals: list[tuple[int, int, str]],
    collar_ms: int = 250,
) -> float:
    """Compute Diarization Error Rate with forgiveness collar."""
    if not ref_intervals:
        return 0.0

    total_ref_time = 0
    confusion_time = 0

    for r_start, r_end, r_spk in ref_intervals:
        dur = max(0, r_end - r_start)
        if dur <= 2 * collar_ms:
            continue
        c_start = r_start + collar_ms
        c_end = r_end - collar_ms
        c_dur = c_end - c_start
        total_ref_time += c_dur

        # Find matching speaker in hypothesis
        matched_dur = 0
        for h_start, h_end, h_spk in hyp_intervals:
            overlap = max(0, min(c_end, h_end) - max(c_start, h_start))
            if overlap > 0 and h_spk == r_spk:
                matched_dur += overlap

        confusion_time += max(0, c_dur - matched_dur)

    if total_ref_time == 0:
        return 0.0
    return min(1.0, confusion_time / total_ref_time)


def run_evaluation(
    gold_dir: Path,
    pred_dir: Path,
    output_report: Path,
    bootstrap_samples: int = 2000,
    seed: int = 42,
) -> int:
    """Evaluate predictions against gold transcripts and output aggregate metrics JSON."""
    gold_files = sorted(list(gold_dir.glob("*.cha")))
    if not gold_files:
        return 2

    valid_sessions = 0
    total_sessions = len(gold_files)
    session_syer: list[float] = []
    session_cer: list[float] = []
    session_wer: list[float] = []
    session_der: list[float] = []

    pos_correct = 0
    pos_total = 0
    uas_correct = 0
    las_correct = 0
    syntax_total = 0

    for g_path in gold_files:
        p_path = pred_dir / g_path.name
        if not p_path.exists():
            continue

        try:
            g_data = extract_session_items(g_path.read_text(encoding="utf-8"))
            p_data = extract_session_items(p_path.read_text(encoding="utf-8"))
            valid_sessions += 1
        except Exception:
            continue

        # SyER
        ref_syl = g_data["syllables"]
        hyp_syl = p_data["syllables"]
        syer = (levenshtein(ref_syl, hyp_syl) / len(ref_syl)) if ref_syl else 0.0
        session_syer.append(min(1.0, syer))

        # CER
        ref_chr = g_data["chars"]
        hyp_chr = p_data["chars"]
        cer = (levenshtein(ref_chr, hyp_chr) / len(ref_chr)) if ref_chr else 0.0
        session_cer.append(min(1.0, cer))

        # WER
        ref_wrd = g_data["words"]
        hyp_wrd = p_data["words"]
        wer = (levenshtein(ref_wrd, hyp_wrd) / len(ref_wrd)) if ref_wrd else 0.0
        session_wer.append(min(1.0, wer))

        # DER
        der = compute_der(g_data["intervals"], p_data["intervals"])
        session_der.append(der)

        # Morphosyntax tier agreement
        n_pos = min(len(g_data["pos"]), len(p_data["pos"]))
        pos_total += n_pos
        for i in range(n_pos):
            if g_data["pos"][i] == p_data["pos"][i]:
                pos_correct += 1

        n_syn = min(len(g_data["heads"]), len(p_data["heads"]))
        syntax_total += n_syn
        for i in range(n_syn):
            head_match = g_data["heads"][i] == p_data["heads"][i]
            rel_match = g_data["rels"][i] == p_data["rels"][i]
            if head_match:
                uas_correct += 1
            if head_match and rel_match:
                las_correct += 1

    pass_rate = valid_sessions / total_sessions if total_sessions > 0 else 0.0

    # Deterministic Bootstrap
    rng = np.random.default_rng(seed)

    def bootstrap_ci(values: list[float]) -> dict[str, Any]:
        if not values:
            return {"mean": 0.0, "ci_95": [0.0, 0.0]}
        arr = np.array(values, dtype=np.float64)
        mean_val = float(np.mean(arr))
        if len(arr) == 1:
            return {"mean": mean_val, "ci_95": [mean_val, mean_val]}

        boot_means = []
        for _ in range(bootstrap_samples):
            resample = rng.choice(arr, size=len(arr), replace=True)
            boot_means.append(float(np.mean(resample)))

        ci_low = float(np.percentile(boot_means, 2.5))
        ci_high = float(np.percentile(boot_means, 97.5))
        return {
            "mean": round(mean_val, 4),
            "ci_95": [round(ci_low, 4), round(ci_high, 4)],
        }

    report = {
        "sessions_evaluated": valid_sessions,
        "format_pass_rate": round(pass_rate, 4),
        "metrics": {
            "syer": bootstrap_ci(session_syer),
            "cer": bootstrap_ci(session_cer),
            "wer": bootstrap_ci(session_wer),
            "der": bootstrap_ci(session_der),
            "pos_accuracy": round(pos_correct / pos_total, 4) if pos_total > 0 else 0.0,
            "uas": round(uas_correct / syntax_total, 4) if syntax_total > 0 else 0.0,
            "las": round(las_correct / syntax_total, 4) if syntax_total > 0 else 0.0,
        },
    }

    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0
