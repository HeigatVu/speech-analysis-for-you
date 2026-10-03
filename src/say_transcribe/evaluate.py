import json
from pathlib import Path
import re
from typing import Any, Sequence
import unicodedata

import numpy as np

from speech_features.formats.chat import InvalidChatError, decode_chat, tier_roles

_MAIN_TIER_LINE = re.compile(r"^\*[A-Z]{3}:\s*(.*)$")
_PUNCTUATION_ONLY = {".", "?", "!", ",", "...", "…"}

# Spoken-domain scoring keeps what a microphone would have heard: words,
# physically spoken fillers and retraced words. Untranscribed spans, nonword
# fragments and annotation syntax carry no scorable surface.
_SPOKEN_ROLES = ("word", "retraced", "filler")


def _spoken_surface(role: str, item: str) -> str:
    """Surface form of a spoken main-tier item with its CHAT markup removed."""
    if role == "filler":
        return item[2:]
    if role == "retraced":
        return item.strip("<>")
    return item


def _uncertainty_code(item: str) -> str | None:
    """Stable code for a skipped item that scoring cannot settle, else ``None``.

    Pauses and bare retrace markers are expected markup rather than uncertainty.
    """
    if item.startswith("["):
        return "REPLACEMENT_ANNOTATION" if ":" in item else None
    if item.startswith("("):
        return None
    if item.startswith("&"):
        return "NONWORD_FRAGMENT"
    return None


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
        "uncertainty": [],
    }


def extract_session_items(cha_text: str) -> dict[str, Any]:
    """Parse .cha transcript and extract evaluation items without logging private data.

    Falls back to a lenient main-tier scan for text-similarity metrics when the
    strict CHAT parser rejects the file; see `_extract_lenient_text_items`.
    """
    try:
        # Untimed utterances are legitimate in a prediction (repetition-guard `xxx`, words
        # the aligner could not place); they simply contribute no interval.
        doc = decode_chat(cha_text, require_timing=False)
    except InvalidChatError:
        return _extract_lenient_text_items(cha_text)

    syllables: list[str] = []
    words: list[str] = []
    chars: list[str] = []
    pos_tags: list[str] = []
    heads: list[str] = []
    rels: list[str] = []
    speaker_intervals: list[tuple[int, int, str]] = []  # (start_ms, end_ms, speaker)
    uncertainty: list[dict[str, Any]] = []

    layers = {a.layer: a.values for a in doc.annotations}
    mor_dict = layers.get("mor", {})
    gra_dict = layers.get("gra", {})

    for index, u in enumerate(doc.utterances):
        codes: set[str] = set()
        if u.start_s is not None and u.end_s is not None:
            start_ms = int(round(u.start_s * 1000.0))
            end_ms = int(round(u.end_s * 1000.0))
            speaker_intervals.append((start_ms, end_ms, u.speaker_id))
            if end_ms <= start_ms:
                codes.add("ZERO_DURATION_UTTERANCE")

        roles = tier_roles([t.text for t in u.tokens])
        for role, t in zip(roles, u.tokens):
            if role == "untranscribed":
                codes.add("UNTRANSCRIBED_SPAN")
                continue
            if role == "skip":
                code = _uncertainty_code(t.text)
                if code is not None:
                    codes.add(code)
                continue

            norm_word = unicodedata.normalize("NFC", _spoken_surface(role, t.text).strip())
            if not norm_word or norm_word in _PUNCTUATION_ONLY:
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

        for code in sorted(codes):
            uncertainty.append({"utterance_index": index, "code": code})

    return {
        "syllables": syllables,
        "words": words,
        "chars": chars,
        "pos": pos_tags,
        "heads": heads,
        "rels": rels,
        "intervals": speaker_intervals,
        "uncertainty": uncertainty,
    }


def items_from_texts(texts: Sequence[str]) -> dict[str, Any]:
    """Tokenize plain transcript text into the item shape of ``extract_session_items``.

    Same conventions: NFC normalization, punctuation-only tokens dropped,
    ``_``-delimited syllable splitting. Morphosyntax and intervals stay empty.
    """
    syllables: list[str] = []
    words: list[str] = []
    chars: list[str] = []
    for text in texts:
        for item in text.split():
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
        "uncertainty": [],
    }


def score_uncapped(gold: dict[str, Any], hyp: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Uncapped SyER/CER/WER with edit counts for participant-level pooling.

    Unlike ``run_evaluation``'s capped metrics, ``rate`` is never clamped and may
    exceed 1.0 for insertion-heavy hypotheses.
    """
    scores: dict[str, dict[str, Any]] = {}
    for key, field in (("syer", "syllables"), ("cer", "chars"), ("wer", "words")):
        ref = gold[field]
        hyp_items = hyp[field]
        edits = levenshtein(ref, hyp_items)
        ref_count = len(ref)
        scores[key] = {
            "edits": edits,
            "ref_count": ref_count,
            "rate": edits / ref_count if ref_count else 0.0,
        }
    return scores


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

    uncertainty_counts: dict[str, int] = {}

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

        for entry in g_data["uncertainty"]:
            uncertainty_counts[entry["code"]] = uncertainty_counts.get(entry["code"], 0) + 1

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
        # A prediction with no timed utterance has nothing to score: report DER absent
        # instead of the 1.0 that compute_der would return for an empty hypothesis.
        if p_data["intervals"] or not g_data["intervals"]:
            session_der.append(compute_der(g_data["intervals"], p_data["intervals"]))

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
            return {"mean": None, "ci_95": None}
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
        "der_sessions_scored": len(session_der),
        "uncertainty": dict(sorted(uncertainty_counts.items())),
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
