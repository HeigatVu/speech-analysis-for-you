from dataclasses import dataclass, replace
from typing import Any, Sequence

from say_transcribe.asr import AsrError
from speech_features.formats.chat import mor_projection
from say_transcribe.word_grouping import GroupedWord


_NO_FEATS_POS = frozenset(
    {"adp", "adv", "cconj", "intj", "num", "part", "sconj", "sym", "x", "punct"}
)


@dataclass(frozen=True)
class MorItem:
    pos: str
    lemma: str
    feats: str | None = None

    def format_mor(self) -> str:
        if self.pos == "cm" and self.lemma == "cm":
            return "cm|cm"
        if self.lemma in {".", "?", "!", "...", "…"}:
            return self.lemma
        suffix = "".join(f"-{value}" for value in _feature_suffixes(self.pos, self.feats))
        return f"{self.pos}|{self.lemma}{suffix}"


def _feature_suffixes(pos: str, feats: str | None) -> list[str]:
    """Vietnamese-relevant handlers from pinned Batchalign3 features.rs."""
    if not feats or pos in _NO_FEATS_POS:
        return []
    values = dict(pair.split("=", 1) for pair in feats.split("|") if "=" in pair)
    agreement = values.get("Number", "")[:1] + (
        "4" if values.get("Person") == "0" else values.get("Person", "")
    )
    keys = {
        "verb": ("VerbForm", "Aspect", "Mood", "Tense", "Polarity", "Polite"),
        "aux": ("VerbForm", "Aspect", "Mood", "Tense", "Polarity", "Polite"),
        "pron": ("PronType", "Case"),
        "det": ("Gender", "Definite", "PronType", "Number"),
        "adj": ("Degree", "Case"),
        "noun": ("Gender", "Number", "Case", "PronType"),
        "propn": ("Gender", "Number", "Case", "PronType"),
    }.get(pos, ())
    suffixes = [values[key] for key in keys if values.get(key) and not (
        (key == "Gender" and values[key] in {"Com", "Com,Neut"})
        or (key == "Degree" and values[key] == "Pos")
        or (key == "Number" and pos in {"noun", "propn"} and values[key] == "Sing")
    )]
    if pos == "pron" and values.get("Reflex") == "Yes":
        suffixes.append("reflx")
    if pos in {"verb", "aux", "pron", "adj"} and agreement:
        suffixes.append(agreement)
    if pos == "det":
        possessor = values.get("NumberPsor", "")[:1] + values.get("PersonPsor", "")
        if possessor:
            suffixes.append(possessor)
    return suffixes


@dataclass(frozen=True)
class GraItem:
    index: int
    head: int
    rel: str

    def format_gra(self) -> str:
        return f"{self.index}|{self.head}|{self.rel}"


@dataclass(frozen=True)
class UtteranceMorphosyntax:
    mor_items: tuple[MorItem, ...]
    gra_items: tuple[GraItem, ...]

    def mor_line(self) -> str:
        return "%mor:\t" + " ".join(item.format_mor() for item in self.mor_items)

    def gra_line(self) -> str:
        return "%gra:\t" + " ".join(item.format_gra() for item in self.gra_items)


class StanzaBackend:
    """Lazy-loaded Stanza backend for Vietnamese morphosyntax."""

    def __init__(self, lang: str = "vi", device: str = "cpu") -> None:
        self.lang = lang
        self.device = device
        self._nlp: Any = None

    def load(self) -> None:
        if self.device == "cuda":
            try:
                import torch

                if not torch.cuda.is_available():
                    raise AsrError("GPU_UNAVAILABLE", "CUDA device requested but not available")
            except ImportError:
                raise AsrError("GPU_UNAVAILABLE", "PyTorch with CUDA not available") from None

        try:
            import stanza

            use_gpu = self.device == "cuda"
            self._nlp = stanza.Pipeline(
                self.lang,
                processors="tokenize,pos,lemma,depparse",
                tokenize_pretokenized=True,
                download_method=None,
                use_gpu=use_gpu,
                verbose=False,
            )
        except AsrError:
            raise
        except Exception:
            raise AsrError("MODEL_UNAVAILABLE", "Failed to load Stanza Vietnamese pipeline") from None

    def parse_pretokenized(self, tokens: Sequence[str]) -> Any:
        if self._nlp is None:
            self.load()
        try:
            doc = self._nlp([[t for t in tokens]])
            return doc
        except Exception:
            raise AsrError("MODEL_UNAVAILABLE", "Stanza parse execution failed") from None


def mor_members(words: Sequence[GroupedWord]) -> tuple[GroupedWord, ...]:
    """Words `%mor` and `%gra` align to: not retraced, filler, `xxx`, pause or annotation."""
    return tuple(
        replace(words[index], word=surface,
                start_ms=words[index].start_ms if surface == words[index].word else None,
                end_ms=words[index].end_ms if surface == words[index].word else None)
        for index, surface in mor_projection([w.word for w in words])
    )


def project_morphosyntax(
    grouped_words: Sequence[GroupedWord],
    backend: StanzaBackend | None = None,
) -> UtteranceMorphosyntax | None:
    """Project grouped Vietnamese words to %mor and %gra tiers using Stanza UD-VTB.

    Returns UtteranceMorphosyntax or None if parsing fails.
    """
    # chatter rejects %mor/%gra on an utterance with no analysable word (E706, E722).
    if not any(c.isalnum() for w in grouped_words for c in w.word):
        return None

    tokens = [w.word for w in mor_members(grouped_words)]
    if backend is None:
        backend = StanzaBackend()

    try:
        doc = backend.parse_pretokenized(tokens)
        if not doc or not doc.sentences:
            return None
        sentence = doc.sentences[0]
        parsed_words = sentence.words
    except Exception:
        return None

    if len(parsed_words) != len(tokens):
        return None

    mor_items: list[MorItem] = []
    gra_items: list[GraItem] = []

    # First pass: find the root index for terminator attachment
    root_index = 0
    for w in parsed_words:
        deprel_clean = (w.deprel or "").upper().replace(":", "-")
        if w.head == 0 or deprel_clean == "ROOT":
            root_index = w.id
            break

    n_words = len(parsed_words)

    for i, w in enumerate(parsed_words, start=1):
        token_text = tokens[i - 1]
        is_final_punct = (i == n_words) and token_text in {".", "?", "!", "...", "…"}

        # %mor projection
        if token_text == ",":
            pos = "cm"
            lemma = "cm"
            feats = None
        elif is_final_punct:
            pos = "punct"
            lemma = token_text
            feats = None
        else:
            pos = (w.upos or "x").lower()
            lemma = w.lemma or token_text  # preserves '_' in Vietnamese compounds
            feats = w.feats if w.feats and w.feats != "None" else None

        mor_items.append(MorItem(pos=pos, lemma=lemma, feats=feats))

        # %gra projection
        if is_final_punct:
            # Terminator gets (n)|root|PUNCT
            head = root_index if root_index > 0 else 0
            rel = "PUNCT"
        else:
            deprel_clean = (w.deprel or "").upper().replace(":", "-")
            head = w.head if w.head is not None else 0
            rel = deprel_clean if deprel_clean else "DEP"

            # Enforce root-head-0 / REL == ROOT joint invariant
            if head == 0 or rel == "ROOT":
                head = 0
                rel = "ROOT"

        gra_items.append(GraItem(index=i, head=head, rel=rel))

    return UtteranceMorphosyntax(
        mor_items=tuple(mor_items),
        gra_items=tuple(gra_items),
    )
