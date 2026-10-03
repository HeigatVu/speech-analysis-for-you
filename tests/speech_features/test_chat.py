"""Tests for the clinical CHAT interchange (Task 3).

Covers the synthetic Batchalign-shaped Vietnamese fixture (`@Begin`,
`@Languages: vie`, `@Participants`, `@ID`, `@Media`, speaker tiers,
utterance and word media bullets, `%mor`, `%gra`, fillers, fragments,
`[/]`, `[//]`, error codes, one unsupported dependent tier, `@End`),
load/save dispatch round trips, verbatim preservation of unsupported
lines/tiers with a structured `UNSUPPORTED_CHAT_TIER` warning, provenance
(origin plus SHA-256 input hash), `INVALID_CHAT` rejection of malformed
input, and the prohibition on importing Batchalign/ASR.
"""

import hashlib
import unicodedata
from pathlib import Path

import pytest

from speech_features.document import (
    InvalidDocumentError,
    SpeechDocument,
    load_document,
    save_document,
)
from speech_features.formats.chat import ChatTierWarning, InvalidChatError

CHAT_FIXTURE = """\
@Begin
@Languages:\tvie
@Participants:\tPAR Nguyễn Thị An, INV Nguyễn Văn Hùng
@ID:\tvie|PAR|Nguyễn Thị An|participant|||||
@ID:\tvie|INV|Nguyễn Văn Hùng|examiner|||||
@Media:\trecording.wav | audio
*INV:\tCon đang làm gì ?
%xaud:\trecording.wav 0.000 1.200
*PAR:\tCon đang đi chơi .
%xaud:\trecording.wav 1.200 2.840
%xaud:\trecording.wav 1.200 1.600 1.610 2.050 2.060 2.500 2.510 2.790 2.800 2.840
%mor:\tCon|pro đang|aux đi|v chơi|v .|punct
%gra:\t1|2|nsubj 2|3|aux 3|4|advmod 4|4|root 5|4|punct
*PAR:\tà &uh con đứa bé- [//] con đi nha .
%xaud:\trecording.wav 2.900 4.400
%mor:\tà|part &uh|co con|pro đứa|n bé|n con|pro đi|v nha|part .|punct
%pho:\tcon@1:do1
*PAR:\ttôi đi [/] tôi đi .
%xaud:\trecording.wav 4.500 5.400
%mor:\ttôi|pro đi|v tôi|pro đi|v .|punct
*PAR:\ttôi [*] đi .
%xaud:\trecording.wav 5.500 6.200
%mor:\ttôi|pro đi|v .|punct
@End
"""


def _write(tmp_path, text=CHAT_FIXTURE, name="session.cha"):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def _semantic_equals(a: SpeechDocument, b: SpeechDocument) -> bool:
    """Compare every field except origin/provenance and load-time warnings."""
    return (
        a.language == b.language
        and a.media == b.media
        and a.speakers == b.speakers
        and a.utterances == b.utterances
        and a.annotations == b.annotations
        and a.raw_tiers == b.raw_tiers
    )


class TestChatLoad:
    def test_loads_documented_subset(self, tmp_path):
        doc = load_document(_write(tmp_path))
        assert doc.language == "vie"
        assert doc.document_id == "recording.wav"
        assert len(doc.media) == 1
        assert (doc.media[0].id, doc.media[0].kind, doc.media[0].path) == (
            "recording.wav",
            "audio",
            "recording.wav",
        )
        assert [(s.id, s.name, s.role) for s in doc.speakers] == [
            ("PAR", "Nguyễn Thị An", "participant"),
            ("INV", "Nguyễn Văn Hùng", "examiner"),
        ]
        assert [u.id for u in doc.utterances] == ["u0001", "u0002", "u0003", "u0004", "u0005"]
        assert [u.speaker_id for u in doc.utterances] == ["INV", "PAR", "PAR", "PAR", "PAR"]
        assert [(u.start_s, u.end_s) for u in doc.utterances] == [
            (0.0, 1.2),
            (1.2, 2.84),
            (2.9, 4.4),
            (4.5, 5.4),
            (5.5, 6.2),
        ]

    def test_token_texts_and_times(self, tmp_path):
        doc = load_document(_write(tmp_path))
        texts = [[t.text for t in u.tokens] for u in doc.utterances]
        assert texts[0] == ["Con", "đang", "làm", "gì", "?"]
        assert texts[1] == ["Con", "đang", "đi", "chơi", "."]
        assert texts[2] == ["à", "&uh", "con", "đứa", "bé-", "[//]", "con", "đi", "nha", "."]
        assert texts[3] == ["tôi", "đi", "[/]", "tôi", "đi", "."]
        assert texts[4] == ["tôi", "[*]", "đi", "."]
        times = [(t.start_s, t.end_s) for t in doc.utterances[1].tokens]
        assert times == [(1.2, 1.6), (1.61, 2.05), (2.06, 2.5), (2.51, 2.79), (2.8, 2.84)]
        assert all(t.start_s is None for t in doc.utterances[2].tokens)

    def test_token_kinds_classify_chat_items(self, tmp_path):
        doc = load_document(_write(tmp_path))
        kinds = [t.kind for t in doc.utterances[2].tokens]
        assert kinds == [
            "word",
            "filler",
            "word",
            "word",
            "fragment",
            "revision",
            "word",
            "word",
            "word",
            "word",
        ]
        assert [t.kind for t in doc.utterances[3].tokens] == [
            "word",
            "word",
            "retracing",
            "word",
            "word",
            "word",
        ]
        assert [t.kind for t in doc.utterances[4].tokens] == ["word", "error", "word", "word"]
        assert all(t.kind == "word" for t in doc.utterances[0].tokens)

    def test_mor_and_gra_annotation_layers(self, tmp_path):
        doc = load_document(_write(tmp_path))
        layers = {a.layer: a for a in doc.annotations}
        assert set(layers) == {"mor", "gra"}
        assert layers["mor"].source == "chat"
        assert layers["mor"].confidence == 1.0
        u2 = doc.utterances[1]
        assert [layers["mor"].values[t.id] for t in u2.tokens] == [
            "Con|pro",
            "đang|aux",
            "đi|v",
            "chơi|v",
            ".|punct",
        ]
        assert [layers["gra"].values[t.id] for t in u2.tokens] == [
            "1|2|nsubj",
            "2|3|aux",
            "3|4|advmod",
            "4|4|root",
            "5|4|punct",
        ]
        u3 = doc.utterances[2]
        assert len(layers["mor"].values) == 22  # 5 + 9 + 5 + 3 items across utterances
        assert u3.tokens[5].id not in layers["mor"].values  # code token carries no %mor item
        assert [layers["mor"].values[t.id] for t in u3.tokens if t.kind != "revision"][:3] == [
            "à|part",
            "&uh|co",
            "con|pro",
        ]

    def test_no_word_grouping_inferred(self, tmp_path):
        doc = load_document(_write(tmp_path))
        for token in (t for u in doc.utterances for t in u.tokens):
            assert token.word_id is None
            assert token.language is None
            assert token.dep_head is None
            assert token.dep_rel is None

    def test_vietnamese_text_preserved(self, tmp_path):
        decomposed = "@Participants:\tPAR Nguye\u0303n Thị An, INV Nguyễn Văn Hùng"
        text = CHAT_FIXTURE.replace(
            "@Participants:\tPAR Nguyễn Thị An, INV Nguyễn Văn Hùng", decomposed
        )
        doc = load_document(_write(tmp_path, text=text))
        par = next(s for s in doc.speakers if s.id == "PAR")
        assert par.name == "Nguyễn Thị An"
        assert par.name == unicodedata.normalize("NFC", par.name)
        assert [t.text for t in doc.utterances[1].tokens] == ["Con", "đang", "đi", "chơi", "."]
        assert doc.utterances[1].tokens[2].text == "đi"
        assert doc.utterances[1].tokens[2].text != "di"

    def test_unknown_dependent_tier_preserved_with_warning(self, tmp_path):
        doc = load_document(_write(tmp_path))
        assert doc.raw_tiers == {"%pho": "%pho:\tcon@1:do1"}
        assert doc.warnings == (ChatTierWarning(tier="%pho", line="%pho:\tcon@1:do1"),)
        assert doc.warnings[0].code == "UNSUPPORTED_CHAT_TIER"

    def test_unknown_header_preserved_with_warning(self, tmp_path):
        text = CHAT_FIXTURE.replace("@End", "@Comment:\tGhi chú\n@End")
        doc = load_document(_write(tmp_path, text=text))
        assert doc.raw_tiers["@Comment"] == "@Comment:\tGhi chú"
        assert doc.warnings == (
            ChatTierWarning(tier="%pho", line="%pho:\tcon@1:do1"),
            ChatTierWarning(tier="@Comment", line="@Comment:\tGhi chú"),
        )

    def test_classify_all_supported_codes(self, tmp_path):
        text = """\
@Begin
@Languages:\tvie
@Participants:\tPAR A
@ID:\tvie|PAR|A|participant|
@Media:\ta.wav | audio
*PAR:\t&uh &=coughs đi- [/] [//] [*] .
%xaud:\ta.wav 0.000 1.000
@End
"""
        doc = load_document(_write(tmp_path, text=text))
        assert [(t.text, t.kind) for t in doc.utterances[0].tokens] == [
            ("&uh", "filler"),
            ("&=coughs", "noise"),
            ("đi-", "fragment"),
            ("[/]", "retracing"),
            ("[//]", "revision"),
            ("[*]", "error"),
            (".", "word"),
        ]


class TestChatProvenance:
    def test_origin_and_input_hash_recorded(self, tmp_path):
        p = _write(tmp_path)
        doc = load_document(p)
        assert doc.source == str(p)
        expected = hashlib.sha256(p.read_bytes()).hexdigest()
        assert doc.source_sha256 == expected
        assert doc.source_sha256 != hashlib.sha256(b"other").hexdigest()

    def test_source_is_snapshot_origin(self, tmp_path):
        p = _write(tmp_path, name="b-0001.cha")
        doc = load_document(p)
        assert doc.document_id == "recording.wav"
        assert doc.source == str(p)


class TestChatRoundTrip:
    def test_semantic_round_trip_through_dispatch(self, tmp_path):
        p = _write(tmp_path)
        doc = load_document(p)
        saved = tmp_path / "session-copy.cha"
        save_document(doc, saved)
        doc2 = load_document(saved)
        assert _semantic_equals(doc, doc2)
        assert doc2.document_id == "recording.wav"
        assert doc2.source == str(saved)

    def test_unsupported_tier_verbatim_in_output(self, tmp_path):
        doc = load_document(_write(tmp_path))
        out = tmp_path / "out.cha"
        save_document(doc, out)
        text = out.read_text(encoding="utf-8")
        assert "%pho:\tcon@1:do1" in text
        assert "@Begin\n" == text[: len("@Begin\n")]
        assert text.rstrip().endswith("@End")
        assert "*PAR:\tCon đang đi chơi ." in text
        assert "%mor:\tCon|pro đang|aux đi|v chơi|v .|punct" in text
        assert "%gra:\t1|2|nsubj 2|3|aux 3|4|advmod 4|4|root 5|4|punct" in text

    def test_double_round_trip_stable(self, tmp_path):
        doc = load_document(_write(tmp_path))
        first = tmp_path / "first.cha"
        second = tmp_path / "second.cha"
        save_document(doc, first)
        save_document(load_document(first), second)
        again = load_document(second)
        assert _semantic_equals(load_document(first), again)
        assert again.document_id == load_document(first).document_id
        assert again.source_sha256 == load_document(first).source_sha256

    def test_save_refuses_overwrite_unless_force(self, tmp_path):
        doc = load_document(_write(tmp_path))
        p = tmp_path / "target.cha"
        p.write_text("original", encoding="utf-8")
        with pytest.raises(FileExistsError, match="overwrite"):
            save_document(doc, p)
        save_document(doc, p, force=True)
        assert _semantic_equals(doc, load_document(p))

    def test_json_round_trip_of_chat_document(self, tmp_path):
        doc = load_document(_write(tmp_path))
        save_document(doc, tmp_path / "doc.json")
        assert _semantic_equals(doc, load_document(tmp_path / "doc.json"))

    def test_no_blank_annotation_tiers_for_code_only_utterance(self, tmp_path):
        text = """\
@Begin
@Languages:\tvie
@Participants:\tPAR A
@ID:\tvie|PAR|A|participant|
@Media:\ta.wav | audio
*PAR:\t[//]
%xaud:\ta.wav 0.000 1.000
@End
"""
        doc = load_document(_write(tmp_path, text=text))
        assert [t.kind for t in doc.utterances[0].tokens] == ["revision"]  # zero content tokens
        out = tmp_path / "out.cha"
        save_document(doc, out)
        encoded = out.read_text(encoding="utf-8")
        assert "%mor:" not in encoded
        assert "%gra:" not in encoded


class TestChatDispatch:
    def test_detected_by_cha_extension(self, tmp_path):
        p = _write(tmp_path)
        assert load_document(p).document_id == "recording.wav"

    def test_detected_by_content(self, tmp_path):
        p = _write(tmp_path, name="session.txt")
        assert load_document(p).document_id == "recording.wav"

    def test_explicit_format_ignores_extension(self, tmp_path):
        p = _write(tmp_path, name="session.data")
        assert load_document(p, format="chat").document_id == "recording.wav"

    def test_save_infers_chat_from_extension(self, tmp_path):
        doc = load_document(_write(tmp_path))
        p = tmp_path / "out.cha"
        save_document(doc, p)
        assert p.read_text(encoding="utf-8").startswith("@Begin")

    def test_no_batchalign_or_asr_imports(self):
        src = Path(__file__).parents[2] / "src" / "speech_features" / "formats" / "chat.py"
        text = src.read_text(encoding="utf-8")
        for forbidden in (
            "import batchalign",
            "from batchalign",
            "import speech_recognition",
            "import whisper",
            "faster_whisper",
        ):
            assert forbidden not in text, forbidden


class TestChatInvalid:
    def _load(self, tmp_path, text):
        return load_document(_write(tmp_path, text=text))

    @pytest.mark.parametrize(
        "label,text",
        [
            ("no-begin", CHAT_FIXTURE.replace("@Begin\n", "")),
            ("no-end", CHAT_FIXTURE.replace("\n@End", "")),
            ("content-after-end", CHAT_FIXTURE + "stray\n"),
            (
                "no-participants",
                CHAT_FIXTURE.replace(
                    "@Participants:\tPAR Nguyễn Thị An, INV Nguyễn Văn Hùng\n", ""
                ),
            ),
            ("no-media", CHAT_FIXTURE.replace("@Media:\trecording.wav | audio\n", "")),
            ("bad-language", CHAT_FIXTURE.replace("@Languages:\tvie", "@Languages:\ten")),
            ("bad-participant-code", CHAT_FIXTURE.replace("*PAR:\ttôi", "*AB:\ttôi")),
            ("unknown-speaker-tier", CHAT_FIXTURE.replace("*PAR:\ttôi", "*XXX:\ttôi")),
            (
                "id-unknown-speaker",
                CHAT_FIXTURE.replace(
                    "@ID:\tvie|INV|Nguyễn Văn Hùng|examiner|||||",
                    "@ID:\tvie|ZZZ|Kẻ lạ|participant|||||",
                ),
            ),
            (
                "bad-utterance-bullet",
                CHAT_FIXTURE.replace(
                    "%xaud:\trecording.wav 0.000 1.200", "%xaud:\trecording.wav nope 1.200"
                ),
            ),
            (
                "missing-utterance-bullet",
                CHAT_FIXTURE.replace("%xaud:\trecording.wav 1.200 2.840\n", ""),
            ),
            (
                "bad-word-bullet-count",
                CHAT_FIXTURE.replace(
                    "%xaud:\trecording.wav 1.200 1.600 1.610 2.050 2.060 2.500 2.510 2.790 2.800 2.840",
                    "%xaud:\trecording.wav 1.200 1.600 1.610",
                ),
            ),
            (
                "mor-count-mismatch",
                CHAT_FIXTURE.replace(
                    "%mor:\tCon|pro đang|aux đi|v chơi|v .|punct", "%mor:\tCon|pro đang|aux"
                ),
            ),
            (
                "gra-count-mismatch",
                CHAT_FIXTURE.replace(
                    "%gra:\t1|2|nsubj 2|3|aux 3|4|advmod 4|4|root 5|4|punct", "%gra:\t1|2|nsubj"
                ),
            ),
            ("stray-line", CHAT_FIXTURE.replace("*PAR:\ttôi [*] đi .", "văn bản lạ")),
            ("duplicate-begin", CHAT_FIXTURE.replace("@Begin\n", "@Begin\n@Begin\n")),
            ("mor-without-utterance", CHAT_FIXTURE.replace("*INV:\tCon đang làm gì ?\n", "")),
        ],
    )
    def test_malformed_chat_rejected_with_invalid_chat(self, tmp_path, label, text):
        with pytest.raises(InvalidChatError) as excinfo:
            self._load(tmp_path, text)
        assert isinstance(excinfo.value, InvalidDocumentError)
        assert excinfo.value.code == "INVALID_CHAT"


REAL_FILE_FIXTURE = (
    "@UTF8\r\n"
    "@Window:\t0_0_0_0_8540_1_9273_0_9273_0\r\n"
    "@Begin\r\n"
    "@Languages:\tvie\r\n"
    "@Participants:\tPAR Nguyễn Văn A, INV Trần Thị B\r\n"
    "@ID:\tvie|PAR|Nguyễn Văn A|participant|||||\r\n"
    "@ID:\tvie|INV|Trần Thị B|examiner|||||\r\n"
    "@Media:\tp001_master.wav | audio\r\n"
    "@Comment:\tThis comment wraps across\r\n"
    "\ttwo indented continuation lines\r\n"
    "*INV:\tMột hai ba bốn .\x151234_3500\x15\r\n"
    "*PAR:\tMột câu dài bị ngắt xuống dòng theo\r\n"
    "\tchuẩn CHAT và mang dấu thời gian\r\n"
    "\tở dòng cuối cùng\x1547178_50522\x15\r\n"
    "*:\tthử nghiệm .\x15852813_854689\x15\r\n"
    "@End\r\n"
)


class TestRealFileShapes:
    """Shapes observed in the pilot corpus (audio/p001/p001.cha)."""

    def _load(self, tmp_path, text=REAL_FILE_FIXTURE):
        return load_document(_write(tmp_path, text=text, name="p001.cha"))

    def test_preamble_headers_before_begin_are_skipped(self, tmp_path):
        doc = self._load(tmp_path)
        assert [u.speaker_id for u in doc.utterances] == ["INV", "PAR", ""]

    def test_wrapped_utterance_unwraps_with_bullet_on_last_line(self, tmp_path):
        doc = self._load(tmp_path)
        wrapped = doc.utterances[1]
        assert (wrapped.start_s, wrapped.end_s) == (47.178, 50.522)
        texts = [t.text for t in wrapped.tokens]
        assert texts[:8] == ["Một", "câu", "dài", "bị", "ngắt", "xuống", "dòng", "theo"]
        assert texts[-3:] == ["dòng", "cuối", "cùng"]

    def test_unlabeled_speaker_tier_warns_and_keeps_times(self, tmp_path):
        doc = self._load(tmp_path)
        tail = doc.utterances[2]
        assert tail.speaker_id == ""
        assert (tail.start_s, tail.end_s) == (852.813, 854.689)
        assert [t.text for t in tail.tokens] == ["thử", "nghiệm", "."]
        assert ChatTierWarning(tier="*", line="unlabeled speaker tier") in doc.warnings

    def test_wrapped_header_continuation_joins_with_space(self, tmp_path):
        doc = self._load(tmp_path)
        assert (
            doc.raw_tiers["@Comment"]
            == "@Comment:\tThis comment wraps across two indented continuation lines"
        )


# --- dependent-tier membership (chatter / Batchalign3 alignment rules) ---------

from speech_features.formats.chat import decode_chat, encode_chat, tier_roles  # noqa: E402


def test_canonical_correction_count_roundtrips():
    text = "@Begin\n@Languages:\tvie\n@Participants:\tPAR Participant\n@Media:\ttest, audio\n*PAR:\ttôi [: chúng tôi] đi . \x150_1000\x15\n%mor:\tpron|chúng pron|tôi verb|đi .\n@End\n"
    doc = decode_chat(text)
    assert "%mor:\tpron|chúng pron|tôi verb|đi ." in encode_chat(doc)


@pytest.mark.parametrize("wor", ["khác \x150_500\x15 .", "tôi \x15500_500\x15 .", "tôi \x15600_500\x15 .", "tôi \x150_1500\x15 ."])
def test_untrusted_wor_is_preserved_without_timing(wor):
    text = f"@Begin\n@Languages:\tvie\n@Participants:\tPAR Participant\n@Media:\ttest, audio\n*PAR:\ttôi . \x150_1000\x15\n%wor:\t{wor}\n@End\n"
    doc = decode_chat(text)
    assert doc.utterances[0].tokens[0].start_s is None
    assert any(w.code == "UNTRUSTED_CHAT_WOR" for w in doc.warnings)
    assert not any(layer.layer == "wor" for layer in doc.annotations)
    assert f"%wor:\t{wor}" in encode_chat(doc)


def test_trusted_wor_sets_only_positive_word_times():
    text = "@Begin\n@Languages:\tvie\n@Participants:\tPAR Participant\n@Media:\ttest, audio\n*PAR:\t<tôi> . \x150_1000\x15\n%wor:\ttôi \x150_500\x15 .\n@End\n"
    doc = decode_chat(text)
    assert doc.utterances[0].tokens[0].end_s == 0.5


def test_mismatched_wor_count_preserves_raw_without_alignment():
    text = "@Begin\n@Languages:\tvie\n@Participants:\tPAR Participant\n@Media:\ttest, audio\n*PAR:\ttôi đi . \x150_1000\x15\n%wor:\ttôi \x150_500\x15 .\n@End\n"
    doc = decode_chat(text)
    assert not any(layer.layer == "wor" for layer in doc.annotations)
    assert "%wor:\ttôi \x150_500\x15 ." in encode_chat(doc)


@pytest.mark.parametrize(
    ("items", "roles"),
    [
        (["tôi", "đi", "."], ["word", "word", "word"]),
        (["tôi", ",", "đi", "."], ["word", "word", "word", "word"]),
        (["tôi", "[/]", "tôi", "đi", "."], ["retraced", "skip", "word", "word", "word"]),
        (
            ["<tôi", "đi>", "[/]", "tôi", "đi", "."],
            ["retraced", "retraced", "skip", "word", "word", "word"],
        ),
        (["đi>", "[//]", "đi"], ["retraced", "skip", "word"]),
        (["&-ờ", "tôi", "."], ["filler", "word", "word"]),
        (["xxx", "XXX.", "yyy", "www"], ["untranscribed"] * 4),
        (["tôi", "(.)", "đi", "(..)"], ["word", "skip", "word", "skip"]),
        (["&+ba", "&~gaga", "&=cười", "ba-"], ["skip"] * 4),
        (["tôi", "[:", "tao]", "đi", "[=", "ghi", "chú]", "[*]"], ["word", "skip", "skip", "word", "skip", "skip", "skip", "skip"]),
    ],
)
def test_tier_roles(items: list[str], roles: list[str]) -> None:
    assert tier_roles(items) == roles


def test_unmatched_group_end_marks_only_the_previous_item_retraced() -> None:
    assert tier_roles(["a", "b>", "[/]", "c"]) == ["word", "retraced", "skip", "word"]


_TIER_ALIGNED = (
    "@Begin\n@Languages:\tvie\n@Participants:\tPAR Participant\n"
    "@ID:\tvie|corpus|PAR|||||Participant|||\n@Media:\tsession, audio\n"
    "*PAR:\t&-ờ tôi [/] tôi đi xxx (.) . \x150_3000\x15\n"
    "%wor:\t&-ờ tôi tôi đi .\n"
    "%mor:\tpron|tôi verb|đi .\n"
    "%gra:\t1|2|NSUBJ 2|0|ROOT 3|2|PUNCT\n"
    "@End\n"
)


def _layer_texts(document, layer):
    values = next(a.values for a in document.annotations if a.layer == layer)
    tokens = document.utterances[0].tokens
    return [t.text for t in tokens if t.id in values]


def test_chatter_aligned_tiers_decode_to_their_own_members() -> None:
    from speech_features.formats.chat import decode_chat

    document = decode_chat(_TIER_ALIGNED, source="t")

    assert _layer_texts(document, "wor") == ["&-ờ", "tôi", "tôi", "đi", "."]
    assert _layer_texts(document, "mor") == ["tôi", "đi", "."]
    assert _layer_texts(document, "gra") == ["tôi", "đi", "."]


def test_chatter_aligned_tiers_round_trip_through_encode() -> None:
    from speech_features.formats.chat import decode_chat, encode_chat

    encoded = encode_chat(decode_chat(_TIER_ALIGNED, source="t"))

    assert "%mor:\tpron|tôi verb|đi ." in encoded
    assert "%wor:\t&-ờ tôi tôi đi ." in encoded
    assert decode_chat(encoded, source="t").annotations == decode_chat(_TIER_ALIGNED, source="t").annotations


def test_mor_that_counts_retraced_words_still_decodes_as_before() -> None:
    from speech_features.formats.chat import decode_chat

    legacy = (
        _TIER_ALIGNED.split("*PAR:")[0]
        + "*PAR:\t&uh tôi [/] tôi đi . \x150_3000\x15\n"
        + "%mor:\tco|uh pron|tôi pron|tôi verb|đi .\n"
        + "@End\n"
    )

    assert len(_layer_texts(decode_chat(legacy, source="t"), "mor")) == 5
