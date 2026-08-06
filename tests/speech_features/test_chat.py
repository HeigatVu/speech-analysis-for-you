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
