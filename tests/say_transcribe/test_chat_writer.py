from pathlib import Path
import pytest

from say_transcribe.chat_writer import UtteranceRecord, format_chat_session, write_chat_file
from say_transcribe.morphosyntax import GraItem, MorItem, UtteranceMorphosyntax
from say_transcribe.word_grouping import GroupedWord
from speech_features.formats.chat import InvalidChatError, decode_chat


def test_untimed_utterance_preserves_text_without_fabricated_bullet():
    utterance = UtteranceRecord(
        speaker="PAR", start_ms=None, end_ms=None, text="xin chào",
        words=(), morphosyntax=None,
    )
    content = format_chat_session("synthetic", "a" * 64, [utterance])
    assert "*PAR:\txin chào .\n" in content
    assert "\x15" not in content
    assert "utterance timing incomplete; align before timing-based analysis" in content
    # Feature extraction must still reject a draft that has not been aligned.
    with pytest.raises(InvalidChatError, match="no media bullet"):
        decode_chat(content)


def test_chat_writer_format_and_provenance_lines(tmp_path: Path):
    sha = "a" * 64
    session_id = "test_session_01"

    utt1 = UtteranceRecord(
        speaker="INV",
        start_ms=0,
        end_ms=1200,
        text="học_sinh đi học .",
        words=(
            GroupedWord(word="học_sinh", start_ms=0, end_ms=600, syllables=()),
            GroupedWord(word="đi", start_ms=600, end_ms=800, syllables=()),
            GroupedWord(word="học", start_ms=800, end_ms=1200, syllables=()),
            GroupedWord(word=".", start_ms=None, end_ms=None, syllables=()),
        ),
        morphosyntax=UtteranceMorphosyntax(
            mor_items=(
                MorItem(pos="noun", lemma="học_sinh"),
                MorItem(pos="verb", lemma="đi"),
                MorItem(pos="verb", lemma="học"),
                MorItem(pos="punct", lemma="."),
            ),
            gra_items=(
                GraItem(index=1, head=2, rel="NSUBJ"),
                GraItem(index=2, head=0, rel="ROOT"),
                GraItem(index=3, head=2, rel="XCOMP"),
                GraItem(index=4, head=2, rel="PUNCT"),
            ),
        ),
    )

    # Utterance 2 missing %mor and %gra (failure path)
    utt2 = UtteranceRecord(
        speaker="PAR",
        start_ms=1300,
        end_ms=2500,
        text="vâng hiểu rồi .",
        words=(
            GroupedWord(word="vâng", start_ms=1300, end_ms=1600, syllables=()),
            GroupedWord(word="hiểu", start_ms=1600, end_ms=2100, syllables=()),
            GroupedWord(word="rồi", start_ms=2100, end_ms=2500, syllables=()),
            GroupedWord(word=".", start_ms=None, end_ms=None, syllables=()),
        ),
        morphosyntax=None,
    )

    cha_text = format_chat_session(
        session_id=session_id,
        source_sha256=sha,
        utterances=[utt1, utt2],
    )

    # Check headers
    assert "@UTF8" in cha_text
    assert "@Begin" in cha_text
    assert "@Languages:\tvie" in cha_text
    assert f"@Media:\t{session_id}, audio" in cha_text

    # Exactly one occurrence of each @Comment
    assert cha_text.count(f"@Comment:\tsource_sha256 {sha}") == 1
    assert cha_text.count("@Comment:\tspeaker labels draft, auto-diarized; review before use") == 1

    # Main tiers
    assert "*INV:\thọc_sinh đi học .\t\x150_1200\x15" in cha_text
    assert "*PAR:\tvâng hiểu rồi .\t\x151300_2500\x15" in cha_text

    # %wor tier in Delaware shape
    assert "%wor:\thọc_sinh \x150_600\x15 đi \x15600_800\x15 học \x15800_1200\x15 ." in cha_text

    # %mor and %gra present for utt1, omitted for utt2
    assert "%mor:\tnoun|học_sinh verb|đi verb|học ." in cha_text
    assert "%gra:\t1|2|NSUBJ 2|0|ROOT 3|2|XCOMP 4|2|PUNCT" in cha_text

    out_file = tmp_path / "test_session_01.cha"
    write_chat_file(out_file, session_id, sha, [utt1, utt2])
    assert out_file.exists()


def test_write_chat_file_refuses_to_overwrite_an_existing_transcript(tmp_path: Path):
    out_file = tmp_path / "p001.cha"
    out_file.write_text("manual transcript, never overwrite", encoding="utf-8")

    with pytest.raises(FileExistsError):
        write_chat_file(out_file, "p001", "a" * 64, [])

    assert out_file.read_text(encoding="utf-8") == "manual transcript, never overwrite"


def test_round_trip_through_speech_features_chat_codec(tmp_path: Path):
    sha = "b" * 64
    session_id = "roundtrip_session"

    utt = UtteranceRecord(
        speaker="PAR",
        start_ms=100,
        end_ms=1500,
        text="tôi là sinh_viên .",
        words=(
            GroupedWord(word="tôi", start_ms=100, end_ms=400, syllables=()),
            GroupedWord(word="là", start_ms=400, end_ms=700, syllables=()),
            GroupedWord(word="sinh_viên", start_ms=700, end_ms=1450, syllables=()),
            GroupedWord(word=".", start_ms=None, end_ms=None, syllables=()),
        ),
        morphosyntax=UtteranceMorphosyntax(
            mor_items=(
                MorItem(pos="pron", lemma="tôi"),
                MorItem(pos="aux", lemma="là"),
                MorItem(pos="noun", lemma="sinh_viên"),
                MorItem(pos="punct", lemma="."),
            ),
            gra_items=(
                GraItem(index=1, head=3, rel="NSUBJ"),
                GraItem(index=2, head=3, rel="COP"),
                GraItem(index=3, head=0, rel="ROOT"),
                GraItem(index=4, head=3, rel="PUNCT"),
            ),
        ),
    )

    out_file = tmp_path / f"{session_id}.cha"
    write_chat_file(out_file, session_id, sha, [utt])

    doc = decode_chat(out_file.read_text(encoding="utf-8"))
    assert len(doc.utterances) == 1
    u = doc.utterances[0]
    assert u.speaker_id == "PAR"
    assert u.start_s == 0.1
    assert u.end_s == 1.5
    assert [t.text for t in u.tokens] == ["tôi", "là", "sinh_viên", "."]

    # Verify annotations preserved (%wor is a first-class layer now)
    layers = {a.layer: a.values for a in doc.annotations}
    assert "wor" in layers
    assert "mor" in layers
    assert "gra" in layers
    assert len(layers["wor"]) == 4
    assert len(layers["mor"]) == 4
    assert len(layers["gra"]) == 4
    assert "@Comment" in doc.raw_tiers


def test_missing_final_punct_gets_mor_gra_terminator(tmp_path):
    from say_transcribe.chat_writer import UtteranceRecord, format_chat_session
    from say_transcribe.morphosyntax import GraItem, MorItem, UtteranceMorphosyntax
    from say_transcribe.word_grouping import GroupedWord

    words = (GroupedWord(word="hở", start_ms=100, end_ms=500, syllables=()),)
    ms = UtteranceMorphosyntax(
        mor_items=(MorItem(pos="noun", lemma="hở"),),
        gra_items=(GraItem(index=1, head=0, rel="ROOT"),),
    )
    utt = UtteranceRecord(
        speaker="PAR", start_ms=100, end_ms=500, text="hở", words=words, morphosyntax=ms
    )
    out = format_chat_session("pX", "0" * 64, [utt])
    mor = next(ln for ln in out.splitlines() if ln.startswith("%mor:")).split("\t")[1].split()
    gra = next(ln for ln in out.splitlines() if ln.startswith("%gra:")).split("\t")[1].split()
    assert mor[-1] == "."
    assert len(mor) == 2
    assert gra[-1] == "2|1|PUNCT"
    assert len(gra) == 2


def test_session_round_trips_through_chat_py_with_spaced_media(tmp_path):
    from speech_features.formats.chat import decode_chat, encode_chat

    from say_transcribe.chat_writer import UtteranceRecord, format_chat_session
    from say_transcribe.morphosyntax import GraItem, MorItem, UtteranceMorphosyntax
    from say_transcribe.word_grouping import GroupedWord

    words = (
        GroupedWord(word="hở", start_ms=0, end_ms=500, syllables=()),
        GroupedWord(word="hở", start_ms=600, end_ms=900, syllables=()),
        GroupedWord(word=".", start_ms=None, end_ms=None, syllables=()),
    )
    ms = UtteranceMorphosyntax(
        mor_items=(MorItem(pos="noun", lemma="hở"), MorItem(pos="noun", lemma="hở"), MorItem(pos="", lemma=".")),
        gra_items=(GraItem(index=1, head=0, rel="ROOT"), GraItem(index=2, head=1, rel="COMPOUND"), GraItem(index=3, head=1, rel="PUNCT")),
    )
    utt = UtteranceRecord(
        speaker="PAR", start_ms=0, end_ms=900, text="hở hở .", words=words, morphosyntax=ms
    )
    raw = format_chat_session("pX", "0" * 64, [utt])
    doc = decode_chat(raw, source="t")
    doc2 = decode_chat(encode_chat(doc), source="t")

    def shape(d):
        layers = {a.layer: a.values for a in d.annotations}
        out = []
        for u in d.utterances:
            ann = tuple(
                (kind, layers[kind][t.id])
                for kind in ("wor", "mor", "gra")
                for t in u.tokens
                if t.id in layers.get(kind, {})
            )
            out.append((u.speaker_id, tuple(t.text for t in u.tokens), ann))
        return out

    assert shape(doc) == shape(doc2)
