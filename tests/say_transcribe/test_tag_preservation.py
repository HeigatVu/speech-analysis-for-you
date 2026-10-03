import re
from types import SimpleNamespace

import pytest

from say_transcribe.cli import cmd_tag
from speech_features.formats.chat import InvalidChatError, decode_chat, encode_chat


class Backend:
    def __init__(self, fail_at=None):
        self.calls = []
        self.fail_at = fail_at

    def parse_pretokenized(self, tokens):
        self.calls.append(tuple(tokens))
        if len(self.calls) == self.fail_at:
            raise RuntimeError("private backend detail")
        words = [SimpleNamespace(id=i, text=t, lemma=t, upos="NOUN", feats=None,
                                 head=0 if i == 1 else 1,
                                 deprel="root" if i == 1 else "dep")
                 for i, t in enumerate(tokens, 1)]
        return SimpleNamespace(sentences=[SimpleNamespace(words=words)])


def source():
    return ("\ufeff@UTF8\r\n@Begin\n@Languages:\tvie\r\n"
            "@Participants:\tPAR Participant, MED Doctor\r\n"
            "@ID:\tvie|test|PAR|||||Participant|||\n"
            "@ID:\tvie|test|MED|||||Doctor|||\r\n"
            "@Media:\tp001, audio\r\n@Comment:\tcustom private header\n"
            "@G:\tTask one\r\n*MED:\ttôi\r\n\tđi . \x150_1000\x15\n"
            "%wor:\ttôi \x150_300\x15 đi \x15300_800\x15 .\r\n"
            "%mor:\tn|tôi\r\n\tn|đi .\n%gra:\t1|0|ROOT\n\t2|1|DEP 3|1|PUNCT\r\n"
            "%com:\tkeep this tier\r\n@G:\tTask two\n"
            "*PAR:\tnhà . \x151000_2000\x15\r\n@End\n").encode()


def without_morphology(data):
    return re.sub(rb"^%(?:mor|gra):[^\n]*(?:\n[ \t][^\n]*)*\n?", b"", data,
                  flags=re.MULTILINE)


def test_tag_changes_only_morphology_blocks(tmp_path, capsys):
    original = source()
    path = tmp_path / "private-name.cha"
    path.write_bytes(original)
    output = tmp_path / "final" / "p001.cha"
    backend = Backend()
    assert cmd_tag(path, output, stanza_backend=backend) == 0
    result = output.read_bytes()
    assert without_morphology(result) == without_morphology(original)
    assert path.read_bytes() == original
    assert result.count(b"%mor:") == 2
    assert backend.calls == [("tôi", "đi", "."), ("nhà", ".")]
    assert "private-name" not in capsys.readouterr().out


def test_tag_partial_failure_never_creates_output(tmp_path, capsys):
    path = tmp_path / "p001.cha"
    path.write_bytes(source())
    output = tmp_path / "new" / "p001.cha"
    assert cmd_tag(path, output, stanza_backend=Backend(fail_at=2)) == 3
    assert not output.exists()
    assert "private backend detail" not in capsys.readouterr().err


def test_tag_replaces_obsolete_morphology_after_review(tmp_path):
    original = re.sub(rb"%mor:[^\n]*\n[ \t][^\n]*\n",
                      lambda _: b"%mor:\tn|obsolete\n", source())
    path = tmp_path / "p001.cha"
    path.write_bytes(original)
    output = tmp_path / "result.cha"
    assert cmd_tag(path, output, stanza_backend=Backend()) == 0
    assert without_morphology(output.read_bytes()) == without_morphology(original)


def test_tag_untimed_and_filler_only(tmp_path):
    path = tmp_path / "p001.cha"
    path.write_text("@Begin\n@Participants:\tMED Doctor\n@Media:\tp001, audio\n"
                    "*MED:\t&-ờ .\n*MED:\tđi .\n@End\n", encoding="utf-8")
    output = tmp_path / "result.cha"
    backend = Backend()
    assert cmd_tag(path, output, stanza_backend=backend) == 0
    assert output.read_bytes().count(b"%mor:") == 1
    assert backend.calls == [("đi", ".")]


def test_untimed_chat_roundtrip_quarantines_wor():
    text = ("@Begin\n@Participants:\tMED Doctor\n@Media:\tp001, audio\n"
            "*MED:\tđi .\n%wor:\tđi \x150_100\x15 .\n@End\n")
    with pytest.raises(InvalidChatError, match="no media bullet"):
        decode_chat(text)
    document = decode_chat(text, require_timing=False)
    assert document.utterances[0].start_s is None
    assert all(token.start_s is None for token in document.utterances[0].tokens)
    assert not any(layer.values for layer in document.annotations if layer.layer == "wor")
    encoded = encode_chat(document)
    assert "%xaud:" not in encoded
    assert decode_chat(encoded, require_timing=False).utterances[0].start_s is None
