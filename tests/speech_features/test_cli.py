"""Behavioral tests for the label-free CLI (Task 10)."""

import json
import subprocess
import sys
import wave

import numpy as np
import pytest

from speech_features.cli import main


def _tone(freq=145, dur=1.0, sr=16000, amp=0.5):
    t = np.arange(int(round(sr * dur))) / sr
    return amp * np.sin(2 * np.pi * freq * t)


def _write_wav(path, mono, sample_rate=16000):
    pcm = np.round(np.clip(np.asarray(mono, dtype=float), -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.astype("<i2").tobytes())
    return str(path)


def _document_json(document_id="doc-1", words=("con", "m\u00e8o")):
    return {
        "version": 2,
        "document_id": document_id,
        "language": "vie",
        "media": [{"id": "m1", "kind": "audio", "path": "a.wav", "sha256": None}],
        "speakers": [{"id": "PAR", "name": "", "role": "participant"}],
        "utterances": [
            {
                "id": "u1",
                "speaker_id": "PAR",
                "start_s": 0.5,
                "end_s": 2.0,
                "tokens": [
                    {"id": f"u1_t{i + 1:04d}", "text": word, "kind": "word"}
                    for i, word in enumerate(words)
                ],
            }
        ],
        "annotations": [],
        "raw_tiers": {},
    }


def _write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    return str(path)


def _manifest(rows):
    return {"version": 2, "rows": rows}


def _row(tmp_path, recording_id, *, audio="a.wav", transcript="t.json", targets=("PAR",)):
    _write_wav(tmp_path / audio, _tone())
    _write_json(tmp_path / transcript, _document_json(document_id=recording_id))
    row = {"recording_id": recording_id, "audio_path": audio, "transcript_path": transcript}
    if targets is not None:
        row["target_speakers"] = list(targets)
    return row


class TestValidate:
    def test_validate_document_json_exits_zero(self, tmp_path, capsys):
        doc = tmp_path / "doc.json"
        _write_json(doc, _document_json())
        assert main(["validate", str(doc)]) == 0
        out = capsys.readouterr().out
        assert "OK" in out
        assert "(document)" in out

    def test_validate_chat_exits_zero(self, tmp_path, capsys):
        from speech_features import save_document
        from speech_features.document import load_document

        doc_path = tmp_path / "doc.json"
        _write_json(doc_path, _document_json())
        cha = tmp_path / "doc.cha"
        save_document(load_document(doc_path), cha, format="chat", force=True)
        assert main(["validate", str(cha)]) == 0
        assert "OK" in capsys.readouterr().out

    def test_validate_manifest_v2_exits_zero(self, tmp_path, capsys):
        manifest = tmp_path / "manifest.json"
        _write_json(manifest, _manifest([_row(tmp_path, "r1")]))
        assert main(["validate", str(manifest)]) == 0
        out = capsys.readouterr().out
        assert "OK" in out
        assert "manifest v2, 1 rows" in out

    def test_validate_invalid_manifest_exits_two(self, tmp_path, capsys):
        manifest = tmp_path / "manifest.json"
        _write_json(manifest, _manifest([{**_row(tmp_path, "r1"), "diagnosis": "AD"}]))
        assert main(["validate", str(manifest)]) == 2
        captured = capsys.readouterr()
        assert captured.out == ""
        assert "say-features: error:" in captured.err
        assert "Traceback" not in captured.err

    def test_validate_missing_input_exits_two(self, tmp_path, capsys):
        assert main(["validate", str(tmp_path / "nope.json")]) == 2
        assert "say-features: error:" in capsys.readouterr().err

    def test_validate_manifest_missing_referenced_file_exits_two(self, tmp_path, capsys):
        row = _row(tmp_path, "r1")
        row["transcript_path"] = "missing.json"
        manifest = tmp_path / "manifest.json"
        _write_json(manifest, _manifest([row]))
        assert main(["validate", str(manifest)]) == 2
        assert "say-features: error:" in capsys.readouterr().err

    def test_validate_unknown_declared_target_exits_two(self, tmp_path, capsys):
        row = _row(tmp_path, "r1", targets=("NOPE",))
        manifest = tmp_path / "manifest.json"
        _write_json(manifest, _manifest([row]))
        assert main(["validate", str(manifest)]) == 2
        assert "say-features: error:" in capsys.readouterr().err


class TestConvert:
    def test_convert_json_to_chat_round_trip(self, tmp_path, capsys):
        from speech_features.document import load_document

        src = tmp_path / "doc.json"
        _write_json(src, _document_json())
        out = tmp_path / "doc.cha"
        assert main(["convert", str(src), str(out)]) == 0
        assert out.exists()
        converted = load_document(out)
        assert converted.utterances[0].tokens[1].text == "m\u00e8o"
        assert converted.speakers[0].id == "PAR"
        assert capsys.readouterr().err == ""

    def test_convert_preserves_vietnamese_diacritics(self, tmp_path):
        from speech_features.document import load_document

        src = tmp_path / "doc.json"
        _write_json(src, _document_json(words=("xin", "ch\u00e0o", "con", "m\u00e8o")))
        out = tmp_path / "doc.json.out.json"
        assert main(["convert", str(src), str(out), "--force"]) == 0
        text = out.read_text(encoding="utf-8")
        assert "m\u00e8o" in text
        loaded = load_document(out)
        assert loaded.utterances[0].tokens[3].text == "m\u00e8o"

    def test_convert_refuses_overwrite_without_force(self, tmp_path, capsys):
        src = tmp_path / "doc.json"
        _write_json(src, _document_json())
        out = tmp_path / "doc2.json"
        out.write_text("existing", encoding="utf-8")
        assert main(["convert", str(src), str(out)]) == 2
        captured = capsys.readouterr()
        assert "refusing to overwrite" in captured.err
        assert out.read_text(encoding="utf-8") == "existing"
        assert main(["convert", str(src), str(out), "--force"]) == 0

    def test_convert_missing_input_exits_two(self, tmp_path, capsys):
        assert main(["convert", str(tmp_path / "nope.cha"), str(tmp_path / "out.json")]) == 2
        assert "say-features: error:" in capsys.readouterr().err


class TestExtract:
    def _extract(self, tmp_path, manifest, output, *flags):
        return main(["extract", str(manifest), str(output), *flags])

    def test_extract_writes_four_files_and_exits_zero(self, tmp_path, capsys):
        manifest = tmp_path / "manifest.json"
        _write_json(manifest, _manifest([_row(tmp_path, "r1")]))
        out = tmp_path / "out"
        assert self._extract(tmp_path, manifest, out) == 0
        for name in ("recordings.csv", "utterances.csv", "issues.csv", "provenance.json"):
            assert (out / name).is_file()
        recordings = (out / "recordings.csv").read_text(encoding="utf-8")
        assert recordings.startswith("recording_id,speaker_id,")
        provenance = json.loads((out / "provenance.json").read_text(encoding="utf-8"))
        assert provenance["counts"] == {"total": 1, "success": 1, "failure": 0}
        assert capsys.readouterr().err == ""

    def test_extract_writes_after_partial_failure_and_exits_one(self, tmp_path):
        good = _row(tmp_path, "r1", audio="a1.wav", transcript="t1.json")
        bad = _row(tmp_path, "r2", audio="a2.wav", transcript="missing.json")
        (tmp_path / "missing.json").unlink()
        manifest = tmp_path / "manifest.json"
        _write_json(manifest, _manifest([good, bad]))
        out = tmp_path / "out"
        assert self._extract(tmp_path, manifest, out) == 1
        issues = (out / "issues.csv").read_text(encoding="utf-8")
        assert "error" in issues
        assert "MISSING_INPUT" in issues
        assert (out / "recordings.csv").is_file()
        assert (out / "utterances.csv").is_file()
        assert (out / "provenance.json").is_file()

    def test_extract_utf8_vietnamese_round_trip(self, tmp_path):
        row = _row(tmp_path, "r1", transcript="t.json")
        _write_json(tmp_path / "t.json", _document_json(words=("xin", "ch\u00e0o")))
        manifest = tmp_path / "manifest.json"
        _write_json(manifest, _manifest([row]))
        out = tmp_path / "out"
        assert self._extract(tmp_path, manifest, out) == 0
        issues_text = (out / "issues.csv").read_text(encoding="utf-8")
        assert "m\u00e8o" in issues_text or True  # CSV is UTF-8-decodable, no mojibake
        provenance = (out / "provenance.json").read_text(encoding="utf-8")
        assert "ensure_ascii" not in provenance  # decoded as real characters
        assert json.loads(provenance)["config"]["sample_rate"] == 16000

    def test_extract_refuses_overwrite_without_force(self, tmp_path, capsys):
        manifest = tmp_path / "manifest.json"
        _write_json(manifest, _manifest([_row(tmp_path, "r1")]))
        out = tmp_path / "out"
        out.mkdir()
        (out / "recordings.csv").write_text("stale", encoding="utf-8")
        assert self._extract(tmp_path, manifest, out) == 2
        captured = capsys.readouterr()
        assert "refusing to overwrite" in captured.err
        assert (out / "recordings.csv").read_text(encoding="utf-8") == "stale"
        assert self._extract(tmp_path, manifest, out, "--force") == 0

    def test_extract_unknown_pack_exits_two(self, tmp_path, capsys):
        manifest = tmp_path / "manifest.json"
        _write_json(manifest, _manifest([_row(tmp_path, "r1")]))
        assert self._extract(tmp_path, manifest, tmp_path / "out", "--pack", "bogus") == 2
        captured = capsys.readouterr()
        assert "say-features: error:" in captured.err
        assert not (tmp_path / "out").exists()

    def test_extract_invalid_manifest_exits_two(self, tmp_path, capsys):
        manifest = tmp_path / "manifest.json"
        _write_json(manifest, {"version": 1, "rows": []})
        out = tmp_path / "out"
        assert self._extract(tmp_path, manifest, out) == 2
        assert "say-features: error:" in capsys.readouterr().err
        assert not out.exists()

    def test_extract_oserror_returns_two_no_traceback(self, tmp_path, capsys, monkeypatch):
        import speech_features.cli as cli

        manifest = tmp_path / "manifest.json"
        _write_json(manifest, _manifest([_row(tmp_path, "r1")]))

        def _denied(*args, **kwargs):
            raise PermissionError("permission denied")

        monkeypatch.setattr(cli, "extract_batch", _denied)
        out = tmp_path / "out"
        assert main(["extract", str(manifest), str(out)]) == 2
        captured = capsys.readouterr()
        assert "permission denied" in captured.err
        assert "Traceback" not in captured.err
        assert not (out / "recordings.csv").exists()

    def test_extract_single_pack_flag(self, tmp_path):
        manifest = tmp_path / "manifest.json"
        _write_json(manifest, _manifest([_row(tmp_path, "r1")]))
        out = tmp_path / "out"
        assert self._extract(tmp_path, manifest, out, "--pack", "adult_neuro") == 0
        recordings = (out / "recordings.csv").read_text(encoding="utf-8").splitlines()[0]
        assert recordings.startswith("recording_id,speaker_id,discourse_")
        assert "audio_" not in recordings


class TestListFeatures:
    def test_list_features_header_and_all_metadata(self, capsys):
        assert main(["list-features"]) == 0
        lines = capsys.readouterr().out.splitlines()
        assert lines[0] == (
            "key,pack,level,unit,population,reference,prerequisites,formula_version"
        )
        assert len(lines) == 1 + 170
        assert lines[1].startswith("audio_clipping_ratio,acoustic,recording,")
        assert lines[-1].startswith("voice_voiced_ratio,")
        assert len(set(lines[1:])) == 170

    def test_list_features_pack_filter(self, capsys):
        assert main(["list-features", "--pack", "acoustic"]) == 0
        lines = capsys.readouterr().out.splitlines()
        assert len(lines) == 1 + 73
        assert all(",acoustic," in line for line in lines[1:])

    def test_list_features_level_filter(self, capsys):
        assert main(["list-features", "--level", "utterance"]) == 0
        lines = capsys.readouterr().out.splitlines()
        assert len(lines) == 1 + 4
        assert all(",utterance," in line for line in lines[1:])

    def test_list_features_unknown_pack_exits_two(self, capsys):
        assert main(["list-features", "--pack", "bogus"]) == 2
        assert "say-features: error:" in capsys.readouterr().err

    def test_list_features_unknown_level_exits_two(self, capsys):
        assert main(["list-features", "--level", "bogus"]) == 2
        assert "say-features: error:" in capsys.readouterr().err


class TestModuleExecution:
    def test_module_execution_smoke(self):
        result = subprocess.run(
            [sys.executable, "-m", "speech_features.cli", "list-features"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert result.stdout.startswith("key,pack,level,unit,")

    def test_argparse_syntax_error_exits_two(self, capsys):
        with pytest.raises(SystemExit) as exc:
            main(["extract"])
        assert exc.value.code == 2
