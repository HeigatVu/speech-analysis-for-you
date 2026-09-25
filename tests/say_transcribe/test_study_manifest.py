import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from say_transcribe import cli
from say_transcribe.asr import AsrResult, AsrSegment
from say_transcribe.denoise import DenoiseError
from say_transcribe.manifest import ManifestError, ManifestRow, load_denoiser_specs, load_manifest
from say_transcribe.study import StudyError, run_study, session_record, verify_source, write_session_record

_ROW_SHA256 = "ab" * 32


def _row(tmp_path: Path, **overrides) -> dict:
    row = {
        "session_id": "s1",
        "participant_id": "p1",
        "audio_path": str(tmp_path / "audio.wav"),
        "channel_index": 0,
        "sha256": _ROW_SHA256,
        "split": "dev",
        "reference_path": str(tmp_path / "ref.cha"),
        "asr_revision": "deadbeefcafe",
    }
    row.update(overrides)
    return row


def _write_manifest(tmp_path: Path, rows: list, denoisers: dict | None = None) -> Path:
    manifest = tmp_path / "manifest.json"
    payload: dict = {"rows": rows}
    if denoisers is not None:
        payload["denoisers"] = denoisers
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    return manifest


def test_load_manifest_parses_valid_rows(tmp_path: Path):
    manifest = _write_manifest(tmp_path, [_row(tmp_path), _row(tmp_path, session_id="s2")])

    rows = load_manifest(manifest)

    assert len(rows) == 2
    assert rows[0] == ManifestRow(
        session_id="s1",
        participant_id="p1",
        audio_path=tmp_path / "audio.wav",
        channel_index=0,
        sha256=_ROW_SHA256,
        split="dev",
        reference_path=tmp_path / "ref.cha",
        asr_revision="deadbeefcafe",
    )


def test_load_manifest_rejects_missing_required_field_redacted(tmp_path: Path):
    row = _row(tmp_path, audio_path=str(tmp_path / "SECRETNAME.wav"))
    del row["channel_index"]
    manifest = _write_manifest(tmp_path, [row])

    with pytest.raises(ManifestError) as excinfo:
        load_manifest(manifest)

    error = excinfo.value
    assert error.code == "INVALID_ARGUMENT"
    assert "row 0" in error.message
    assert "channel_index" in error.message
    assert "SECRETNAME" not in error.message
    assert str(tmp_path) not in error.message


@pytest.mark.parametrize(
    "override",
    [
        {"sha256": "not-hex"},
        {"sha256": "ab" * 31},
        {"split": "test"},
        {"channel_index": -1},
        {"channel_index": True},
        {"session_id": "../escape"},
        {"participant_id": ""},
        {"asr_revision": ""},
        {"audio_path": ""},
        {"reference_path": "   "},
    ],
)
def test_load_manifest_rejects_invalid_field_values(tmp_path: Path, override):
    manifest = _write_manifest(tmp_path, [_row(tmp_path, **override)])

    with pytest.raises(ManifestError):
        load_manifest(manifest)


def test_load_manifest_rejects_duplicate_session_ids(tmp_path: Path):
    manifest = _write_manifest(tmp_path, [_row(tmp_path), _row(tmp_path)])

    with pytest.raises(ManifestError) as excinfo:
        load_manifest(manifest)

    assert "duplicate" in excinfo.value.message


def test_load_manifest_rejects_malformed_documents(tmp_path: Path):
    for name, content in (
        ("bad.json", "{not json"),
        ("no_rows.json", json.dumps({"sessions": []})),
        ("empty_rows.json", json.dumps({"rows": []})),
        ("rows_not_list.json", json.dumps({"rows": {}})),
        ("row_not_object.json", json.dumps({"rows": ["s1"]})),
    ):
        manifest = tmp_path / name
        manifest.write_text(content, encoding="utf-8")
        with pytest.raises(ManifestError):
            load_manifest(manifest)


def test_hash_mismatch_raises_before_any_decode(tmp_path: Path, monkeypatch):
    events: list[str] = []
    manifest = _write_manifest(tmp_path, [_row(tmp_path, sha256="cd" * 32)])

    def spy_hash(path):
        events.append("hash")
        return "ab" * 32

    def unexpected_decode(*args, **kwargs):
        events.append("decode")
        raise AssertionError("decoder must not run before the hash gate")

    monkeypatch.setattr("say_transcribe.study.compute_sha256", spy_hash)
    monkeypatch.setattr("say_transcribe.study.read_wav", unexpected_decode)
    monkeypatch.setattr("say_transcribe.study.transcribe", unexpected_decode)

    with pytest.raises(StudyError) as excinfo:
        run_study(manifest_path=manifest, out_dir=tmp_path / "out", asr_backend=object())

    assert excinfo.value.code == "SOURCE_HASH_MISMATCH"
    assert events == ["hash"]


def test_verify_source_accepts_matching_hash(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("say_transcribe.study.compute_sha256", lambda path: "AB" * 32)
    row = load_manifest(_write_manifest(tmp_path, [_row(tmp_path)]))[0]

    assert verify_source(row) == "AB" * 32


def test_verify_source_wraps_missing_audio_without_paths(tmp_path: Path):
    row = load_manifest(_write_manifest(tmp_path, [_row(tmp_path)]))[0]

    with pytest.raises(StudyError) as excinfo:
        verify_source(row)

    assert excinfo.value.code == "INVALID_ARGUMENT"
    assert str(tmp_path) not in excinfo.value.message
    assert excinfo.value.__cause__ is None


def test_session_record_rejects_missing_reference(tmp_path: Path):
    row = load_manifest(_write_manifest(tmp_path, [_row(tmp_path)]))[0]
    arms = {
        "N0": _fake_result("tôi là", _ROW_SHA256),
        "N1": _fake_result("tôi là", _ROW_SHA256),
    }

    with pytest.raises(StudyError) as excinfo:
        session_record(row, arms)

    assert excinfo.value.code == "INVALID_ARGUMENT"
    assert str(tmp_path) not in excinfo.value.message
    assert excinfo.value.__cause__ is None


def test_session_record_rejects_empty_reference(tmp_path: Path):
    (tmp_path / "ref.cha").write_text("@Begin\n@End\n", encoding="utf-8")
    row = load_manifest(_write_manifest(tmp_path, [_row(tmp_path)]))[0]
    arms = {
        "N0": _fake_result("tôi là", _ROW_SHA256),
        "N1": _fake_result("tôi là", _ROW_SHA256),
    }

    with pytest.raises(StudyError) as excinfo:
        session_record(row, arms)

    assert excinfo.value.code == "INVALID_ARGUMENT"
    assert "scorable" in excinfo.value.message


def test_write_session_record_wraps_unwritable_output(tmp_path: Path):
    out_dir = tmp_path / "out"
    out_dir.write_text("not a directory", encoding="utf-8")

    with pytest.raises(StudyError) as excinfo:
        write_session_record({"session_id": "s1"}, out_dir)

    assert excinfo.value.code == "STUDY_FAILED"
    assert str(tmp_path) not in excinfo.value.message


def test_write_session_record_restricts_permissions(tmp_path: Path):
    out_dir = tmp_path / "private_out"
    written = write_session_record({"session_id": "s1", "text": "secret"}, out_dir)
    assert written.is_file()
    assert out_dir.stat().st_mode & 0o777 == 0o700
    assert written.stat().st_mode & 0o777 == 0o600


def _fake_result(text: str, source_sha256: str) -> AsrResult:
    return AsrResult(
        source_sha256=source_sha256,
        segments=(AsrSegment(start_ms=0, end_ms=1000, text=text, words=()),),
        warnings=(),
    )


# One second of 16 kHz audio, so a stub run has a signal the reference fits in.
_STUB_SAMPLES = 16000


def _profile_stub(samples):
    """A P0 profile result shaped like the real one, loudness report included."""
    return SimpleNamespace(
        samples=samples,
        sample_rate=16000,
        loudness=SimpleNamespace(
            input_i=-23.0,
            input_tp=-1.0,
            input_lra=7.0,
            output_i=-23.0,
            output_tp=-1.0,
            output_lra=7.0,
            normalization_type="linear",
        ),
    )


def _verified_reference(tmp_path: Path, name: str = "ref.cha") -> Path:
    """A reference shaped like a verified one: strict CHAT plus participant timing."""
    path = tmp_path / name
    path.write_text(
        "@UTF8\n@Begin\n@Languages:\tvie\n"
        "@Participants:\tPAR Participant, INV Investigator\n"
        "@ID:\tvie|corpus|PAR|||||Participant|||\n"
        "@ID:\tvie|corpus|INV|||||Investigator|||\n"
        "@Media:\tsynthetic, audio\n"
        "*PAR:\ttôi là sinh_viên .\t\x150_2000\x15\n"
        "@End\n",
        encoding="utf-8",
    )
    return path


def _stub_feature_layer(monkeypatch) -> None:
    """Keep pipeline tests offline: the eGeMAPS gate and extractors are stubbed."""
    monkeypatch.setattr("say_transcribe.study._opensmile_available", lambda: True)
    monkeypatch.setattr(
        "say_transcribe.study.extract_acoustic_features",
        lambda audio, rate, **kwargs: ({"audio_rms_dbfs": -20.0}, ()),
    )
    monkeypatch.setattr(
        "say_transcribe.study.extract_egemaps_features",
        lambda audio, rate, **kwargs: ({"egemaps_f0": 1.0}, (), {}),
    )


def test_native_arms_use_compare_code_paths_and_score_both(tmp_path: Path, monkeypatch):
    events: list[str] = []
    _verified_reference(tmp_path)
    _stub_feature_layer(monkeypatch)
    manifest = _write_manifest(tmp_path, [_row(tmp_path)])

    def spy_hash(path):
        events.append("hash")
        return _ROW_SHA256

    monkeypatch.setattr("say_transcribe.study.compute_sha256", spy_hash)
    monkeypatch.setattr(
        "say_transcribe.study.transcribe",
        lambda **kwargs: (events.append("transcribe"), _fake_result("tôi là", _ROW_SHA256))[1],
    )
    monkeypatch.setattr(
        "say_transcribe.study.read_wav",
        lambda path: (events.append("decode"), SimpleNamespace(sample_rate=16000, sample_width=2))[1],
    )
    monkeypatch.setattr("say_transcribe.study.extract_channel", lambda audio, channel: np.zeros(_STUB_SAMPLES))
    monkeypatch.setattr("say_transcribe.study.resample_to_16kHz", lambda *args: np.zeros(_STUB_SAMPLES))
    monkeypatch.setattr(
        "say_transcribe.study.apply_p0_profile",
        lambda samples, rate, width: _profile_stub(np.zeros(_STUB_SAMPLES)),
    )
    monkeypatch.setattr("say_transcribe.study.get_speech_windows", lambda audio, *, threshold: [(0, _STUB_SAMPLES)])
    monkeypatch.setattr("say_transcribe.study.merge_asr_windows", lambda windows: windows)
    monkeypatch.setattr(
        "say_transcribe.study.transcribe_windows",
        lambda audio, windows, backend: (events.append("transcribe_windows"), ())[1],
    )
    monkeypatch.setattr(
        "say_transcribe.study.result_from_windows",
        lambda audio, sha, windows: (
            events.append("result_from_windows"),
            _fake_result("tôi là sinh_viên", _ROW_SHA256),
        )[1],
    )

    out_dir = tmp_path / "out"
    run_study(manifest_path=manifest, out_dir=out_dir, asr_backend=object())

    # The N0/N1 arms run through the exact compare command's code paths,
    # and the hash gate runs before any decode.
    assert events[0] == "hash"
    assert events.index("hash") < events.index("transcribe") < events.index("decode")
    assert "transcribe_windows" in events and "result_from_windows" in events

    record = json.loads((out_dir / "s1.json").read_text(encoding="utf-8"))
    assert record["session_id"] == "s1"
    assert record["split"] == "dev"
    # Without a denoisers config only the native + P0 arms run.
    assert set(record["arms"]) == {"N0", "N1", "P0"}
    for arm in record["arms"].values():
        assert set(arm["scores"]) == {"syer", "cer", "wer"}
        assert arm["segments"][0]["text"]


def test_denoiser_arms_run_when_configured(tmp_path: Path, monkeypatch):
    _verified_reference(tmp_path)
    _stub_feature_layer(monkeypatch)
    for name in ("python", "worker", "checkpoint"):
        (tmp_path / name).write_text("stub", encoding="utf-8")
    manifest = _write_manifest(
        tmp_path,
        [_row(tmp_path)],
        denoisers={
            "PD": {
                "python": str(tmp_path / "python"),
                "worker": str(tmp_path / "worker"),
                "checkpoint": str(tmp_path / "checkpoint"),
            }
        },
    )
    seen_specs = []

    monkeypatch.setattr("say_transcribe.study.compute_sha256", lambda path: _ROW_SHA256)
    monkeypatch.setattr(
        "say_transcribe.study.transcribe",
        lambda **kwargs: _fake_result("tôi là", _ROW_SHA256),
    )
    monkeypatch.setattr(
        "say_transcribe.study.read_wav",
        lambda path: SimpleNamespace(sample_rate=16000, sample_width=2),
    )
    monkeypatch.setattr("say_transcribe.study.extract_channel", lambda audio, channel: np.zeros(_STUB_SAMPLES))
    monkeypatch.setattr("say_transcribe.study.resample_to_16kHz", lambda *args: np.zeros(_STUB_SAMPLES))
    monkeypatch.setattr(
        "say_transcribe.study.apply_p0_profile",
        lambda samples, rate, width: _profile_stub(np.zeros(_STUB_SAMPLES)),
    )
    monkeypatch.setattr(
        "say_transcribe.study.denoise_pcm",
        lambda samples, rate, spec: (seen_specs.append(spec), np.zeros(_STUB_SAMPLES))[1],
    )
    monkeypatch.setattr("say_transcribe.study.get_speech_windows", lambda audio, *, threshold: [(0, _STUB_SAMPLES)])
    monkeypatch.setattr("say_transcribe.study.merge_asr_windows", lambda windows: windows)
    monkeypatch.setattr("say_transcribe.study.transcribe_windows", lambda audio, windows, backend: ())
    monkeypatch.setattr(
        "say_transcribe.study.result_from_windows",
        lambda audio, sha, windows: _fake_result("tôi là sinh_viên", _ROW_SHA256),
    )

    out_dir = tmp_path / "out"
    run_study(manifest_path=manifest, out_dir=out_dir, asr_backend=object())

    record = json.loads((out_dir / "s1.json").read_text(encoding="utf-8"))
    assert set(record["arms"]) == {"N0", "N1", "P0", "PD"}
    assert [spec.name for spec in seen_specs] == ["PD"]


def test_session_record_scores_against_reference(tmp_path: Path):
    ref = tmp_path / "ref.cha"
    ref.write_text("@Begin\n*PAR:\ttôi là .\n@End\n", encoding="utf-8")
    row = load_manifest(_write_manifest(tmp_path, [_row(tmp_path)]))[0]
    arms = {
        "N0": _fake_result("tôi là", _ROW_SHA256),
        "N1": _fake_result("tôi là sinh viên năm hai ba", _ROW_SHA256),
    }

    record = session_record(row, arms)

    assert record["arms"]["N0"]["scores"]["syer"]["rate"] == 0.0
    assert record["arms"]["N1"]["scores"]["syer"]["rate"] > 1.0


def test_load_denoiser_specs_parses_and_validates(tmp_path: Path):
    config = {
        "python": "/venv/bin/python",
        "worker": "/opt/worker.py",
        "checkpoint": "/models/df3",
    }
    manifest = _write_manifest(tmp_path, [_row(tmp_path)], denoisers={"PD": config})

    specs = load_denoiser_specs(manifest)

    assert set(specs) == {"PD"}
    assert specs["PD"].name == "PD"
    assert specs["PD"].python == Path("/venv/bin/python")
    assert specs["PD"].config is None

    pinned = load_denoiser_specs(
        _write_manifest(
            tmp_path,
            [_row(tmp_path)],
            denoisers={"PF": dict(config, config="/opt/inference.toml")},
        )
    )
    assert pinned["PF"].config == Path("/opt/inference.toml")

    assert load_denoiser_specs(_write_manifest(tmp_path, [_row(tmp_path)])) == {}

    with pytest.raises(ManifestError):
        load_denoiser_specs(_write_manifest(tmp_path, [_row(tmp_path)], denoisers={"N0": config}))
    broken = dict(config)
    del broken["checkpoint"]
    with pytest.raises(ManifestError):
        load_denoiser_specs(_write_manifest(tmp_path, [_row(tmp_path)], denoisers={"PD": broken}))
    with pytest.raises(ManifestError):
        load_denoiser_specs(
            _write_manifest(tmp_path, [_row(tmp_path)], denoisers={"PD": dict(config, worker="")})
        )
    with pytest.raises(ManifestError):
        load_denoiser_specs(
            _write_manifest(tmp_path, [_row(tmp_path)], denoisers={"PD": dict(config, config="")})
        )
    # The FullSubNet worker needs its pinned recipe config; the DeepFilterNet
    # worker takes none, so declaring it there would only fail inside the worker.
    with pytest.raises(ManifestError) as missing_config:
        load_denoiser_specs(_write_manifest(tmp_path, [_row(tmp_path)], denoisers={"PF": config}))
    assert "config" in missing_config.value.message
    with pytest.raises(ManifestError) as unsupported_config:
        load_denoiser_specs(
            _write_manifest(
                tmp_path,
                [_row(tmp_path)],
                denoisers={"PD": dict(config, config="/opt/inference.toml")},
            )
        )
    assert "not supported" in unsupported_config.value.message

    valid_sha = "a" * 64
    with_sha = load_denoiser_specs(
        _write_manifest(
            tmp_path,
            [_row(tmp_path)],
            denoisers={"PD": dict(config, sha256=valid_sha)},
        )
    )
    assert with_sha["PD"].sha256 == valid_sha

    with pytest.raises(ManifestError) as bad_sha:
        load_denoiser_specs(
            _write_manifest(
                tmp_path,
                [_row(tmp_path)],
                denoisers={"PD": dict(config, sha256="not-valid-hex")},
            )
        )
    assert "sha256" in bad_sha.value.message


def test_mid_run_master_swap_raises_before_writing(tmp_path: Path, monkeypatch):
    _verified_reference(tmp_path)
    _stub_feature_layer(monkeypatch)
    manifest = _write_manifest(tmp_path, [_row(tmp_path)])
    hashes = iter([_ROW_SHA256, "cd" * 32])
    monkeypatch.setattr("say_transcribe.study.compute_sha256", lambda path: next(hashes))
    monkeypatch.setattr(
        "say_transcribe.study.transcribe", lambda **kwargs: _fake_result("tôi là", _ROW_SHA256)
    )
    monkeypatch.setattr(
        "say_transcribe.study.read_wav",
        lambda path: SimpleNamespace(sample_rate=16000, sample_width=2),
    )
    monkeypatch.setattr("say_transcribe.study.extract_channel", lambda audio, channel: np.zeros(_STUB_SAMPLES))
    monkeypatch.setattr("say_transcribe.study.resample_to_16kHz", lambda *args: np.zeros(_STUB_SAMPLES))
    monkeypatch.setattr(
        "say_transcribe.study.apply_p0_profile",
        lambda samples, rate, width: _profile_stub(np.zeros(_STUB_SAMPLES)),
    )
    monkeypatch.setattr("say_transcribe.study.get_speech_windows", lambda audio, *, threshold: [(0, _STUB_SAMPLES)])
    monkeypatch.setattr("say_transcribe.study.merge_asr_windows", lambda windows: windows)
    monkeypatch.setattr("say_transcribe.study.transcribe_windows", lambda audio, windows, backend: ())
    monkeypatch.setattr(
        "say_transcribe.study.result_from_windows",
        lambda audio, sha, windows: _fake_result("tôi là sinh_viên", _ROW_SHA256),
    )

    out_dir = tmp_path / "out"
    with pytest.raises(StudyError) as excinfo:
        run_study(manifest_path=manifest, out_dir=out_dir, asr_backend=object())

    assert excinfo.value.code == "SOURCE_HASH_MISMATCH"
    assert not (out_dir / "s1.json").exists()


def test_broken_denoiser_raises_and_never_skips_the_arm(tmp_path: Path, monkeypatch):
    _verified_reference(tmp_path)
    _stub_feature_layer(monkeypatch)
    for name in ("python", "worker", "checkpoint"):
        (tmp_path / name).write_text("stub", encoding="utf-8")
    manifest = _write_manifest(
        tmp_path,
        [_row(tmp_path)],
        denoisers={
            "PD": {
                "python": str(tmp_path / "python"),
                "worker": str(tmp_path / "worker"),
                "checkpoint": str(tmp_path / "checkpoint"),
            }
        },
    )
    monkeypatch.setattr("say_transcribe.study.compute_sha256", lambda path: _ROW_SHA256)
    monkeypatch.setattr(
        "say_transcribe.study.transcribe", lambda **kwargs: _fake_result("tôi là", _ROW_SHA256)
    )
    monkeypatch.setattr(
        "say_transcribe.study.read_wav",
        lambda path: SimpleNamespace(sample_rate=16000, sample_width=2),
    )
    monkeypatch.setattr("say_transcribe.study.extract_channel", lambda audio, channel: np.zeros(_STUB_SAMPLES))
    monkeypatch.setattr("say_transcribe.study.resample_to_16kHz", lambda *args: np.zeros(_STUB_SAMPLES))
    monkeypatch.setattr(
        "say_transcribe.study.apply_p0_profile",
        lambda samples, rate, width: _profile_stub(np.zeros(_STUB_SAMPLES)),
    )

    def broken_denoise(samples, rate, spec):
        raise DenoiseError("DENOISER_UNAVAILABLE", "denoiser worker failed")

    monkeypatch.setattr("say_transcribe.study.denoise_pcm", broken_denoise)

    with pytest.raises(DenoiseError) as excinfo:
        run_study(manifest_path=manifest, out_dir=tmp_path / "out", asr_backend=object())

    assert excinfo.value.code == "DENOISER_UNAVAILABLE"


def test_preprocess_study_cli_exit_codes(tmp_path: Path, monkeypatch, capsys):
    manifest = _write_manifest(tmp_path, [_row(tmp_path)])

    monkeypatch.setattr(cli, "run_study", lambda **kwargs: 0)
    assert cli.main(["preprocess-study", str(manifest), "--out", str(tmp_path / "out")]) == 0

    def hash_mismatch(**kwargs):
        raise StudyError("SOURCE_HASH_MISMATCH", "Master audio does not match manifest SHA-256")

    monkeypatch.setattr(cli, "run_study", hash_mismatch)
    assert cli.main(["preprocess-study", str(manifest), "--out", str(tmp_path / "out")]) == 2
    assert "[SOURCE_HASH_MISMATCH]" in capsys.readouterr().err

    def invalid_manifest(**kwargs):
        raise ManifestError("manifest row 0: missing required field 'split'")

    monkeypatch.setattr(cli, "run_study", invalid_manifest)
    assert cli.main(["preprocess-study", str(manifest), "--out", str(tmp_path / "out")]) == 2
    assert "[INVALID_ARGUMENT]" in capsys.readouterr().err

    def broken_denoiser(**kwargs):
        raise DenoiseError("DENOISER_UNAVAILABLE", "denoiser environment is missing")

    monkeypatch.setattr(cli, "run_study", broken_denoiser)
    assert cli.main(["preprocess-study", str(manifest), "--out", str(tmp_path / "out")]) == 3
    assert "[DENOISER_UNAVAILABLE]" in capsys.readouterr().err
