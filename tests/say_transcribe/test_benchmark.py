import json
import struct
import wave

from say_transcribe.asr import AsrError, compute_sha256
from say_transcribe.benchmark import (
    render_markdown,
    run_asr_benchmark,
    run_diarization_benchmark,
    write_report,
)
from say_transcribe.cli import main
from say_transcribe.diarize import DiarizationTurn

CHA_TWO_TURN = """@UTF8
@Begin
@Languages:\tvie
@Participants:\tPAR Participant, INV Investigator
@ID:\tvie|corpus|PAR|||||Participant|||
@ID:\tvie|corpus|INV|||||Investigator|||
@Media:\tp001, audio
*PAR:\txin chào .\t\x150_1000\x15
*INV:\ttạm biệt .\t\x151000_2000\x15
@End
"""


def test_the_default_factory_asks_for_segment_timestamps(monkeypatch):
    """A text-only benchmark must not pay 4.4GB for word timings.

    Word and segment timestamps decode identical text (measured 2026-10-03);
    word mode only adds the cross-attention alignment that costs the extra VRAM,
    and nothing the benchmark scores reads it.
    """
    from say_transcribe import benchmark as benchmark_module

    seen: dict[str, object] = {}

    def fake_make_asr_backend(model, device="cpu", revision=None, **kwargs):
        seen.update(model=model, device=device, **kwargs)
        return object()

    monkeypatch.setattr(benchmark_module, "make_asr_backend", fake_make_asr_backend)

    benchmark_module._default_asr_factory("phowhisper-large", "cuda")

    assert seen["device"] == "cuda"
    assert seen["timestamps"] == "segment"


def _make_wav(path, seconds=0.5, rate=16000):
    n = int(rate * seconds)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(struct.pack(f"<{n}h", *([0] * n)))
    return path


def _make_manifest(tmp_path, rows):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"rows": rows}), encoding="utf-8")
    return manifest


def _manifest_row(tmp_path, session_id="p001", sha=None):
    audio = _make_wav(tmp_path / f"{session_id}_master.wav")
    reference = tmp_path / f"{session_id}.cha"
    reference.write_text(CHA_TWO_TURN, encoding="utf-8")
    return {
        "session_id": session_id,
        "participant_id": "par1",
        "audio_path": str(audio),
        "channel_index": 0,
        "sha256": sha if sha is not None else compute_sha256(audio),
        "split": "dev",
        "reference_path": str(reference),
        "asr_revision": "rev-a",
    }


class FakeAsrBackend:
    def __init__(self, texts):
        self.texts = texts

    def transcribe_audio(self, audio_16k_mono):
        return [
            {
                "start_ms": 0,
                "end_ms": 1000,
                "text": self.texts[0],
                "words": [],
            },
            {
                "start_ms": 1000,
                "end_ms": 2000,
                "text": self.texts[1],
                "words": [],
            },
        ]


def test_run_asr_benchmark_scores_models_and_redacts(tmp_path):
    from say_transcribe.manifest import load_manifest

    manifest = _make_manifest(tmp_path, [_manifest_row(tmp_path)])
    rows = load_manifest(manifest)
    texts = {
        "phowhisper-large": ("xin chào", "tạm biệt"),
        "qwen3-asr": ("a", "b"),
    }
    report = run_asr_benchmark(
        rows, backend_factory=lambda model, device: FakeAsrBackend(texts[model])
    )

    assert report["task"] == "asr"
    by_model = {row["model"]: row for row in report["rows"]}
    assert by_model["phowhisper-large"]["status"] == "ok"
    assert by_model["phowhisper-large"]["wer"] == 0.0
    assert by_model["qwen3-asr"]["status"] == "ok"
    assert by_model["qwen3-asr"]["wer"] == 1.0
    assert report["aggregate"]["phowhisper-large"]["cells_scored"] == 1
    assert report["aggregate"]["qwen3-asr"]["cells_scored"] == 1

    serialized = json.dumps(report) + render_markdown(report)
    assert str(tmp_path) not in serialized
    assert "xin chào" not in serialized

    json_path, md_path = write_report(report, tmp_path / "out")
    assert json_path.is_file() and md_path.is_file()
    assert "# Benchmark report: asr" in md_path.read_text(encoding="utf-8")


def test_run_asr_benchmark_error_cells_and_hash_mismatch(tmp_path):
    from say_transcribe.manifest import load_manifest

    manifest = _make_manifest(
        tmp_path,
        [
            _manifest_row(tmp_path, session_id="p001", sha="ab" * 32),
            _manifest_row(tmp_path, session_id="p002"),
        ],
    )
    rows = load_manifest(manifest)
    calls: list[str] = []

    def factory(model, device):
        calls.append(model)
        raise AsrError("MODEL_UNAVAILABLE", "load failed")

    report = run_asr_benchmark(rows, backend_factory=factory)

    by_session = {}
    for row in report["rows"]:
        by_session.setdefault(row["session_id"], []).append(row)
    assert [r["code"] for r in by_session["p001"]] == ["SOURCE_HASH_MISMATCH"]
    assert all(r["code"] == "MODEL_UNAVAILABLE" for r in by_session["p002"])
    assert all(model in calls for model in ("phowhisper-large", "qwen3-asr"))
    assert report["aggregate"]["phowhisper-large"]["cells_scored"] == 0
    assert report["aggregate"]["qwen3-asr"]["cells_scored"] == 0


def test_run_diarization_benchmark_role_agreement(tmp_path):
    from say_transcribe.manifest import load_manifest

    manifest = _make_manifest(tmp_path, [_manifest_row(tmp_path)])
    rows = load_manifest(manifest)

    class FakePyannote:
        def diarize(self, audio_16k_mono, sample_rate=16000):
            return (
                DiarizationTurn(start_ms=0, end_ms=1000, cluster_id="A"),
                DiarizationTurn(start_ms=1000, end_ms=2000, cluster_id="B"),
            )

    class FakeWavlm:
        def diarize(self, audio_16k_mono, sample_rate=16000, segments=None):
            assert segments is not None
            return (DiarizationTurn(start_ms=0, end_ms=2000, cluster_id="Y"),)

    diarizers = {"pyannote": FakePyannote(), "wavlm": FakeWavlm()}
    report = run_diarization_benchmark(
        rows,
        asr_factory=lambda model, device: FakeAsrBackend(("xin chào", "tạm biệt")),
        diarizer_factory=lambda name, device: diarizers[name],
    )

    by_diarizer = {row["diarizer"]: row for row in report["rows"]}
    assert by_diarizer["pyannote"]["status"] == "ok"
    assert by_diarizer["pyannote"]["der"] == 0.0
    assert by_diarizer["pyannote"]["n_speakers"] == 2
    assert by_diarizer["wavlm"]["status"] == "ok"
    assert by_diarizer["wavlm"]["der"] > 0.0
    assert report["aggregate"]["pyannote"]["cells_scored"] == 1

    serialized = json.dumps(report) + render_markdown(report)
    assert str(tmp_path) not in serialized


def test_cli_benchmark_dispatch_and_invalid_manifest(tmp_path, monkeypatch):
    manifest = _make_manifest(tmp_path, [_manifest_row(tmp_path)])
    out_dir = tmp_path / "out"
    canned = {"task": "asr", "rows": [], "aggregate": {}}
    monkeypatch.setattr(
        "say_transcribe.cli.run_asr_benchmark",
        lambda rows, device="cpu": canned,
    )

    ret = main(
        ["benchmark", "--task", "asr", "--manifest", str(manifest), "--out", str(out_dir)]
    )
    assert ret == 0
    assert (out_dir / "benchmark-asr.json").is_file()
    assert (out_dir / "benchmark-asr.md").is_file()

    bad = tmp_path / "bad.json"
    bad.write_text("{}", encoding="utf-8")
    ret = main(["benchmark", "--task", "asr", "--manifest", str(bad), "--out", str(out_dir)])
    assert ret == 2

def test_diarization_unmapped_cluster_scores_unknown_not_par(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from say_transcribe.manifest import load_manifest

    manifest = _make_manifest(tmp_path, [_manifest_row(tmp_path)])
    rows = load_manifest(manifest)

    class FakePyannote:
        def diarize(self, audio_16k_mono, sample_rate=16000, **kwargs):
            return (DiarizationTurn(start_ms=0, end_ms=1000, cluster_id="Z"),)

    # No role assignment survives: the turn's cluster is unmapped.
    monkeypatch.setattr(
        "say_transcribe.benchmark.assign_speakers",
        lambda *args, **kwargs: SimpleNamespace(cluster_to_role=None),
    )
    report = run_diarization_benchmark(
        rows,
        asr_factory=lambda model, device: FakeAsrBackend(("xin chào", "tạm biệt")),
        diarizer_factory=lambda name, device: FakePyannote(),
    )

    row = next(r for r in report["rows"] if r["diarizer"] == "pyannote")
    assert row["status"] == "ok"
    # An unmapped cluster must not earn PAR credit against the gold PAR turn.
    assert row["der"] > 0.0
