import hashlib
import io
import wave
from pathlib import Path

import numpy as np
import pytest

from say_transcribe.annotations import (
    AnnotationValidationError,
    SessionAnnotations,
    TaskAnnotation,
    TurnAnnotation,
)
from say_transcribe.audio import AudioPreparationError, read_wav
from say_transcribe.prepare import prepare_task_clips


def _make_wav_bytes(
    channels: int,
    sample_rate: int,
    sample_width: int,
    data: np.ndarray,
) -> bytes:
    """Helper to synthesize PCM WAV bytes in-memory."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        if sample_width == 2:
            raw = (data.astype(np.int16)).tobytes()
        elif sample_width == 4:
            raw = (data.astype(np.int32)).tobytes()
        else:
            raise ValueError(f"unsupported width {sample_width}")
        wf.writeframes(raw)
    return buf.getvalue()


@pytest.fixture
def stereo_pilot_audio(tmp_path: Path):
    """Create a 48 kHz stereo WAV where Left is active (sine) and Right is silent."""
    sample_rate = 48000
    duration_s = 2.0
    n_samples = int(sample_rate * duration_s)
    t = np.linspace(0, duration_s, n_samples, endpoint=False)
    left = (np.sin(2 * np.pi * 440 * t) * 20000).astype(np.int16)
    right = np.zeros(n_samples, dtype=np.int16)
    interleaved = np.empty((n_samples, 2), dtype=np.int16)
    interleaved[:, 0] = left
    interleaved[:, 1] = right

    wav_bytes = _make_wav_bytes(2, sample_rate, 2, interleaved)
    sha256 = hashlib.sha256(wav_bytes).hexdigest()

    audio_file = tmp_path / "pilot_session_01.wav"
    audio_file.write_bytes(wav_bytes)
    return audio_file, sha256, sample_rate, 2, interleaved


def test_hash_mismatch_fails_before_processing(stereo_pilot_audio, tmp_path):
    audio_file, sha256, sample_rate, _, _ = stereo_pilot_audio
    annotations = SessionAnnotations(
        schema_version="1.0.0",
        source_audio="pilot_session_01.wav",
        source_sha256="0" * 64,  # wrong hash
        channel_index=0,
        sample_rate=sample_rate,
        review_status="approved",
        tasks=(
            TaskAnnotation(
                task_id="task_01",
                start_ms=0,
                end_ms=1000,
                turns=(TurnAnnotation("t1", "participant", 0, 500),),
            ),
        ),
    )

    with pytest.raises(AnnotationValidationError) as exc_info:
        prepare_task_clips(annotations, audio_root=tmp_path, output_dir=tmp_path / "out")
    assert exc_info.value.code == "SOURCE_HASH_MISMATCH"


def test_hash_read_failure_does_not_leak_source_path(stereo_pilot_audio, tmp_path, monkeypatch):
    audio_file, sha256, sample_rate, _, _ = stereo_pilot_audio
    annotations = SessionAnnotations(
        schema_version="1.0.0",
        source_audio=audio_file.name,
        source_sha256=sha256,
        channel_index=0,
        sample_rate=sample_rate,
        review_status="approved",
        tasks=(TaskAnnotation("task_01", 0, 500, ()),),
    )
    original_open = Path.open

    def fail_source_open(path, *args, **kwargs):
        if path == audio_file:
            raise OSError(str(audio_file))
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_source_open)

    with pytest.raises(AnnotationValidationError) as exc_info:
        prepare_task_clips(annotations, tmp_path, tmp_path / "clips")

    assert exc_info.value.code == "INVALID_ANNOTATIONS"
    assert str(tmp_path) not in str(exc_info.value)
    assert exc_info.value.__cause__ is None


def test_active_left_silent_right_prevents_mean_downmix(stereo_pilot_audio, tmp_path):
    audio_file, sha256, sample_rate, sample_width, interleaved = stereo_pilot_audio
    annotations = SessionAnnotations(
        schema_version="1.0.0",
        source_audio="pilot_session_01.wav",
        source_sha256=sha256,
        channel_index=0,  # select active left
        sample_rate=sample_rate,
        review_status="approved",
        tasks=(
            TaskAnnotation(
                task_id="task_01",
                start_ms=100,
                end_ms=600,
                turns=(),
            ),
        ),
    )

    out_dir = tmp_path / "clips"
    clips = prepare_task_clips(annotations, audio_root=tmp_path, output_dir=out_dir)
    assert len(clips) == 1

    # Verify native clip on disk
    clip_path = out_dir / "task_01.wav"
    assert clip_path.exists()
    assert clip_path.stat().st_mode & 0o077 == 0

    with wave.open(str(clip_path), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getframerate() == sample_rate
        assert wf.getsampwidth() == sample_width
        raw = wf.readframes(wf.getnframes())
        samples = np.frombuffer(raw, dtype=np.int16)

    # Expected slice
    start_sample = int(0.100 * sample_rate)
    end_sample = int(0.600 * sample_rate)
    expected_left = interleaved[start_sample:end_sample, 0]

    # Must match active left exactly; if it was mean-downmixed, max amplitude would be halved
    np.testing.assert_array_equal(samples, expected_left)
    assert np.max(np.abs(samples)) > 15000  # Left channel is loud


def test_silent_right_channel_selected(stereo_pilot_audio, tmp_path):
    audio_file, sha256, sample_rate, _, _ = stereo_pilot_audio
    annotations = SessionAnnotations(
        schema_version="1.0.0",
        source_audio="pilot_session_01.wav",
        source_sha256=sha256,
        channel_index=1,  # select silent right
        sample_rate=sample_rate,
        review_status="approved",
        tasks=(
            TaskAnnotation(
                task_id="task_01",
                start_ms=0,
                end_ms=500,
                turns=(),
            ),
        ),
    )

    out_dir = tmp_path / "clips_right"
    prepare_task_clips(annotations, audio_root=tmp_path, output_dir=out_dir)

    with wave.open(str(out_dir / "task_01.wav"), "rb") as wf:
        raw = wf.readframes(wf.getnframes())
        samples = np.frombuffer(raw, dtype=np.int16)

    # Right channel must be entirely silent (zeros)
    assert np.all(samples == 0)


def test_in_memory_16khz_asr_view(stereo_pilot_audio, tmp_path):
    audio_file, sha256, sample_rate, _, _ = stereo_pilot_audio
    annotations = SessionAnnotations(
        schema_version="1.0.0",
        source_audio="pilot_session_01.wav",
        source_sha256=sha256,
        channel_index=0,
        sample_rate=sample_rate,
        review_status="approved",
        tasks=(
            TaskAnnotation(
                task_id="task_01",
                start_ms=0,
                end_ms=1000,
                turns=(),
            ),
        ),
    )

    out_dir = tmp_path / "clips_asr"
    clips = prepare_task_clips(annotations, audio_root=tmp_path, output_dir=out_dir)
    clip = clips[0]

    asr_view = clip.asr_16k_mono
    assert isinstance(asr_view, np.ndarray)
    assert asr_view.dtype == np.float32
    assert asr_view.ndim == 1
    # 1.0 second at 16 kHz = 16000 samples (+/- 1 sample tolerance)
    assert abs(len(asr_view) - 16000) <= 2
    assert np.max(np.abs(asr_view)) <= 1.0


def test_invalid_channel_index_fails_with_stable_code(stereo_pilot_audio, tmp_path):
    audio_file, sha256, sample_rate, _, _ = stereo_pilot_audio
    annotations = SessionAnnotations(
        schema_version="1.0.0",
        source_audio="pilot_session_01.wav",
        source_sha256=sha256,
        channel_index=2,  # Stereo has only channels 0 and 1
        sample_rate=sample_rate,
        review_status="approved",
        tasks=(
            TaskAnnotation(
                task_id="task_01",
                start_ms=0,
                end_ms=500,
                turns=(),
            ),
        ),
    )

    with pytest.raises(AudioPreparationError) as exc_info:
        prepare_task_clips(annotations, audio_root=tmp_path, output_dir=tmp_path / "out")
    assert exc_info.value.code == "INVALID_AUDIO_CHANNEL"


def test_span_exceeding_audio_length_fails_with_stable_code(stereo_pilot_audio, tmp_path):
    audio_file, sha256, sample_rate, _, _ = stereo_pilot_audio
    annotations = SessionAnnotations(
        schema_version="1.0.0",
        source_audio="pilot_session_01.wav",
        source_sha256=sha256,
        channel_index=0,
        sample_rate=sample_rate,
        review_status="approved",
        tasks=(
            TaskAnnotation(
                task_id="task_01",
                start_ms=0,
                end_ms=10000,  # 10s exceeds 2s file
                turns=(),
            ),
        ),
    )

    with pytest.raises(AudioPreparationError) as exc_info:
        prepare_task_clips(annotations, audio_root=tmp_path, output_dir=tmp_path / "out")
    assert exc_info.value.code == "INVALID_AUDIO_SPAN"


@pytest.mark.parametrize("start_ms,end_ms", [(-100, 500), (500, 500), (600, 500)])
def test_invalid_task_span_writes_no_clip(stereo_pilot_audio, tmp_path, start_ms, end_ms):
    _, sha256, sample_rate, _, _ = stereo_pilot_audio
    annotations = SessionAnnotations(
        schema_version="1.0.0",
        source_audio="pilot_session_01.wav",
        source_sha256=sha256,
        channel_index=0,
        sample_rate=sample_rate,
        review_status="approved",
        tasks=(TaskAnnotation("task_01", start_ms, end_ms, ()),),
    )
    output_dir = tmp_path / "invalid_spans"

    with pytest.raises(AudioPreparationError) as exc_info:
        prepare_task_clips(annotations, audio_root=tmp_path, output_dir=output_dir)

    assert exc_info.value.code == "INVALID_AUDIO_SPAN"
    assert not output_dir.exists()


@pytest.mark.parametrize("task_id", ["../escaped", "bad\x00id"])
def test_task_id_cannot_escape_output_directory(stereo_pilot_audio, tmp_path, task_id):
    _, sha256, sample_rate, _, _ = stereo_pilot_audio
    annotations = SessionAnnotations(
        schema_version="1.0.0",
        source_audio="pilot_session_01.wav",
        source_sha256=sha256,
        channel_index=0,
        sample_rate=sample_rate,
        review_status="approved",
        tasks=(TaskAnnotation(task_id, 0, 500, ()),),
    )

    with pytest.raises(AnnotationValidationError) as exc_info:
        prepare_task_clips(annotations, audio_root=tmp_path, output_dir=tmp_path / "clips")

    assert exc_info.value.code == "INVALID_ANNOTATIONS"
    assert not (tmp_path / "escaped.wav").exists()


def test_clip_cannot_replace_source_wav(stereo_pilot_audio, tmp_path):
    audio_file, sha256, sample_rate, _, _ = stereo_pilot_audio
    annotations = SessionAnnotations(
        schema_version="1.0.0",
        source_audio=audio_file.name,
        source_sha256=sha256,
        channel_index=0,
        sample_rate=sample_rate,
        review_status="approved",
        tasks=(TaskAnnotation(audio_file.stem, 0, 500, ()),),
    )

    with pytest.raises(AnnotationValidationError) as exc_info:
        prepare_task_clips(annotations, audio_root=tmp_path, output_dir=tmp_path)

    assert exc_info.value.code == "INVALID_ANNOTATIONS"
    assert hashlib.sha256(audio_file.read_bytes()).hexdigest() == sha256


def test_existing_clip_is_not_overwritten(stereo_pilot_audio, tmp_path):
    _, sha256, sample_rate, _, _ = stereo_pilot_audio
    annotations = SessionAnnotations(
        schema_version="1.0.0",
        source_audio="pilot_session_01.wav",
        source_sha256=sha256,
        channel_index=0,
        sample_rate=sample_rate,
        review_status="approved",
        tasks=(TaskAnnotation("task_01", 0, 500, ()),),
    )
    output_dir = tmp_path / "clips"
    output_dir.mkdir()
    existing_clip = output_dir / "task_01.wav"
    existing_clip.write_bytes(b"existing output")

    with pytest.raises(AnnotationValidationError) as exc_info:
        prepare_task_clips(annotations, tmp_path, output_dir)

    assert exc_info.value.code == "INVALID_ANNOTATIONS"
    assert existing_clip.read_bytes() == b"existing output"


def test_write_failure_leaves_no_clip_or_private_path(stereo_pilot_audio, tmp_path, monkeypatch):
    _, sha256, sample_rate, _, _ = stereo_pilot_audio
    annotations = SessionAnnotations(
        schema_version="1.0.0",
        source_audio="pilot_session_01.wav",
        source_sha256=sha256,
        channel_index=0,
        sample_rate=sample_rate,
        review_status="approved",
        tasks=(TaskAnnotation("task_01", 0, 500, ()),),
    )
    original_open = wave.open

    def fail_output(file, mode):
        if mode == "wb":
            raise OSError(str(tmp_path / "private.wav"))
        return original_open(file, mode)

    monkeypatch.setattr(wave, "open", fail_output)
    output_dir = tmp_path / "clips"

    with pytest.raises(AudioPreparationError) as exc_info:
        prepare_task_clips(annotations, tmp_path, output_dir)

    assert exc_info.value.code == "AUDIO_WRITE_FAILED"
    assert str(tmp_path) not in str(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert list(output_dir.iterdir()) == []


def test_output_directory_error_does_not_leak_path(stereo_pilot_audio, tmp_path, monkeypatch):
    _, sha256, sample_rate, _, _ = stereo_pilot_audio
    annotations = SessionAnnotations(
        schema_version="1.0.0",
        source_audio="pilot_session_01.wav",
        source_sha256=sha256,
        channel_index=0,
        sample_rate=sample_rate,
        review_status="approved",
        tasks=(TaskAnnotation("task_01", 0, 500, ()),),
    )
    output_dir = tmp_path / "clips"
    original_mkdir = Path.mkdir

    def fail_output_mkdir(path, *args, **kwargs):
        if path == output_dir:
            raise OSError(str(output_dir))
        return original_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", fail_output_mkdir)

    with pytest.raises(AudioPreparationError) as exc_info:
        prepare_task_clips(annotations, tmp_path, output_dir)

    assert exc_info.value.code == "AUDIO_WRITE_FAILED"
    assert str(tmp_path) not in str(exc_info.value)
    assert exc_info.value.__cause__ is None


@pytest.mark.parametrize(
    "source_audio,review_status,audio_root,code",
    [
        ("../pilot_session_01.wav", "approved", "inputs", "INVALID_ANNOTATIONS"),
        ("bad\x00.wav", "approved", ".", "INVALID_ANNOTATIONS"),
        ("pilot_session_01.wav", "draft", ".", "UNAPPROVED_ANNOTATIONS"),
    ],
)
def test_untrusted_annotations_cannot_prepare_clips(
    stereo_pilot_audio, tmp_path, source_audio, review_status, audio_root, code
):
    _, sha256, sample_rate, _, _ = stereo_pilot_audio
    annotations = SessionAnnotations(
        schema_version="1.0.0",
        source_audio=source_audio,
        source_sha256=sha256,
        channel_index=0,
        sample_rate=sample_rate,
        review_status=review_status,
        tasks=(TaskAnnotation("task_01", 0, 500, ()),),
    )
    output_dir = tmp_path / "clips"

    with pytest.raises(AnnotationValidationError) as exc_info:
        prepare_task_clips(annotations, tmp_path / audio_root, output_dir)

    assert exc_info.value.code == code
    assert not output_dir.exists()


def test_sample_rate_mismatch_fails_before_output(stereo_pilot_audio, tmp_path):
    _, sha256, _, _, _ = stereo_pilot_audio
    annotations = SessionAnnotations(
        schema_version="1.0.0",
        source_audio="pilot_session_01.wav",
        source_sha256=sha256,
        channel_index=0,
        sample_rate=44100,
        review_status="approved",
        tasks=(TaskAnnotation("task_01", 0, 500, ()),),
    )
    output_dir = tmp_path / "clips"

    with pytest.raises(AnnotationValidationError) as exc_info:
        prepare_task_clips(annotations, tmp_path, output_dir)

    assert exc_info.value.code == "INVALID_ANNOTATIONS"
    assert not output_dir.exists()


def test_redacted_provenance_and_errors(stereo_pilot_audio, tmp_path):
    audio_file, sha256, sample_rate, _, _ = stereo_pilot_audio
    annotations = SessionAnnotations(
        schema_version="1.0.0",
        source_audio="pilot_session_01.wav",
        source_sha256=sha256,
        channel_index=0,
        sample_rate=sample_rate,
        review_status="approved",
        tasks=(
            TaskAnnotation(
                task_id="task_01",
                start_ms=0,
                end_ms=500,
                turns=(),
            ),
        ),
    )

    out_dir = tmp_path / "clips_prov"
    clips = prepare_task_clips(annotations, audio_root=tmp_path, output_dir=out_dir)
    prov = clips[0].provenance

    # Provenance must not leak absolute paths
    assert str(tmp_path) not in prov["relative_audio_path"]
    assert "source_sha256" in prov
    assert prov["channel_index"] == 0
    assert "start_sample" in prov
    assert "end_sample" in prov
    assert "sample_rate" in prov
    # Pinned exact key set: guards against silently widening provenance disclosure
    assert set(prov) == {
        "relative_audio_path",
        "source_sha256",
        "channel_index",
        "start_sample",
        "end_sample",
        "sample_rate",
        "task_id",
    }


def test_one_task_span_failure_writes_no_clips(stereo_pilot_audio, tmp_path):
    """A later task's invalid span must not leave an earlier task's clip on disk."""
    audio_file, sha256, sample_rate, _, _ = stereo_pilot_audio
    annotations = SessionAnnotations(
        schema_version="1.0.0",
        source_audio="pilot_session_01.wav",
        source_sha256=sha256,
        channel_index=0,
        sample_rate=sample_rate,
        review_status="approved",
        tasks=(
            TaskAnnotation(task_id="task_01", start_ms=0, end_ms=500, turns=()),
            TaskAnnotation(task_id="task_02", start_ms=0, end_ms=10000, turns=()),  # exceeds 2s file
        ),
    )

    out_dir = tmp_path / "clips_atomic"
    with pytest.raises(AudioPreparationError) as exc_info:
        prepare_task_clips(annotations, audio_root=tmp_path, output_dir=out_dir)
    assert exc_info.value.code == "INVALID_AUDIO_SPAN"
    assert list(out_dir.glob("*.wav")) == []


@pytest.mark.parametrize("code_producing_error", ["hash_mismatch", "bad_channel", "bad_span"])
def test_errors_do_not_leak_absolute_paths_or_chain_raw_exceptions(
    stereo_pilot_audio, tmp_path, code_producing_error
):
    audio_file, sha256, sample_rate, _, _ = stereo_pilot_audio
    bad_hash = code_producing_error == "hash_mismatch"
    channel_index = 5 if code_producing_error == "bad_channel" else 0
    end_ms = 10000 if code_producing_error == "bad_span" else 500

    annotations = SessionAnnotations(
        schema_version="1.0.0",
        source_audio="pilot_session_01.wav",
        source_sha256="0" * 64 if bad_hash else sha256,
        channel_index=channel_index,
        sample_rate=sample_rate,
        review_status="approved",
        tasks=(
            TaskAnnotation(task_id="task_01", start_ms=0, end_ms=end_ms, turns=()),
        ),
    )

    with pytest.raises(Exception) as exc_info:
        prepare_task_clips(annotations, audio_root=tmp_path, output_dir=tmp_path / "out")

    assert str(tmp_path) not in str(exc_info.value)
    assert exc_info.value.__cause__ is None


def test_corrupt_wav_error_does_not_chain_raw_exception(tmp_path):
    """read_wav's decode failure must not chain the raw stdlib exception,
    which may carry the absolute path in its own message or traceback."""
    garbage = b"not a real wav file"
    sha256 = hashlib.sha256(garbage).hexdigest()
    audio_file = tmp_path / "corrupt.wav"
    audio_file.write_bytes(garbage)

    annotations = SessionAnnotations(
        schema_version="1.0.0",
        source_audio="corrupt.wav",
        source_sha256=sha256,
        channel_index=0,
        sample_rate=16000,
        review_status="approved",
        tasks=(TaskAnnotation(task_id="task_01", start_ms=0, end_ms=50, turns=()),),
    )

    with pytest.raises(AudioPreparationError) as exc_info:
        prepare_task_clips(annotations, audio_root=tmp_path, output_dir=tmp_path / "out")
    assert exc_info.value.code == "INVALID_AUDIO"
    assert exc_info.value.__cause__ is None


def test_truncated_stereo_wav_fails_with_stable_code(tmp_path):
    wav_bytes = _make_wav_bytes(2, 16000, 2, np.zeros((100, 2), dtype=np.int16))
    audio_file = tmp_path / "truncated.wav"
    audio_file.write_bytes(wav_bytes[:-2])

    with pytest.raises(AudioPreparationError) as exc_info:
        read_wav(audio_file)

    assert exc_info.value.code == "INVALID_AUDIO"
    assert exc_info.value.__cause__ is None


def test_unsupported_sample_width_fails_with_stable_code(tmp_path):
    """8-bit PCM is not one of the two supported widths."""
    sample_rate = 16000
    n_samples = 800
    data = np.linspace(0, 255, n_samples, dtype=np.uint8)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(1)
        wf.setframerate(sample_rate)
        wf.writeframes(data.tobytes())
    wav_bytes = buf.getvalue()
    sha256 = hashlib.sha256(wav_bytes).hexdigest()

    audio_file = tmp_path / "eight_bit.wav"
    audio_file.write_bytes(wav_bytes)

    annotations = SessionAnnotations(
        schema_version="1.0.0",
        source_audio="eight_bit.wav",
        source_sha256=sha256,
        channel_index=0,
        sample_rate=sample_rate,
        review_status="approved",
        tasks=(TaskAnnotation(task_id="task_01", start_ms=0, end_ms=50, turns=()),),
    )

    with pytest.raises(AudioPreparationError) as exc_info:
        prepare_task_clips(annotations, audio_root=tmp_path, output_dir=tmp_path / "out")
    assert exc_info.value.code == "INVALID_AUDIO"


def test_mono_int32_input_passes_through_and_scales_correctly(tmp_path):
    """Single-channel int32 input: extract_channel returns the array unchanged,
    and resample_to_16kHz scales by the int32 full-scale value, not int16's."""
    sample_rate = 16000
    n_samples = 1600
    max_int32 = 2147483647
    data = np.full(n_samples, max_int32 // 2, dtype=np.int32)
    wav_bytes = _make_wav_bytes(1, sample_rate, 4, data)
    sha256 = hashlib.sha256(wav_bytes).hexdigest()

    audio_file = tmp_path / "mono32.wav"
    audio_file.write_bytes(wav_bytes)

    annotations = SessionAnnotations(
        schema_version="1.0.0",
        source_audio="mono32.wav",
        source_sha256=sha256,
        channel_index=0,
        sample_rate=sample_rate,
        review_status="approved",
        tasks=(TaskAnnotation(task_id="task_01", start_ms=0, end_ms=100, turns=()),),
    )

    out_dir = tmp_path / "clips_mono32"
    clips = prepare_task_clips(annotations, audio_root=tmp_path, output_dir=out_dir)
    asr_view = clips[0].asr_16k_mono

    # 2147483647 // 2 / 2147483648.0 ~= 0.5, not the int16 scaling (~16384.0)
    assert 0.49 < np.max(asr_view) < 0.51
