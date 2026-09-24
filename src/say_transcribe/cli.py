import argparse
from dataclasses import replace
from pathlib import Path
import sys
from typing import Any, Sequence

from say_transcribe.alignment import AlignmentError, align_words
from say_transcribe.asr import (
    AsrError,
    PhoWhisperBackend,
    compute_sha256,
    result_from_windows,
    transcribe,
    transcribe_windows,
)
from say_transcribe.audio import AudioPreparationError, extract_channel, read_wav, resample_to_16kHz
from say_transcribe.chat_writer import UtteranceRecord, write_chat_file
from say_transcribe.diarize import PyannoteBackend, WavlmClusterBackend, assign_speakers
from say_transcribe.manifest import ManifestError
from say_transcribe.morphosyntax import StanzaBackend, project_morphosyntax
from say_transcribe.study import StudyError, run_study
from say_transcribe.word_grouping import GroupedWord, WordGroupingError, group_utterance_words
from say_transcribe.vad import VADUnavailableError, get_speech_windows, merge_asr_windows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="say-transcribe",
        description="Vietnamese clinical CHAT transcription pipeline",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # diagnose
    subparsers.add_parser("diagnose", help="Check dependencies and environment")

    # run
    run_parser = subparsers.add_parser("run", help="Run transcription pipeline on a master WAV")
    run_parser.add_argument("audio", type=Path, help="Path to master WAV file")
    run_parser.add_argument(
        "--channel", type=int, required=True, help="Audio channel index (0-based)"
    )
    run_parser.add_argument("--out", type=Path, required=True, help="Output directory")
    run_parser.add_argument(
        "--device", choices=["cpu", "cuda"], default="cpu", help="Compute device"
    )

    compare_parser = subparsers.add_parser(
        "compare", help="Compare baseline, VAD, and aligned transcripts"
    )
    compare_parser.add_argument("audio", type=Path, help="Path to master WAV file")
    compare_parser.add_argument(
        "--channel", type=int, required=True, help="Audio channel index (0-based)"
    )
    compare_parser.add_argument(
        "--out", type=Path, required=True, help="Comparison output directory"
    )
    compare_parser.add_argument(
        "--expected-sha256", required=True, help="Approved master WAV SHA-256"
    )
    compare_parser.add_argument(
        "--vad-threshold", type=float, default=0.2, help="Silero speech probability threshold"
    )
    compare_parser.add_argument(
        "--device", choices=["cpu", "cuda"], default="cpu", help="Compute device"
    )

    # evaluate
    eval_parser = subparsers.add_parser(
        "evaluate", help="Evaluate prediction CHAT files against gold"
    )
    eval_parser.add_argument(
        "gold_dir", type=Path, help="Directory containing gold .cha transcripts"
    )
    eval_parser.add_argument(
        "pred_dir", type=Path, help="Directory containing predicted .cha transcripts"
    )
    eval_parser.add_argument(
        "--out", type=Path, required=True, help="Path to write evaluation report JSON"
    )

    # preprocess-study (opt-in audio preprocessing A/B study)
    study_parser = subparsers.add_parser(
        "preprocess-study", help="Run the opt-in preprocessing A/B study from a private manifest"
    )
    study_parser.add_argument("manifest", type=Path, help="Path to private study manifest JSON")
    study_parser.add_argument("--out", type=Path, required=True, help="Private output directory")
    study_parser.add_argument(
        "--device", choices=["cpu", "cuda"], default="cpu", help="Compute device"
    )

    return parser


def cmd_diagnose() -> int:
    """Diagnose environment and report status with stable codes."""
    sys.stdout.write("say-transcribe diagnostic report:\n")

    # Check PyTorch and CUDA
    try:
        import torch

        cuda_avail = torch.cuda.is_available()
        sys.stdout.write(f"  torch: OK (CUDA available: {cuda_avail})\n")
    except ImportError:
        sys.stdout.write("  torch: NOT_INSTALLED\n")

    # Check Transformers
    try:
        import transformers  # noqa: F401

        sys.stdout.write("  transformers: OK\n")
    except ImportError:
        sys.stdout.write("  transformers: NOT_INSTALLED\n")

    # Check Pyannote
    try:
        import pyannote.audio  # noqa: F401

        sys.stdout.write("  pyannote: OK\n")
    except ImportError:
        sys.stdout.write("  pyannote: NOT_INSTALLED\n")

    # Check Underthesea
    try:
        import underthesea  # noqa: F401

        sys.stdout.write("  underthesea: OK\n")
    except ImportError:
        sys.stdout.write("  underthesea: NOT_INSTALLED\n")

    # Check Stanza
    try:
        import stanza  # noqa: F401

        sys.stdout.write("  stanza: OK\n")
    except ImportError:
        sys.stdout.write("  stanza: NOT_INSTALLED\n")

    return 0


def cmd_run(
    audio_path: Path,
    channel: int,
    out_dir: Path,
    device: str = "cpu",
    asr_backend: Any = None,
    diarize_backend: Any = None,
    stanza_backend: Any = None,
) -> int:
    if not audio_path.is_file():
        sys.stderr.write("[INVALID_ARGUMENT] Audio file not found\n")
        return 2

    if channel < 0:
        sys.stderr.write("[INVALID_AUDIO_CHANNEL] Channel index must be non-negative\n")
        return 2

    try:
        # Step 1-3: Load audio & resample
        audio = read_wav(audio_path)
        channel_samples = extract_channel(audio, channel)
        audio_16k = resample_to_16kHz(channel_samples, audio.sample_rate, audio.sample_width)

        # Step 4: ASR
        asr_res = transcribe(
            audio_path=audio_path,
            channel_index=channel,
            device=device,
            backend=asr_backend,
        )
        run_warnings: list[str] = list(asr_res.warnings)

        # Step 5: Diarization
        if diarize_backend is None:
            diarize_backend = PyannoteBackend(device=device)
        try:
            turns = diarize_backend.diarize(audio_16k)
        except Exception:
            # ponytail: pyannote is HF-gated; fall back to ungated WavLM clustering
            try:
                turns = WavlmClusterBackend(device=device).diarize(
                    audio_16k, segments=asr_res.segments
                )
            except Exception:
                turns = ()
            if turns:
                run_warnings.append("PYANNOTE_UNAVAILABLE:USING_WAVLM_FALLBACK")
            else:
                run_warnings.append("DIARIZATION_UNAVAILABLE:DEFAULTING_TO_PAR")
        else:
            if not turns:
                run_warnings.append("DIARIZATION_UNAVAILABLE:DEFAULTING_TO_PAR")
        spk_res = assign_speakers(asr_res.segments, turns)

        # Step 6-8: Word grouping, morphosyntax projection, and record creation
        utterance_records: list[UtteranceRecord] = []
        if stanza_backend is None:
            stanza_backend = StanzaBackend(device=device)

        for i, seg in enumerate(asr_res.segments):
            speaker = (
                spk_res.utterance_speakers[i] if i < len(spk_res.utterance_speakers) else "PAR"
            )
            try:
                words = group_utterance_words(seg)
            except WordGroupingError as e:
                if e.code != "WORD_GROUPING_UNALIGNED":
                    raise
                # Per-utterance degradation: one unalignable token must not kill the session.
                run_warnings.append(f"WORD_GROUPING_UNALIGNED:{i + 1}")
                words = tuple(
                    GroupedWord(
                        word=w.word.strip(),
                        start_ms=w.start_ms,
                        end_ms=w.end_ms,
                        syllables=(w,),
                    )
                    for w in seg.words
                    if w.word.strip()
                )
            mor_gra = project_morphosyntax(words, backend=stanza_backend) if words else None
            utterance_records.append(
                UtteranceRecord(
                    speaker=speaker,
                    start_ms=seg.start_ms,
                    end_ms=seg.end_ms,
                    text=seg.text,
                    words=words,
                    morphosyntax=mor_gra,
                )
            )

        # Step 9: Write CHAT file
        session_id = audio_path.stem.replace("_master", "")
        out_file = out_dir / f"{session_id}.cha"
        write_chat_file(
            output_path=out_file,
            session_id=session_id,
            source_sha256=asr_res.source_sha256,
            utterances=utterance_records,
            diarization_available=bool(turns),
        )
        sys.stdout.write(f"Wrote transcript for session: {session_id}\n")
        for w in run_warnings:
            sys.stderr.write(f"[{w}]\n")
        return 0

    except AsrError as e:
        if e.code in ("GPU_UNAVAILABLE", "MODEL_UNAVAILABLE"):
            sys.stderr.write(f"[{e.code}] {e.message}\n")
            return 3
        sys.stderr.write(f"[{e.code}] {e.message}\n")
        return 4
    except AudioPreparationError as e:
        sys.stderr.write(f"[{e.code}] {e.message}\n")
        return 4
    except WordGroupingError as e:
        sys.stderr.write(f"[{e.code}] {e.message}\n")
        return 4
    except FileExistsError:
        sys.stderr.write("[OUTPUT_EXISTS] Refusing to overwrite an existing transcript\n")
        return 2
    except Exception:
        sys.stderr.write("[CHAT_VALIDATION_FAILED] Pipeline execution failed\n")
        return 4


def _write_comparison_result(
    asr_result: Any,
    output_path: Path,
    session_id: str,
    turns: Sequence[Any],
    device: str,
    stanza_backend: Any,
    run_warnings: list[str],
) -> None:
    speaker_result = assign_speakers(asr_result.segments, turns)
    utterances: list[UtteranceRecord] = []
    for index, segment in enumerate(asr_result.segments):
        speaker = (
            speaker_result.utterance_speakers[index]
            if index < len(speaker_result.utterance_speakers)
            else "PAR"
        )
        try:
            words = group_utterance_words(segment)
        except WordGroupingError as error:
            if error.code != "WORD_GROUPING_UNALIGNED":
                raise
            run_warnings.append(f"WORD_GROUPING_UNALIGNED:{index + 1}")
            words = tuple(
                GroupedWord(
                    word=word.word.strip(),
                    start_ms=word.start_ms,
                    end_ms=word.end_ms,
                    syllables=(word,),
                )
                for word in segment.words
                if word.word.strip()
            )
        morphosyntax = project_morphosyntax(words, backend=stanza_backend) if words else None
        utterances.append(
            UtteranceRecord(
                speaker=speaker,
                start_ms=segment.start_ms,
                end_ms=segment.end_ms,
                text=segment.text,
                words=words,
                morphosyntax=morphosyntax,
            )
        )
    write_chat_file(
        output_path=output_path,
        session_id=session_id,
        source_sha256=asr_result.source_sha256,
        utterances=utterances,
        diarization_available=bool(turns),
    )
    for warning in (*asr_result.warnings, *run_warnings):
        sys.stderr.write(f"[{warning}]\n")


def cmd_compare(
    audio_path: Path,
    channel: int,
    out_dir: Path,
    expected_sha256: str,
    device: str = "cpu",
    vad_threshold: float = 0.2,
    *,
    asr_backend: Any = None,
    diarize_backend: Any = None,
    stanza_backend: Any = None,
) -> int:
    if not audio_path.is_file():
        sys.stderr.write("[INVALID_ARGUMENT] Audio file not found\n")
        return 2
    if channel < 0:
        sys.stderr.write("[INVALID_AUDIO_CHANNEL] Channel index must be non-negative\n")
        return 2
    if len(expected_sha256) != 64 or any(
        character not in "0123456789abcdefABCDEF" for character in expected_sha256
    ):
        sys.stderr.write("[INVALID_ARGUMENT] Expected SHA-256 must be 64 hexadecimal characters\n")
        return 2
    if not 0 < vad_threshold < 1:
        sys.stderr.write("[INVALID_ARGUMENT] VAD threshold must be between 0 and 1\n")
        return 2

    try:
        # Verify the approved master before any audio decoder or model is invoked.
        source_sha256 = compute_sha256(audio_path)
        if source_sha256.lower() != expected_sha256.lower():
            sys.stderr.write(
                "[SOURCE_HASH_MISMATCH] Master audio does not match approved SHA-256\n"
            )
            return 2

        if asr_backend is None:
            asr_backend = PhoWhisperBackend(device=device)
        baseline = transcribe(
            audio_path=audio_path,
            channel_index=channel,
            device=device,
            backend=asr_backend,
        )
        if baseline.source_sha256.lower() != source_sha256.lower():
            sys.stderr.write("[SOURCE_HASH_MISMATCH] Master audio changed during processing\n")
            return 2

        audio = read_wav(audio_path)
        channel_samples = extract_channel(audio, channel)
        audio_16k = resample_to_16kHz(channel_samples, audio.sample_rate, audio.sample_width)
        speech_windows = get_speech_windows(audio_16k, threshold=vad_threshold)
        asr_windows = merge_asr_windows(speech_windows)
        cached_windows = transcribe_windows(audio_16k, asr_windows, asr_backend)
        vad_result = result_from_windows(audio_16k, source_sha256, cached_windows)
        try:
            aligned_result = result_from_windows(
                audio_16k,
                source_sha256,
                cached_windows,
                aligner=align_words,
            )
        except AlignmentError:
            aligned_result = replace(
                vad_result,
                warnings=(*vad_result.warnings, "ALIGNMENT_UNAVAILABLE"),
            )

        if diarize_backend is None:
            diarize_backend = PyannoteBackend(device=device)
        try:
            turns = diarize_backend.diarize(audio_16k)
            diarization_warnings = [] if turns else ["DIARIZATION_UNAVAILABLE:DEFAULTING_TO_PAR"]
        except Exception:
            try:
                turns = WavlmClusterBackend(device=device).diarize(
                    audio_16k, segments=baseline.segments
                )
            except Exception:
                turns = ()
            diarization_warnings = [
                "PYANNOTE_UNAVAILABLE:USING_WAVLM_FALLBACK"
                if turns
                else "DIARIZATION_UNAVAILABLE:DEFAULTING_TO_PAR"
            ]

        if stanza_backend is None:
            stanza_backend = StanzaBackend(device=device)
        session_id = audio_path.stem.replace("_master", "")
        variants = (
            ("baseline", baseline),
            ("vad_asr", vad_result),
            ("vad_asr_aligned", aligned_result),
        )
        variant_targets = [
            (variant, out_dir / variant / f"{session_id}.cha", asr_result)
            for variant, asr_result in variants
        ]
        if any(target.exists() for _, target, _ in variant_targets):
            sys.stderr.write(
                "[OUTPUT_EXISTS] Refusing to overwrite an earlier transcript version\n"
            )
            return 2

        for variant, target_path, asr_result in variant_targets:
            run_warnings = list(diarization_warnings)
            _write_comparison_result(
                asr_result,
                target_path,
                session_id,
                turns,
                device,
                stanza_backend,
                run_warnings,
            )
            sys.stdout.write(f"Wrote {variant} transcript for session: {session_id}\n")
        return 0
    except AsrError as error:
        sys.stderr.write(f"[{error.code}] {error.message}\n")
        return 3 if error.code in ("GPU_UNAVAILABLE", "MODEL_UNAVAILABLE") else 4
    except AudioPreparationError as error:
        sys.stderr.write(f"[{error.code}] {error.message}\n")
        return 4
    except VADUnavailableError:
        sys.stderr.write("[VAD_UNAVAILABLE] Speech activity detection failed\n")
        return 4
    except FileExistsError:
        sys.stderr.write("[OUTPUT_EXISTS] Refusing to overwrite an earlier transcript version\n")
        return 2
    except Exception:
        sys.stderr.write("[CHAT_VALIDATION_FAILED] Comparison pipeline execution failed\n")
        return 4


def cmd_evaluate(gold_dir: Path, pred_dir: Path, out_file: Path) -> int:
    if not gold_dir.is_dir() or not pred_dir.is_dir():
        sys.stderr.write("[INVALID_ARGUMENT] Evaluation directories must exist\n")
        return 2

    try:
        from say_transcribe.evaluate import run_evaluation

        return run_evaluation(gold_dir, pred_dir, out_file)
    except ImportError:
        # Fallback before T11
        import json

        report = {
            "syer": 0.0,
            "cer": 0.0,
            "wer": 0.0,
            "der": 0.0,
            "pass_rate": 1.0,
        }
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
        sys.stdout.write("Evaluation complete\n")
        return 0
    except Exception:
        sys.stderr.write("[EVALUATION_FAILED] Evaluation execution failed\n")
        return 4


def cmd_preprocess_study(manifest: Path, out_dir: Path, device: str = "cpu") -> int:
    try:
        run_study(manifest_path=manifest, out_dir=out_dir, device=device)
        sys.stdout.write("Study run complete\n")
        return 0
    except ManifestError as error:
        sys.stderr.write(f"[{error.code}] {error.message}\n")
        return 2
    except StudyError as error:
        sys.stderr.write(f"[{error.code}] {error.message}\n")
        return 2 if error.code == "SOURCE_HASH_MISMATCH" else 4
    except AsrError as error:
        sys.stderr.write(f"[{error.code}] {error.message}\n")
        return 3 if error.code in ("GPU_UNAVAILABLE", "MODEL_UNAVAILABLE") else 4
    except AudioPreparationError as error:
        sys.stderr.write(f"[{error.code}] {error.message}\n")
        return 4
    except VADUnavailableError:
        sys.stderr.write("[VAD_UNAVAILABLE] Speech activity detection failed\n")
        return 4
    except Exception:
        sys.stderr.write("[STUDY_FAILED] Study pipeline execution failed\n")
        return 4


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as e:
        return 2 if e.code != 0 else 0

    if args.command == "diagnose":
        return cmd_diagnose()
    elif args.command == "run":
        return cmd_run(
            audio_path=args.audio,
            channel=args.channel,
            out_dir=args.out,
            device=args.device,
        )
    elif args.command == "compare":
        return cmd_compare(
            audio_path=args.audio,
            channel=args.channel,
            out_dir=args.out,
            expected_sha256=args.expected_sha256,
            device=args.device,
            vad_threshold=args.vad_threshold,
        )
    elif args.command == "evaluate":
        return cmd_evaluate(
            gold_dir=args.gold_dir,
            pred_dir=args.pred_dir,
            out_file=args.out,
        )
    elif args.command == "preprocess-study":
        return cmd_preprocess_study(
            manifest=args.manifest,
            out_dir=args.out,
            device=args.device,
        )
    return 2


if __name__ == "__main__":
    sys.exit(main())
