from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from say_transcribe.asr import AsrError, AsrSegment

DRAFT_SPEAKER_NOTE = "speaker labels draft, auto-diarized; review before use"


@dataclass(frozen=True)
class DiarizationTurn:
    start_ms: int
    end_ms: int
    cluster_id: str


@dataclass(frozen=True)
class SpeakerDiarizationResult:
    utterance_speakers: tuple[str, ...]  # "PAR" or "INV" per utterance
    is_draft: bool = True
    draft_comment: str = DRAFT_SPEAKER_NOTE
    cluster_to_role: dict[str, str] | None = None


class PyannoteBackend:
    """Lazy-loaded Pyannote speaker diarization backend."""

    def __init__(
        self,
        model_id: str = "pyannote/speaker-diarization-3.1",
        device: str = "cpu",
        auth_token: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.auth_token = auth_token
        self._pipeline: Any = None

    def load(self) -> None:
        if self.device == "cuda":
            try:
                import torch

                if not torch.cuda.is_available():
                    raise AsrError("GPU_UNAVAILABLE", "CUDA device requested but not available")
            except ImportError:
                raise AsrError("GPU_UNAVAILABLE", "PyTorch with CUDA not available") from None

        try:
            from pyannote.audio import Pipeline
            from huggingface_hub import get_token
            import torch

            token = self.auth_token or get_token()
            self._pipeline = Pipeline.from_pretrained(
                self.model_id,
                token=token,
            )
            if self._pipeline is None:
                raise AsrError("MODEL_UNAVAILABLE", "Pyannote pipeline could not be loaded")
            if self.device == "cuda":
                self._pipeline.to(torch.device("cuda"))
        except AsrError:
            raise
        except Exception:
            raise AsrError("MODEL_UNAVAILABLE", "Failed to load pyannote diarization model") from None

    def diarize(
        self, audio_16k_mono: np.ndarray, sample_rate: int = 16000
    ) -> Sequence[DiarizationTurn]:
        """Run diarization on 16kHz float32 audio."""
        if self._pipeline is None:
            self.load()

        try:
            import torch

            # pyannote expects torch.Tensor of shape (channels, samples)
            tensor = torch.from_numpy(audio_16k_mono).unsqueeze(0)
            result = self._pipeline({"waveform": tensor, "sample_rate": sample_rate})
            diarization = getattr(result, "speaker_diarization", result)

            turns: list[DiarizationTurn] = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                turns.append(
                    DiarizationTurn(
                        start_ms=int(round(turn.start * 1000.0)),
                        end_ms=int(round(turn.end * 1000.0)),
                        cluster_id=str(speaker),
                    )
                )
            return turns
        except Exception:
            raise AsrError("MODEL_UNAVAILABLE", "Diarization inference execution failed") from None


def assign_speakers(
    asr_segments: Sequence[AsrSegment],
    turns: Sequence[DiarizationTurn],
) -> SpeakerDiarizationResult:
    """Assign ASR utterances to PAR or INV based on diarization overlap and total talk time.

    The speaker cluster with the larger total talk time becomes PAR; the other becomes INV.
    """
    if not asr_segments:
        return SpeakerDiarizationResult(
            utterance_speakers=(),
            is_draft=True,
            draft_comment=DRAFT_SPEAKER_NOTE,
            cluster_to_role={},
        )

    # Calculate total talk time per cluster from diarization turns
    cluster_talk_time: dict[str, int] = {}
    for turn in turns:
        dur = max(0, turn.end_ms - turn.start_ms)
        cluster_talk_time[turn.cluster_id] = cluster_talk_time.get(turn.cluster_id, 0) + dur

    # Rank clusters by total talk time
    sorted_clusters = sorted(
        cluster_talk_time.keys(), key=lambda c: cluster_talk_time[c], reverse=True
    )

    cluster_to_role: dict[str, str] = {}
    if sorted_clusters:
        cluster_to_role[sorted_clusters[0]] = "PAR"
        if len(sorted_clusters) > 1:
            cluster_to_role[sorted_clusters[1]] = "INV"
        # Any additional clusters map to INV
        for extra in sorted_clusters[2:]:
            cluster_to_role[extra] = "INV"

    utterance_speakers: list[str] = []

    for seg in asr_segments:
        if seg.start_ms is None or seg.end_ms is None:
            utterance_speakers.append("PAR")
            continue

        # Find overlapping turns
        overlap_per_cluster: dict[str, int] = {}
        for turn in turns:
            overlap = max(0, min(seg.end_ms, turn.end_ms) - max(seg.start_ms, turn.start_ms))
            if overlap > 0:
                overlap_per_cluster[turn.cluster_id] = (
                    overlap_per_cluster.get(turn.cluster_id, 0) + overlap
                )

        if overlap_per_cluster:
            best_cluster = max(overlap_per_cluster.keys(), key=lambda c: overlap_per_cluster[c])
            utterance_speakers.append(cluster_to_role.get(best_cluster, "PAR"))
        else:
            # Fallback to nearest turn if no direct overlap
            if turns:
                seg_mid = (seg.start_ms + seg.end_ms) / 2.0
                nearest_turn = min(
                    turns,
                    key=lambda t: abs(((t.start_ms + t.end_ms) / 2.0) - seg_mid),
                )
                utterance_speakers.append(cluster_to_role.get(nearest_turn.cluster_id, "PAR"))
            else:
                utterance_speakers.append("PAR")

    return SpeakerDiarizationResult(
        utterance_speakers=tuple(utterance_speakers),
        is_draft=True,
        draft_comment=DRAFT_SPEAKER_NOTE,
        cluster_to_role=cluster_to_role,
    )


class WavlmClusterBackend:
    """Ungated diarization fallback: WavLM-SV window embeddings + agglomerative clustering.

    Same DiarizationTurn contract as PyannoteBackend. ponytail: k=speakers is fixed at
    2 (SPEC §4 two-party sessions); raise n_speakers if multi-party data arrives.
    """

    def __init__(
        self,
        model_id: str = "microsoft/wavlm-base-sv",
        device: str = "cpu",
        n_speakers: int = 2,
        win_ms: int = 1500,
        hop_ms: int = 750,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.n_speakers = n_speakers
        self.win_ms = win_ms
        self.hop_ms = hop_ms
        self._model: Any = None
        self._fe: Any = None

    def load(self) -> None:
        if self.device == "cuda":
            try:
                import torch

                if not torch.cuda.is_available():
                    raise AsrError("GPU_UNAVAILABLE", "CUDA device requested but not available")
            except ImportError:
                raise AsrError("GPU_UNAVAILABLE", "PyTorch with CUDA not available") from None

        try:
            import torch
            from transformers import AutoFeatureExtractor, WavLMForXVector

            self._fe = AutoFeatureExtractor.from_pretrained(self.model_id)
            self._model = WavLMForXVector.from_pretrained(self.model_id)
            if self.device == "cuda":
                self._model.to(torch.device("cuda"))
            self._model.eval()
        except AsrError:
            raise
        except Exception:
            raise AsrError("MODEL_UNAVAILABLE", "Failed to load WavLM speaker model") from None

    def _embed(self, wav: np.ndarray) -> np.ndarray:
        """L2-normalized xvector embedding for one utterance."""
        import torch

        # ponytail: per-utterance level boost for the quiet INV mic on this corpus
        # (analysis view only; source audio is never altered). Gain capped at 30x so
        # near-dead windows do not amplify digital noise into "speaker" signal.
        rms = float(np.sqrt((wav**2).mean()))
        if rms > 1e-6:
            wav = np.clip(wav * min(0.1 / rms, 30.0), -1.0, 1.0)

        feats = self._fe(wav, sampling_rate=16000, return_tensors="pt")
        dev = next(self._model.parameters()).device
        with torch.no_grad():
            vec = self._model(input_values=feats["input_values"].to(dev)).embeddings
        vec = vec.squeeze(0).float().cpu().numpy()
        norm = float(np.linalg.norm(vec)) or 1.0
        return vec / norm

    def diarize(
        self,
        audio_16k_mono: np.ndarray,
        sample_rate: int = 16000,
        segments: Sequence[AsrSegment] | None = None,
    ) -> Sequence[DiarizationTurn]:
        if self._model is None:
            self.load()

        try:
            from scipy.cluster.hierarchy import fcluster, linkage
        except Exception:
            raise AsrError("MODEL_UNAVAILABLE", "scipy clustering unavailable") from None

        try:
            # Cluster at the ASR-utterance level: each segment is one speaker's turn,
            # so embeddings stay single-speaker (fixed windows straddle turn changes
            # and an energy gate starves quiet dual-mono recordings).
            pairs = []
            for seg in segments or ():
                if seg.start_ms is None or seg.end_ms is None or seg.end_ms - seg.start_ms < 300:
                    continue
                start = int(seg.start_ms * 16)
                end = min(int(seg.end_ms * 16), len(audio_16k_mono))
                chunk = audio_16k_mono[start:end]
                if len(chunk) >= 4800:
                    pairs.append((seg, self._embed(chunk.astype(np.float32))))
            if not pairs:
                return ()

            matrix = np.stack([emb for _, emb in pairs])
            if len(pairs) == 1:
                labels = [1]
            else:
                labels = fcluster(
                    linkage(matrix, method="average"), t=self.n_speakers, criterion="maxclust"
                )
            return tuple(
                DiarizationTurn(
                    start_ms=seg.start_ms,
                    end_ms=seg.end_ms,
                    cluster_id=str(int(label) - 1),
                )
                for (seg, _), label in zip(pairs, labels)
            )
        except Exception:
            raise AsrError("MODEL_UNAVAILABLE", "Diarization inference execution failed") from None
