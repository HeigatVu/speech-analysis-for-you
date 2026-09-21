from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


class AudioProfileError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message


@dataclass(frozen=True)
class AudioProfile:
    profile_id: str
    target_sample_rate: int
    highpass_hz: float
    lowpass_hz: float
    max_trailing_padding_samples: int
    gsm_sample_rate: int = 8000


TELL_INSPIRED_V1 = AudioProfile(
    profile_id="tell-inspired-v1",
    target_sample_rate=16000,
    highpass_hz=200.0,
    lowpass_hz=3400.0,
    max_trailing_padding_samples=319,
    gsm_sample_rate=8000,
)

_KNOWN_PROFILES: Mapping[str, AudioProfile] = {
    TELL_INSPIRED_V1.profile_id: TELL_INSPIRED_V1,
}


def get_audio_profile(profile_id: str) -> AudioProfile:
    """Retrieve a known audio profile by ID."""
    if profile_id not in _KNOWN_PROFILES:
        raise AudioProfileError("UNKNOWN_PROFILE", f"Unknown profile: {profile_id}")
    return _KNOWN_PROFILES[profile_id]
