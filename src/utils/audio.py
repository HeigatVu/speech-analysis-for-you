import os
from scipy.io import wavfile
from pydub import AudioSegment
import numpy as np
from numpy.typing import NDArray

def load_audio_pydub(audio_path: str, mono: bool = True) -> tuple[int, NDArray]:
    """ Load audio file
    Input:
        audio_path: path of audio file
        mono: option for change audio to mono or not
    Output:
        sampling rate and numpy array of audio file
    """
    # open file with pydub
    audio_file = AudioSegment.from_file(audio_path)
    # Extract sampling rate
    sr_audio_file = audio_file.frame_rate
    # Ensure mono channel
    if mono and audio_file.channels != 1:
        audio_file = audio_file.set_channels(1)

    return sr_audio_file, audio_file


# Second will format in MM:SS.mmm
def format_time(seconds:int) -> str:
    """ Format time to MM:SS.mmm format
    Input:
        seconds: duration
    Output:
        string with formatted 
            - Zero-padded to 2 digits
            - 6 value with zero-padded to 2 digits + dot + 3 decimal places

    """
    minute_part = int(seconds//60)
    second_part = seconds%60
    return f"{minute_part:02d}:{second_part:06.3f}"


def normalize_with_zscore(mono_audio_file:NDArray) -> NDArray:
    """Normalizing with z-score scaling"""
    # Make audio signal centered
    audio_file_centered = mono_audio_file - np.mean(mono_audio_file)
    # Normalized with std (avoid division by zero)
    audio_file_norm = audio_file_centered / (np.std(audio_file_centered) + 1e-10)

    return audio_file_norm

def normalize_with_minmax(mono_audio_file:NDArray) -> NDArray:
    """Normalizing with min-max scaling"""
    min_value = np.min(np.abs(mono_audio_file))
    max_value = np.max(np.abs(mono_audio_file))
    x_std = (mono_audio_file - min_value) / (max_value - min_value)
    audio_file_norm = x_std * (1-0) + 0

    return audio_file_norm