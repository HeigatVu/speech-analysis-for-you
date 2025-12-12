import os
from scipy.io import wavfile
import numpy as np
from numpy.typing import NDArray

# Second will format in MM:SS.mmm
def format_time(seconds:int) -> str:
    minute_part = int(seconds//60)
    second_part = seconds%60
    return f"{minute_part:02d}:{second_part:06.3f}"


# Stereo to mono channel
def mono_channel_converter(audio_file:NDArray) -> NDArray:
    # Check channel > 1
    if audio_file.shape[1] > 1:
        # Average channel
        audio_file = np.mean(audio_file, axis=1, keepdims=True)

    return audio_file

def normalize_with_mean_std(audio_file:NDArray) -> NDArray:
    # Make audio signal to center
    audio_file_centered = audio_file - np.mean(audio_file)
    # Normalized witn std
    audio_file_norm = audio_file_centered / (np.std(audio_file_centered) + 1e-10)
    
    return audio_file_norm


if __name__ == "__main__":
    pass