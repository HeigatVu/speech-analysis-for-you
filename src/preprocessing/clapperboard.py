import os
import sys
import json
from tqdm import tqdm
import glob
import typing
from collections import defaultdict
import hydra
from omegaconf import DictConfig

import numpy as np
import pandas as pd
from scipy import signal
from scipy.io import wavfile

# # Add project root to Python path for imports
# project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)

import src.utils.audio as uAudio
import src.utils.visualization as uVisualization
import src.utils.file_io as uFile

def clapperboard_detection(
                            audio_path:str,
                            config:DictConfig,
                            threshold:int=0.5,
                            min_distance_sec:int=3,
                            save_json:bool=True,
                            output_json:str="",
                            file_name:str="",
                            ) -> dict:

    # Load audio
    sr_raw_audio, raw_audio = wavfile.read(audio_path)
    _, clapperboard_sound_effect = wavfile.read(f'{config["project"]["project_path"]}/data/clapperboard-sound-effect.wav')

    # Convert to mono channel
    mono_raw_audio = uAudio.mono_channel_converter(raw_audio)
    mono_clapperboard_sound_effect = uAudio.mono_channel_converter(clapperboard_sound_effect)

    # Normalize audio and cross-corelation
    raw_audio_norm = uAudio.normalize_with_mean_std(mono_raw_audio)
    mono_clapperboard_sound_effect_norm = uAudio.normalize_with_mean_std(mono_clapperboard_sound_effect)
    correlation = signal.correlate(raw_audio_norm, mono_clapperboard_sound_effect_norm, mode="valid")

    # Visualization
    lag = np.arange(len(correlation))
    visualization = uVisualization.correlation_visualization(lag, correlation, threshold)

    # Avoid overlap clapperboard sound effect and find peak of correlation
    min_distance_samples = min_distance_sec*sr_raw_audio

    peaks, _ = signal.find_peaks(
        correlation,
        height=threshold,
        distance=min_distance_samples,
    )

    # Convert clapperboard position
    clapperboard_position_samples = peaks.tolist()
    correlation_values = correlation[peaks].tolist()

    clapperboard_position_seconds = (peaks/sr_raw_audio).tolist()
    clapperboard_position_ms = (peaks/sr_raw_audio * 1000).tolist()

    # Store the result
    result = defaultdict(list)
    result["positions_samples"] = clapperboard_position_samples
    result["positions_seconds"] = clapperboard_position_seconds
    result["positions_ms"] = clapperboard_position_ms
    result["correlation_scores"] = correlation_values

    # Create output folder for json and save json
    if save_json and output_json and file_name:
        uFile.create_dir(output_json, config)
        uFile.save_json(result, output_json, file_name, config)

    return result

@hydra.main(config_path="../config", config_name="main", version_base=None)
def main(config: DictConfig) -> str:
    # Get config parameters
    audio_path = config.get("audio_path", "")
    threshold = config.get("threshold", 0.5)
    min_distance_sec = config.get("min_distance_sec", 3)
    save_json = config.get("save_json", True)
    output_json = config.get("output_json", "")
    file_name = config.get("file_name", "")
    
    # Extract file_name from audio_path if not provided
    if not file_name and audio_path:
        full_file_name = audio_path.split('/')[-1]
        file_name = full_file_name.split('.')[0]
    
    result = clapperboard_detection(
        audio_path=audio_path,
        config=config,
        threshold=threshold,
        min_distance_sec=min_distance_sec,
        save_json=save_json,
        output_json=output_json,
        file_name=file_name
    )
    
    return f"Splitting audio successfully"


if __name__ == "__main__":
    # Option 1: Call main() directly - Hydra will automatically load config
    # 
    # Usage from command line:
    #   python -m src.preprocessing.clapperboard audio_path=/path/to/audio.wav output_json=path
    #
    # For hardcoded values when running directly (no CLI args), set them here:
    # Set default values if not provided via command line (sys.argv[0] is script name)
    if len(sys.argv) == 1:  # Only script name, no arguments
        audio_file = "/home2/ducvu/speech-analysis-for-you/data/raw/participant_001.wav"
        output_json = "splitedAudio/clapperboard_position"
        
        # Override config via command-line style arguments (Hydra will parse these)
        sys.argv.extend([
            f"audio_path={audio_file}",
            f"output_json={output_json}"
        ])
        # Note: file_name will be auto-extracted from audio_path in main() function
    
    # Call main() - @hydra.main decorator will handle config loading and argument parsing
    main()


