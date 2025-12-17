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
from scipy import signal

import src.utils.audio as uAudio
import src.utils.visualization as uVisualization
import src.utils.file_io as uFile

def clapperboard_detection(
                            audio_path:str,
                            config:DictConfig,
                            threshold:int=0.5,
                            min_distance_sec:int=1.0,
                            save_json:bool=True,
                            output_json:str="",
                            file_name:str="",
                            save_viz_cor_path:str=os.getcwd(),
                            ) -> dict:
    """ Detecting clapperboard to split task
    Input: 
        audio_path: path of audio file
        config: configuration file to take parameter and path
        threshold: threshold to take correlation
        min_distance_sec: time for detecting new peak after the first peak
    Output:
        dictionary contains all information of clapperboard

    """

    clapperboard_path = f'{config["project"]["project_path"]}/data/clapperboard-sound-effect.wav'

    # Load audio with pydub with sample_rate, mono audio_file
    sr_raw_audio, mono_raw_audio = uAudio.load_audio_pydub(audio_path, mono=True)
    _, mono_clapperboard_sound_effect = uAudio.load_audio_pydub(
                                                            clapperboard_path,
                                                            mono=True,
                                                        )

    # Convert to numpy array
    mono_raw_audio_array = np.array(mono_raw_audio.get_array_of_samples(), dtype=float)
    mono_clapperboard_sound_effect_array = np.array(
                                                mono_clapperboard_sound_effect.get_array_of_samples(), 
                                                dtype=float
                                            )

    # Normalize audio and cross-corelation
    mono_raw_audio_array_norm = uAudio.normalize_with_minmax(mono_raw_audio_array)
    mono_clapperboard_sound_effect_array_norm = uAudio.normalize_with_minmax(mono_clapperboard_sound_effect_array)
    
    # Cross-correlation
    correlation = signal.correlate(
                            mono_raw_audio_array_norm, 
                            mono_clapperboard_sound_effect_array_norm, 
                            mode="full"
                        )
    correlation = correlation / np.max(np.abs(correlation))

    # Visualization
    lag = np.arange(-len(mono_clapperboard_sound_effect_array_norm) + 1, len(mono_raw_audio_array_norm))
    visualization = uVisualization.correlation_visualization(
                                            lag, 
                                            correlation, 
                                            threshold, 
                                            save_path=save_viz_cor_path
                                        )

    # Avoid overlap clapperboard sound effect and find peak of correlation
    min_distance_samples = int(min_distance_sec*sr_raw_audio)
    peaks, _ = signal.find_peaks(
                    correlation,
                    height=threshold,
                    distance=min_distance_samples,
                )

    # Convert clapperboard position to actual audio position
    clapperboard_position_samples = []
    correlation_values = []

    for peak_idx in peaks:
        actual_position = lag[peak_idx]
        if 0 <= actual_position < len(mono_raw_audio_array_norm):
            clapperboard_position_samples.append(actual_position)
            correlation_values.append(correlation[peak_idx])

    # Convert to seconds
    clapperboard_positions_seconds = [i / sr_raw_audio for i in clapperboard_position_samples]
    clapperboard_positions_ms = [int(pos / sr_raw_audio * 1000) for pos in clapperboard_position_samples]

    # Create result dictionary
    result = {
        "audio_file": audio_path,
        "slate_file": clapperboard_path,
        "sample_rate": sr_raw_audio,
        "audio_duration_sec": len(mono_raw_audio_array_norm) / sr_raw_audio,
        "audio_duration_ms": len(mono_raw_audio_array_norm),
        "slate_duration_sec": len(mono_clapperboard_sound_effect_array_norm) / sr_raw_audio,
        "slate_duration_ms": len(mono_clapperboard_sound_effect_array_norm),
        "num_slates_found": len(clapperboard_position_samples),
        "threshold_used": threshold,
        "min_distance_sec": min_distance_sec,
        "slate_positions": [
            {
                "index": i,
                "sample": int(pos_sample),
                "time_sec": round(pos_sec, 3),
                "time_ms": pos_ms,
                "time_formatted": uAudio.format_time(pos_sec),
                "correlation_score": round(corr, 4)
            }
            for i, (pos_sample, pos_sec, pos_ms, corr) in enumerate(
                zip(clapperboard_position_samples, clapperboard_positions_seconds, 
                    clapperboard_positions_ms, correlation_values)
            )
        ]
    }

    # Create output folder for json and save json
    if save_json and output_json and file_name:
        uFile.create_dir(output_json, config)
        uFile.save_json(result, output_json, file_name, config)

    return result


def split_audio_by_position(
                                audio_file:str,
                                splited_json_path:str,
                                output_dir:str,
                                remove_clapperboard:bool=True,
                                clapperboard_buffer_ms:int=500,
                                min_segment_duration_sec:float=1.0,
                                question_list:list=[],
                                file_name:str="",
                            ) -> list:

    pass
    return f"Splitting audio successfully"

@hydra.main(config_path="../config", config_name="main", version_base=None)
def main(config: DictConfig) -> str:

    # Get config preprocessing parameters
    raw_audio_path = config.preprocessing.raw_audio_path
    splited_threshold = config.preprocessing.splited_threshold
    min_distance_sec = config.preprocessing.min_distance_sec
    save_json = config.preprocessing.save_json_splited_position
    output_json_splited_position_path = config.preprocessing.output_json_splited_position_path
    save_viz_cor_path = config.preprocessing.output_img_correlation
    # Extract file_name from audio_path if not provided
    if raw_audio_path:
        full_file_name = raw_audio_path.split('/')[-1]
        file_name = full_file_name.split('.')[0]
    
    result = clapperboard_detection(
        audio_path=raw_audio_path,
        config=config,
        threshold=splited_threshold,
        min_distance_sec=min_distance_sec,
        save_json=save_json,
        output_json=output_json_splited_position_path,
        file_name=file_name,
        save_viz_cor_path=save_viz_cor_path,
    )
    
    return f"Splitting audio successfully"


if __name__ == "__main__":
    main()

