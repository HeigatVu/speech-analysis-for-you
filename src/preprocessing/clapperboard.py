import os
import sys
import json
from tqdm import tqdm
import glob
import typing
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
        "sample_rate": sr_raw_audio,
        "audio_duration_sec": len(mono_raw_audio_array_norm) / sr_raw_audio,
        "audio_duration_ms": int(len(mono_raw_audio_array_norm) / sr_raw_audio * 1000),
        "clapperboard_duration_sec": len(mono_clapperboard_sound_effect_array_norm) / sr_raw_audio,
        "clapperboard_duration_ms": int(len(mono_clapperboard_sound_effect_array_norm) / sr_raw_audio * 1000),
        "num_clapperboard_found": len(clapperboard_position_samples),
        "threshold_used": threshold,
        "min_distance_sec": min_distance_sec,
        "clapperboard_positions": [
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
                            # splited_json_path:str,
                            result_splited_position:dict,
                            output_dir:str,
                            config:DictConfig,
                            remove_clapperboard:bool=True,
                            clapperboard_buffer_ms:int=500.0,
                            min_segment_duration_sec:float=1.0,
                            save_json:bool=True,
                            ) -> list:
    """ Split audio follwing json file
    Input:
        audio_file: path of audio path
        // splited_json_path: json for saving position of clapperboard
        result_splited_position: result from clapperboard detection above
        output_dir: output of splited audio
        config: configuration of splited audio
        remove_clapperboard: option for including clapperboard in splitted audio
        clapperboard_buffer_ms:  adds padding to avoid cutting off audio
        min_segment_duration_sec:  keeps segments longer than the minimum duration
        question_list: list of tasks in research
        file_name: name of file to create directory
        save_json: backup json to double check clappboard detection
    Output:
        None
    """


    # # Load json file
    # data = uFile.load_json(splited_json_path)

    # Load audio
    _, raw_audio = uAudio.load_audio_pydub(audio_file, mono=False)
    audio_length_ms = len(raw_audio)
    print(f"Audio length: {audio_length_ms}ms ({audio_length_ms/1000:.2f}s)")

    # Extract clapperboard position in milliseconds
    clapperboard_time_ms = [pos["time_ms"] for pos in result_splited_position["clapperboard_positions"]]
    print(f"Clapperboard positions: {clapperboard_time_ms}")
    print(f"Number of clapperboards: {len(clapperboard_time_ms)}")

    clapperboard_duration_ms = result_splited_position["clapperboard_duration_ms"]
    print(f"Clapperboard duration: {clapperboard_duration_ms}ms, Buffer: {clapperboard_buffer_ms}ms")

    # Create output folder
    uFile.create_dir(output_dir, config)

    # Calculate segment boudaries
    segments = []
    segment_counter = 0
    for i in range(len(clapperboard_time_ms)+1):
        if i == 0: 
        # First clapperboard position in audio
            start = 0
            if remove_clapperboard:
                end = clapperboard_time_ms[0]
            else:
                end = clapperboard_time_ms[0] + clapperboard_duration_ms
            label = "intro"
            clap_ref_index = 0
        elif i < len(clapperboard_time_ms):
            prev_clap_end = clapperboard_time_ms[i - 1] + clapperboard_duration_ms
            
            if remove_clapperboard:
                start = prev_clap_end + clapperboard_buffer_ms
                end = clapperboard_time_ms[i]
            else:
                start = prev_clap_end + clapperboard_buffer_ms
                end = clapperboard_time_ms[i] + clapperboard_duration_ms
            
            label = f"segment_{i}"
            clap_ref_index = i

        elif i == len(clapperboard_time_ms):
            prev_clap_end = clapperboard_time_ms[-1] + clapperboard_duration_ms
            start = prev_clap_end + clapperboard_buffer_ms
            end = audio_length_ms
            label = "outro"
            clap_ref_index = len(clapperboard_time_ms) - 1

        duration_sec = (end - start) / 1000.0
        if duration_sec >= min_segment_duration_sec:
            segments.append({
                "segment_id": segment_counter,
                "clapperboard_index": clap_ref_index,
                "start_ms": int(start),
                "end_ms": int(end),
                "duration_sec": round(duration_sec, 3),
                "clapperboard_time": uAudio.format_time(clapperboard_time_ms[clap_ref_index] / 1000),
                "label": label,
            })
            segment_counter += 1
        else:
            print(f"Skip Segment {segment_counter}: {label} [{start}ms - {end}ms] = {duration_sec:.3f}s\n")


    # Export json and audio file
    audio_name = result_splited_position["audio_file"].split('/')[-1]
    file_name = audio_name.split('.')[0]
    # Save segment
    for seg in tqdm(segments, desc=f"Processing segments {file_name}"):
        # Extract segment
        segment_audio = raw_audio[seg["start_ms"]:seg["end_ms"]]
        output_path = os.path.join(output_dir, f"{file_name}_{seg["segment_id"]:03d}.wav")
        # Export file
        segment_audio.export(output_path, format="wav")

    if save_json:
        uFile.save_json(segments, output_dir, f"{file_name}_segments", config)




@hydra.main(config_path="../config", config_name="main", version_base=None)
def main(config: DictConfig) -> str:
    # Get config preprocessing parameters
    raw_audio_path = config.preprocessing.raw_audio_path
    splited_threshold = config.preprocessing.splited_threshold
    min_distance_sec = config.preprocessing.min_distance_sec
    save_json = config.preprocessing.save_json_splited_position
    output_json_splited_position_path = config.preprocessing.output_json_splited_position_path
    save_viz_cor_path = config.preprocessing.output_img_correlation
    output_audio_segment_path = config.preprocessing.output_audio_segment_path
    min_segment_duration_sec = config.preprocessing.min_segment_duration_sec
    clappboard_buffer_ms = config.preprocessing.clapperboard_buffer_ms

    # Extract file_name from audio_path if not provided
    if raw_audio_path:
        full_file_name = raw_audio_path.split('/')[-1]
        file_name = full_file_name.split('.')[0]
    
    # Clapperboard position
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

    # Spliting original audio
    split_audio_by_position(
        audio_file=raw_audio_path,
        result_splited_position=result,
        output_dir=output_audio_segment_path,
        config=config,
        clapperboard_buffer_ms=clappboard_buffer_ms,
        min_segment_duration_sec=min_segment_duration_sec,
    )

    return f"Splitting audio successfully"


if __name__ == "__main__":
    main()

