import os
from omegaconf import DictConfig
import hydra
from tqdm import tqdm
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
                            save_viz_cor_path:str=None,
                            clapperboard_path:str=None,
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

    # Load audio with pydub with sample_rate, mono audio_file
    sr_raw_audio, mono_raw_audio = uAudio.load_audio_pydub(audio_path, mono=True)
    _, mono_clap = uAudio.load_audio_pydub(
                                                            clapperboard_path,
                                                            mono=True,
                                                        )

    # Convert to numpy array
    mono_raw_audio_array = np.array(mono_raw_audio.get_array_of_samples(), dtype=float)
    mono_clap_array = np.array(
                                                mono_clap.get_array_of_samples(), 
                                                dtype=float
                                            )

    # Normalize audio and cross-corelation
    mono_raw_audio_norm = uAudio.normalize_with_minmax(mono_raw_audio_array)
    mono_clap_norm = uAudio.normalize_with_minmax(mono_clap_array)
    
    # Cross-correlation
    correlation = signal.correlate(
                            mono_raw_audio_norm, 
                            mono_clap_norm, 
                            mode="full"
                        )
    correlation = correlation / np.max(np.abs(correlation))

    # Visualization
    lag = np.arange(-len(mono_clap_norm) + 1, len(mono_raw_audio_norm))
    
    if save_viz_cor_path:
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
        if 0 <= actual_position < len(mono_raw_audio_norm):
            clapperboard_position_samples.append(actual_position)
            correlation_values.append(correlation[peak_idx])

    # Convert to seconds
    clapperboard_positions_seconds = [i / sr_raw_audio for i in clapperboard_position_samples]
    clapperboard_positions_ms = [int(pos / sr_raw_audio * 1000) for pos in clapperboard_position_samples]

    result = {
        "audio_file": audio_path,
        "sample_rate": sr_raw_audio,
        "audio_duration_ms": int(len(mono_raw_audio_norm) / sr_raw_audio * 1000),
        "clapperboard_duration_ms": int(len(mono_clap_norm) / sr_raw_audio * 1000),
        "clapperboard_positions": [
            {
                "time_sec": round(pos_sec, 3),
                "time_ms": pos_ms,
                "correlation_score": round(corr, 4)
            }
            for pos_sec, pos_ms, corr in zip(clapperboard_positions_seconds, clapperboard_positions_ms, correlation_values)
        ]
    }

    # Create output folder for json and save json
    if save_json and output_json and file_name:
        uFile.create_dir(output_json, config)
        uFile.save_json(result, output_json, file_name, config)

    return result


def split_tasks_first_pass(
                            audio_file:str,
                            result_splited_position:dict,
                            output_base_dir:str,
                            participant_name:str,
                            config:DictConfig,
                            remove_clapperboard:bool=True,
                            clapperboard_buffer_ms:int=500,
                            save_json:bool=True,
                            task_names:list=None,
                            ) -> list:
    """ Split audio to each task
    """


    # # Load json file
    # data = uFile.load_json(splited_json_path)

    # Load audio
    _, raw_audio = uAudio.load_audio_pydub(audio_file, mono=False)
    audio_length_ms = len(raw_audio)


    claps_ms = [pos["time_ms"] for pos in result_splited_position["clapperboard_positions"]]
    clap_dur_ms = result_splited_position["clapperboard_duration_ms"]

    segments = []
    
    for i in range(len(claps_ms) + 1):
        if i == 0:
            start = 0
            end = claps_ms[0] if remove_clapperboard else claps_ms[0] + clap_dur_ms
        elif i < len(claps_ms):
            start = claps_ms[i - 1] + clap_dur_ms + clapperboard_buffer_ms
            end = claps_ms[i] if remove_clapperboard else claps_ms[i] + clap_dur_ms
        else:
            start = claps_ms[-1] + clap_dur_ms + clapperboard_buffer_ms
            end = audio_length_ms

        label = task_names[i] if i < len(task_names) else f"extra_segment_{i}"
        
        segments.append({
            "label": label,
            "start_ms": int(start),
            "end_ms": int(end)
        })

    for seg in tqdm(segments, desc=f"Exporting Tasks for {participant_name}"):
        task_dir = os.path.join(output_base_dir, seg["label"])
        os.makedirs(task_dir, exist_ok=True)
        
        output_path = os.path.join(task_dir, f"{participant_name}.wav")
        raw_audio[seg["start_ms"]:seg["end_ms"]].export(output_path, format="wav")

    if save_json:
        uFile.save_json(segments, output_base_dir, f"{participant_name}_task_segments", config)

    return segments

def split_onset_second_pass(
    task_audio_file: str,
    result_splited_position: dict,
    participant_name: str,
    remove_clapperboard: bool = True
):
    """Splits a task audio file into -setup.wav and -participant.wav
    """
    _, raw_audio = uAudio.load_audio_pydub(task_audio_file, mono=False)
    
    if not result_splited_position["clapperboard_positions"]:
        print(f"Skipping {task_audio_file} - No participant onset clap found.")
        return

    clap_ms = result_splited_position["clapperboard_positions"][0]["time_ms"]
    clap_dur_ms = result_splited_position["clapperboard_duration_ms"]
    task_dir = os.path.dirname(task_audio_file)

    # 1. Setup Phase
    setup_audio = raw_audio[0:clap_ms]
    setup_path = os.path.join(task_dir, f"{participant_name}-setup.wav")
    setup_audio.export(setup_path, format="wav")

    # 2. Participant Phase
    part_start = clap_ms if remove_clapperboard else clap_ms + clap_dur_ms
    participant_audio = raw_audio[part_start:len(raw_audio)]
    participant_path = os.path.join(task_dir, f"{participant_name}-participant.wav")
    participant_audio.export(participant_path, format="wav")



def run_split_audio_file(config: DictConfig, audio_path:str) -> dict:
    """
    """
    task_clap_path = config.preprocessing.task_clapperboard_path
    participant_clap_path = config.preprocessing.participant_clapperboard_path
    output_base_dir = config.preprocessing.output_audio_segment_path
    task_names = config.preprocessing.task_names

        
    participant_name = os.path.basename(audio_path).split('.')[0]

    # --- STAGE 1: Splitting Audio by Task ---
    task_result = clapperboard_detection(
        audio_path=audio_path,
        config=config,
        threshold=config.preprocessing.splited_threshold,
        min_distance_sec=config.preprocessing.min_distance_sec,
        save_json=config.preprocessing.save_json_splited_position,
        output_json=config.preprocessing.output_json_splited_position_path,
        file_name=f"{participant_name}_task_claps",
        save_viz_cor_path=config.preprocessing.output_img_correlation,
        clapperboard_path=task_clap_path
    )

    task_segments = split_tasks_first_pass(
        audio_file=audio_path,
        result_splited_position=task_result,
        output_base_dir=output_base_dir,
        participant_name=participant_name,
        task_names=task_names,
        config=config,
        clapperboard_buffer_ms=config.preprocessing.clapperboard_buffer_ms,
    )


    for seg in task_segments:
                if seg["label"] in ["intro", "outro"] or seg["label"].startswith("extra"):
                    continue # Skip non-task folders
                    
                task_audio_filepath = os.path.join(output_base_dir, seg["label"], f"{participant_name}.wav")
                
                if os.path.exists(task_audio_filepath):
                    part_result = clapperboard_detection(
                        audio_path=task_audio_filepath,
                        config=config,
                        threshold=config.preprocessing.splited_threshold,
                        min_distance_sec=0.5, # Shorter distance since we only expect 1 clap here
                        save_json=False,
                        clapperboard_path=participant_clap_path
                    )
                    
                    split_onset_second_pass(
                        task_audio_file=task_audio_filepath,
                        result_splited_position=part_result,
                        participant_name=participant_name,
                    )