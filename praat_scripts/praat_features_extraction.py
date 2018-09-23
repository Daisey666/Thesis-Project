# Imports
import os
import subprocess
import collections
import pandas as pd
from joblib import Parallel, delayed

# Global variables
EXT = ".wav"
EXT_SIZE = len(EXT)
SAMPLE_RATE = 44100
# Change in case OS is not MacOS
PRAAT = "/Applications/Praat.app/Contents/MacOS/Praat"
RUN_OPTIONS = "--run"
# Add full path to praat script
PRAAT_SCRIPT = "audio_info_extraction.praat"
GLOBAL_REPORT = "_global_report"
PITCH = "_pitch.Pitch"
PITCH_TIER = "_pitch_tier.csv"
POINT_PROCESS = "_point_process.PointProcess"
INTENSITY = "_intensity.Intensity"
INTENSITY_TIER = "_intensity_tier.csv"
VOICE_REPORT = "_voice_report.csv"


def extract_info_from_complete_event_segments(df_fn, praat_parameters, dest_paths):
    # TODO controllare che i nomi dei nuovi file vengano inseriti correttamente
    segments_df = pd.read_csv(df_fn, dtype={"start_time": int, "end_time": int, "class": str, "audio_segment_file":str})
    info_file_names = []
    new_columns = ["pitch_file", "pitch_tier_file", "point_process_file", "intensity_file", "intensity_tier_file", "voice_report_file"]
    for wav_file in segments_df["audio_segment_file"].values:
        pitch_file = dest_paths.pitch_path + os.path.basename.split(wav_file)[:-EXT_SIZE] + PITCH
        pitch_tier_file = dest_paths.pitch_tier_path + os.path.basename.split(wav_file)[:-EXT_SIZE] + PITCH_TIER
        point_process_file = dest_paths.point_process_path + os.path.basename.split(wav_file)[:-EXT_SIZE] + POINT_PROCESS
        intensity_file = dest_paths.intensity_path + os.path.basename.split(wav_file)[:-EXT_SIZE] + INTENSITY
        intensity_tier_file = dest_paths.intensity_tier_path + os.path.basename.split(wav_file)[:-EXT_SIZE] + INTENSITY_TIER
        voice_report_file = dest_paths.voice_report_path + os.path.basename.split(wav_file)[:-EXT_SIZE] + VOICE_REPORT
        info_file_names.append((pitch_file, pitch_tier_file, point_process_file, intensity_file, intensity_tier_file, voice_report_file))
        subprocess.call([PRAAT, RUN_OPTIONS, PRAAT_SCRIPT, wav_file, pitch_file, pitch_tier_file, point_process_file, intensity_file, intensity_tier_file, voice_report_file, str(praat_parameters.pitch_min), str(praat_parameters.pitch_max), str(praat_parameters.max_period_factor), str(praat_parameters.max_amplitude_factor), str(praat_parameters.silence_threshold), str(praat_parameters.voicing_thresholding), praat_parameters.subtract_mean])
    ext_info_cols = pd.DataFrame(data=info_file_names, columns=new_columns)
    segments_df = segments_df.join(ext_info_cols)
    segments_df.to_csv(df_fn, index=False)


def audio_info_extraction_serial(audio_df_list, praat_parameters, dest_paths):
    for df_fn in audio_df_list:
        extract_info_from_complete_event_segments(df_fn, praat_parameters, dest_paths)


def audio_info_extraction_parallel(audio_df_list, praat_parameters, dest_paths, n):
    Parallel(n_jobs=n, backend="threading")(delayed(extract_info_from_complete_event_segments)(x, praat_parameters, dest_paths) for x in audio_df_list)


def extract_audio_informations(audio_info_df_fn, praat_parameters, dest_paths, parallel=False, n_jobs=-1):
    # Note that praat_parameters and dest_paths must be passed as named tuples
    df = pd.read_csv(audio_info_df_fn, dtype={"complete_event_file": str, "segmented_event_file": str, "segment_boundaries_df": str})
    audio_df_list = df["segment_boundaries_df"].values
    if parallel:
        audio_info_extraction_parallel(audio_df_list, praat_parameters, dest_paths, n_jobs)
    else:
        audio_info_extraction_serial(audio_df_list, praat_parameters, dest_paths)
