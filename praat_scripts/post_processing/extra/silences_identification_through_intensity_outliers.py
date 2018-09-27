# Imports
import os
import subprocess
import collections
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from operator import itemgetter

# Global variables
EXT = ".wav"
EXT_SIZE = len(EXT)
SILENCES_FROM_INTENSITY = "_silences_from_intensity.csv"
# Change in case OS is not MacOS
PRAAT = "/Applications/Praat.app/Contents/MacOS/Praat"
RUN_OPTIONS = "--run"
# Add full path to praat script
PRAAT_SCRIPT = "silences_info_extraction.praat"

DTYPE = {"complete_event_file": str,
         "segmented_event_file": str,
         "segment_boundaries_df": str,
         "parameters_file": str,
         "pitch_file": str,
         "pitch_tier_file": str,
         "point_process_file": str,
         "intensity_file": str,
         "intensity_tier_file": str,
         "voice_report_file": str,
         "clean_intensity_tier_file": str,
         "clean_pitch_tier_file": str}
DTYPE_PARAM = {"dx_pitch": float,
               "dx_intensity": float,
               "window_size_vr": float,
               "window_shift_vr": float}
DTYPE_INTENSITY_TIER = {"Time": float, "Intensity": float}


def gen_file_name(audio_fn, dest_path):
    csil = dest_path + os.path.basename.split(audio_fn)[:-EXT_SIZE] + SILENCES_FROM_INTENSITY
    return csil


def get_time_step(param_df_fn):
    param_df = pd.read_csv(param_df_fn, dtype=DTYPE_PARAM)
    ts = param_df["dx_intensity"].values[0]
    return ts


def extract_silences_as_intensity_outliers(audio_fn, silences_df_fn, time_step, intensity_tier_df_fn, praat_parameters):
    # TODO controllare che i nomi dei nuovi file vengano inseriti correttamente
    subprocess.call([PRAAT, RUN_OPTIONS, PRAAT_SCRIPT, audio_fn,
                     intensity_tier_df_fn,
                     silences_df_fn,
                     str(praat_parameters.silence_threshold_db),
                     str(praat_parameters.minimum_silent_interval_duration),
                     str(praat_parameters.minimum_sounding_interval_duration),
                     str(time_step)])
    return 1


def silences_identification_through_intensity_serial(audio_info_df, audio_info_df_fn, audio_param_it_list, praat_parameters, dest_path, new_column):
    audio_sil_ws_it_list = list(map(lambda x: list(x[0], gen_file_name(x[0], dest_path), get_time_step(x[1]), x[2]), audio_param_it_list))
    for audio_fn, sil_df_fn, win_size, it_df_fn in audio_sil_ws_it_list:
        extract_silences_as_intensity_outliers(audio_fn, sil_df_fn, win_size, it_df_fn, praat_parameters)
    sil_df_fn_list = [(x, y) for x, y, z, t in audio_sil_ws_it_list]
    sil_df = pd.DataFrame(data=sil_df_fn_list, columns=new_column)
    audio_info_df = audio_info_df.join(sil_df)
    audio_info_df.to_csv(audio_info_df_fn, index=False)


def silences_identification_through_intensity_parallel(audio_info_df, audio_info_df_fn, audio_param_it_list, praat_parameters, dest_path, new_column, n):
    audio_sil_ws_it_list = list(map(lambda x: list(x[0], gen_file_name(x[0], dest_path), get_time_step(x[1]), x[2]), audio_param_it_list))
    Parallel(n_jobs=n, backend="threading")(delayed(extract_silences_as_intensity_outliers)(x, y, z, t, praat_parameters) for x, y, z, t in audio_sil_ws_it_list)
    sil_df_fn_list = [(x, y) for x, y, z, t in audio_sil_ws_it_list]
    sil_df = pd.DataFrame(data=sil_df_fn_list, columns=new_column)
    audio_info_df = audio_info_df.join(sil_df)
    audio_info_df.to_csv(audio_info_df_fn, index=False)


def identify_silences(audio_info_df_fn, praat_parameters, dest_path, parallel=False, n_jobs=-1):
    df = pd.read_csv(audio_info_df_fn, dtype=DTYPE)
    audio_param_it_list = df[["complete_event_file", "parameters_file", "clean_intensity_tier_file"]].values
    new_column = ["complete_event_file", "silences_from_intensity_file"]
    if parallel:
        silences_identification_through_intensity_parallel(df, audio_info_df_fn, audio_param_it_list, praat_parameters, dest_path, new_column, n_jobs)
    else:
        silences_identification_through_intensity_serial(df, audio_info_df_fn, audio_param_it_list, praat_parameters, dest_path, new_column)
