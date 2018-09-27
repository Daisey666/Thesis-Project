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


def extract_silences_as_intensity_outliers(silences_df_fn, window_size, intensity_tier_df_fn, n_sigma):
    intensity_tier_df = pd.read_csv(intensity_tier_df_fn, dtype=DTYPE_INTENSITY_TIER)
    intensity_values = intensity_tier_df["Intensity"]
    mean = np.mean(intensity_values, axis=0)
    sd = np.std(intensity_values, axis=0)
    query = "Intensity < " + str(mean) + " - " + str(n_sigma * sd) + " or Intensity > " + str(mean) + " + " + str(n_sigma * sd)
    intensity_tier_df = intensity_tier_df.query(query)
    ts = intensity_tier_df["Time"].values
    tmp_intervals = list(map(lambda x: list((x - (window_size / 2), x + (window_size + 2))), ts))
    intervals = sorted(tmp_intervals, key=itemgetter(0))
    consecutives = []
    st = intervals[0][0]
    for i in range(len(intervals)-1):
        if intervals[i][1] != intervals[i+1][0]:
            consecutives.append((st, intervals[i][1]))
            st = intervals[i+1][0]
    consecutives.append((st, intervals[-1][1]))
    consecutives[0][0] = 0
    #TODO check how to set end time efficiently
    segments = [(x, y, "silence") for x, y in consecutives]
    start = [x[1] for x in consecutives[:-1]]
    end = [x[0] for x in consecutives[1:]]
    segments += [(x, y, "speech") for x, y in list(zip(start, end))]
    segments = sorted(segments, key=itemgetter(0))
    silences_df = pd.DataFrame(data=segments, columns=["start_time", "end_time", "silence"])
    silences_df.to_csv(silences_df_fn, index=False)


def silences_identification_through_intensity_serial(audio_info_df, audio_info_df_fn, audio_param_it_list, dest_path, new_column, n_sigma):
    audio_sil_ws_it_list = list(map(lambda x: list(x[0], gen_file_name(x[0], dest_path), get_win_size(x[1]), x[2]), audio_param_it_list))
    for tmp, sil_df_fn, win_size, it_df_fn in audio_sil_ws_it_list:
        extract_silences_as_intensity_outliers(sil_df_fn, win_size, it_df_fn, n_sigma)
    sil_df_fn_list = [(x, y) for x, y, z, t in audio_sil_ws_it_list]
    sil_df = pd.DataFrame(data=sil_df_fn_list, columns=new_column)
    audio_info_df = audio_info_df.join(sil_df)
    audio_info_df.to_csv(audio_info_df_fn, index=False)


def silences_identification_through_intensity_parallel(audio_info_df, audio_info_df_fn, audio_param_it_list, dest_path, new_column, n_sigma, n):
    audio_sil_ws_it_list = list(map(lambda x: list(x[0], gen_file_name(x[0], dest_path), get_win_size(x[1]), x[2]), audio_param_it_list))
    Parallel(n_jobs=n, backend="threading")(delayed(extract_silences_as_intensity_outliers)(y, z, t, n_sigma) for x, y, z, t in audio_sil_ws_it_list)
    sil_df_fn_list = [(x, y) for x, y, z, t in audio_sil_ws_it_list]
    sil_df = pd.DataFrame(data=sil_df_fn_list, columns=new_column)
    audio_info_df = audio_info_df.join(sil_df)
    audio_info_df.to_csv(audio_info_df_fn, index=False)


def identify_silences(audio_info_df_fn, dest_path, n_sigma=2, parallel=False, n_jobs=-1):
    df = pd.read_csv(audio_info_df_fn, dtype=DTYPE)
    audio_param_it_list = df[["complete_event_file", "parameters_file", "clean_intensity_tier_file"]].values
    new_column = ["complete_event_file", "silences_from_intensity_file"]
    if parallel:
        silences_identification_through_intensity_parallel(df, audio_info_df_fn, audio_param_it_list, dest_path, new_column, n_sigma, n_jobs)
    else:
        silences_identification_through_intensity_serial(df, audio_info_df_fn, audio_param_it_list, dest_path, new_column, n_sigma)
