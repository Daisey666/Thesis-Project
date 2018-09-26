# Imports
import os
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from operator import itemgetter

# Global variables
EXT = ".wav"
EXT_SIZE = len(EXT)
SILENCES_FROM_INTENSITY = "_silences_from_intensity.csv"


def extract_silences_as_intensity_outliers(df_fn, dest_path, win_size,
                                           n_sigma):
    segments_df = pd.read_csv(df_fn, dtype={"start_time": int, "end_time": int,
                                            "class": str,
                                            "audio_segment_file": str,
                                            "parameters_file": str,
                                            "pitch_file": str,
                                            "pitch_tier_file": str,
                                            "point_process_file": str,
                                            "intensity_file": str,
                                            "intensity_tier_file": str,
                                            "voice_report_file": str,
                                            "clean_pitch_tier_file": str,
                                            "clean_intensity_tier": str})
    audio_intensity_fn_list = segments_df[["audio_segment_file",
                                           "intensity_tier_file"]].values
    intensity_values = np.array([])
    for tmp, intensity_tier in audio_intensity_fn_list:
        tmp_df = pd.read_csv(intensity_tier, dtype={"Time": float,
                                                    "Intensity": float})
        intensity_values = np.concatenate((intensity_values,
                                           tmp_df["Intensity"].values))
    mean_i = np.mean(intensity_values, axis=0)
    sd_i = np.std(intensity_values, axis=0)
    silences = []
    for audio_fn, intensity_tier in audio_intensity_fn_list:
        tmp_df = pd.read_csv(intensity_tier, dtype={"Time": float,
                                                    "Intensity": float})
        silences_from_intensity_df_fn = dest_path + os.path.basename.split(audio_fn)[:-EXT_SIZE] + SILENCES_FROM_INTENSITY
        silences.append((audio_fn, silences_from_intensity_df_fn))
        tmp_intervals = list(map(lambda x: list((x[0] - (win_size / 2), x[0] + (win_size / 2), x[1])), tmp_df[["Time", "Intensity"]].values)
        intervals = list(filter(lambda x: (x[2] < mean_i - (n_sigma * sd_i)) or (x[2] > mean_i + (n_sigma * sd_i)), tmp_intervals))
        intervals = sorted(tmp_data, key=itemgetter(0))
        consecutives = []
        st = intervals[0][0]
        for i in range(len(intervals)-1):
            if intervals[i][1] != intervals[i+1][0]:
                consecutives.append((st, intervals[i][1]))
                st = intervals[i+1][0]
        consecutives.append((st, intervals[-1][1]))
        segments = [(x, y, "silence") for x, y, z in consecutives]
        sample_rate, signal = wavfile.read(audio_fn)
        len = float(len(signal) / sample_rate)
        start = [x[1] for x in intervals[:-1]]
        end = [x[0] for x in intervals[1:]]
        segments += [(x, y, "speech") for x, y in list(zip(start, end))]
        segments = sorted(segments, key=itemgetter(0))
        df = pd.DataFrame(data=segments, columns=["start_time", "end_time", "silence"])
        df.to_csv(silences_from_intensity_df_fn, index=False)
    silences_from_intensity_col = pd.DataFrame(data=clean_intensity_tiers, columns=["audio_segment_file", "silences_from_intensity"])
    segments_df = segments_df.join(silences_from_intensity_col)
    segments_df.to_csv(df_fn, index=False)


def silences_identification_through_intensity_serial(audio_df_list, dest_path, win_size, n_sigma):
    for df_fn in audio_df_list:
        extract_silences_as_intensity_outliers(df_fn, dest_path, win_size, n_sigma)


def silences_identification_through_intensity_parallel(audio_df_list, dest_path, win_size, n_sigma, n):
    Parallel(n_jobs=n, backend="threading")(delayed(extract_silences_as_intensity_outliers)(x, dest_path, win_size, n_sigma) for x in audio_df_list)


def identify_silences(audio_info_df_fn, dest_path, win_size, n_sigma=2, parallel=False, n_jobs=-1):
    df = pd.read_csv(audio_info_df_fn, dtype={"complete_event_file": str, "segmented_event_file": str, "segment_boundaries_df": str})
    audio_df_list = df["segment_boundaries_df"].values
    if parallel:
        silences_identification_through_intensity_parallel(audio_df_list, dest_path, win_size, n_sigma, n_jobs)
    else:
        silences_identification_through_intensity_serial(audio_df_list, dest_path, win_size, n_sigma)
