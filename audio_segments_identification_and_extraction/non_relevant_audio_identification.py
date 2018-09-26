# Imports
import pandas as pd
import numpy as np
from scipy.io import wavfile
from joblib import Parallel, delayed


DTYPE = {"complete_event_file": str,
         "segmented_event_file": str,
         "segment_boundaries_df": str}
DTYPE_SEGMENT_BOUNDARIES = {"start_time": int,
                            "end_time": int,
                            "class": str}


def split_segment(start, end, n_split):
    splits = []
    len = (end - start) // n_split
    for i in range(n_split):
        if i != n_split - 1:
            splits.append((start + (i * len), (start + (i * len) + len), "non_relevant"))
        else:
            splits.append((start + (i * len), end, "non_relevant"))


def get_segments(start, end, s_list):
    # Evental splitting of a segment is heuristic
    avg_len = np.mean([x[1] - x[0] for x in s_list])
    start_points = [start] + [x[1] for x in s_list]
    end_points = [x[0] for x in s_list] + [end]
    segments = list(zip(start_points, end_points))
    non_relevant_segments = []
    for sp, ep in segments:
        if ep - sp < 2 * avg_len:
            non_relevant_segments.append((sp, ep, "non_relevant"))
        else:
            non_relevant_segments += split_segment(sp, ep, (ep - sp) // avg_len)
    return non_relevant_segments


def complete_time_boundaries_df(audio_fn, df_fn):
    sample_rate, signal = wavfile.read(audio_fn)
    start = 0
    end = len(signal)
    segments_df = pd.read_csv(df_fn, dtype=DTYPE_SEGMENT_BOUNDARIES)
    seg_s_e_list = segments_df[["start_time", "end_time"]].values
    non_relevant_segments = get_segments(start, end, seg_s_e_list)
    new_df = pd.DataFrame(data=non_relevant_segments, columns=["start_time", "end_time", "class"])
    segments_df = segments_df.append(new_df, ignore_index=True)
    segments_df = segments_df.sort_values("start_time")
    segments_df.to_csv(df_fn, index=False)


def extract_time_boundaries_serial(audio_df_list):
    for audio_fn, df_fn in audio_df_list:
        complete_time_boundaries_df(audio_fn, df_fn)


def extract_time_boundaries_parallel(audio_df_list, n):
    Parallel(n_jobs=n, backend="threading")(delayed(complete_time_boundaries_df)(x, y) for x, y in audio_df_list)


def extract_non_relevant_segments_positions(audio_info_df_fn, parallel=False, n_jobs=-1):
    df = pd.read_csv(audio_info_df_fn, dtype=DTYPE)
    audio_df_list = df[["complete_event_file", "segment_boundaries_df"]].values
    if parallel:
        extract_time_boundaries_parallel(audio_df_list, n_jobs)
    else:
        extract_time_boundaries_serial(audio_df_list)
