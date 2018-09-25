# Imports
import os
import pandas as pd
import numpy as np
from joblib import Parallel, delayed

# Global variables
TAB_EXT = ".csv"
TAB_EXT_SIZE = len(TAB_EXT)
CLEAN_PITCH_TIER = "_clean.csv"

# TODO check if outliers must be put to zero or deleted
# TODO check if outliers are outside percentile or outside distribution


def delete_pitch_ouliers_from_complete_event(df_fn, dest_path, n_sigma):
    segments_df = pd.read_csv(df_fn, dtype={"start_time": int, "end_time": int,
                                            "class": str,
                                            "audio_segment_file": str,
                                            "parameters_file": str,
                                            "pitch_file": str,
                                            "pitch_tier_file": str,
                                            "point_process_file": str,
                                            "intensity_file": str,
                                            "intensity_tier_file": str,
                                            "voice_report_file": str})
    pitch_tier_list = segments_df["pitch_tier_file"].values
    pitch_values = np.array([])
    for pitch_tier in pitch_tier_list:
        tmp_df = pd.read_csv(pitch_tier, dtype={"Time": float, "F0": float})
        pitch_values = np.concatenate((pitch_values, tmp_df["F0"].values))
    mean = np.mean(pitch_values, axis=0)
    sd = np.std(pitch_values, axis=0)
    clean_pitch_tiers = []
    for pitch_tier in pitch_tier_list:
        tmp_df = pd.read_csv(pitch_tier, dtype={"Time": float, "F0": float})
        clean_pitch_tier_df_fn = dest_path + os.path.basename.split(pitch_tier)[:-TAB_EXT_SIZE] + CLEAN_PITCH_TIER
        clean_pitch_tiers.append((pitch_tier, clean_pitch_tier_df_fn))
        query = "F0 >= " + str(mean) + " - " + str(n_sigma * sd) + " and F0 <= " + str(mean) + " + " + str(n_sigma * sd)
        clean_pitch_tier_df = tmp_df.query(query)
        clean_pitch_tier_df.to_csv(clean_pitch_tier_df_fn, index=False)
    clean_pitch_tiers_col = pd.DataFrame(data=clean_pitch_tiers, columns=["pitch_tier_file", "clean_pitch_tier_file"])
    segments_df = segments_df.join(clean_pitch_tiers_col)
    segments_df.to_csv(df_fn, index=False)


def pitch_outliers_identification_serial(audio_df_list, dest_path, n_sigma):
    for df_fn in audio_df_list:
        delete_pitch_ouliers_from_complete_event(df_fn, dest_path, n_sigma)


def pitch_outliers_identification_parallel(audio_df_list, dest_path, n_sigma, n):
    Parallel(n_jobs=n, backend="threading")(delayed(delete_pitch_ouliers_from_complete_event)(x, dest_path, n_sigma) for x in audio_df_list)


def identify_ouliers(audio_info_df_fn, dest_path, n_sigma=2, parallel=False, n_jobs=-1):
    df = pd.read_csv(audio_info_df_fn, dtype={"complete_event_file": str, "segmented_event_file": str, "segment_boundaries_df": str})
    audio_df_list = df["segment_boundaries_df"].values
    if parallel:
        pitch_outliers_identification_parallel(audio_df_list, dest_path, n_sigma, n_jobs)
    else:
        pitch_outliers_identification_serial(audio_df_list, dest_path, n_sigma)
