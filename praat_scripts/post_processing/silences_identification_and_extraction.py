# Imports
import os
import pandas as pd
import numpy as np
from joblib import Parallel, delayed

# Global variables
TAB_EXT = ".csv"
TAB_EXT_SIZE = len(TAB_EXT)
CLEAN_INTENSITY_TIER = "_clean.csv"

# TODO check what has to be done with intensity outliers
# TODO add silences intervals identification


def delete_pitch_ouliers_from_complete_event(df_fn, dest_path, n_sigma):
    segments_df = pd.read_csv(df_fn, dtype={"start_time": int, "end_time": int, "class": str, "audio_segment_file":str, "pitch_file": str, "pitch_tier_file": str, "point_process_file": str, "intensity_file": str, "intensity_tier_file": str, "voice_report_file": str})
    intensity_tier_list = segments_df["intensity_tier_file"].values
    intensity_values = np.array([])
    for intensity_tier in intensity_tier_list:
        tmp_df = pd.read_csv(intensity_tier, dtype={"Time (s)": float, "Intensity (dB)": float})
        tmp_df = tmp_df.rename(columns={"Time (s)": "Time", "Intensity (dB)": "Intensity"})
        intensity_values = np.concatenate((intensity_values, tmp_df["Intensity"].values))
        tmp_df.to_csv(intensity_tier, index=False)
    mean = np.mean(intensity_values, axis=0)
    sd = np.std(intensity_values, axis=0)
    clean_intensity_tiers_list = []
    for intensity_tier in intensity_tier_list:
        tmp_df = pd.read_csv(intensity_tier, dtype={"Time": float, "Intensity": float})
        clean_intensity_tier_df_fn = dest_path + os.path.basename.split(intensity_tier)[:-TAB_EXT_SIZE] + CLEAN_INTENSITY_TIER
        clean_intensity_tiers_list.append(clean_intensity_tier_df_fn)
        query = "Intensity >= " + str(mean) + " - " + str(n_sigma * sd) + " and Intensity <= " + str(mean) + " + " + str(n_sigma * sd)
        clean_pitch_tier_df = tmp_df.query(query)
        clean_pitch_tier_df.to_csv(clean_intensity_tier_df_fn, index=False)
    clean_intensity_tiers_col = pd.DataFrame(data=clean_intensity_tiers_list, columns=["clean_intensity_tier"])
    segments_df = segments_df.join(clean_intensity_tiers_col)
    segments_df.to_csv(df_fn, index=False)


def intensity_outliers_identification_serial(audio_df_list, dest_path, n_sigma):
    for df_fn in audio_df_list:
        delete_pitch_ouliers_from_complete_event(df_fn, dest_path, n_sigma)


def intensity_outliers_identification_parallel(audio_df_list, dest_path, n_sigma, n):
    Parallel(n_jobs=n, backend="threading")(delayed(delete_pitch_ouliers_from_complete_event)(x, dest_path, n_sigma) for x in audio_df_list)


def identify_outliers(audio_info_df_fn, dest_path, n_sigma=2, parallel=False, n_jobs=-1):
    df = pd.read_csv(audio_info_df_fn, dtype={"complete_event_file": str, "segmented_event_file": str, "segment_boundaries_df": str})
    audio_df_list = df["segment_boundaries_df"].values
    if parallel:
        intensity_outliers_identification_parallel(audio_df_list, dest_path, n_sigma, n_jobs)
    else:
        intensity_outliers_identification_serial(audio_df_list, dest_path, n_sigma)
