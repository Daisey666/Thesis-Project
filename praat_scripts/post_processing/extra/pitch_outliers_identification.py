# Imports
import os
import pandas as pd
import numpy as np
from joblib import Parallel, delayed

# Global variables
TAB_EXT = ".csv"
TAB_EXT_SIZE = len(TAB_EXT)
CLEAN_PITCH_TIER = "_clean.csv"

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
         "silences_from_intensity_file": str,
         "clean_intensity_tier": str}
DTYPE_PITCH_TIER = {"Time": float, "F0": float}

# TODO check if outliers must be put to zero or deleted
# TODO check if outliers are outside percentile or outside distribution

def gen_file_name(pitch_tier_fn, dest_path):
    cpt = dest_path + os.path.basename.split(pitch_tier_fn)[:-TAB_EXT_SIZE] + CLEAN_PITCH_TIER
    return cpt


def delete_pitch_ouliers_from_complete_event(pitch_tier_fn, clean_pitch_tier_fn, n_sigma):
    pt_tmp_df = pd.read_csv(pitch_tier_fn, dtype=DTYPE_PITCH_TIER)
    pitch_values = pt_tmp_df["F0"].values
    mean = np.mean(pitch_values, axis=0)
    sd = np.std(pitch_values, axis=0)
    query = "F0 >= " + str(mean) + " - " + str(n_sigma * sd) + " and F0 <= " + str(mean) + " + " + str(n_sigma * sd)
    clean_pitch_tier_df = pt_tmp_df.query(query)
    clean_pitch_tier_df.to_csv(clean_pitch_tier_fn, index=False)


def pitch_outliers_identification_serial(audio_info_df, audio_info_df_fn, pitch_tier_list, dest_path, new_column, n_sigma):
    pt_clean_pt_list = list(map(lambda x: list(x, gen_file_name(x, dest_path)), pitch_tier_list))
    for pt_fn, clean_pt_fn in pt_clean_pt_list:
        delete_pitch_ouliers_from_complete_event(pt_fn, clean_pt_fn, n_sigma)
    clean_pt_df = pd.DataFrame(data=pt_clean_pt_list, columns=new_column)
    audio_info_df = audio_info_df.join(clean_pt_df)
    audio_info_df.to_csv(audio_info_df_fn, index=False)


def pitch_outliers_identification_parallel(audio_info_df, audio_info_df_fn, pitch_tier_list, dest_path, new_column, n_sigma, n):
    pt_clean_pt_list = list(map(lambda x: list(x, gen_file_name(x, dest_path)), pitch_tier_list))
    Parallel(n_jobs=n, backend="threading")(delayed(delete_pitch_ouliers_from_complete_event)(x, y, n_sigma) for x, y in pt_clean_pt_list)
    clean_pt_df = pd.DataFrame(data=pt_clean_pt_list, columns=new_column)
    audio_info_df = audio_info_df.join(clean_pt_df)
    audio_info_df.to_csv(audio_info_df_fn, index=False)


def identify_ouliers(audio_info_df_fn, dest_path, n_sigma=2, parallel=False, n_jobs=-1):
    df = pd.read_csv(audio_info_df_fn, dtype=DTYPE)
    pt_list = df["pitch_tier_file"].values
    new_column = ["pitch_tier_file", "clean_pitch_tier_file"]
    if parallel:
        pitch_outliers_identification_parallel(df, audio_info_df_fn, pt_list, dest_path, new_column, n_sigma, n_jobs)
    else:
        pitch_outliers_identification_serial(df, audio_info_df_fn, pt_list, dest_path, new_column, n_sigma)
