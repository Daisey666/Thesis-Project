# Imports
import os
import pandas as pd
import numpy as np
from joblib import Parallel, delayed

# Global variables
TAB_EXT = ".csv"
TAB_EXT_SIZE = len(TAB_EXT)
CLEAN_INTENSITY_TIER = "_clean.csv"

DTYPE = {"complete_event_file": str,
         "segmented_event_file": str,
         "segment_boundaries_df": str,
         "parameters_file": str,
         "pitch_file": str,
         "pitch_tier_file": str,
         "point_process_file": str,
         "intensity_file": str,
         "intensity_tier_file": str,
         "voice_report_file": str}
DTYPE_INTENSITY_TIER_OLD = {"Time (s)": float, "Intensity (dB)": float}

# TODO check what has to be done with intensity outliers
# TODO add silences intervals identification

def gen_file_name(intensity_tier_fn, dest_path):
    cit = dest_path + os.path.basename.split(intensity_tier_fn)[:-TAB_EXT_SIZE] + CLEAN_INTENSITY_TIER
    return cit


def delete_intensity_ouliers_from_complete_event(intensity_tier_fn, clean_intensity_tier_fn, n_sigma):
    it_tmp_df = pd.read_csv(intensity_tier_fn, dtype=DTYPE_INTENSITY_TIER_OLD)
    it_tmp_df = it_tmp_df.rename(columns={"Time (s)": "Time", "Intensity (dB)": "Intensity"})
    it_tmp_df.to_csv(intensity_tier_fn, index=False)
    intensity_values = it_tmp_df["Intensity"].values
    mean = np.mean(intensity_values, axis=0)
    sd = np.std(intensity_values, axis=0)
    query = "Intensity >= " + str(mean) + " - " + str(n_sigma * sd) + " and Intensity <= " + str(mean) + " + " + str(n_sigma * sd)
    clean_intensity_tier_df = it_tmp_df.query(query)
    clean_intensity_tier_df.to_csv(clean_intensity_tier_fn, index=False)


def intensity_outliers_identification_serial(audio_info_df, audio_info_df_fn, intensity_tier_list, dest_path, new_column, n_sigma):
    it_clean_it_list = list(map(lambda x: list(x, gen_file_name(x, dest_path)), intensity_tier_list))
    for it_fn, clean_it_fn in it_clean_it_list:
        delete_intensity_ouliers_from_complete_event(it_fn, clean_it_fn, n_sigma)
    clean_it_df = pd.DataFrame(data=it_clean_it_list, columns=new_column)
    audio_info_df = audio_info_df.join(clean_it_df)
    audio_info_df.to_csv(audio_info_df_fn, index=False)


def intensity_outliers_identification_parallel(audio_info_df, audio_info_df_fn, intensity_tier_list, dest_path, new_column, n_sigma, n):
    it_clean_it_list = list(map(lambda x: list(x, gen_file_name(x, dest_path)), intensity_tier_list))
    Parallel(n_jobs=n, backend="threading")(delayed(delete_intensity_ouliers_from_complete_event)(x, y, n_sigma) for x, y in it_clean_it_list)
    clean_it_df = pd.DataFrame(data=it_clean_it_list, columns=new_column)
    audio_info_df = audio_info_df.join(clean_it_df)
    audio_info_df.to_csv(audio_info_df_fn, index=False)


def identify_intensity_outliers(audio_info_df_fn, dest_path, n_sigma=2, parallel=False, n_jobs=-1):
    df = pd.read_csv(audio_info_df_fn, dtype=DTYPE)
    it_list = df["intensity_tier_file"].values
    new_column = ["intensity_tier_file", "clean_intensity_tier_file"]
    if parallel:
        intensity_outliers_identification_parallel(df, audio_info_df_fn, it_list, dest_path, new_column, n_sigma, n_jobs)
    else:
        intensity_outliers_identification_serial(df, audio_info_df_fn, it_list, dest_path, new_column, n_sigma)
