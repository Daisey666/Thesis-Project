# Imports
import os
import pandas as pd
import numpy as np
from joblib import Parallel, delayed

# Global variables
TAB_EXT = ".csv"
TAB_EXT_SIZE = len(TAB_EXT)
CLEAN_VOICE_REPORT = "_clean.csv"

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
         "clean_intensity_tier": str,
         "clean_pitch_tier_file": str,
         "silences_from_intensity_file": str}
DTYPE_VOICE_REPORT_OLD = {"start_time": float,
                          "end_time": float,
                          "harmonicity": str,
                          "jitter": str,
                          "shimmer": str}


def clean_voice_report_with_delete(voice_report_df_fn):
    voice_report_df = pd.read_csv(voice_report_df_fn, dtype=DTYPE_VOICE_REPORT_OLD)
    query = "harmonicity == --undefined-- or jitter == --undefined-- or shimmer == --undefined--"
    voice_report_df = voice_report_df.query(query)
    voice_report_df.to_csv(voice_report_df_fn, index=False)


def clean_voice_report(voice_report_df_fn):
    voice_report_df = pd.read_csv(voice_report_df_fn, dtype=DTYPE_VOICE_REPORT_OLD)
    voice_report_df = voice_report_df.replace("--undefined--", np.nan)
    voice_report_df.to_csv(voice_report_df_fn, index=False)


def undefined_entries_identification_serial(vr_df_fn_list, clean_fun):
    for vr_df_fn in vr_df_fn_list:
        clean_fun(vr_df_fn)


def undefined_entries_identification_parallel(vr_df_fn_list, clean_fun, n):
    Parallel(n_jobs=n, backend="threading")(delayed(clean_fun)(x) for x in vr_df_fn_list)


def identify_undefined_entries(audio_info_df_fn, with_delete=False, parallel=False, n_jobs=-1):
    df = pd.read_csv(audio_info_df_fn, dtype=DTYPE)
    vr_df_fn_list = df["voice_report_file"].values
    if with_delete:
        f = clean_voice_report_with_delete
    else:
        f = clean_voice_report
    if parallel:
        undefined_entries_identification_parallel(vr_df_fn_list, f, n_jobs)
    else:
        undefined_entries_identification_serial(vr_df_fn_list, f)
