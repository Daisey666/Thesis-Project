# Imports
import os
import pandas as pd
import numpy as np
from joblib import Parallel, delayed

# Global variables
TAB_EXT = ".csv"
TAB_EXT_SIZE = len(TAB_EXT)
CLEAN_VOICE_REPORT = "_clean.csv"

# TODO check what has to be done with intensity outliers
# TODO add silences intervals identification


def clean_voice_report(df_fn, dest_path, n_sigma):
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
                                            "clean_intensity_tier": str,
                                            "silences_from_intensity": str})
    audio_fn_vr_list = segments_df[["audio_segment_file", "voice_report_file"]].values
    clean_voice_reports = []
    for audio_fn, voice_report_fn in audio_fn_vr_list:
        tmp_df = pd.read_csv(voice_report_fn, dtype={"start_time": str, "end_time": str, "harmonicity": str, "jitter": str, "shimmer": str})
        clean_voice_report_df_fn = dest_path + os.path.basename.split(voice_report_fn)[:-TAB_EXT_SIZE] + CLEAN_VOICE_REPORT
        query = "harmonicity == --undefined-- or jitter == --undefined-- or shimmer == --undefined--"
        clean_voice_report_df = tmp_df.query(query)
        clean_voice_report_df.to_csv(clean_voice_report_df_fn, index=False)
        clean_voice_reports.append((audio_fn, clean_voice_report_df_fn))
    clean_voice_report_col = pd.DataFrame(data=clean_voice_reports, columns=["audio_segment_file", "clean_voice_report"])
    segments_df = segments_df.join(clean_voice_report_col)
    segments_df.to_csv(df_fn, index=False)


def undefined_entries_identification_serial(audio_df_list, dest_path, n_sigma):
    for df_fn in audio_df_list:
        clean_voice_report(df_fn, dest_path)


def undefined_entries_identification_parallel(audio_df_list, dest_path, n_sigma, n):
    Parallel(n_jobs=n, backend="threading")(delayed(clean_voice_report)(x, dest_path) for x in audio_df_list)


def identify_undefined_entries(audio_info_df_fn, dest_path, parallel=False, n_jobs=-1):
    df = pd.read_csv(audio_info_df_fn, dtype={"complete_event_file": str, "segmented_event_file": str, "segment_boundaries_df": str})
    audio_df_list = df["segment_boundaries_df"].values
    if parallel:
        undefined_entries_identification_parallel(audio_df_list, dest_path, n_jobs)
    else:
        undefined_entries_identification_serial(audio_df_list, dest_path)
