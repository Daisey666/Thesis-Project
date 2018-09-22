# Imports
import pandas as pd
from scipy import signal
from scipy.io import wavfile


def split_segment(start, end, n_split):


def get_segments():




def extract_non_relevant_segments_positions(audio_info_df_fn, parallel=False, n_jobs=-1):
    df = pd.read_csv(audio_info_df_fn, dtype={"complete_event_file": str, "segmented_event_file": str, "segment_boundaries_df": str})
    audio_df_list = [[x[1]["complete_event_file"], x[1]["segment_boundaries_df"]] for x in df.iterrows()]
    if parallel:
        extract_time_boundaries_parallel(audio_df_list, n_jobs)
    else:
        extract_time_boundaries_serial(audio_df_list)
