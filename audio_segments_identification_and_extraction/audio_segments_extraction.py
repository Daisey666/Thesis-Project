# Imports
import os
import pandas as pd
from scipy.io import wavfile
from joblib import Parallel, delayed

# Global variables
EXT = ".wav"
EXT_SIZE = len(EXT)
Z_FILL_PARAM = 5


def extract_all_audio_segments_from_single_file(audio_fn, df_fn, dest_path):
    # TODO check if order is preserved
    sample_rate, signal = wavfile.read(audio_fn)
    segments_df = pd.read_csv(df_fn, dtype={"start_time": int, "end_time": int, "class": str})
    file_names = []
    counter = 0
    for row in segments_df.iterrows():
        fn = dest_path + os.path.basename.split(audio_fn)[:-EXT_SIZE] + str(counter).zfill(Z_FILL_PARAM) + EXT
        file_names += fn
        start = row[1]["start_time"]
        end = row[1]["end_time"]
        wavfile.write(fn, sample_rate, signal[start, end])
    seg_path_col = pd.DataFrame(data=file_names, columns=["segment_file"])
    segments_df.join(seg_path_col)
    segments_df.to_csv(df_fn, index=False)

def audio_segments_extraction_serial(audio_df_list, dest_path):
    for audio_fn, df_fn in audio_df_list:
        extract_all_audio_segments_from_single_file(audio_fn, df_fn, dest_path)


def audio_segments_extraction_parallel(audio_df_list, dest_path, n):
    Parallel(n_jobs=n, backend="threading")(delayed(extract_all_audio_segments_from_single_file)(x, y, dest_path) for x, y in audio_df_list)


def extract_audio_segments(audio_info_df_fn, dest_path, parallel=False, n_jobs=-1):
    # TODO find better solution for destination paths
    df = pd.read_csv(audio_info_df_fn, dtype={"complete_event_file": str, "segmented_event_file": str, "segment_boundaries_df": str})
    audio_df_list = [[x[1]["complete_event_file"], x[1]["segment_boundaries_df"]] for x in df.iterrows()]
    if parallel:
        audio_segments_extraction_parallel(audio_df_list, dest_path, n_jobs)
    else:
        audio_segments_extraction_serial(audio_df_list, dest_path)
