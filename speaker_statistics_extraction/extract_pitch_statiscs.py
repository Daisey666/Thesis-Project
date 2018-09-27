# Imports
import os
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# Global variables
EXT = ".wav"
EXT_SIZE = len(EXT)
PITCH_SPEAKER_STATS = "_pitch_speaker_stats.csv"

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
         "silences_from_intensity_file": str,
         "speech_info_file": str,
         "silences_info_file": str}
DTYPE_PITCH_TIER = {"Time": float,
                    "F0": float}
DTYPE_SPEECH_INFO = {"word": str,
                     "start_time": float,
                     "end_time": float,
                     "speaker_tag": float}


def gen_file_name(audio_fn, dest_path):
    pstatfn = dest_path + os.path.basename.split(audio_fn)[:-EXT_SIZE] + PITCH_SPEAKER_STATS
    return pstatfn


def extract_speaker_pitch_stats(pitch_tier_df_fn, speech_info_df_fn, pitch_stats_df_fn):
    pitch_tier_df = pd.read_csv(pitch_tier_df_fn, dtype=DTYPE_PITCH_TIER)
    speech_info_df = pd.read_csv(speech_info_df_fn, dtype=DTYPE_SPEECH_INFO)
    time_pitch_values = pitch_tier_df.values
    pitch_stats_columns = ["speaker", "min", "max", "mean", "median", "q1", "q2", "q3", "std"]
    pitch_stats = []
    for speaker in speech_info_df.speaker_tag.unique():
        spk_df = speech_info_df[speech_info_df["speaker_tag"] == speaker]
        speech_intervals = spk_df[["start_time", "end_time"]].values
        time_stamps, tmp = np.where((time_pitch_values[:, 0, None] >= speech_intervals[:, 0, None]) & (time_pitch_values[:, 0, None] < speech_intervals[:, 1, None]))
        speaker_pitches = time_pitch_values[time_stamps, 1]
        pitch_stats.append((speaker,
                            np.amin(speaker_pitches),
                            np.amax(speaker_pitches),
                            np.mean(speaker_pitches),
                            np.median(speaker_pitches),
                            np.percentile(speaker_pitches, 25),
                            np.percentile(speaker_pitches, 50),
                            np.percentile(speaker_pitches, 75),
                            np.std(speaker_pitches)))
    pitch_stats_df = pd.DataFrame(data=pitch_stats, columns=pitch_stats_columns)
    pitch_stats_df.to_csv(pitch_stats_df_fn, index=False)
    return 1


def pitch_stats_extraction_serial(audio_info_df_fn, audio_info_df, fn_list, dest_path):
    speakers_info = [(x, y, z, gen_file_name(x, dest_path)) for x, y, z in fn_list]
    for audio_fn, pitch_tier_df_fn, speech_info_df_fn, pitch_stats_df_fn in speakers_info:
        extract_speaker_pitch_stats(pitch_tier_df_fn, speech_info_df_fn, pitch_stats_df_fn)
    speakers_pitch_col = [(x, t) for x, y, z, t in speakers_info]
    speakers_pitch_df = pd.DataFrame(data=speakers_pitch_col, columns=["complete_event_file", "speaker_pitch_stats_file"])
    audio_info_df = audio_info_df.join(speakers_pitch_df)
    audio_info_df.to_csv(audio_info_df_fn, index=False)


def pitch_stats_extraction_parallel(audio_info_df_fn, audio_info_df, fn_list, dest_path, n):
    speakers_info = [(x, y, z, gen_file_name(x, dest_path)) for x, y, z in fn_list]
    Parallel(n_jobs=n, backend="threading")(delayed(extract_speaker_pitch_stats)(y, z, t) for x, y, z, t in speakers_info)
    speakers_pitch_col = [(x, t) for x, y, z, t in speakers_info]
    speakers_pitch_df = pd.DataFrame(data=speakers_pitch_col, columns=["complete_event_file", "speaker_pitch_stats_file"])
    audio_info_df = audio_info_df.join(speakers_pitch_df)
    audio_info_df.to_csv(audio_info_df_fn, index=False)


def speaker_pitch_statistics_extraction(audio_info_df_fn, dest_path, parallel=False, n_jobs=-1):
    # Note that praat_parameters and dest_paths must be passed as named tuples
    df = pd.read_csv(audio_info_df_fn, dtype=DTYPE)
    fn_list = df[["complete_event_file", "clean_pitch_tier_file", "speech_info_file"]].values
    if parallel:
        pitch_stats_extraction_parallel(audio_info_df_fn, df, fn_list, dest_path, n_jobs)
    else:
        pitch_stats_extraction_serial(audio_info_df_fn, df, fn_list, dest_path)
