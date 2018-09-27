# Imports
import os
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# Global variables
EXT = ".wav"
EXT_SIZE = len(EXT)
INTENSITY_SPEAKER_STATS = "_intensity_speaker_stats.csv"

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
         "clean_intensity_tier_file": str,
         "clean_pitch_tier_file": str,
         "silences_from_intensity_file": str,
         "speech_info_file": str,
         "silences_info_file": str,
         "speaker_pitch_stats_file": str}
DTYPE_INTENSITY_TIER = {"Time": float,
                        "Intensity": float}
DTYPE_SPEECH_INFO = {"word": str,
                     "start_time": float,
                     "end_time": float,
                     "speaker_tag": float}


def gen_file_name(audio_fn, dest_path):
    istatfn = dest_path + os.path.basename.split(audio_fn)[:-EXT_SIZE] + INTENSITY_SPEAKER_STATS
    return istatfn


def extract_speaker_intensity_stats(intensity_tier_df_fn, speech_info_df_fn, intensity_stats_df_fn):
    intensity_tier_df = pd.read_csv(intensity_tier_df_fn, dtype=DTYPE_INTENSITY_TIER)
    speech_info_df = pd.read_csv(speech_info_df_fn, dtype=DTYPE_SPEECH_INFO)
    time_intensity_values = intensity_tier_df.values
    intensity_stats_columns = ["speaker", "min", "max", "mean", "median", "q1", "q2", "q3", "std"]
    intensity_stats = []
    for speaker in speech_info_df.speaker_tag.unique():
        spk_df = speech_info_df[speech_info_df["speaker_tag"] == speaker]
        speech_intervals = spk_df[["start_time", "end_time"]].values
        time_stamps, tmp = np.where((time_intensity_values[:, 0, None] >= speech_intervals[:, 0, None]) & (time_intensity_values[:, 0, None] < speech_intervals[:, 1, None]))
        speaker_intensities = time_intensity_values[time_stamps, 1]
        intensity_stats.append((speaker,
                                np.amin(speaker_intensities),
                                np.amax(speaker_intensities),
                                np.mean(speaker_intensities),
                                np.median(speaker_intensities),
                                np.percentile(speaker_intensities, 25),
                                np.percentile(speaker_intensities, 50),
                                np.percentile(speaker_intensities, 75),
                                np.std(speaker_intensities)))
    intensity_stats_df = pd.DataFrame(data=intensity_stats, columns=intensity_stats_columns)
    intensity_stats_df.to_csv(intensity_stats_df_fn, index=False)
    return 1


def intensity_stats_extraction_serial(audio_info_df_fn, audio_info_df, fn_list, dest_path):
    speakers_info = [(x, y, z, gen_file_name(x, dest_path)) for x, y, z in fn_list]
    for audio_fn, intensity_tier_df_fn, speech_info_df_fn, intensity_stats_df_fn in speakers_info:
        extract_speaker_intensity_stats(intensity_tier_df_fn, speech_info_df_fn, intensity_stats_df_fn)
    speakers_intensity_col = [(x, t) for x, y, z, t in speakers_info]
    speakers_intensity_df = pd.DataFrame(data=speakers_intensity_col, columns=["complete_event_file", "speaker_intensity_stats_file"])
    audio_info_df = audio_info_df.join(speakers_intensity_df)
    audio_info_df.to_csv(audio_info_df_fn, index=False)


def intensity_stats_extraction_parallel(audio_info_df_fn, audio_info_df, fn_list, dest_path, n):
    speakers_info = [(x, y, z, gen_file_name(x, dest_path)) for x, y, z in fn_list]
    Parallel(n_jobs=n, backend="threading")(delayed(extract_speaker_intensity_stats)(y, z, t) for x, y, z, t in speakers_info)
    speakers_intensity_col = [(x, t) for x, y, z, t in speakers_info]
    speakers_intensity_df = pd.DataFrame(data=speakers_intensity_col, columns=["complete_event_file", "speaker_intensity_stats_file"])
    audio_info_df = audio_info_df.join(speakers_intensity_df)
    audio_info_df.to_csv(audio_info_df_fn, index=False)


def speaker_intensity_statistics_extraction(audio_info_df_fn, dest_path, parallel=False, n_jobs=-1):
    # Note that praat_parameters and dest_paths must be passed as named tuples
    df = pd.read_csv(audio_info_df_fn, dtype=DTYPE)
    fn_list = df[["complete_event_file", "clean_intensity_tier_file", "speech_info_file"]].values
    if parallel:
        intensity_stats_extraction_parallel(audio_info_df_fn, df, fn_list, dest_path, n_jobs)
    else:
        intensity_stats_extraction_serial(audio_info_df_fn, df, fn_list, dest_path)
