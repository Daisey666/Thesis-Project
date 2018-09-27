# Imports
import os
import collections
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# Global variables
EXT = ".wav"
EXT_SIZE = len(EXT)
HARMONICITY_SPEAKER_STATS = "_harmonicity_speaker_stats.csv"
JITTER_SPEAKER_STATS = "_jitter_speaker_stats.csv"
SHIMMER_SPEAKER_STATS = "_shimmer_speaker_stats.csv"

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
         "silences_info_file": str,
         "speaker_pitch_stats_file": str,
         "speaker_intensity_stats_file": str}
DTYPE_VOICE_REPORT = {"start_time": float,
                      "end_time": float,
                      "harmonicity": float,
                      "jitter": float,
                      "shimmer": float}
DTYPE_SPEECH_INFO = {"word": str,
                     "start_time": float,
                     "end_time": float,
                     "speaker_tag": float}

Voice_Report_Stats_Paths = collections.namedtuple("VoiceReportStatsPaths", "harmonicity_stats_df_fn jitter_stats_df_fn shimmer_stats_df_fn")


def gen_file_names(audio_fn, dest_paths):
    base_fn = os.path.basename.split(audio_fn)[:-EXT_SIZE]
    vrstatsfn = Voice_Report_Stats_Paths(harmonicity_stats_df_fn=dest_paths.harmonicity_dest_path + base_fn + HARMONICITY_SPEAKER_STATS,
                                         jitter_stats_df_fn=dest_paths.jitter_dest_path + base_fn + JITTER_SPEAKER_STATS,
                                         shimmer_stats_df_fn=dest_paths.shimmer_dest_path + base_fn + SHIMMER_SPEAKER_STATS)
    return vrstatsfn


def extract_speaker_voice_report_stats(voice_report_df_fn, speech_info_df_fn, voice_report_df_fn_nt):
    # Note that I'm using the central value of the voice report intervals
    voice_report_df = pd.read_csv(voice_report_df_fn, dtype=DTYPE_VOICE_REPORT)
    speech_info_df = pd.read_csv(speech_info_df_fn, dtype=DTYPE_SPEECH_INFO)
    time_h_j_s_values = np.array([((x[0] + x[1]) / 2, x[2], x[3]) for x in voice_report_df.values])
    stats_columns = ["speaker", "min", "max", "mean", "median", "q1", "q2", "q3", "std"]
    harmonicity_stats = []
    jitter_stats = []
    shimmer_stats = []
    for speaker in speech_info_df.speaker_tag.unique():
        spk_df = speech_info_df[speech_info_df["speaker_tag"] == speaker]
        speech_intervals = spk_df[["start_time", "end_time"]].values
        time_stamps, tmp = np.where((time_h_j_s_values[:, 0, None] >= speech_intervals[:, 0, None]) & (time_h_j_s_values[:, 0, None] < speech_intervals[:, 1, None]))
        speaker_harmonicities = time_h_j_s_values[time_stamps, 1]
        speaker_jitters = time_h_j_s_values[time_stamps, 2]
        speaker_shimmers = time_h_j_s_values[time_stamps, 3]
        harmonicity_stats.append((speaker,
                                  np.nanmin(speaker_harmonicities),
                                  np.nanmax(speaker_harmonicities),
                                  np.nanmean(speaker_harmonicities),
                                  np.nanmedian(speaker_harmonicities),
                                  np.nanpercentile(speaker_harmonicities, 25),
                                  np.nanpercentile(speaker_harmonicities, 50),
                                  np.nanpercentile(speaker_harmonicities, 75),
                                  np.nanstd(speaker_harmonicities)))
        jitter_stats.append((speaker,
                             np.nanmin(speaker_jitters),
                             np.nanmax(speaker_jitters),
                             np.nanmean(speaker_jitters),
                             np.nanmedian(speaker_jitters),
                             np.nanpercentile(speaker_jitters, 25),
                             np.nanpercentile(speaker_jitters, 50),
                             np.nanpercentile(speaker_jitters, 75),
                             np.nanstd(speaker_jitters)))
        shimmer_stats.append((speaker,
                              np.nanmin(speaker_shimmers),
                              np.nanmax(speaker_shimmers),
                              np.nanmean(speaker_shimmers),
                              np.nanmedian(speaker_shimmers),
                              np.nanpercentile(speaker_shimmers, 25),
                              np.nanpercentile(speaker_shimmers, 50),
                              np.nanpercentile(speaker_shimmers, 75),
                              np.nanstd(speaker_shimmers)))
    harmonicity_stats_df = pd.DataFrame(data=harmonicity_stats, columns=stats_columns)
    harmonicity_stats_df.to_csv(voice_report_df_fn_nt.harmonicity_stats_df_fn, index=False)
    jitter_stats_df = pd.DataFrame(data=jitter_stats, columns=stats_columns)
    jitter_stats_df.to_csv(voice_report_df_fn_nt.jitter_stats_df_fn, index=False)
    shimmer_stats_df = pd.DataFrame(data=shimmer_stats, columns=stats_columns)
    shimmer_stats_df.to_csv(voice_report_df_fn_nt.shimmer_stats_df_fn, index=False)
    return 1


def voice_report_stats_extraction_serial(audio_info_df_fn, audio_info_df, fn_list, dest_paths):
    speakers_info = [(x, y, z, gen_file_names(x, dest_paths)) for x, y, z in fn_list]
    for audio_fn, voice_report_tier_df_fn, speech_info_df_fn, voice_report_stats_df_fn in speakers_info:
        extract_speaker_voice_report_stats(voice_report_tier_df_fn, speech_info_df_fn, voice_report_stats_df_fn)
    speakers_voice_report_col = [(x, t.harmonicity_stats_df_fn, t.jitter_stats_df_fn, t.shimmer_stats_df_fn) for x, y, z, t in speakers_info]
    speakers_voice_report_df = pd.DataFrame(data=speakers_voice_report_col, columns=["complete_event_file", "speaker_harmonicity_stats_file", "speaker_jitter_stats_file", "speaker_shimmer_stats_file"])
    audio_info_df = audio_info_df.join(speakers_voice_report_df)
    audio_info_df.to_csv(audio_info_df_fn, index=False)


def voice_report_stats_extraction_parallel(audio_info_df_fn, audio_info_df, fn_list, dest_paths, n):
    speakers_info = [(x, y, z, gen_file_names(x, dest_paths)) for x, y, z in fn_list]
    Parallel(n_jobs=n, backend="threading")(delayed(extract_speaker_voice_report_stats)(y, z, t) for x, y, z, t in speakers_info)
    speakers_voice_report_col = [(x, t.harmonicity_stats_df_fn, t.jitter_stats_df_fn, t.shimmer_stats_df_fn) for x, y, z, t in speakers_info]
    speakers_voice_report_df = pd.DataFrame(data=speakers_voice_report_col, columns=["complete_event_file", "speaker_harmonicity_stats_file", "speaker_jitter_stats_file", "speaker_shimmer_stats_file"])
    audio_info_df = audio_info_df.join(speakers_voice_report_df)
    audio_info_df.to_csv(audio_info_df_fn, index=False)


def speaker_voice_report_statistics_extraction(audio_info_df_fn, dest_paths, parallel=False, n_jobs=-1):
    # Note that praat_parameters and dest_paths must be passed as named tuples
    df = pd.read_csv(audio_info_df_fn, dtype=DTYPE)
    fn_list = df[["complete_event_file", "clean_voice_report_tier_file", "speech_info_file"]].values
    if parallel:
        voice_report_stats_extraction_parallel(audio_info_df_fn, df, fn_list, dest_paths, n_jobs)
    else:
        voice_report_stats_extraction_serial(audio_info_df_fn, df, fn_list, dest_paths)
