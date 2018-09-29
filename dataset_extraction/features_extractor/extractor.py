# Imports
import collections
import pandas as pd
import numpy as np
import essentia.standard as es
from scipy.interpolate import Rbf

# Global variables
EXT = ".wav"
EXT_SIZE = len(EXT)
SAMPLE_RATE = 44100
# Data types
DTYPE_GLOBAL = {"complete_event_file": str,
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
                "speaker_pitch_stats_file": str,
                "speaker_intensity_stats_file": str,
                "speaker_harmonicity_stats_file": str,
                "speaker_jitter_stats_file": str,
                "speaker_shimmer_stats_file": str}
DTYPE_SEGMENT_BOUNDARIES = {"start_time": float,
                            "end_time": float,
                            "class": str,
                            "audio_segment_file": str}
DTYPE_PARAM = {"dx_pitch": float,
               "dx_intensity": float,
               "window_size_vr": float,
               "window_shift_vr": float}
DTYPE_VOICE_REPORT = {"start_time": float,
                      "end_time": float,
                      "harmonicity": float,
                      "jitter": float,
                      "shimmer": float}
DTYPE_INTENSITY_TIER = {"Time": float,
                        "Intensity": float}
DTYPE_PITCH_TIER = {"Time": float,
                    "F0": float}
DTYPE_SILENCES_FROM_INTENSITY = {"tmin": float,
                                 "text": str,
                                 "tmax": float}
DTYPE_SPEECH_INFO = {"word": str,
                     "start_time": float,
                     "end_time": float,
                     "speaker_tag": float}
DTYPE_STATS = {"speaker": str,
               "min": float,
               "max": float,
               "mean": float,
               "median": float,
               "q1": float,
               "q2": float,
               "q3": float,
               "std": float}

File_Names_Tuple = collections.namedtuple("FileNamesTuple", "audio_fn segments_df_fn param_df_fn pitch_tier_df_fn intensity_tier_df_fn voice_report_df_fn silences_df_fn speech_df_fn pitch_stats_df_fn intensity_stats_df_fn harmonicity_stats_df_fn jitter_stats_df_fn shimmer_stats_df_fn")


# NOTE sig len is expressed in number of samples, so is for win size and hop size
def extract_praat_feature(feature_df, speech_info_df, silences_info_df, feature_stats_df, win_size, hop_size, sig_len, norm="mean", zero_out=True, sil_subst="mean"):
    feature_arr = np.array([])
    time_feature_values = feature_df.values
    for speaker, norm_value in feature_stats_df[["speaker", norm]].values:
        spk_df = speech_info_df[speech_info_df["speaker_tag"] == speaker]
        speech_intervals = spk_df[["start_time", "end_time"]].values
        ts, tmp = np.where((time_feature_values[:, 0, None] >= speech_intervals[:, 0]) & (time_feature_values[:, 0, None] < speech_intervals[:, 1]))
        feature_arr = np.concatenate((feature_arr, time_feature_values[ts]), axis=0)
    # TODO check if to sort
    rbfi = Rbf(feature_arr[:, 0], feature_arr[:, 1], function="gaussian")
    xi = np.linspace(0, sig_len / SAMPLE_RATE, num=sig_len)
    di = rbfi(xi)
    if zero_out:
        tmp_df = silences_info_df[silences_info_df["text"] == "silent"]
        for tmin, tmax in tmp_df[["tmin", "tmax"]].values:
            di[tmin:tmax] = 0
    else:
        # TODO handle silences differently from zeroing out
        return 1
    feature = list(map(lambda x: np.mean(x), [di[t:min(di.shape[0], t + win_size)] for t in range(0, di.shape[0], hop_size)]))
    return np.array(feature)


def get_features(fn_tuple, win_size, hop_size, norm, zero_out, sil_subst):
    audio = es.MonoLoader(filename=fn_tuple.audio_fn, sampleRate=SAMPLE_RATE)()
    sig_len = len(audio)
    speech_info_df = pd.read_csv(fn_tuple.speech_df_fn, dtype=DTYPE_SPEECH_INFO)
    silences_df = pd.read_csv(fn_tuple.silences_df_fn, dtype=DTYPE_SILENCES_FROM_INTENSITY)
    pitch = extract_praat_feature(pd.read_csv(fn_tuple.pitch_tier_df_fn, dtype=DTYPE_PITCH_TIER), speech_info_df, silences_df, pd.read_csv(fn_tuple.pitch_stats_df_fn, dtype=DTYPE_STATS), win_size, hop_size, sig_len)
    intensity = extract_praat_feature(pd.read_csv(fn_tuple.intensity_tier_df_fn, dtype=DTYPE_INTENSITY_TIER), speech_info_df, silences_df, pd.read_csv(fn_tuple.intensity_stats_df_fn, dtype=DTYPE_STATS), win_size, hop_size, sig_len)
    voice_report_df = pd.read_csv(fn_tuple.voice_report_df_fn, DTYPE_VOICE_REPORT)
    harmonicity = extract_praat_feature(voice_report_df[["start_time", "end_time", "harmonicity"]], speech_info_df, silences_df, pd.read_csv(fn_tuple.harmonicity_stats_df_fn, dtype=DTYPE_STATS), win_size, hop_size, sig_len)
    jitter = extract_praat_feature(voice_report_df[["start_time", "end_time", "jitter"]], speech_info_df, silences_df, pd.read_csv(fn_tuple.jitter_stats_df_fn, dtype=DTYPE_STATS), win_size, hop_size, sig_len)
    shimmer = extract_praat_feature(voice_report_df[["start_time", "end_time", "shimmer"]], speech_info_df, silences_df, pd.read_csv(fn_tuple.shimmer_stats_df_fn, dtype=DTYPE_STATS), win_size, hop_size, sig_len)
    spectrum = es.spectrum(audio, size=sig_len)
    chromagram = es.
