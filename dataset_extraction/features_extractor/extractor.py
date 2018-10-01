# Imports
import collections
import essentia
import librosa
import math
import os
import pandas as pd
import numpy as np
import essentia.standard as es
from scipy.interpolate import Rbf
from joblib import Parallel, delayed
from functools import reduce
# NOTE all windows hops ecc must be expressed in number of samples
# Global variables
EXT = ".wav"
EXT_SIZE = len(EXT)
SAMPLE_RATE = 44100
TAB_EXT = ".csv"
FEATURE_JSON = "_features.json"
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
DTYPE_SEGMENT_BOUNDARIES = {"start_time": int,
                            "end_time": int,
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
Features_Tuple = collections.namedtuple("FeaturesTuple", "pitch delta_pitch delta_delta_pitch intensity delta_intensity delta_delta_intensity harmonicity jitter shimmer chromagram mfccs melbands melbands_log short_term_energy short_term_entropy spectral_centroid spectral_spread spectral_entropy spectral_flux spectral_roll_off")


def store_reults(file_name, features, format="json"):
    pool = essentia.Pool()
    pool.add("pitch", features.pitch)
    pool.add("delta_pitch", features.delta_pitch)
    pool.add("delta_delta_pitch", features.delta_delta_pitch)
    pool.add("intensity", features.intensity)
    pool.add("delta_intensity", features.delta_intensity)
    pool.add("delta_delta_intensity", features.delta_delta_intensity)
    pool.add("chromagram", features.chromagram)
    pool.add("mfccs", features.mfccs)
    pool.add("melbands", features.melbands)
    pool.add("melbands_log", features.melbands_log)
    pool.add("short_term_energy", features.short_term_energy)
    pool.add("short_term_entropy", features.short_term_entropy)
    pool.add("spectral_centroid", features.spectral_centroid)
    pool.add("spectral_spread", features.spectral_spread)
    pool.add("spectral_entropy", features.spectral_entropy)
    pool.add("spectral_flux", features.spectral_flux)
    pool.add("spectral_roll_off", features.spectral_roll_off)
    es.YamlOutput(filename=file_name, format=format)(pool)


def gen_file_names_tuple(fn_arr):
    # TODO rewrite in a more acceptale way
    fn_tuple = File_Names_Tuple(audio_fn=fn_arr[0],
                                segments_df_fn=fn_arr[2],
                                param_df_fn=fn_arr[3],
                                pitch_tier_df_fn=fn_arr[11],
                                intensity_tier_df_fn=fn_arr[10],
                                voice_report_df_fn=fn_arr[9],
                                silences_df_fn=fn_arr[12],
                                speech_df_fn=fn_arr[13],
                                pitch_stats_df_fn=fn_arr[15],
                                intensity_stats_df_fn=fn_arr[16],
                                harmonicity_stats_df_fn=fn_arr[17],
                                jitter_stats_df_fn=fn_arr[18],
                                shimmer_stats_df_fn=fn_arr[19])
    return fn_tuple


# NOTE sig len is expressed in number of samples, so is for win size and hop size
def extract_praat_feature(feature_df, speech_info_df, silences_info_df, feature_stats_df, sig_len, win_size=1024, hop_size=512, norm="mean", zero_out=True, sil_subst="mean", deltas=False, interpolation="gaussian"):
    feature_arr = np.array([])
    time_feature_values = feature_df.values
    for speaker, norm_value in feature_stats_df[["speaker", norm]].values:
        spk_df = speech_info_df[speech_info_df["speaker_tag"] == speaker]
        speech_intervals = spk_df[["start_time", "end_time"]].values
        ts, tmp = np.where((time_feature_values[:, 0, None] >= speech_intervals[:, 0]) & (time_feature_values[:, 0, None] < speech_intervals[:, 1]))
        feature_arr = np.concatenate((feature_arr, time_feature_values[ts]), axis=0)
    # TODO check if to sort
    rbfi = Rbf(feature_arr[:, 0], feature_arr[:, 1], function=interpolation)
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


def extract_delta(signal):
    # TODO check if it's ok to extract delta in this way
    delta_sig = signal - np.concatenate((np.array([0]), signal[:-1]))
    return delta_sig


def get_features(fn_tuple, features_fn_df, win_size=1024, hop_size=512, norm="mean", zero_out=True, sil_subst="mean"):
    audio = es.MonoLoader(filename=fn_tuple.audio_fn, sampleRate=SAMPLE_RATE)()
    sig_len = len(audio)
    speech_info_df = pd.read_csv(fn_tuple.speech_df_fn, dtype=DTYPE_SPEECH_INFO)
    silences_df = pd.read_csv(fn_tuple.silences_df_fn, dtype=DTYPE_SILENCES_FROM_INTENSITY)
    pitch = extract_praat_feature(pd.read_csv(fn_tuple.pitch_tier_df_fn, dtype=DTYPE_PITCH_TIER), speech_info_df, silences_df, pd.read_csv(fn_tuple.pitch_stats_df_fn, dtype=DTYPE_STATS), win_size=win_size, hop_size=hop_size, norm=norm, zero_out=zero_out, sil_subst=sil_subst)
    delta_pitch = extract_delta(pitch)
    delta_delta_pitch = extract_delta(delta_pitch)
    intensity = extract_praat_feature(pd.read_csv(fn_tuple.intensity_tier_df_fn, dtype=DTYPE_INTENSITY_TIER), speech_info_df, silences_df, pd.read_csv(fn_tuple.intensity_stats_df_fn, dtype=DTYPE_STATS), win_size=win_size, hop_size=hop_size, norm=norm, zero_out=zero_out, sil_subst=sil_subst)
    delta_intensity = extract_delta(intensity)
    delta_delta_intensity = extract_delta(delta_intensity)
    voice_report_df = pd.read_csv(fn_tuple.voice_report_df_fn, DTYPE_VOICE_REPORT)
    harmonicity = extract_praat_feature(voice_report_df[["start_time", "end_time", "harmonicity"]], speech_info_df, silences_df, pd.read_csv(fn_tuple.harmonicity_stats_df_fn, dtype=DTYPE_STATS), win_size, hop_size, sig_len)
    jitter = extract_praat_feature(voice_report_df[["start_time", "end_time", "jitter"]], speech_info_df, silences_df, pd.read_csv(fn_tuple.jitter_stats_df_fn, dtype=DTYPE_STATS), win_size, hop_size, sig_len)
    shimmer = extract_praat_feature(voice_report_df[["start_time", "end_time", "shimmer"]], speech_info_df, silences_df, pd.read_csv(fn_tuple.shimmer_stats_df_fn, dtype=DTYPE_STATS), win_size, hop_size, sig_len)
    spectrum = es.Spectrum()
    energy = es.Energy()
    mfcc = es.MFCC()
    centroid_t = es.SpectralCentroidTime()
    entropy = es.Entropy()
    flux = es.Flux()
    roll_off = es.RollOff()
    log_norm = es.UnaryOperator(type='log')
    chroma = []
    mfccs = []
    melbands = []
    melbands_log = []
    short_term_energy = []
    short_term_entropy = []
    spectral_centroid = []
    spectral_spread = []
    spectral_entropy = []
    spectral_flux = []
    spectral_roll_off = []
    for frame in es.FrameGenerator(audio, frameSize=win_size, hopSize=hop_size, lastFrameToEndOfFile=True, startFromZero=True):
        spec = spectrum(frame)
        mfcc_bands, mfcc_coeffs = mfcc(spec)
        mfccs.append(mfcc_coeffs)
        melbands.append(mfcc_bands)
        melbands_log.append(log_norm(mfcc_bands))
        short_term_energy.append(energy(frame))
        short_term_entropy.append(entropy(np.absolute(frame)))
        spectral_centroid.append(centroid_t(spec))
        spectral_spread.append(np.std(spec))
        spectral_entropy.append(entropy(np.absolute(spec)))
        spectral_flux.append(flux(spec))
        spectral_roll_off.append(roll_off(spec))
    cqt_lbs = np.abs(librosa.cqt(audio, SAMPLE_RATE))
    chroma_map_lbs = librosa.filters.cq_to_chroma(cqt_lbs.shape[0])
    chroma = chroma_map_lbs.dot(cqt_lbs)
    chroma = librosa.util.normalize(chroma, axis=0)
    mfccs = essentia.array(mfccs)
    chroma = essentia.array(chroma[:, 1:]).T
    melbands = essentia.array(melbands)
    melbands_log = essentia.array(melbands_log)
    short_term_energy = essentia.array(short_term_energy)
    short_term_entropy = essentia.array(short_term_entropy)
    spectral_centroid = essentia.array(spectral_centroid)
    spectral_spread = essentia.array(spectral_spread)
    spectral_entropy = essentia.array(spectral_entropy)
    spectral_flux = essentia.array(spectral_flux)
    spectral_roll_off = essentia.array(spectral_roll_off)
    segments_df = pd.read_csv(fn_tuple.segments_df_fn, dtype=DTYPE_SEGMENT_BOUNDARIES)
    segments_df = segments_df.join(features_fn_df)
    # TODO add normalization of other features, word embeddings and speech rate
    for start, end, file_name in segments_df[["start_time", "end_time", "features_file"]].values:
        start_win = start // hop_size
        end_win = end // hop_size
        features = Features_Tuple(pitch=pitch[start_win:end_win],
                                  delta_pitch=delta_pitch[start_win:end_win],
                                  delta_delta_pitch=delta_delta_pitch[start_win:end_win],
                                  intensity=intensity[start_win:end_win],
                                  delta_intensity=delta_intensity[start_win:end_win],
                                  delta_delta_intensity=delta_delta_intensity[start_win:end_win],
                                  harmonicity=harmonicity[start_win:end_win],
                                  jitter=jitter[start_win:end_win],
                                  shimmer=shimmer[start_win:end_win],
                                  chromagram=chroma[start_win:end_win],
                                  mfccs=mfccs[start_win:end_win],
                                  melbands=melbands[start_win:end_win],
                                  melbands_log=melbands_log[start_win:end_win],
                                  short_term_energy=short_term_energy[start_win:end_win],
                                  short_term_entropy=short_term_entropy[start_win:end_win],
                                  spectral_centroid=spectral_centroid[start_win:end_win],
                                  spectral_spread=spectral_spread[start_win:end_win],
                                  spectral_entropy=spectral_entropy[start_win:end_win],
                                  spectral_flux=spectral_flux[start_win:end_win],
                                  spectral_roll_off=spectral_roll_off[start_win:end_win])
        store_reults(file_name, features)


def gen_feat_file_name(segment_fn, dest_path):
    fn = dest_path + os.path.basename.split(segment_fn)[:-EXT_SIZE] + FEATURE_JSON
    return fn


def gen_feat_file_names(audio_fn, segments_df_fn, dest_path):
    segments_df = pd.read_csv(segments_df_fn, dtype=DTYPE_SEGMENT_BOUNDARIES)
    fn_list = [(audio_fn, x, gen_feat_file_name(x, dest_path)) for x in segments_df["audio_segment_file"].values]
    return fn_list


def extract_features_serial(audio_info_df, dest_path, win_size=1024, hop_size=512, norm="mean", zero_out=True, sil_subst="mean"):
    tmp_ds_map = list(map(lambda x: gen_feat_file_names(x[0], x[1]), audio_info_df[["complete_event_file", "segment_boundaries_df"]].values))
    tmp_ds_map = reduce(lambda x, y: x + y, tmp_ds_map)
    ds_map_df = pd.DataFrame(data=tmp_ds_map, columns=["complete_event_file", "audio_segment_file", "features_file"])
    for row in audio_info_df.values:
        get_features(gen_file_names_tuple(row), ds_map_df[ds_map_df["complete_event_file"] == row[0]], win_size=win_size, hop_size=hop_size, norm=norm, zero_out=zero_out, sil_subst=sil_subst)
    ds_map_df_fn = dest_path + "data_set_map" + TAB_EXT
    ds_map_df.to_csv(ds_map_df_fn, index=False)


def extract_features_parallel(audio_info_df, dest_path, n, win_size=1024, hop_size=512, norm="mean", zero_out=True, sil_subst="mean"):
    tmp_ds_map = list(map(lambda x: gen_feat_file_names(x[0], x[1]), audio_info_df[["complete_event_file", "segment_boundaries_df"]].values))
    tmp_ds_map = reduce(lambda x, y: x + y, tmp_ds_map)
    ds_map_df = pd.DataFrame(data=tmp_ds_map, columns=["complete_event_file", "segment_boundaries_df", "features_file"])
    Parallel(n_jobs=n, backend="threading")(delayed(get_features)(gen_file_names_tuple(x), ds_map_df[ds_map_df["complete_event_file"] == x[0]], win_size=win_size, hop_size=hop_size, norm=norm, zero_out=zero_out, sil_subst=sil_subst) for x in audio_info_df.values)
    ds_map_df_fn = dest_path + "data_set_map" + TAB_EXT
    ds_map_df.to_csv(ds_map_df_fn, index=False)


def features_extraction(audio_info_df_fn, dest_path, win_size=1024, hop_size=512, norm="mean", zero_out=True, sil_subst="mean", parallel=False, n_jobs=-1):
    df = pd.read_csv(audio_info_df_fn, dtype=DTYPE_GLOBAL)
    if parallel:
        extract_features_parallel(df, dest_path, n_jobs, win_size=win_size, hop_size=hop_size, norm=norm, zero_out=zero_out, sil_subst=sil_subst)
    else:
        extract_features_serial(df, dest_path, win_size=win_size, hop_size=hop_size, norm=norm, zero_out=zero_out, sil_subst=sil_subst)
