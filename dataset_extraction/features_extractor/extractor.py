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
from gensim.models import Word2Vec
# NOTE all windows hops ecc must be expressed in number of samples
# Global variables
EXT = ".wav"
EXT_SIZE = len(EXT)
SAMPLE_RATE = 44100
TAB_EXT = ".csv"
FEATURE_JSON = "_features.json"
WORD_VECTOR_MODEL = "resources/word_vectors/wiki_iter=5_algorithm=skipgram_window=10_size=300_neg-samples=10.m"
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
Features_Tuple = collections.namedtuple("FeaturesTuple", "pitch delta_pitch delta_delta_pitch intensity delta_intensity delta_delta_intensity harmonicity jitter shimmer silences chromagram mfccs melbands melbands_log short_term_energy short_term_entropy spectral_centroid spectral_spread spectral_entropy spectral_flux spectral_roll_off zero_crossing_rate onsets onset_rate word_embeddings")


def store_reults(file_name, features, format="json"):
    pool = essentia.Pool()
    pool.add("pitch", features.pitch)
    pool.add("delta_pitch", features.delta_pitch)
    pool.add("delta_delta_pitch", features.delta_delta_pitch)
    pool.add("intensity", features.intensity)
    pool.add("delta_intensity", features.delta_intensity)
    pool.add("delta_delta_intensity", features.delta_delta_intensity)
    pool.add("harmonicity", features.harmonicity)
    pool.add("jitter", features.jitter)
    pool.add("shimmer", features.shimmer)
    pool.add("silences", features.silences)
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
    pool.add("zero_crossing_rate", features.zero_crossing_rate)
    pool.add("onsets", features.onsets)
    pool.add("onset_rate", features.onset_rate)
    pool.add("word_embeddings", features.word_embeddings)
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


def extract_delta(signal):
    delta_sig = np.concatenate((np.array([0]), signal[:-1])) - signal
    return delta_sig


# NOTE sig len is expressed in number of samples, so is for win size and hop size
def extract_praat_feature(feature_df, speech_info_df, silences_info_df, feature_stats_df, sig_len, win_size=1024, hop_size=512, norm="mean", zero_out=True, sil_subst="mean", deltas=False, interpolation="gaussian"):
    feature_arr = np.array([])
    time_feature_values = feature_df.values
    for speaker, norm_value in feature_stats_df[["speaker", norm]].values:
        spk_df = speech_info_df[speech_info_df["speaker_tag"] == speaker]
        speech_intervals = spk_df[["start_time", "end_time"]].values
        ts, tmp = np.where((time_feature_values[:, 0, None] >= speech_intervals[:, 0]) & (time_feature_values[:, 0, None] < speech_intervals[:, 1]))
        time_feature_values[ts, 1] -= norm_value
        feature_arr = np.concatenate((feature_arr, time_feature_values[ts]), axis=0)
    # TODO check if to sort
    rbfi = Rbf(feature_arr[:, 0], feature_arr[:, 1], function=interpolation)
    xi = np.linspace(0, sig_len / SAMPLE_RATE, num=sig_len)
    di = rbfi(xi)
    if deltas:
        delta_di = extract_delta(di)
        delta_delta_di = extract_delta(delta_di)
        if zero_out:
            tmp_df = silences_info_df[silences_info_df["text"] == "silent"]
            for tmin, tmax in tmp_df[["tmin", "tmax"]].values:
                di[tmin:tmax] = 0
                delta_di[tmin:tmax] = 0
                delta_delta_di[tmin:tmax] = 0
        else:
            # TODO handle silences differently from zeroing out
            return 1
        feature = list(map(lambda x: np.mean(x), [di[t:min(di.shape[0], t + win_size)] for t in range(0, di.shape[0], hop_size)]))
        delta_feature = list(map(lambda x: np.mean(x), [delta_di[t:min(delta_di.shape[0], t + win_size)] for t in range(0, delta_di.shape[0], hop_size)]))
        delta_delta_feature = list(map(lambda x: np.mean(x), [delta_delta_di[t:min(delta_delta_di.shape[0], t + win_size)] for t in range(0, delta_delta_di.shape[0], hop_size)]))
        return np.array(feature), np.array(delta_feature), np.array(delta_delta_feature)
    else:
        if zero_out:
            tmp_df = silences_info_df[silences_info_df["text"] == "silent"]
            for tmin, tmax in tmp_df[["tmin", "tmax"]].values:
                di[tmin:tmax] = 0
        else:
            # TODO handle silences differently from zeroing out
            return 1
        feature = list(map(lambda x: np.mean(x), [di[t:min(di.shape[0], t + win_size)] for t in range(0, di.shape[0], hop_size)]))
        return np.array(feature)


def extract_silences(silences_info_df, n_win, hop_size=512):
    silences = np.zeros(n_win)
    tmp_df = silences_info_df[silences_info_df["text"] == "silent"]
    for tmin, tmax in tmp_df[["tmin", "tmax"]].values:
        silences[tmin//hop_size:tmax//hop_size] = 1
    return silences


def extract_word_embeddings(speech_info_df, n_win, hop_size=512):
    model = Word2Vec.load(WORD_VECTOR_MODEL)
    word_embeddings = np.empty((n_win, 300), np.zeros(300))
    for word, start_time, end_time in speech_info_df["word", "start_time", "end_time"].values:
        word_embeddings[start_time//hop_size:end_time//hop_size] = model.wv[word]
    return word_embeddings


def extract_speaker_stats(time_feature_values, speech_info_df):
    # NOTE tecnically it should handle also multidimensional arrays
    feature_stats_columns = ["speaker", "min", "max", "mean", "median", "q1", "q2", "q3", "std"]
    feature_stats = []
    for speaker in speech_info_df.speaker_tag.unique():
        spk_df = speech_info_df[speech_info_df["speaker_tag"] == speaker]
        speech_intervals = spk_df[["start_time", "end_time"]].values
        time_stamps, tmp = np.where((time_feature_values[:, 0, None] >= speech_intervals[:, 0, None]) & (time_feature_values[:, 0, None] < speech_intervals[:, 1, None]))
        speaker_features = time_feature_values[time_stamps, 1]
        feature_stats.append((speaker,
                              np.amin(speaker_features, axis=0),
                              np.amax(speaker_features, axis=0),
                              np.mean(speaker_features, axis=0),
                              np.median(speaker_features, axis=0),
                              np.percentile(speaker_features, 25, axis=0),
                              np.percentile(speaker_features, 50, axis=0),
                              np.percentile(speaker_features, 75, axis=0),
                              np.std(speaker_features, axis=0)))
    feature_stats_df = pd.DataFrame(data=feature_stats, columns=feature_stats_columns)
    return feature_stats_df


def post_process_feature(feature, speech_info_df, silences_info_df, sig_len, win_size=1024, hop_size=512, norm="mean", zero_out=True, sil_subst="mean"):
    # NOTE I assumed each window represents its central value
    # NOTE this may not handle correctly multidimensional features, but apparently (from repl tests) it should
    # TODO check if dividing by sample rate is correct
    time_feature_values = np.array([(((i * hop_size) + (win_size / 2)) / SAMPLE_RATE, feature[i]) for i in range(feature.shape[0])])
    feature_stats_df = extract_speaker_stats(time_feature_values, speech_info_df)
    for speaker, norm_value in feature_stats_df[["speaker", norm]].values:
        spk_df = speech_info_df[speech_info_df["speaker_tag"] == speaker]
        speech_intervals = spk_df[["start_time", "end_time"]].values
        ts, tmp = np.where((time_feature_values[:, 0, None] >= speech_intervals[:, 0]) & (time_feature_values[:, 0, None] < speech_intervals[:, 1]))
        # NOTE I'm note shure this is correct
        feature[ts] -= norm_value
    if zero_out:
        tmp_df = silences_info_df[silences_info_df["text"] == "silent"]
        for tmin, tmax in tmp_df[["tmin", "tmax"]].values:
            feature[(tmin * SAMPLE_RATE) // hop_size:(tmax * SAMPLE_RATE) // hop_size] = 0
    else:
        # TODO handle silences differently from zeroing out
        return 1
    return feature


def get_features(fn_tuple, features_fn_df, win_size=1024, hop_size=512, norm="mean", zero_out=True, sil_subst="mean", interpolation="gaussian"):
    audio = es.MonoLoader(filename=fn_tuple.audio_fn, sampleRate=SAMPLE_RATE)()
    sig_len = len(audio)
    speech_info_df = pd.read_csv(fn_tuple.speech_df_fn, dtype=DTYPE_SPEECH_INFO)
    silences_df = pd.read_csv(fn_tuple.silences_df_fn, dtype=DTYPE_SILENCES_FROM_INTENSITY)
    pitch, delta_pitch, delta_delta_pitch = extract_praat_feature(pd.read_csv(fn_tuple.pitch_tier_df_fn, dtype=DTYPE_PITCH_TIER), speech_info_df, silences_df, pd.read_csv(fn_tuple.pitch_stats_df_fn, dtype=DTYPE_STATS), win_size=win_size, hop_size=hop_size, norm=norm, zero_out=zero_out, sil_subst=sil_subst, deltas=True, interpolation=interpolation)
    intensity, delta_intensity, delta_delta_intensity = extract_praat_feature(pd.read_csv(fn_tuple.intensity_tier_df_fn, dtype=DTYPE_INTENSITY_TIER), speech_info_df, silences_df, pd.read_csv(fn_tuple.intensity_stats_df_fn, dtype=DTYPE_STATS), win_size=win_size, hop_size=hop_size, norm=norm, zero_out=zero_out, sil_subst=sil_subst, deltas=True, interpolation=interpolation)
    voice_report_df = pd.read_csv(fn_tuple.voice_report_df_fn, DTYPE_VOICE_REPORT)
    harmonicity = extract_praat_feature(voice_report_df[["start_time", "end_time", "harmonicity"]], speech_info_df, silences_df, pd.read_csv(fn_tuple.harmonicity_stats_df_fn, dtype=DTYPE_STATS), win_size, hop_size, sig_len)
    jitter = extract_praat_feature(voice_report_df[["start_time", "end_time", "jitter"]], speech_info_df, silences_df, pd.read_csv(fn_tuple.jitter_stats_df_fn, dtype=DTYPE_STATS), win_size, hop_size, sig_len)
    shimmer = extract_praat_feature(voice_report_df[["start_time", "end_time", "shimmer"]], speech_info_df, silences_df, pd.read_csv(fn_tuple.shimmer_stats_df_fn, dtype=DTYPE_STATS), win_size, hop_size, sig_len)
    spectrum = es.Spectrum()
    energy = es.Energy()
    mfcc = es.MFCC()
    centroid_t = es.SpectralCentroidTime()
    entropy = es.Entropy()
    power_spectrum = es.PowerSpectrum()
    flux = es.Flux()
    roll_off = es.RollOff()
    zero_crossing = es.ZeroCrossingRate()
    onset = es.OnsetRate()
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
    zero_crossing_rate = []
    onsets = []
    onset_rate = []
    for frame in es.FrameGenerator(audio, frameSize=win_size, hopSize=hop_size, lastFrameToEndOfFile=True, startFromZero=True):
        spec = spectrum(frame)
        mfcc_bands, mfcc_coeffs = mfcc(spec)
        mfccs.append(mfcc_coeffs)
        melbands.append(mfcc_bands)
        melbands_log.append(log_norm(mfcc_bands))
        short_term_energy.append(energy(frame))
        short_term_entropy.append(entropy(np.power(frame, 2)))
        spectral_centroid.append(centroid_t(spec))
        spectral_spread.append(np.std(spec))
        tmp_power_spectral_density = power_spectrum(spec[:-1])  # Spectrum must have even length
        tmp_power_spectral_density_normalized = tmp_power_spectral_density / np.sum(tmp_power_spectral_density)
        spectral_entropy.append(entropy(tmp_power_spectral_density_normalized))
        spectral_flux.append(flux(spec))
        spectral_roll_off.append(roll_off(spec))
        zero_crossing_rate.append(zero_crossing(frame))
        ons, onsr = onset(frame)
        onsets.append(ons)
        onset_rate.append(onsr)
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
    zero_crossing_rate = essentia.array(zero_crossing_rate)
    onsets = essentia.array(onsets)
    onset_rate = essentia.array(onset_rate)
    silences = extract_silences(silences_df, pitch.shape[0], hop_size=hop_size)
    word_embeddings = extract_word_embeddings(speech_info_df, pitch.shape[0], hop_size=hop_size)
    mfccs = post_process_feature(mfccs, speech_info_df, silences_df, sig_len, win_size=win_size, hop_size=hop_size, norm=norm, zero_out=zero_out, sil_subst=sil_subst)
    chroma = post_process_feature(chroma, speech_info_df, silences_df, sig_len, win_size=win_size, hop_size=hop_size, norm=norm, zero_out=zero_out, sil_subst=sil_subst)
    melbands = post_process_feature(melbands, speech_info_df, silences_df, sig_len, win_size=win_size, hop_size=hop_size, norm=norm, zero_out=zero_out, sil_subst=sil_subst)
    melbands_log = post_process_feature(melbands_log, speech_info_df, silences_df, sig_len, win_size=win_size, hop_size=hop_size, norm=norm, zero_out=zero_out, sil_subst=sil_subst)
    short_term_energy = post_process_feature(short_term_energy, speech_info_df, silences_df, sig_len, win_size=win_size, hop_size=hop_size, norm=norm, zero_out=zero_out, sil_subst=sil_subst)
    short_term_entropy = post_process_feature(short_term_entropy, speech_info_df, silences_df, sig_len, win_size=win_size, hop_size=hop_size, norm=norm, zero_out=zero_out, sil_subst=sil_subst)
    spectral_centroid = post_process_feature(spectral_centroid, speech_info_df, silences_df, sig_len, win_size=win_size, hop_size=hop_size, norm=norm, zero_out=zero_out, sil_subst=sil_subst)
    spectral_spread = post_process_feature(spectral_spread, speech_info_df, silences_df, sig_len, win_size=win_size, hop_size=hop_size, norm=norm, zero_out=zero_out, sil_subst=sil_subst)
    spectral_entropy = post_process_feature(spectral_entropy, speech_info_df, silences_df, sig_len, win_size=win_size, hop_size=hop_size, norm=norm, zero_out=zero_out, sil_subst=sil_subst)
    spectral_flux = post_process_feature(spectral_flux, speech_info_df, silences_df, sig_len, win_size=win_size, hop_size=hop_size, norm=norm, zero_out=zero_out, sil_subst=sil_subst)
    spectral_roll_off = post_process_feature(spectral_roll_off, speech_info_df, silences_df, sig_len, win_size=win_size, hop_size=hop_size, norm=norm, zero_out=zero_out, sil_subst=sil_subst)
    zero_crossing_rate = post_process_feature(zero_crossing_rate, speech_info_df, silences_df, sig_len, win_size=win_size, hop_size=hop_size, norm=norm, zero_out=zero_out, sil_subst=sil_subst)
    # onsets = post_process_feature(onsets, speech_info_df, silences_df, sig_len, win_size=win_size, hop_size=hop_size, norm=norm, zero_out=zero_out, sil_subst=sil_subst)
    onset_rate = post_process_feature(onset_rate, speech_info_df, silences_df, sig_len, win_size=win_size, hop_size=hop_size, norm=norm, zero_out=zero_out, sil_subst=sil_subst)
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
                                  silences=silences[start_win:end_win],
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
                                  spectral_roll_off=spectral_roll_off[start_win:end_win],
                                  zero_crossing_rate=zero_crossing_rate[start_win:end_win],
                                  onsets=onsets[start_win:end_win],
                                  onset_rate=onset_rate[start_win:end_win],
                                  word_embeddings=word_embeddings[start_win:end_win])
        store_reults(file_name, features)


# TODO gestire variabili interpolazione e postprocessing


def gen_feat_file_name(segment_fn, dest_path):
    fn = dest_path + os.path.basename.split(segment_fn)[:-EXT_SIZE] + FEATURE_JSON
    return fn


def gen_feat_file_names(audio_fn, segments_df_fn, dest_path):
    segments_df = pd.read_csv(segments_df_fn, dtype=DTYPE_SEGMENT_BOUNDARIES)
    fn_list = [(audio_fn, x[0], gen_feat_file_name(x[0], dest_path), x[1]) for x in segments_df[["audio_segment_file", "class"]].values]
    return fn_list


def extract_features_serial(audio_info_df, dest_path, win_size=1024, hop_size=512, norm="mean", zero_out=True, sil_subst="mean"):
    tmp_ds_map = list(map(lambda x: gen_feat_file_names(x[0], x[1]), audio_info_df[["complete_event_file", "segment_boundaries_df"]].values))
    tmp_ds_map = reduce(lambda x, y: x + y, tmp_ds_map)
    ds_map_df = pd.DataFrame(data=tmp_ds_map, columns=["complete_event_file", "audio_segment_file", "features_file", "class"])
    for row in audio_info_df.values:
        get_features(gen_file_names_tuple(row), ds_map_df[ds_map_df["complete_event_file"] == row[0]], win_size=win_size, hop_size=hop_size, norm=norm, zero_out=zero_out, sil_subst=sil_subst)
    ds_map_df_fn = dest_path + "data_set_map" + TAB_EXT
    ds_map_df.to_csv(ds_map_df_fn, index=False)


def extract_features_parallel(audio_info_df, dest_path, n, win_size=1024, hop_size=512, norm="mean", zero_out=True, sil_subst="mean"):
    tmp_ds_map = list(map(lambda x: gen_feat_file_names(x[0], x[1]), audio_info_df[["complete_event_file", "segment_boundaries_df"]].values))
    tmp_ds_map = reduce(lambda x, y: x + y, tmp_ds_map)
    ds_map_df = pd.DataFrame(data=tmp_ds_map, columns=["complete_event_file", "segment_boundaries_df", "features_file", "class"])
    Parallel(n_jobs=n, backend="threading")(delayed(get_features)(gen_file_names_tuple(x), ds_map_df[ds_map_df["complete_event_file"] == x[0]], win_size=win_size, hop_size=hop_size, norm=norm, zero_out=zero_out, sil_subst=sil_subst) for x in audio_info_df.values)
    ds_map_df_fn = dest_path + "data_set_map" + TAB_EXT
    ds_map_df.to_csv(ds_map_df_fn, index=False)


def features_extraction(audio_info_df_fn, dest_path, win_size=1024, hop_size=512, norm="mean", zero_out=True, sil_subst="mean", parallel=False, n_jobs=-1):
    df = pd.read_csv(audio_info_df_fn, dtype=DTYPE_GLOBAL)
    if parallel:
        extract_features_parallel(df, dest_path, n_jobs, win_size=win_size, hop_size=hop_size, norm=norm, zero_out=zero_out, sil_subst=sil_subst)
    else:
        extract_features_serial(df, dest_path, win_size=win_size, hop_size=hop_size, norm=norm, zero_out=zero_out, sil_subst=sil_subst)
