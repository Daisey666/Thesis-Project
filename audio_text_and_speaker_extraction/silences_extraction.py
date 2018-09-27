# Imports
import os
import pandas as pd
from joblib import Parallel, delayed
from scipy.io import wavfile
from operator import itemgetter

# Global variables
TAB_EXT = ".csv"
TAB_EXT_SIZE = len(TAB_EXT)
SILENCES_INFO = "_silences_info.csv"

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
         "speech_info_file": str}
DTYPE_SPEECH_INFO = {"word": str,
                     "start_time": float,
                     "end_time": float,
                     "speaker_tag": float}


def gen_file_name(audio_fn, dest_path):
    sfn = dest_path + os.path.basename.split(audio_fn)[:-TAB_EXT_SIZE] + SILENCES_INFO
    return sfn


def extract_silences_from_speech_infos(audio_fn, silence_info_df_fn, speech_info_df_fn):
    speech_df = pd.read_csv(speech_info_df_fn, dtype=DTYPE_SPEECH_INFO)
    speech_se = speech_df[["start_time", "end_time"]].values
    consecutives = []
    st = speech_se[0][0]
    for i in range(len(speech_se)-1):
        if speech_se[i][1] != speech_se[i+1][0]:
            consecutives.append((st, speech_se[i][1]))
            st = speech_se[i+1][0]
    consecutives.append((st, speech_se[-1][1]))
    segments = [(x, y, "speech") for x, y in consecutives]
    sample_rate, signal = wavfile.read(audio_fn)
    len = float(len(signal) / sample_rate)
    start = [0] + [x[1] for x in speech_se]
    end = [x[0] for x in speech_se] + [len]
    segments += [(x, y, "silence") for x, y in list(zip(start, end))]
    segments = sorted(segments, key=itemgetter(0))
    df = pd.DataFrame(data=segments, columns=["start_time", "end_time", "silence"])
    df.to_csv(silence_info_df_fn, index=False)


def silences_extraction_serial(audio_df_list, audio_info_df, ai_df_fn, praat_parameters, dest_path):
    silence_infos = [(x, gen_file_name(x, dest_path), y) for x, y in audio_df_list]
    for audio_fn, silence_info_df_fn, speech_info_df_fn in silence_infos:
        extract_silences_from_speech_infos(audio_fn, silence_info_df_fn, speech_info_df_fn)
    silences_info_col = pd.DataFrame(data=silence_infos, columns=["complete_event_file", "silences_info_file", "speech_info_file"])
    audio_info_df = audio_info_df.join(silences_info_col)
    audio_info_df.to_csv(ai_df_fn, index=False)


def silences_extraction_parallel(audio_df_list, audio_info_df, ai_df_fn, praat_parameters, dest_path, n):
    silence_infos = [(x, gen_file_name(x, dest_path), y) for x, y in audio_df_list]
    Parallel(n_jobs=n, backend="threading")(delayed(extract_silences_from_speech_infos)(x, y, z) for x, y, z in silence_infos)
    silences_info_col = pd.DataFrame(data=silence_infos, columns=["complete_event_file", "silences_info_file", "speech_info_file"])
    audio_info_df = audio_info_df.join(silences_info_col)
    audio_info_df.to_csv(ai_df_fn, index=False)


def extract_silence_informations(audio_info_df_fn, dest_path, parallel=False, n_jobs=-1):
    # Note that praat_parameters and dest_paths must be passed as named tuples
    df = pd.read_csv(audio_info_df_fn, dtype=DTYPE)
    audio_df_list = df[["complete_event_file", "speech_info_file"]].values
    if parallel:
        silences_extraction_parallel(audio_df_list, df, audio_info_df_fn, dest_path, n_jobs)
    else:
        silences_extraction_serial(audio_df_list, df, audio_info_df_fn, dest_path)
