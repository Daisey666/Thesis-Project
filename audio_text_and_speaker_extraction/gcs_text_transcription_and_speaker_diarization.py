# Imports
import os
import pandas as pd
from joblib import Parallel, delayed
from google.cloud import speech_v1p1beta1 as speech

# Global variables
EXT = ".wav"
EXT_SIZE = len(EXT)
SPEECH_INFO = "_speech_info.csv"
SAMPLE_RATE = 44100

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
         "silences_from_intensity_file": str}


def extract_text_trascription_and_speaker_diarization(audio_fn, speech_info_df_fn):
    client = speech.SpeechClient()
    speech_file = audio_fn
    with open(speech_file, 'rb') as audio_file:
        content = audio_file.read()
    audio = speech.types.RecognitionAudio(content=content)
    config = speech.types.RecognitionConfig(
        encoding=speech.enums.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=SAMPLE_RATE,
        language_code='it-IT',
        enable_speaker_diarization=True,
        diarization_speaker_count=2)
    response = client.recognize(config, audio)
    result = response.results[-1]
    # TODO check correct fields of response
    words_info = [(x.word, x.start_time, x.end_time, x.speaker_tag) for x in result.alternatives[0].words]
    words_info_df = pd.DataFrame(data=words_info, columns=["word", "start_time", "end_time", "speaker_tag"])
    words_info_df.to_csv(words_info_df, index=False)


def extract_speech_info_serial(audio_list, audio_info_df, ai_df_fn, dest_path):
    word_infos = [(x, dest_path + os.path.basename.split(x)[:-EXT_SIZE] + SPEECH_INFO) for x in audio_list]
    for audio_fn, speech_info_df_fn in word_infos:
        extract_text_trascription_and_speaker_diarization(audio_fn, speech_info_df_fn)
    speech_info_col = pd.DataFrame(data=word_infos, columns=["complete_event_file", "speech_info_file"])
    audio_info_df = audio_info_df.join(speech_info_col)
    audio_info_df.to_csv(ai_df_fn, index=False)


def extract_speech_info_parallel(audio_list, audio_info_df, ai_df_fn, dest_path, n):
    word_infos = [(x, dest_path + os.path.basename.split(x)[:-EXT_SIZE] + SPEECH_INFO) for x in audio_list]
    Parallel(n_jobs=n, backend="threading")(delayed(extract_text_trascription_and_speaker_diarization)(x, y) for x, y in word_infos)
    speech_info_col = pd.DataFrame(data=word_infos, columns=["complete_event_file", "speech_info_file"])
    audio_info_df = audio_info_df.join(speech_info_col)
    audio_info_df.to_csv(ai_df_fn, index=False)


def extract_speech_informations(audio_info_df_fn, dest_path, parallel=False, n_jobs=-1):
    df = pd.read_csv(audio_info_df_fn, dtype=DTYPE)
    audio_list = df["complete_event_file"].values
    if parallel:
        extract_speech_info_parallel(audio_list, df, audio_info_df_fn, dest_path, n_jobs)
    else:
        extract_speech_info_serial(audio_list, df, audio_info_df_fn, dest_path)
