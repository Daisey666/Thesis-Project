# Imports
import os
import pandas as pd
import speech_recognition as sr
from pocketsphinx import AudioFile
from joblib import Parallel, delayed

# Global variables
LANGUAGE = "it-IT"
EXT = ".wav"
EXT_SIZE = len(EXT)
SPEECH_INFO = "_speech_info.csv"
SAMPLE_RATE = 44100


def extract_text_trascription(audio_fn, speech_info_df_fn):
    r = sr.Recognizer()
    with sr.AudioFile(audio_fn) as source:
        audio = r.record(source)  # read the entire audio file
    try:
        ps_text_decoder =  r.recognize_sphinx(audio, language=LANGUAGE, show_all=True)
    except sr.UnknownValueError:
        print("Sphinx could not understand audio")
    except sr.RequestError as e:
        print("Sphinx error; {0}".format(e))



    #words_info = [(x.word, x.start_time, x.end_time, x.speaker_tag) for x in result.alternatives[0].words]
    #words_info_df = pd.DataFrame(data=words_info, columns=["word", "start_time", "end_time", "speaker_tag"])
    #words_info_df.to_csv(words_info_df, index=False)


def extract_speech_info_serial(audio_list, audio_info_df, ai_df_fn, dest_path):
    word_infos = [(x, dest_path + os.path.basename.split(x)[:-EXT_SIZE] + SPEECH_INFO) for x in audio_list]
    for audio_fn, speech_info_df_fn in word_infos:
        extract_text_trascription(audio_fn, speech_info_df_fn)
    speech_info_col = pd.DataFrame(data=word_infos, columns=["complete_event_file", "speech_info_df"])
    audio_info_df = audio_info_df.join(speech_info_col)
    audio_info_df.to_csv(ai_df_fn, index=False)


def extract_speech_info_parallel(audio_list, audio_info_df, ai_df_fn, dest_path, n):
    word_infos = [(x, dest_path + os.path.basename.split(x)[:-EXT_SIZE] + SPEECH_INFO) for x in audio_list]
    Parallel(n_jobs=n, backend="threading")(delayed(extract_text_trascription)(x, y) for x, y in word_infos)
    speech_info_col = pd.DataFrame(data=word_infos, columns=["complete_event_file", "speech_info_df"])
    audio_info_df = audio_info_df.join(speech_info_col)
    audio_info_df.to_csv(ai_df_fn, index=False)


def extract_speech_transcriptions(audio_info_df_fn, dest_path, parallel=False, n_jobs=-1):
    df = pd.read_csv(audio_info_df_fn, dtype={"complete_event_file": str, "segmented_event_file": str, "segment_boundaries_df": str})
    audio_list = df["complete_event_file"].values
    if parallel:
        extract_speech_info_parallel(audio_list, df, audio_info_df_fn, dest_path, n_jobs)
    else:
        extract_speech_info_serial(audio_list, df, audio_info_df_fn, dest_path)
