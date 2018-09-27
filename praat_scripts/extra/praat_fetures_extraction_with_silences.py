# Imports
import os
import subprocess
import collections
import pandas as pd
from joblib import Parallel, delayed

# Global variables
EXT = ".wav"
EXT_SIZE = len(EXT)
SAMPLE_RATE = 44100
# Change in case OS is not MacOS
PRAAT = "/Applications/Praat.app/Contents/MacOS/Praat"
RUN_OPTIONS = "--run"
# Add full path to praat script
PRAAT_SCRIPT = "audio_info_extraction.praat"
PARAM = "_param_file.csv"
PITCH = "_pitch.Pitch"
PITCH_TIER = "_pitch_tier.csv"
POINT_PROCESS = "_point_process.PointProcess"
INTENSITY = "_intensity.Intensity"
INTENSITY_TIER = "_intensity_tier.csv"
VOICE_REPORT = "_voice_report.csv"
SILENCES = "_silences.csv"

DTYPE = {"complete_event_file": str,
         "segmented_event_file": str,
         "segment_boundaries_df": str}

Praat_File_Paths = collections.namedtuple("PraatFilePaths", "params pitch pitch_tier point_process intensity intensity_tier voice_report silences")


def gen_file_names(audio_fn, dest_paths):
    base_fn = os.path.basename.split(audio_fn)[:-EXT_SIZE]
    pfp = Praat_File_Paths(params=dest_paths.parameters_path + base_fn + PARAM,
                           pitch=dest_paths.pitch_path + base_fn + PITCH,
                           pitch_tier=dest_paths.pitch_tier_path + base_fn + PITCH_TIER,
                           point_process=dest_paths.point_process_path + base_fn + POINT_PROCESS,
                           intensity=dest_paths.intensity_path + base_fn + INTENSITY,
                           intensity_tier=dest_paths.intensity_tier_path + base_fn + INTENSITY_TIER,
                           voice_report=dest_paths.voice_report_path + base_fn + VOICE_REPORT,
                           silences=dest_paths.silences_path + base_fn + SILENCES)
    return pfp


def extract_info_from_complete_event(audio_fn, praat_parameters, praat_fns):
    # TODO controllare che i nomi dei nuovi file vengano inseriti correttamente
    subprocess.call([PRAAT, RUN_OPTIONS, PRAAT_SCRIPT, audio_fn,
                     praat_fns.params,
                     praat_fns.pitch,
                     praat_fns.pitch_tier,
                     praat_fns.point_process,
                     praat_fns.intensity,
                     praat_fns.intensity_tier,
                     praat_fns.voice_report,
                     praat_fns.silences,
                     str(praat_parameters.window_size),
                     str(praat_parameters.window_shift),
                     str(praat_parameters.pitch_min),
                     str(praat_parameters.pitch_max),
                     str(praat_parameters.max_period_factor),
                     str(praat_parameters.max_amplitude_factor),
                     str(praat_parameters.silence_threshold),
                     str(praat_parameters.voicing_thresholding),
                     str(praat_parameters.minimum_pitch),
                     str(praat_parameters.silence_threshold_db),
                     str(praat_parameters.minimum_silent_interval_duration),
                     str(praat_parameters.minimum_sounding_interval_duration),
                     str(praat_parameters.time_decimals)])


def audio_info_extraction_serial(audio_info_df, audio_info_df_fn, new_columns, praat_parameters, dest_paths):
    audio_event_list = audio_info_df["complete_event_file"].values
    audio_pfp_list = list(map(lambda x: list(x, gen_file_names(x)), audio_event_list))
    for audio_fn, praat_file_paths in audio_pfp_list:
        extract_info_from_complete_event(audio_fn, praat_parameters, praat_file_paths)
    praat_fn_cols = [(x[0], x[1].params, x[1].pitch, x[1].pitch_tier, x[1].point_process, x[1].intensity, x[1].intensity_tier, x[1].voice_report, x[1].silences) for x in audio_pfp_list]
    praat_fn_df = pd.DataFrame(data=praat_fn_cols, columns=new_columns)
    audio_info_df = audio_info_df.join(praat_fn_df)
    audio_info_df.to_csv(audio_info_df_fn, index=False)


def audio_info_extraction_parallel(audio_info_df, audio_info_df_fn, new_columns, praat_parameters, dest_paths, n):
    audio_event_list = audio_info_df["complete_event_file"].values
    audio_pfp_list = list(map(lambda x: list(x, gen_file_names(x)), audio_event_list))
    Parallel(n_jobs=n, backend="threading")(delayed(extract_info_from_complete_event)(x, praat_parameters, y) for x, y in audio_pfp_list)
    praat_fn_cols = [(x[0], x[1].params, x[1].pitch, x[1].pitch_tier, x[1].point_process, x[1].intensity, x[1].intensity_tier, x[1].voice_report, x[1].silences) for x in audio_pfp_list]
    praat_fn_df = pd.DataFrame(data=praat_fn_cols, columns=new_columns)
    audio_info_df = audio_info_df.join(praat_fn_df)
    audio_info_df.to_csv(audio_info_df_fn, index=False)


def extract_audio_informations(audio_info_df_fn, praat_parameters, dest_paths, parallel=False, n_jobs=-1):
    # Note that praat_parameters and dest_paths must be passed as named tuples
    df = pd.read_csv(audio_info_df_fn, dtype=DTYPE)
    new_columns = ["complete_event_file", "parameters_file", "pitch_file", "pitch_tier_file", "point_process_file", "intensity_file", "intensity_tier_file", "voice_report_file", "silences_from_intensity_file"]
    if parallel:
        audio_info_extraction_parallel(df, audio_info_df_fn, new_columns, praat_parameters, dest_paths, n_jobs)
    else:
        audio_info_extraction_serial(df, audio_info_df_fn, new_columns, praat_parameters, dest_paths)
