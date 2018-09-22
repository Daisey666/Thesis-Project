# Imports
import subprocess
import os
from joblib import Parallel, delayed

# Global variables
SUITE_SW = "ffmpeg"
INPUT_OPTION = "-i"
# 44100 Hz sample rate
SAMPLE_RATE_OPTION = "-ar 44100"
# Pulse code modulated with signed, 16 bit, little endian samples
SAMPLE_FORMAT_OPTION = "-acodec pcm_s16le"
# single channel (mono)
CHANNEL_OPTION = "-ac 1"
# Uncompressed output
OUTPUT_FORMAT = ".wav"


def extract_audio_single_file(input_path, file_in, output_path):
    input = input_path + file_in
    input_format_len = len(os.path.splitext(input)[1])
    output = output_path + input[:-input_format_len] + OUTPUT_FORMAT
    subprocess.call([SUITE_SW, INPUT_OPTION, input, SAMPLE_RATE_OPTION, SAMPLE_FORMAT_OPTION, CHANNEL_OPTION, output])


def extract_audio_serial(directory_paths):
    for src, dst in directory_paths:
        file_list = os.listdir(src)
        for file in file_list:
            extract_audio_single_file(src, file, dst)


def extract_audio_parallel(directory_paths, n):
    file_list = []
    for src, dst in directory_paths:
        file_list_tmp = os.listdir(src)
        file_list += [[src, x, dst] for x in file_list_tmp]
    Parallel(n_jobs=n, prefer="threads")(delayed(extract_audio_single_file)(x, y, z) for x, y, z in file_list)


def extract_audio(directory_paths, parallel=False, n_jobs=-1):
    if parallel:
        extract_audio_parallel(directory_paths, n_jobs)
    else:
        extract_audio_serial(directory_paths)
