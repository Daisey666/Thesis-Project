# Imports
import numpy as np
import os
import math
import essentia.standard as es
import acoustid as ai
import pandas as pd
from scipy import signal as sgnl
from joblib import Parallel, delayed
from operator import itemgetter

# Global variables
EXT = ".wav"
EXT_SIZE = len(EXT)
SAMPLE_RATE = 44100
MAX = 32767
# int16 max value
# TODO check assumption on minimm length
WINDOW_SIZE = 40    # 50
# Expressed in number of prints per second according to chromaprint
STEP = 4    # 4
# Expressed in number of prints per second according to chromaprint
FP_LEN = 0.1238
# Expressed in seconds according to chromaprint
MAX_SCORE_DECREASE = 0.9
MAX_CORR_DECREASE = 1
Z_FILL_PARAM = 5


# Utils
def fp_to_sample(time):
    return math.floor(time * FP_LEN * SAMPLE_RATE)


def get_uncompressed_chromaprint(audio):
    # Use Essentia wrapper for Chromaprint together with AcoustId to extract
    # the uncompressed audio fingeprint.
    fp_char = es.Chromaprinter()(audio)
    fp = ai.chromaprint.decode_fingerprint(fp_char)[0]
    return fp


def invert(arr):
    # For convenience, a reversed dictionary is created to access the
    # timestamps given the 20-bit values.
    # Make a dictionary that with the array elements as keys and their
    # positions positions as values.
    map = {}
    for i, a in enumerate(arr):
        map.setdefault(a, []).append(i)
    return map


popcnt_table_8bit = [
    0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,
    1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
    1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
    2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,
    1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
    2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,
    2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,
    3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8,
]


def popcnt(x):
    # Count the number of set bits in the given 32-bit integer.
    return (popcnt_table_8bit[(x >> 0) & 0xFF] +
            popcnt_table_8bit[(x >> 8) & 0xFF] +
            popcnt_table_8bit[(x >> 16) & 0xFF] +
            popcnt_table_8bit[(x >> 24) & 0xFF])


def ber(offset, fp_full, fp_frag):
    # Compare the short snippet against the full track at given offset.
    errors = 0
    count = 0
    for a, b in zip(fp_full[offset:], fp_frag):
        errors += popcnt(a ^ b)
        count += 1
    return max(0.0, 1.0 - 2.0 * errors / (32.0 * count))


def get_single_segment_location(fp_full, fp_frag):
    # The identification proccess starts by finding the common items
    # (or frames) of the chromaprints. At this stage, only the 20 most
    # significative bits are used.
    full_20bit = [x & (1 << 20 - 1) for x in fp_full]
    short_20bit = [x & (1 << 20 - 1) for x in fp_frag]
    common = set(full_20bit) & set(short_20bit)
    i_full_20bit = invert(full_20bit)
    i_short_20bit = invert(short_20bit)
    # Now the offsets among the common items are stored:
    offsets = {}
    for a in common:
        for i in i_full_20bit[a]:
            for j in i_short_20bit[a]:
                o = i - j
                offsets[o] = offsets.get(o, 0) + 1
    # All the detected offsets are filtered and scored. The criterium for
    # filtering is the greatest number of common events. In this example, only
    # the 20 top offsets are considered. The final score is computed by
    # measuring the bit-wise distance of the 32-bit vectors given the proposed
    # offsets.
    matches = []
    for count, offset in sorted([(v, k) for k, v in offsets.items()], reverse=True)[:20]:
        matches.append((ber(offset, fp_full, fp_frag), offset))
    matches.sort(reverse=True)
    score, offset = matches[0]
    return score, offset


def get_correlation_score(sig_full, sig_frag):
    corr = sgnl.correlate(sig_full, sig_frag, mode='same')
    score = np.max(corr)
    return score


def get_segments_location(full_audio_fn, audio_segments_fn):
    full_audio = es.MonoLoader(filename=full_audio_fn, sampleRate=SAMPLE_RATE)()
    full_audio_normalized = full_audio / MAX
    full_audio_fp = get_uncompressed_chromaprint(full_audio)
    audio_segments = es.MonoLoader(filename=audio_segments_fn, sampleRate=SAMPLE_RATE)()
    audio_segments_normalized = audio_segments / MAX
    audio_segments_fp = get_uncompressed_chromaprint(audio_segments)
    segments_info = []
    full_audio_index = 0
    full_audio_fp_index = 0
    audio_segments_index = 0
    audio_segments_fp_index = 0
    window_size = fp_to_sample(WINDOW_SIZE)
    while audio_segments_fp_index + WINDOW_SIZE < len(audio_segments_fp):
        frag_fp = audio_segments_fp[audio_segments_fp_index:audio_segments_fp_index+WINDOW_SIZE]
        frag = audio_segments_normalized[int(audio_segments_index):int(audio_segments_index+window_size)]
        try:
            score, offset = get_single_segment_location(full_audio_fp[full_audio_fp_index:], frag_fp)
        except IndexError:
            full_audio_index = 0
            full_audio_fp_index = 0
            score, offset = get_single_segment_location(full_audio_fp[full_audio_fp_index:], frag_fp)
        full_audio_fp_index += offset
        full_audio_index += fp_to_sample(offset)
        full_frag = full_audio_normalized[int(full_audio_index):int(full_audio_index+window_size)]
        correlation = get_correlation_score(full_frag, frag)
        old_score = score
        new_score = score
        old_correlation = correlation
        new_correlation = correlation
        n_steps = 1
        while ((new_score >= old_score*MAX_SCORE_DECREASE) or (new_correlation >= old_correlation*MAX_CORR_DECREASE)) and ((offset + WINDOW_SIZE + (STEP * n_steps)) <= len(audio_segments_fp)):
            shift_fp = STEP * n_steps
            shift = fp_to_sample(shift_fp)
            frag_fp = audio_segments_fp[int(audio_segments_fp_index+shift_fp):int(audio_segments_fp_index+WINDOW_SIZE+shift_fp)]
            frag = audio_segments_normalized[int(audio_segments_index+shift):int(audio_segments_index+window_size+shift)]
            old_score = new_score
            try:
                new_score, tmp_offset = get_single_segment_location(full_audio_fp[full_audio_fp_index+shift_fp:full_audio_fp_index+WINDOW_SIZE+shift_fp], frag_fp)
            except IndexError:
                break
            old_correlation = new_correlation
            full_frag = full_audio_normalized[int(full_audio_index+shift):int(full_audio_index+window_size+shift)]
            new_correlation = get_correlation_score(full_frag, frag_fp)
            n_steps += 1
        n_steps -= 1
        frag_len_fp = (STEP * n_steps) + WINDOW_SIZE
        frag_len = fp_to_sample(frag_len_fp)
        segments_info.append((full_audio_index, full_audio_index+frag_len, "relevant"))
        full_audio_fp_index += frag_len_fp
        audio_segments_fp_index += frag_len_fp
        full_audio_index += frag_len
        audio_segments_index += frag_len
    return segments_info


def generate_time_boundaries_df(full_audio_fn, audio_segments_fn, df_fn):
    tmp_data = get_segments_location(full_audio_fn, audio_segments_fn)
    tmp_data = sorted(tmp_data, key=itemgetter(0))
    df = pd.DataFrame(data=tmp_data, columns=["start_time", "end_time", "class"])
    df.to_csv(df_fn, index=False)


def extract_time_boundaries_serial(audio_info_list):
    for full_audio_fn, audio_segments_fn, df_fn in audio_info_list:
        generate_time_boundaries_df(full_audio_fn, audio_segments_fn, df_fn)


def extract_time_boundaries_parallel(audio_info_list, n):
    Parallel(n_jobs=n, backend="threading")(delayed(generate_time_boundaries_df)(x, y, z) for x, y, z in audio_info_list)


def identify_audio_segments_positions(complete_event_files_path, segmented_event_files_path, data_frames_path, parallel=False, n_jobs=-1):
    # TODO check if events and segments are correctly coupled
    full_event_audio_list = os.listdir(complete_event_files_path)
    full_event_audio_list = [complete_event_files_path + x for x in full_event_audio_list]
    segmented_audio_list = os.listdir(segmented_event_files_path)
    segmented_audio_list = [segmented_event_files_path + x for x in segmented_audio_list]
    data_frames_list = [data_frames_path + x[:-EXT_SIZE] + ".csv" for x in full_event_audio_list]
    audio_info_list = (full_event_audio_list, segmented_audio_list, data_frames_list)
    if parallel:
        extract_time_boundaries_parallel(audio_info_list, n_jobs)
    else:
        extract_time_boundaries_serial(audio_info_list)
    df = pd.DataFrame(data=audio_info_list, columns=["complete_event_file", "segmented_event_file", "segment_boundaries_df"])
    df.to_csv(data_frames_path + "segments_info.csv", index=False)
