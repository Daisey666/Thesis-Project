# Imports
import math
import pandas as pd
import numpy as np
# Global variables
DTYPE_MAP = {"complete_event_file": str,
             "audio_segment_file": str,
             "features_file": str,
             "class": str}


def select_train_and_test_data(data_set_map_df_fn, split, percentage, dest_path):
    data_set_map_df = pd.read_csv(data_set_map_df_fn, dtype=DTYPE_MAP)
    pos_class = data_set_map_df[data_set_map_df["class"] == "relevant"].values
    n_pos_class_samples = pos_class.shape[0]
    neg_class = np.shuffle(data_set_map_df[data_set_map_df["class"] == "non_relevant"].values)
    n_neg_class_samples = math.floor(percentage * n_pos_class_samples) if (percentage <= 1) else n_pos_class_samples + math.floor((neg_class.shape[0] - n_pos_class_samples) * (percentage - 1))
    neg_class = neg_class[:n_neg_class_samples]
    data_set = np.shuffle(np.concatenate((pos_class, neg_class)))
    n_train = math.floor(split * data_set.shape[0])
    train = pd.DataFrame(data=data_set[:n_train, 1:], columns=["audio_segment_file", "features_file", "class"])
    train.to_csv(dest_path + "train_set.csv", index=False)
    test = pd.DataFrame(data=data_set[n_train:, 1:], columns=["audio_segment_file", "features_file", "class"])
    test.to_csv(dest_path + "test_set.csv")
