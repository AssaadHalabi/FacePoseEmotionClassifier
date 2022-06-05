import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.data import AUTOTUNE
from tensorflow.data import Dataset

from data_pipeline import process_labels, parse_img


class AffectNetDatasetBuilder():
    def __init__(self, train_df, val_df):
        self.train_df = train_df
        self.val_df = val_df

    def build(self, batch_size=32, augmentations=[], shuffle=False, seed=0):
        train_dataset = Dataset.from_tensor_slices((self.train_df["file_paths"],
                                                    self.train_df["expression"],
                                                    self.train_df["valence"],
                                                    self.train_df["arousal"]))

        val_dataset = Dataset.from_tensor_slices((self.val_df["file_paths"],
                                                  self.val_df["expression"],
                                                  self.val_df["valence"],
                                                  self.val_df["arousal"]))

        if shuffle:
            train_dataset = train_dataset.shuffle(
                len(self.train_df), seed=seed, reshuffle_each_iteration=True)

        train_dataset = train_dataset.map(process_labels, num_parallel_calls=AUTOTUNE) \
                                     .map(parse_img, num_parallel_calls=AUTOTUNE)

        for aug in augmentations:
            train_dataset = train_dataset.map(aug, num_parallel_calls=AUTOTUNE)

        train_dataset = train_dataset.batch(batch_size) \
                                     .prefetch(buffer_size=AUTOTUNE)

        val_dataset = val_dataset.map(process_labels, num_parallel_calls=AUTOTUNE) \
                                 .map(parse_img, num_parallel_calls=AUTOTUNE) \
                                 .batch(batch_size) \
                                 .prefetch(buffer_size=AUTOTUNE)

        print(
            f"Loaded {len(self.train_df)} training images, and {len(self.val_df)} validation images.")
        return train_dataset, val_dataset

    def get_class_weights(self):
        class_weights = compute_class_weight(class_weight="balanced",
                                             classes=np.unique(
                                                 self.train_df["expression"]),
                                             y=self.train_df["expression"])
        # expression_class_weight = {i:weight for i, weight in enumerate(expression_class_weight)}
        return class_weights
