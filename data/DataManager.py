from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
from preprocessing.PreProcessor import PreProcessor
from sklearn.model_selection import KFold, train_test_split

from data.CustomDataset import CustomDataset


class DataManager:
    def __init__(self, path, target):
        self.path = path
        self.target = target
        self.preprocessor = PreProcessor()

    def _load_data_from_pickle(self):
        with Path(self.path).open("rb") as f:
            return pickle.load(f)

    def _load_data_from_csv(self):
        return pd.read_csv(self.path)

    def load_data(self):
        if self.path.endswith(".csv"):
            data = self._load_data_from_csv()
        elif self.path.endswith(".pkl"):
            data = self._load_data_from_pickle()
        else:
            raise ValueError("File format not supported")
        return data

    def k_fold_train_test_split(self, k_folds, test_size, val_size, random_state):
        data = self.preprocessor.preprocess(self.load_data(), self.target)
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)

        datasets = []

        for train_index, _ in kf.split(data):
            train_data, test_data = train_test_split(
                data.iloc[train_index],
                test_size=test_size,
                random_state=random_state,
            )
            train_data, val_data = train_test_split(
                train_data,
                test_size=val_size,
                random_state=random_state,
            )

            datasets.append(
                {
                    "train": CustomDataset(train_data, self.target),
                    "test": CustomDataset(test_data, self.target),
                    "val": CustomDataset(val_data, self.target),
                },
            )

        return datasets
