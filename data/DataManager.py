from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
from preprocessing.PreProcessor import PreProcessor
from sklearn.model_selection import KFold, train_test_split

from data.CustomDataset import CustomDataset


class DataManager:
    def __init__(self, dir_path, dataset_name, target_col):
        self.dataset_path = f"{dir_path}/{dataset_name}"
        self.results_path = f"{dir_path}/results.pkl"
        self.dataset_name = dataset_name
        self.dir_path = dir_path

        self.target = target_col
        self.preprocessor = PreProcessor()

    def _load_data_from_pickle(self):
        with Path(self.dataset_path).open("rb") as f:
            return pickle.load(f)

    def _load_data_from_csv(self):
        return pd.read_csv(self.dataset_path)

    def load_data(self):
        if self.dataset_path.endswith(".csv"):
            data = self._load_data_from_csv()
        elif self.dataset_path.endswith(".pkl"):
            data = self._load_data_from_pickle()
        else:
            raise ValueError("File format not supported")
        return data

    def k_fold_train_test_split(self, k_folds, test_size, val_size, random_state):
        # Preprocess the data (Missing values, encoding, outliers, scaling,...)
        data = self.preprocessor.preprocess(self.load_data(), self.target)

        # Initialize KFold
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)

        # List to store datasets
        datasets = []

        # Iterate through KFold splits
        for train_index, test_index in kf.split(data):
            # Split data into train and test sets
            test_data = data.iloc[test_index]
            train_data = data.iloc[train_index]

            # Split train data into train and validation sets
            train_data, val_data = train_test_split(
                train_data,
                # Calculate the actual validation size based on the remaining data
                # val size is given as a percentage to the total data and
                # thus has to be scaled
                test_size=val_size / (1 - test_size),
                random_state=random_state,
            )

            # Create CustomDataset instances and append to datasets list
            datasets.append(
                {
                    "train": CustomDataset(train_data, self.target),
                    "test": CustomDataset(test_data, self.target),
                    "val": CustomDataset(val_data, self.target),
                },
            )

        return datasets

    def store_results(self, dict):
        with open(self.results_path, "wb") as f:
            pickle.dump(dict, f)

    def load_results(self):
        # check if exists
        if not Path(self.results_path).exists():
            return None
        with open(self.results_path, "rb") as f:
            return pickle.load(f)
