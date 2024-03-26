from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
import openml
from preprocessing.PreProcessor import PreProcessor
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold

from data.CustomDataset import CustomDataset


class DataManager:
    def __init__(self, dir_path, dataset_name, target_col):
        self.dataset_path = f"{dir_path}/{dataset_name}"
        self.results_path = f"{dir_path}/fine_tune_results"
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

    def _load_data_from_openml(self, tid: int = 168746):
        oml_task = openml.tasks.get_task(
            tid,
            download_data=True,
            download_qualities=False,
        )
        data, *_ = oml_task.get_dataset().get_data(dataset_format="dataframe")

        return data

    def k_fold_train_test_split(self, k_folds, test_size, val_size, random_state):
        # Preprocess the data (Missing values, encoding, outliers, scaling,...)
        data = self.preprocessor.preprocess(self._load_data_from_openml(), self.target)

        # Initialize StratifiedKFold
        kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_state)

        # List to store datasets
        datasets = []

        x_data = data.drop(columns=[self.target])
        y_data = data[self.target]

        # Iterate through StratifiedKFold splits
        for train_index, test_index in kf.split(x_data, y_data):
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

    def store_results(self, data):
        # Store DataFrame using pickle
        data.to_pickle(f"{self.results_path}.pkl")

    def load_results(self):
        file_path = f"{self.results_path}.pkl"
        # check if exists
        if not Path(file_path).exists():
            return None
        return pd.read_pickle(file_path)
