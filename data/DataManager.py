"""This module provides the DataManager class for data management tasks."""

from __future__ import annotations

from pathlib import Path

import joblib
import openml
import pandas as pd
from preprocessing.PreProcessor import PreProcessor
from sklearn.model_selection import StratifiedKFold, train_test_split

from data.CustomDataset import CustomDataset


class DataManager:
    """DataManager class facilitates data management tasks such as loading local
    datasets or downloading datasets from OpenML.

    """

    def __init__(self, dir_path, dataset_name=None, target_col=None, dataset_id=None):
        """Initialize the DataManager class.

        Parameters:
            dir_path (str): The directory path where the dataset is located or where it
             will be downloaded to.
            dataset_name (str, optional): The name of the local dataset if available in
             the dir_path. Default is None.
            target_col (str, optional): The name of the target column in the dataset.
             Required if dataset_name is provided.
            dataset_id (int, optional): The ID of the dataset to be downloaded from
             OpenML. If both dataset_name and dataset_id are provided, dataset_id will
              be used. Default is None.
        """
        assert (
            dataset_name is not None or dataset_id is not None
        ), "Either dataset_name or dataset_id must be provided"

        assert (
            dataset_name is None or target_col is not None
        ), "If dataset_name is provided, target_col must be provided"

        self._use_openml = dataset_id is not None

        self.dir_path = dir_path
        self.dataset_name = dataset_name
        self.target = target_col
        self.dataset_id = dataset_id

        if self.dataset_name is not None:
            self.dataset_path = f"{dir_path}/{self.dataset_name}"

        self.results_path = f"{dir_path}/fine_tune_results"
        self.preprocessor = PreProcessor()

    # ----- ----- ----- ----- ----- load data from a local dataset
    def _load_data_from_pickle(self):
        with Path(self.dataset_path).open("rb") as f:
            return joblib.load(f)

    def _load_data_from_csv(self):
        return pd.read_csv(self.dataset_path)

    # ----- ----- ----- ----- ----- load data from a local dataset or OpenML
    def load_data(self):
        if self._use_openml:
            data_df, target_str, name = self._load_data_from_openml(self.dataset_id)

        else:
            target_str = self.target
            name = self.dataset_name
            if self.dataset_path.endswith(".csv"):
                data_df = self._load_data_from_csv()
            elif self.dataset_path.endswith(".pkl"):
                data_df = self._load_data_from_pickle()
            else:
                raise ValueError("File format not supported")
        return data_df, target_str, name

    # ----- ----- ----- ----- ----- load data from openml
    def _load_data_from_openml(self, tid: int = 168746):
        oml_task = openml.tasks.get_task(
            tid,
            download_data=True,
            download_qualities=False,
        )
        dataset = oml_task.get_dataset()
        name = dataset.name
        data_df, *_ = dataset.get_data(dataset_format="dataframe")
        target = oml_task.target_name
        return data_df, target, name

    # ----- ----- ----- ----- ----- create k-fold splits (strategy: StratifiedKFold)
    def k_fold_train_test_split(self, k_folds, val_size, random_state):
        # Preprocess the data (Missing values, encoding, outliers, scaling,...)
        data_df, target, name = self.load_data()
        data = self.preprocessor.preprocess(data_df, target)

        # Initialize StratifiedKFold
        kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_state)

        # List to store datasets
        datasets = []

        x_data = data.drop(columns=[target])
        y_data = data[target]

        # Iterate through StratifiedKFold splits
        for train_index, test_index in kf.split(x_data, y_data):
            # Split data into train and test sets
            test_data = data.iloc[test_index]
            train_data = data.iloc[train_index]

            # Split train data into train and validation sets
            # Calculate the actual validation size based on the remaining data
            # val size is given as a percentage to the total data and
            # thus has to be scaled
            val_split_size_normalized = val_size / (1 - len(test_index) / data.shape[0])
            train_data, val_data = train_test_split(
                train_data,
                test_size=val_split_size_normalized,
                random_state=random_state,
            )

            # Create CustomDataset instances and append to datasets list
            datasets.append(
                {
                    "train": CustomDataset(train_data, target, name),
                    "test": CustomDataset(test_data, target, name),
                    "val": CustomDataset(val_data, target, name),
                },
            )

        return datasets

    # ----- ----- ----- ----- ----- store and load results
    def store_results(self, data):
        # Store DataFrame using pickle
        data.to_pickle(f"{self.results_path}.pkl")

    def load_results(self):
        file_path = f"{self.results_path}.pkl"
        # check if exists
        if not Path(file_path).exists():
            return None
        return pd.read_pickle(file_path)
