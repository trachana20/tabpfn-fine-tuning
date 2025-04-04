"""This module provides the DataManager class for data management tasks."""

from __future__ import annotations

from pathlib import Path
from gan_module import GAN

import openml
import pandas as pd

from preprocessing.PreProcessor import PreProcessor
from sklearn.model_selection import (
    StratifiedKFold,
    StratifiedShuffleSplit,
)
import warnings

# Filter out the warning related to interactive backend
warnings.filterwarnings(
    "ignore",
    message="Backend TkAgg is interactive backend. Turning interactive mode on.",
)

# Filter out the future warning related to openml
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading.",
)


class DataManager:
    """DataManager class facilitates data management tasks such as loading local
    datasets or downloading datasets from OpenML.

    """

    def __init__(self, dir_path, dataset_id):
        self.dir_path = dir_path
        self.dataset_id = dataset_id

        self.results_path = f"{dir_path}/fine_tune_results"
        self.preprocessor = PreProcessor()

    def load_manual_dataset(self) -> pd.DataFrame:
        """Load a dataset from a local file."""
        data_df = pd.read_csv(self.dir_path)
        return data_df

    def split_train_test_validation(
        self,
        data_df,
        test_index,
        train_index,
        val_size,
        target,
        name,
        random_state=None,
    ):
        # Split data into train and test sets
        test_data = data_df.iloc[test_index]
        train_val_data = data_df.iloc[train_index]

        # Split train data into train and validation sets
        # Calculate the actual validation size based on the remaining data
        # val size is given as a percentage to the total data and
        # thus has to be scaled
        val_split_size_normalized = val_size / (1 - len(test_index) / data_df.shape[0])

        # Separate features and target
        X_train = train_val_data.drop(columns=[target])
        y_train = train_val_data[target]

        # Implement stratified train-val split
        strat_shuffle_split = StratifiedShuffleSplit(
            n_splits=1,
            test_size=val_split_size_normalized,
            random_state=random_state,
        )

        train_indices, val_indices = next(strat_shuffle_split.split(X_train, y_train))

        # Create train and validation sets
        train_data = train_val_data.iloc[train_indices]
        val_data = train_val_data.iloc[val_indices]

        return train_data, val_data, test_data

    # ----- ----- ----- ----- ----- create k-fold splits (strategy: StratifiedKFold)
    def k_fold_train_test_split(self, k_folds, val_size, random_state):
        # Preprocess the data (Missing values, encoding, outliers, scaling,...)
        task = openml.tasks.get_task(
            task_id=self.dataset_id,
            download_qualities=True,
            download_features_meta_data=True,
            download_splits=True,
            download_data=True,
        )
        # ignore future warning! We use version, where defaults are correct
        dataset = task.get_dataset()
        target = task.target_name
        name = dataset.name

        data_df, _, categorical_indicator, attribute_names = dataset.get_data(
            dataset_format="dataframe",
        )

        # List to store datasets
        datasets = []

        task_splits = task.download_split()
        task_splits = None

        if task_splits is not None:
            for repeat in range(task_splits.repeats):
                for fold in range(task_splits.folds):
                    for sample in range(task_splits.samples):
                        split = task_splits.get(repeat=repeat, fold=fold, sample=sample)

                        # Create CustomDataset instances and append to datasets list
                        train_data, val_data, test_data = (
                            self.split_train_test_validation(
                                data_df=data_df,
                                test_index=split.test,
                                train_index=split.train,
                                val_size=val_size,
                                target=target,
                                name=name,
                                random_state=random_state,
                            )
                        )
                        train_data, val_data, test_data = self.preprocessor.preprocess(
                            train_data=train_data,
                            val_data=val_data,
                            test_data=test_data,
                            target=target,
                            categorical_indicator=categorical_indicator,
                            attribute_names=attribute_names,
                        )

                        datasets.append(
                            {
                                "train": {
                                    "data": train_data,
                                    "target": target,
                                    "name": name,
                                },
                                "val": {
                                    "data": val_data,
                                    "target": target,
                                    "name": name,
                                },
                                "test": {
                                    "data": test_data,
                                    "target": target,
                                    "name": name,
                                },
                            },
                        )
        else:
            # Initialize StratifiedKFold
            kf = StratifiedKFold(
                n_splits=k_folds,
                shuffle=True,
                random_state=random_state,
            )

            x_data = data_df.drop(columns=[target])
            y_data = data_df[target]

            # Iterate through StratifiedKFold splits
            for train_index, test_index in kf.split(x_data, y_data):
                # Create CustomDataset instances and append to datasets list
                train_data, val_data, test_data = self.split_train_test_validation(
                    data_df=data_df,
                    test_index=test_index,
                    train_index=train_index,
                    val_size=val_size,
                    target=target,
                    name=name,
                    random_state=random_state,
                )
                train_data, val_data, test_data = self.preprocessor.preprocess(
                    train_data=train_data,
                    val_data=val_data,
                    test_data=test_data,
                    target=target,
                    categorical_indicator=categorical_indicator,
                    attribute_names=attribute_names,
                )

                datasets.append(
                    {
                        "train": {"data": train_data, "target": target, "name": name},
                        "val": {"data": val_data, "target": target, "name": name},
                        "test": {"data": test_data, "target": target, "name": name},
                    },
                )

        return datasets, categorical_indicator, target, attribute_names

    def k_fold_train_test_split_gan_augmentation(self, k_folds, val_size, random_state):
        # Preprocess the data (Missing values, encoding, outliers, scaling,...)
        synthetic_dataset = None
        datasets = []
        task = openml.tasks.get_task(
            task_id=self.dataset_id,
            download_qualities=True,
            download_features_meta_data=True,
            download_splits=True,
            download_data=True,
        )
        # ignore future warning! We use version, where defaults are correct
        dataset = task.get_dataset()
        target = task.target_name
        name = dataset.name
        data_df, _, categorical_indicator, attribute_names = dataset.get_data(
            dataset_format="dataframe",
        )
        #perform augmentation on the dataset i.e. do cosine similarity with the manual dataset and add the most similar rows to the dataset
        # Use the augmented dataset as the previous dataset and continue with the process

        task_splits = task.download_split()
        test_indices = [task_splits.get(repeat=0, fold=i, sample=0).test for i in range(task_splits.folds)]
        test_data_df = data_df.iloc[test_indices[0]]
        # get the dataset without the test data
        if data_df.shape[0] < 1000:
            # This is done so that the sum of the entire data generated is 1000
            num_samples = 1000 - data_df.shape[0] - test_data_df.shape[0]
            synthetic_dataset = GAN.create_synthetic_data(data_df, categorical_indicator, input_dim=100, epochs=100, batch_size=128, num_samples=num_samples)
            # preprocess manual dataset
            synthetic_dataset, _, _ = self.preprocessor.preprocess(
                train_data=synthetic_dataset,
                val_data=synthetic_dataset,
                test_data=synthetic_dataset,
                target=target,
                categorical_indicator=categorical_indicator,
                attribute_names=attribute_names,
            )
            # check the size of data_df and synthetic_dataset. data_df + Manual Dataset should be less than 1000
            if data_df.shape[0] + synthetic_dataset.shape[0] > 1000:
                # sample the manual dataset to make the total size 1000
                synthetic_dataset = synthetic_dataset.sample(n = num_samples)
        if task_splits is not None:
            for repeat in range(task_splits.repeats):
                for fold in range(task_splits.folds):
                    for sample in range(task_splits.samples):
                        split = task_splits.get(repeat=repeat, fold=fold, sample=sample)

                        # Create CustomDataset instances and append to datasets list
                        train_data, val_data, test_data = (
                            self.split_train_test_validation(
                                data_df=data_df,
                                test_index=split.test,
                                train_index=split.train,
                                val_size=val_size,
                                target=target,
                                name=name,
                                random_state=random_state,
                            )
                        )
                        train_data, val_data, test_data = self.preprocessor.preprocess(
                            train_data=train_data,
                            val_data=val_data,
                            test_data=test_data,
                            target=target,
                            categorical_indicator=categorical_indicator,
                            attribute_names=attribute_names,
                        )
                        # perform augmentation on the dataset i.e. do cosine similarity with the manual dataset and add the most similar rows to the dataset
                        # Use the augmented dataset as the previous dataset and continue with the process
                        if synthetic_dataset is not None:
                            train_data = self.preprocessor.augment_dataset(train_data, synthetic_dataset, target)
                        datasets.append(
                            {
                                "train": {
                                    "data": train_data,
                                    "target": target,
                                    "name": name,
                                },
                                "val": {
                                    "data": val_data,
                                    "target": target,
                                    "name": name,
                                },
                                "test": {
                                    "data": test_data,
                                    "target": target,
                                    "name": name,
                                },
                            },
                        )
        else:
            # Initialize StratifiedKFold
            kf = StratifiedKFold(
                n_splits=k_folds,
                shuffle=True,
                random_state=random_state,
            )

            x_data = data_df.drop(columns=[target])
            y_data = data_df[target]

            # Iterate through StratifiedKFold splits
            for train_index, test_index in kf.split(x_data, y_data):
                # Create CustomDataset instances and append to datasets list
                train_data, val_data, test_data = self.split_train_test_validation(
                    data_df=data_df,
                    test_index=test_index,
                    train_index=train_index,
                    val_size=val_size,
                    target=target,
                    name=name,
                    random_state=random_state,
                )
                train_data, val_data, test_data = self.preprocessor.preprocess(
                    train_data=train_data,
                    val_data=val_data,
                    test_data=test_data,
                    target=target,
                    categorical_indicator=categorical_indicator,
                    attribute_names=attribute_names,
                )
                if synthetic_dataset is not None:
                    train_data = self.preprocessor.augment_dataset(train_data, synthetic_dataset, target)
                datasets.append(
                    {
                        "train": {"data": train_data, "target": target, "name": name},
                        "val": {"data": val_data, "target": target, "name": name},
                        "test": {"data": test_data, "target": target, "name": name},
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
