from sklearn.model_selection import RandomizedSearchCV
import numpy as np
import openml
import pandas as pd
from models.FineTuneTabPFNClassifier import FineTuneTabPFNClassifier
from tabpfn import TabPFNClassifier
from sklearn.metrics import accuracy_score
from preprocessing.PreProcessor import PreProcessor
import torch
setup_config = {
    "results_path": "results/",
}
# XGBoost model for hyperparameter optimization


# tabpfn_classifier = TabPFNClassifier(batch_size_inference=5)
# fine_tune_model = FineTuneTabPFNClassifier(
#     tabpfn_classifier=tabpfn_classifier,
#     weights_path="model_weights/FullWeightFineTuneTabPFN.pth"
# )

# class SklearnWrapper:
#     def __init__(self, model, **kwargs):
#         self.model = model
#         self.kwargs = kwargs
    
#     def fit(self, X, y):
#         self.model.fit(X, y, **self.kwargs)
#         return self
    
#     def predict(self, X):
#         return self.model.predict(X)
    
#     def score(self, X, y):
#         return accuracy_score(y, self.predict(X))

#     def get_params(self, deep=True):
#         return {'model': self.model, **self.kwargs}
    
#     def set_params(self, **params):
#         for key, value in params.items():
#             if key == 'model':
#                 self.model = value
#             else:
#                 self.kwargs[key] = value
#         return self

# # Wrap your model
# model = SklearnWrapper(fine_tune_model)
# import pth file for model using torch
# Define the hyperparameter search space
param_dist = {
    "epochs": [50, 100, 150, 200],
    "batch_size": [2, 4, 8],
    "learning_rate": [1e-4, 1e-5, 1e-6, 1e-7],
    "early_stopping_threshold": [0.01, 0.05, 0.1, 0.2],
}

# Example of using RandomizedSearchCV with cross-validation
# Assuming `model` is your FineTuneTabPFNClassifier_full_weight instance
random_search = RandomizedSearchCV(
    estimator=model, 
    param_distributions=param_dist, 
    n_iter=20,  # Number of hyperparameter settings sampled
    scoring='accuracy',  # Scoring metric
    n_jobs=-1,  # Number of jobs to run in parallel
    cv=5,  # Number of folds in cross-validation
    random_state=42  # For reproducibility
)

preprocessor = PreProcessor()
# Assuming X_train and y_train are your training data and labels
# get data from open ml titanic dataset
task = openml.tasks.get_task(
    task_id=168746,
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

# preprocess manual dataset
manual_dataset, _, _ = preprocessor.preprocess(
    train_data=data_df,
    val_data=data_df,
    test_data=data_df,
    target=target,
    categorical_indicator=categorical_indicator,
    attribute_names=attribute_names,
    name = name
)
# take only 1000 rows for training
manual_dataset = manual_dataset.head(1000)
X_train = manual_dataset.drop(columns=[target])
y_train = manual_dataset[target]
print("########## fitting##########")
random_search.fit(X_train, y_train)

# Get the best hyperparameters

best_hyperparameters = random_search.best_params_

print("Best hyperparameters:", best_hyperparameters)
#print accuracy
print("Best accuracy:", random_search.best_score_)
