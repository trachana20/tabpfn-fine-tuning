from __future__ import annotations

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from data.DataManager import DataManager
from gym.Evaluator import Evaluator
from logger.Logger import Logger
from models.FineTuneTabPFNClassifier import FineTuneTabPFNClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from tabpfn import TabPFNClassifier
from utils import set_seed_globally

# Step 0: Define hyperparameters which are valid for all models and model
# specific hyperparameters

setup_config = {
    "project_name": "Finetune-TabPFN",
    "results_path": "results/",
    "random_states": [0, 1],
    "k_folds": 5,
    # val_size is percentage w.r.t. the total dataset-rows ]0,1[
    "val_size": 0.25,
    "num_workers": 0,
    "dataset_ids": [168746],  # , 23381]
    "log_wandb": False,
    "models": [
        FineTuneTabPFNClassifier,
        RandomForestClassifier,
        DecisionTreeClassifier,
        TabPFNClassifier,
    ],
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}
# Create a lookup dictionary which contains the architectural and training
# Hyperparameters of the models

# Step 1: Define the model, criterion, optimizer, device and evaluator
modelkwargs_dict = {
    FineTuneTabPFNClassifier: {
        "architecture": {"tabpfn_classifier": TabPFNClassifier(), "path": ""},
        "training": {},
    },
}

dataset_mapper = {168746: "Titanic"}


logger = Logger(
    project_name=setup_config["project_name"],
    log_wandb=setup_config["log_wandb"],
    results_path=f"{setup_config['results_path']}",
)
logger.setup_wandb(setup_config=setup_config)

evaluator = Evaluator(logger=logger if setup_config["log_wandb"] else None)


results_df = None
if os.path.exists(f"{setup_config['results_path']}results_df.pkl"):
    results_df = pd.read_pickle(f"{setup_config['results_path']}results_df.pkl")
else:
    # Step 2: run the evaluation and training loop
    # ---------- ---------- ---------- ---------- ---------- ---------- RANDOM STATES LOOP
    for random_state in setup_config["random_states"]:
        set_seed_globally(random_state)

        # ---------- ---------- ---------- ---------- ----------  DATASET ID LOOP
        for dataset_id in setup_config["dataset_ids"]:
            # Step 3: Load  data
            data_manager = DataManager(
                dir_path="data/dataset",
                dataset_id=dataset_id,
            )
            data_k_folded = data_manager.k_fold_train_test_split(
                k_folds=setup_config["k_folds"],
                val_size=setup_config["val_size"],
                random_state=random_state,
            )

            # ---------- ---------- ---------- ---------- ----------  FOLD LOOP
            for fold_i, fold in enumerate(data_k_folded):
                train_dataset = fold["train"]
                test_dataset = fold["test"]
                val_dataset = fold["val"]

                # iterate over all models and train on fold
                # ---------- ---------- ---------- ---------- ----------  MODEL LOOP
                for model_fn in setup_config["models"]:
                    model_architecture_kwargs = modelkwargs_dict.get(model_fn, {}).get(
                        "architecture",
                        {},
                    )
                    model_training_kwargs = modelkwargs_dict.get(model_fn, {}).get(
                        "training",
                        {},
                    )

                    # create a model which uses modelkwargs
                    model = model_fn(**model_architecture_kwargs)

                    # evaluate the model given the right setting
                    trained_model, performance_metrics = (
                        evaluator.main_train_and_evaluate_model(
                            model=model,
                            train_dataset=train_dataset,
                            val_dataset=val_dataset,
                            random_state=random_state,
                            dataset_id=dataset_id,
                            fold_i=fold_i,
                            **model_training_kwargs,
                        )
                    )

                    # add settings to performance metrics dictionary
                    performance_metrics.update(
                        {
                            "random_state": random_state,
                            "dataset_id": dataset_id,
                            "fold": fold_i,
                            "model": type(model).__name__,
                        },
                    )

                    if results_df is None:
                        results_df = pd.DataFrame([performance_metrics])
                    else:
                        results_df.loc[len(results_df)] = performance_metrics

    os.makedirs(f"{setup_config['results_path']}", exist_ok=True)
    results_df.to_pickle(f"{setup_config['results_path']}results_df.pkl")
    logger.save_results()


# ----------------- Visualize results -----------------


os.makedirs(f"{setup_config['results_path']}/plots/model_performance/", exist_ok=True)


def visualize_performance_across_models():
    for metric in [
        "accuracy",
        "auc",
        "f1",
        "cross_entropy",
        "time_fit",
        "time_predict",
    ]:
        for dataset_id in setup_config["dataset_ids"]:
            dataset_name = dataset_mapper[dataset_id]

            selected_df = results_df[results_df["dataset_id"] == dataset_id][
                ["model", metric]
            ]

            # Create barplot
            plt.figure(figsize=(10, 6))
            ax = sns.barplot(
                data=selected_df,
                x="model",
                y=metric,
                errorbar=("ci", 95),
                capsize=0.1,
                err_kws={"linewidth": 1},
            )

            # Adjust bar labels position
            threshold = 0.025
            for c in ax.containers:
                # Filter the labels
                labels = [v if v > threshold else "" for v in c.datavalues]
                ax.bar_label(c, labels=labels, label_type="center")

            # Add labels and title
            plt.xlabel("Model")
            plt.ylabel(metric.capitalize().replace("_", " "))
            plt.title(
                f"Dataset: {dataset_name} - Average {metric.capitalize()} by Model [95% CI]",
            )

            # Rotate x-axis labels for better readability
            plt.xticks(rotation=25, ha="right")

            # Show plot
            plt.tight_layout()
            plt.savefig(
                f"{setup_config['results_path']}/plots/model_performance/{metric}_{dataset_name}.png",
            )
            plt.close()


# ----------------- ----------------- ----------------- -----------------
# ----------------- ----------------- ----------------- Visualize results
# ----------------- ----------------- ----------------- -----------------

visualize_performance_across_models()
