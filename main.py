from __future__ import annotations

import os
import time
import uuid
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from data.DataManager import DataManager
from data.FullRealDataDataset import FullRealDataDataset
from data.RealDataDataset import RealDataDataset
from gym.Evaluator import Evaluator
from gym.Trainer import Trainer
from gym.Visualizer import Visualizer
from models.FineTuneTabPFNClassifier import FineTuneTabPFNClassifier
from preprocessing.PreProcessor import PreProcessor
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from tabpfn import TabPFNClassifier
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader
from utils import set_seed_globally

# Step 0: Define hyperparameters which are valid for all models and model
# specific hyperparameters

# Generate unique identifier for each run
unique_id = f"{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4()}"

setup_config = {
    "project_name": "Finetune-TabPFN",
    "results_path": f"results/{unique_id}/",
    "random_states": [1],
    "k_folds": 5,
    # val_size is percentage w.r.t. the total dataset-rows ]0,1[
    "val_size": 0.2,
    "num_workers": 0,
    "dataset_mapping": {3778: "plasma_retinol"},
    "log_wandb": False,
    "models": {
        "FineTuneTabPFNClassifier_full_weight": FineTuneTabPFNClassifier,
        "RandomForestClassifier": RandomForestClassifier,
        "DecisionTreeClassifier": DecisionTreeClassifier,
        "TabPFNClassifier": TabPFNClassifier,
    },
    "dataset_dir": "data/dataset/",
    "dataset_augmentations": {"FullRealDataDataset": FullRealDataDataset},
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

# Create a lookup dictionary which contains the architectural and training
# Hyperparameters of the models
modelkwargs_dict = {
    # key: finetune models have to start with FineTuneTabPFNClassifier...
    "FineTuneTabPFNClassifier_full_weight": {
        "architectural": {
            # finetune models need a normal TabPFNClassifier instance
            "tabpfn_classifier": TabPFNClassifier(batch_size_inference=5),
            # weights_path is the location from where the model is loaded if path exists
            # or if not exists where the weights are stored in
            "weights_path": "model_weights/FullWeightFineTuneTabPFN.pth",
            # fine_tune_type determines the fine-tuning function, to be used
            "fine_tune_type": "full_weight_fine_tuning",
        },
        "training": {
            "epochs": 100,
            "batch_size": 2,
            "learning_rate": 1e-6,
            "criterion": CrossEntropyLoss,
            "optimizer": Adam,
            "early_stopping_threshold": 0.1,
        },
    },
}

augmentationkwargs_dict = {
    # batch elements are row-granularity! we introduce one edge case which is
    # "full" -> use all rows in the dataset. Number of rows is only exactly known later
    "FullRealDataDataset": {"batch_size": "full"},
    "BatchedRealDataDataset": {"batch_size": 50},
}

# Create results path if not exist
os.makedirs(setup_config['results_path'], exist_ok=True)

visualizer = Visualizer(path=setup_config['results_path'])
evaluator = Evaluator(visualizer=Visualizer)
trainer = Trainer(visualizer=visualizer)
preprocessor = PreProcessor()
results_df = None

results_pickle_path = os.path.join(setup_config['results_path'], "results_df_rag_100ep1_baseline.pkl")

if os.path.exists(results_pickle_path):
    results_df = pd.read_pickle(results_pickle_path)
else:
    # Step 2: run the evaluation and training loop
    # ---------- ---------- ---------- ---------- ---------- ---------- RANDOM STATES LOOP
    for random_state in setup_config["random_states"]:
        set_seed_globally(random_state)

        # ---------- ---------- ---------- ---------- ----------  DATASET ID LOOP
        for dataset_id, dataset_name in setup_config["dataset_mapping"].items():
            # Step 3: Load  data
            print("Dataset: ", dataset_name)
            data_manager = DataManager(
                dir_path=setup_config["dataset_dir"] + dataset_name + ".csv",
                dataset_id=dataset_id if dataset_id != 0 else None,
            )
            data_k_folded = data_manager.k_fold_train_test_split(
                k_folds=setup_config["k_folds"],
                val_size=setup_config["val_size"],
                random_state=random_state,
            )

            # ---------- ---------- ---------- ---------- ----------  FOLD LOOP
            for fold_i, fold in enumerate(data_k_folded):
                train_data = fold["train"]
                val_data = fold["val"]
                test_data = fold["test"]

                # iterate over all models and train on fold
                # ---------- ---------- ---------- ---------- ----------  MODEL LOOP
                for model_name, model_fn in setup_config["models"].items():
                    model_architectural_kwargs = modelkwargs_dict.get(
                        model_name,
                        {},
                    ).get(
                        "architectural",
                        {},
                    )
                    model_training_kwargs = modelkwargs_dict.get(model_name, {}).get(
                        "training",
                        {},
                    )

                    # validation and test data is never augmented
                    val_dataset = RealDataDataset(
                        data=val_data["data"],
                        target=val_data["target"],
                        name=val_data["name"],
                    )

                    test_dataset = RealDataDataset(
                        data=test_data["data"],
                        target=test_data["target"],
                        name=test_data["name"],
                    )

                    if "FineTuneTabPFNClassifier" in model_name:
                        for augmentation, augmentation_fn in setup_config[
                            "dataset_augmentations"
                        ].items():
                            augmentation_kwargs = augmentationkwargs_dict.get(
                                augmentation, {}
                            )

                            fine_tuning_configuration = {
                                "random_state": random_state,
                                "dataset_id": dataset_id,
                                "dataset_name": dataset_name,
                                "fold": fold_i,
                                "model": model_fn,
                                "augmentation": augmentation,
                            }

                            # perform data augmentation
                            train_data["data"] = preprocessor.augment_dataset(
                                train_data["data"],
                                test_data["data"],
                                train_data["target"]
                            )
                            train_dataset = augmentation_fn(
                                data=train_data["data"],
                                target=train_data["target"],
                                name=train_data["name"],
                            )

                            batch_size = augmentation_kwargs.get(
                                "batch_size",
                                1,
                            )
                            batch_size = (
                                train_dataset.number_rows
                                if batch_size == "full"
                                else batch_size
                            )

                            model = trainer.fine_tune_model(
                                train_loader=DataLoader(
                                    dataset=train_dataset,
                                    shuffle=True,
                                    num_workers=0,
                                    collate_fn=train_dataset._collate_fn,
                                    batch_size=batch_size,
                                ),
                                val_dataset=val_dataset,
                                fine_tune_type=model_architectural_kwargs[
                                    "fine_tune_type"
                                ],
                                device=setup_config["device"],
                                fine_tuning_configuration=fine_tuning_configuration,
                                **modelkwargs_dict.get(model_name, {}),
                            )

                            # evaluate the model given the right setting
                            trained_model, performance_metrics = (
                                evaluator.fit_and_predict_model(
                                    model=model,
                                    train_dataset=train_dataset,
                                    test_dataset=test_dataset,
                                    random_state=random_state,
                                    dataset_id=dataset_id,
                                    fold_i=fold_i,
                                    **model_training_kwargs,
                                )
                            )
                            # add settings to performance metrics dictionary
                            fine_tuning_configuration.update(performance_metrics)

                            if results_df is None:
                                results_df = pd.DataFrame([fine_tuning_configuration])
                            else:
                                results_df.loc[len(results_df)] = (
                                    fine_tuning_configuration
                                )
                    else:
                        # create a model which uses modelkwargs
                        model = model_fn(**model_architectural_kwargs)

                        # evaluate the model given the right setting
                        trained_model, performance_metrics = (
                            evaluator.fit_and_predict_model(
                                model=model,
                                train_dataset=train_dataset,
                                test_dataset=test_dataset,
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
                                "model": model_fn,
                                "augmentation": "none",
                            },
                        )

                    if results_df is None:
                        results_df = pd.DataFrame([performance_metrics])
                    else:
                        results_df.loc[len(results_df)] = performance_metrics
    visualizer.save_training_logs_as_csv()

    os.makedirs(f"{setup_config['results_path']}", exist_ok=True)
    results_df.to_pickle(results_pickle_path)
    visualizer.save_training_logs_as_csv()

# ----------------- Visualize results -----------------

os.makedirs(f"{setup_config['results_path']}/plots/model_performance/", exist_ok=True)

def bar_plot_dataset_performance_across_folds(
        results_df,
        metric,
        dataset_id,
        plot_settings,
):
    dataset_name = setup_config["dataset_mapping"][dataset_id]

    selected_df = results_df[results_df["dataset_id"] == dataset_id][["model", metric]]

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

    if plot_settings.get(metric, {}).get("y_scale") == "log":
        plt.yscale("log")

    # Adjust bar labels position
    if plot_settings.get(metric, {}).get("bar_label") == "top":
        ax.bar_label(ax.containers[0])

    elif plot_settings.get(metric, {}).get("bar_label") == "center":
        threshold = 0.025
        for c in ax.containers:
            # Filter the labels
            labels = [v if v > threshold else "" for v in c.datavalues]
            ax.bar_label(c, labels=labels, label_type="center")

    # Add labels and title
    plt.xlabel("Model")

    if plot_settings.get(metric, {}).get("y_label"):
        plt.ylabel(f"{plot_settings[metric]['y_label']} [Across Folds]")
    else:
        plt.ylabel(f"{metric.capitalize().replace('_', '')} [Across Folds]")

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

def bar_plot_performance_across_datasets(results_df, metric, plot_settings):
    selected_df = results_df[["model", metric]]

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

    if plot_settings.get(metric, {}).get("y_scale") == "log":
        plt.yscale("log")

    # Adjust bar labels position
    if plot_settings.get(metric, {}).get("bar_label") == "top":
        ax.bar_label(ax.containers[0])

    elif plot_settings.get(metric, {}).get("bar_label") == "center":
        threshold = 0.025
        for c in ax.containers:
            # Filter the labels
            labels = [v if v > threshold else "" for v in c.datavalues]
            ax.bar_label(c, labels=labels, label_type="center")

    # Add labels and title
    plt.xlabel("Model")

    if plot_settings.get(metric, {}).get("y_label"):
        plt.ylabel(f"{plot_settings[metric]['y_label']} [Across Datasets]")
    else:
        plt.ylabel(f"{metric.capitalize().replace('_', ' ')} [Across Datasets]")

    plt.title(
        f"Average {metric.capitalize()} by Model [95% CI]",
    )

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=25, ha="right")

    # Show plot
    plt.tight_layout()
    plt.savefig(
        f"{setup_config['results_path']}/plots/model_performance/{metric}_across_datasets.png",
    )
    plt.close()

# ----------------- ----------------- ----------------- -----------------
# ----------------- ----------------- ----------------- Visualize results
# ----------------- ----------------- ----------------- -----------------

plot_settings = {
    "time_fit": {"y_scale": "log", "y_label": "Time [s]", "bar_label": "top"},
    "time_predict": {"y_scale": "log", "y_label": "Time [s]", "bar_label": "top"},
    "accuracy": {"bar_label": "center"},
    "auc": {"bar_label": "center"},
    "f1": {"bar_label": "center"},
    "log_loss": {"bar_label": "center"},
}
performance_metrics = [
    "accuracy",
    "auc",
    "f1",
    "log_loss",
    "time_fit",
    "time_predict",
]

for metric in performance_metrics:
    for dataset_id in setup_config["dataset_mapping"]:
        bar_plot_dataset_performance_across_folds(
            results_df=results_df,
            metric=metric,
            dataset_id=dataset_id,
            plot_settings=plot_settings,
        )

for metric in performance_metrics:
    bar_plot_performance_across_datasets(
        results_df=results_df,
        metric=metric,
        plot_settings=plot_settings,
    )
