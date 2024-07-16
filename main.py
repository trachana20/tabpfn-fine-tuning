from __future__ import annotations

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from data.DataManager import DataManager
from data.FullRealDataDataset import FullRealDataDataset
from data.RealDataDataset import RealDataDataset
from gym.Evaluator import Evaluator
from gym.Trainer import Trainer
from gym.Visualizer import Visualizer
from models.FineTuneTabPFNClassifier import FineTuneTabPFNClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from tabpfn import TabPFNClassifier
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader
from utils import set_seed_globally
from sklearn.neighbors import NearestNeighbors
from sklearn.impute import SimpleImputer
# Step 0: Define hyperparameters which are valid for all models and model
# specific hyperparameters



# def augment_dataset(df):
#     augmented_data = df.copy()
#     for i in range(len(df)):
#         target_instance = df.iloc[i]
#         augmented_features = augment_data_with_retrieval(df, target_instance)
#         for col in augmented_features.index:
#             augmented_data.at[i, col] = augmented_features[col]
#     return augmented_data

from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_cosine_similarity(X_train, X_test):
    # Normalize X_train and X_test
    X_train_normalized = X_train / np.linalg.norm(X_train, axis=1, keepdims=True)
    X_test_normalized = X_test / np.linalg.norm(X_test, axis=1, keepdims=True)

    # Compute cosine similarity between X_train and X_test
    cosine_sim = cosine_similarity(X_test_normalized, X_train_normalized)

    return cosine_sim

# Function to augment X_train
def augment_X_train(X_train, X_test, top_k=5):
    # Calculate cosine similarity
    cosine_sim = calculate_cosine_similarity(X_train.values, X_test.values)

    # Initialize an empty array to store the augmented rows
    augmented_rows = []

    # Iterate through each row in X_test
    for i in range(cosine_sim.shape[0]):
        # Get indices of top k similar rows in X_train
        top_indices = np.argsort(cosine_sim[i])[-top_k:][::]

        # Calculate the mean of the top k rows
        mean_row = np.mean(X_train.iloc[top_indices, :].values, axis=0)
        
        augmented_rows.append(mean_row)

    # Append augmented_rows to X_train
    X_train_augmented = np.vstack([X_train.values, augmented_rows])

    # Convert back to DataFrame with original columns
    X_train_augmented = pd.DataFrame(X_train_augmented, columns=X_train.columns)

    return X_train_augmented

def augment_dataset(train_data, test_data, target_instance):
    # Convert to DataFrame if needed
    train_df = pd.DataFrame(train_data["data"])
    test_df = pd.DataFrame(test_data["data"])

    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    imputed_train_data = imputer.fit_transform(train_df)
    imputed_test_data = imputer.transform(test_df)

    # Convert back to DataFrame to ensure compatibility with augment_X_train
    train_df = pd.DataFrame(imputed_train_data, columns=train_df.columns)
    test_df = pd.DataFrame(imputed_test_data, columns=test_df.columns)

    # Ensure no NaN values after imputation
    assert not train_df.isnull().values.any(), "Train DataFrame contains NaN values after imputation"
    assert not test_df.isnull().values.any(), "Test DataFrame contains NaN values after imputation"

    # Augment train data
    augmented_df = augment_X_train(train_df, test_df)

    # Check if augmented DataFrame contains NaN values
    if augmented_df.isnull().values.any():
        print("DataFrame contains NaN values after augmentation")
    else:
        print("DataFrame does not contain NaN values after augmentation")

    if str(target_instance) in augmented_df.columns:
    # for every row in train data [data] change the survived column to 1 if the value is greater than 0.5
        augmented_df[target_instance] = augmented_df[target_instance].apply(lambda x: 1 if x > 0.5 else 0)
        # train_data["data"][target_instance] = train_data["data"][target_instance].apply(lambda x: int(1) if x > 0.5 else int(0))
    train_data["data"] = augmented_df
    return train_data



# def retrieve_similar_data(data, target_instance, n_neighbors=5):
#     # Ensure data is a DataFrame
#     data = pd.DataFrame(data)
#     # Using Nearest Neighbors to find similar instances
#     neighbors = NearestNeighbors(n_neighbors=n_neighbors)
#     neighbors.fit(data)
#     # Finding the nearest neighbors for the target instance
#     distances, indices = neighbors.kneighbors([target_instance])
#     similar_data = data.iloc[indices[0]]
#     return similar_data

# def augment_data_with_retrieval(data, target_instance):
#     augmented_data = data.copy()
#     # if 'survived' in augmented_data.columns:
#     #     augmented_data = augmented_data.drop(columns=['survived'])
#     augmented_rows = []
#     for i in range(len(target_instance)):
#         # if 'survived' in data.columns:
#         #     target_instance = data.drop(columns=['survived']).iloc[i]
#         # else:
#         target_instance = target_instance.iloc[i]
#         similar_data = retrieve_similar_data(augmented_data, target_instance)
#         if 'survived' in similar_data.columns:
#             similar_data = similar_data.drop(columns=['survived'])
#         augmented_features = similar_data.mean()
#         augmented_rows = augmented_rows.append(augmented_features, ignore_index=True)
#     augmented_data.append(augmented_rows, ignore_index=True)
#     if 'survived' in data.columns:
#         augmented_data['survived'] = data['survived']
#     return augmented_data



setup_config = {
    "project_name": "Finetune-TabPFN",
    "results_path": "results/",
    "random_states": [0, 1],
    "k_folds": 5,
    # val_size is percentage w.r.t. the total dataset-rows ]0,1[
    "val_size": 0.2,
    "num_workers": 0,
    "dataset_mapping": {168746: "Titanic", 9982:"Dress-Sales"},
    "log_wandb": False,
    "models": {
        "FineTuneTabPFNClassifier_full_weight": FineTuneTabPFNClassifier,
        "RandomForestClassifier": RandomForestClassifier,
        "DecisionTreeClassifier": DecisionTreeClassifier,
        "TabPFNClassifier": TabPFNClassifier,
    },
    "dataset_augmentations": {"FullRealDataDataset": FullRealDataDataset},
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}
# Create a lookup dictionary which contains the architectural and training
# Hyperparameters of the models

# Step 1: Define the model, criterion, optimizer, device and evaluator
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


visualizer = Visualizer(path=f"{setup_config['results_path']}")


evaluator = Evaluator(visualizer=Visualizer)
trainer = Trainer(visualizer=visualizer)

results_df = None


if os.path.exists(f"{setup_config['results_path']}results_df_rag_100ep1_baseline.pkl"):
    results_df = pd.read_pickle(f"{setup_config['results_path']}results_df_rag_100ep1_baseline.pkl")
else:
    # Step 2: run the evaluation and training loop
    # ---------- ---------- ---------- ---------- ---------- ---------- RANDOM STATES LOOP
    for random_state in setup_config["random_states"]:
        set_seed_globally(random_state)

        # ---------- ---------- ---------- ---------- ----------  DATASET ID LOOP
        for dataset_id, dataset_name in setup_config["dataset_mapping"].items():
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
                    # train_dataset = RealDataDataset(
                    #     data=train_data["data"],
                    #     target=train_data["target"],
                    #     name=train_data["name"],
                    # )

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

                            # depending on the setting we augment the training data via
                            # different methods. Therefore we overwrite the train_dataset
                            # train_dataset = augmentation_fn(
                            #     data=train_data["data"],
                            #     target=train_data["target"],
                            #     name=train_data["name"],
                            train_data = augment_dataset(train_data, test_data, train_data["target"])
                            train_dataset = augmentation_fn(
                                data=train_data["data"],
                                target=train_data["target"],
                                name=train_data["name"],
                            )
                            # train_dataset = RealDataDataset(
                            #     data=train_dataset["data"],
                            #     target=train_dataset["target"],
                            #     name=train_dataset["name"],
                            # )


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
                                    # train_dataset=train_dataset,
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
    results_df.to_pickle(f"{setup_config['results_path']}results_df_rag_100ep1_baseline.pkl")
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
