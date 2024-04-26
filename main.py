from __future__ import annotations

import torch
from data.CustomDataloader import CustomDataLoader
from data.DataManager import DataManager
from gym.Evaluator import Evaluator
from logger.Logger import Logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from tabpfn import TabPFNClassifier
from torch import nn
from torch.optim import Adam
from utils import set_seed_globally

# Step 0: Define hyperparameters which are valid for all models and model
# specific hyperparameters

setup_config = {
    "random_states": [0, 1],
    "k_folds": 5,
    # val_size is percentage w.r.t. the total dataset-rows ]0,1[
    "val_size": 0.25,
    "num_workers": 0,
    "dataset_ids": [168746],  # , 23381]
    "log_wandb": True,
    "project_name": "Finetune-TabPFN",
    "models": [RandomForestClassifier, DecisionTreeClassifier, TabPFNClassifier],
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}
# Create a lookup dictionary which contains the architectural and training
# Hyperparameters of the models

# Step 1: Define the model, criterion, optimizer, device and evaluator
modelkwargs_dict = {
    TabPFNClassifier: {
        "architecture": {},
        "training": {
            "epochs": 10,
            "learning_rate": 1e-3,
            "criterion": nn.CrossEntropyLoss,
            "optimizer": Adam,
        },
    },
}


logger = Logger(
    project_name=setup_config["project_name"],
    log_wandb=setup_config["log_wandb"],
    results_path="results/",
)
logger.setup_wandb(setup_config=setup_config)

evaluator = Evaluator(logger=logger if setup_config["log_wandb"] else None)


# Step 2: run the evaluation and training loop
# ---------- ---------- ---------- ---------- ---------- ---------- RANDOM STATES LOOP
for random_state in setup_config["random_states"]:
    set_seed_globally(random_state)

    for dataset_id in setup_config["dataset_ids"]:
        # ---------- ---------- ---------- ---------- ----------  DATASET ID LOOP
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
            train_data_loader = CustomDataLoader(
                fold["train"],
                sequence_length=fold["train"].number_rows,
                shuffle=True,
                num_workers=setup_config["num_workers"],
            )

            test_data_loader = CustomDataLoader(
                fold["test"],
                sequence_length=fold["test"].number_rows,
                shuffle=True,
                num_workers=setup_config["num_workers"],
            )

            val_data_loader = CustomDataLoader(
                fold["val"],
                sequence_length=fold["val"].number_rows,
                shuffle=True,
                num_workers=setup_config["num_workers"],
            )

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

                model = model_fn()  # TODO create a modelbuilder which uses modelkwargs
                # Fine-tune the model
                trained_model = evaluator.main_train_and_evaluate_model(
                    model=model,
                    train_loader=train_data_loader,
                    val_loader=val_data_loader,
                    random_state=random_state,
                    dataset_id=dataset_id,
                    fold_i=fold_i,
                    **model_training_kwargs,
                )

logger.save_results()
