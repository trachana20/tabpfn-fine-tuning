from __future__ import annotations

import pandas as pd
import torch
from data.CustomDataloader import CustomDataLoader
from data.DataManager import DataManager
from evaluation.model_evaluation import evaluate_accuracy
from gym.Trainer import Trainer
from logger.WandBLogger import WandBLogger
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from tabpfn import TabPFNClassifier
from torch import nn
from torch.optim import Adam

# Step 0: Define hyperparameters which are valid for all models and model
# specific hyperparameters
setup_config = {
    "n_seeds": [0, 1, 2, 3, 4, 5],
    "k_folds": 5,
    # val_size is percentage w.r.t. the total dataset-rows ]0,1[
    "val_size": 0.25,
    "num_workers": 0,
    "dataset_ids": [168746],  # , 23381]
    "log_wandb": False,
    "name": "Finetune-TabPFN",
    "models": [RandomForestClassifier, DecisionTreeClassifier, TabPFNClassifier],
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

modelkwargs_dict = {TabPFNClassifier: {"epochs": 10, "learning_rate": 1e-3}}

# Step 1: Define the model, criterion, optimizer, device and trainer
criterion = nn.CrossEntropyLoss()
optimizer = Adam


wandb_logger = None
if setup_config["log_wandb"]:
    wandb_logger = WandBLogger(name=setup_config["name"])
    wandb_logger._setup_wandb(setup_config=setup_config)

trainer = Trainer(
    name=setup_config["name"],
    logger=wandb_logger,
)


# Step 2: run the evaluation and training loop
for random_state in setup_config["n_seeds"]:
    torch.manual_seed(random_state)  # Set seed for torch RNG
    torch.cuda.manual_seed(random_state)  # Set seed for CUDA RNG
    torch.cuda.manual_seed_all(random_state)  # Set seed for all CUDA devices
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior for cudnn
    # Disable cudnn benchmark for reproducibility
    torch.backends.cudnn.benchmark = False

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
            for model_fn in setup_config["models"]:
                model = model_fn()  # TODO create a modelbuilder which uses modelkwargs

                # Fine-tune the model
                trained_model = trainer.main_train_and_evaluate_model(
                    model=model,
                    train_loader=train_data_loader,
                    val_loader=val_data_loader,
                    random_state=random_state,
                    dataset_id=dataset_id,
                    fold_i=fold_i,
                )
