from __future__ import annotations

import pandas as pd
import torch
from data.CustomDataloader import CustomDataLoader
from data.DataManager import DataManager
from evaluation.model_evaluation import evaluate_accuracy
from gym.Trainer import Trainer
from tabpfn import TabPFNClassifier
from torch import nn
from torch.optim import Adam

# Step 0: Define hyperparameters
n_seeds = [0, 1, 2, 3, 4, 5]
k_folds = 5
# val_size is percentage w.r.t. the total dataset-rows ]0,1[
val_size = 0.25

min_single_eval_pos = 100
epochs = 50
learning_rate = 0.00001
num_workers = 0
dataset_ids = [168746, 23381]
log_wandb = True
project_name = "Finetune-TabPFN"


# Step 1: Define the model, criterion, optimizer, device and trainer
criterion = nn.CrossEntropyLoss()
optimizer = Adam
device = "cuda" if torch.cuda.is_available() else "cpu"

tabpfn_classifier = TabPFNClassifier()
trainer = Trainer(
    project_name,
    tabpfn_classifier,
    learning_rate,
    criterion,
    optimizer,
    log_wandb,
    device,
)


results_df = pd.DataFrame(columns=["random_state", "fold", "pre_eval", "post_eval"])

# Step 2: run the evaluation and training loop
for random_state in n_seeds:
    torch.manual_seed(random_state)  # Set seed for torch RNG
    torch.cuda.manual_seed(random_state)  # Set seed for CUDA RNG
    torch.cuda.manual_seed_all(random_state)  # Set seed for all CUDA devices
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior for cudnn
    # Disable cudnn benchmark for reproducibility
    torch.backends.cudnn.benchmark = False

    for dataset_id in dataset_ids:
        # Step 3: Load  data
        data_manager = DataManager(
            dir_path="data/dataset",
            dataset_id=dataset_id,
        )
        data_k_folded = data_manager.k_fold_train_test_split(
            k_folds=k_folds,
            val_size=val_size,
            random_state=random_state,
        )

        for fold_i, fold in enumerate(data_k_folded):
            train_data_loader = CustomDataLoader(
                fold["train"],
                sequence_length=fold["train"].number_rows,
                min_single_eval_pos=min_single_eval_pos,
                shuffle=True,
                num_workers=num_workers,
            )

            test_data_loader = CustomDataLoader(
                fold["test"],
                sequence_length=fold["test"].number_rows,
                min_single_eval_pos=min_single_eval_pos,
                shuffle=True,
                num_workers=num_workers,
            )

            val_data_loader = CustomDataLoader(
                fold["val"],
                sequence_length=fold["val"].number_rows,
                min_single_eval_pos=min_single_eval_pos,
                shuffle=True,
                num_workers=num_workers,
            )

            # Pre-tuning evaluation
            pre_eval_metrics = evaluate_accuracy(
                val_data_loader,
                tabpfn_classifier,
                device,
            )

            # Fine-tune the model
            tabpfn_classifier = trainer.fine_tune_model(
                train_loader=train_data_loader,
                val_loader=val_data_loader,
                epochs=epochs,
            )

            # Post-tuning evaluation
            post_eval_metrics = evaluate_accuracy(
                val_data_loader,
                tabpfn_classifier,
                device,
            )

            new_rows_df = pd.DataFrame(
                [
                    {
                        "random_state": random_state,
                        "fold": fold_i,
                        "pre_eval": pre_eval_metrics["accuracy"],
                        "post_eval": post_eval_metrics["accuracy"],
                    },
                ],
            )
            results_df = pd.concat([results_df, new_rows_df], ignore_index=True)


# store the results
data_manager.store_results(results_df)
