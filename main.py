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
test_size = 0.0
val_size = 0.5

sequence_length = 150
min_single_eval_pos = 100
epochs = 10
# learning_rate = 0.00001
learning_rate = 0.01

num_workers = 0

# Step 1: Load  data
data_manager = DataManager(
    dir_path="data/dataset",
    dataset_id=168746,
)


# Step 2: Define the model, criterion, optimizer, device and trainer
criterion = nn.CrossEntropyLoss()
optimizer = Adam
device = "cuda" if torch.cuda.is_available() else "cpu"

tabpfn_classifier = TabPFNClassifier()
trainer = Trainer(tabpfn_classifier, learning_rate, criterion, optimizer, device)


results_df = pd.DataFrame(columns=["random_state", "fold", "pre_eval", "post_eval"])

# Step 3: run the evaluation and training loop
for random_state in n_seeds:
    torch.manual_seed(random_state)  # Set seed for torch RNG
    torch.cuda.manual_seed(random_state)  # Set seed for CUDA RNG
    torch.cuda.manual_seed_all(random_state)  # Set seed for all CUDA devices
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior for cudnn
    # Disable cudnn benchmark for reproducibility
    torch.backends.cudnn.benchmark = False

    data_k_folded = data_manager.k_fold_train_test_split(
        k_folds=k_folds,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
    )

    for fold_i, fold in enumerate(data_k_folded):
        train_data_loader = CustomDataLoader(
            fold["train"],
            sequence_length=sequence_length,
            min_single_eval_pos=min_single_eval_pos,
            shuffle=True,
            num_workers=num_workers,
        )

        test_data_loader = CustomDataLoader(
            fold["test"],
            sequence_length=sequence_length,
            min_single_eval_pos=min_single_eval_pos,
            shuffle=True,
            num_workers=num_workers,
        )

        val_data_loader = CustomDataLoader(
            fold["val"],
            sequence_length=sequence_length,
            min_single_eval_pos=min_single_eval_pos,
            shuffle=True,
            num_workers=num_workers,
        )

        # Pre-tuning evaluation
        pre_eval_metrics = evaluate_accuracy(val_data_loader, tabpfn_classifier, device)

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
