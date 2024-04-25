from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from torch import nn

from collections import defaultdict
import wandb

if TYPE_CHECKING:
    from data.CustomDataloader import CustomDataLoader
    from tabpfn import TabPFNClassifier


class Trainer:
    def __init__(
        self,
        name: str,
        logger=None,
    ):
        self.logger = logger

    def main_train_and_evaluate_model(
        self,
        model,
        train_loader,
        val_loader,
        random_state,
        dataset_id,
        fold_i,
        **model_kwargs,
    ):
        if isinstance(model, TabPFNClassifier):
            return self.fine_tune_tabpfn_model(
                model,
                train_loader,
                val_loader,
                **model_kwargs,
            )
        else:
            return self.train_sklearn_model(
                model,
                train_loader,
                val_loader,
                **model_kwargs,
            )

    def train_sklearn_model(self, model, train_loader, val_loader, **model_kwargs):
        print("")

    def fine_tune_tabpfn_model(
        self,
        train_loader: CustomDataLoader,
        val_loader: CustomDataLoader,
        epochs: int,
    ):
        tabpfn_model = self.tabpfn_classifier.model[2]

        tabpfn_model.train()
        tabpfn_model.to(self.device)

        num_classes = train_loader.dataset.num_classes

        for _epoch_i in range(epochs):
            epoch_loss = 0.0  # initialize epoch loss
            num_batches = len(train_loader)
            for _batch_i, (x_train, y_train, x_query, y_query) in enumerate(
                train_loader,
            ):
                self.optimizer.zero_grad()

                # x_data shape: sequence_length, num_features
                #  -> sequence_length, batch_size=1, num_features
                x_data = (
                    torch.cat([x_train, x_query], dim=0).unsqueeze(1).to(self.device)
                )

                # prepare x_data with padding with zeros up to 100 features
                x_data = nn.functional.pad(
                    x_data,
                    (0, 100 - x_data.shape[-1]),
                ).float()

                # y_train shape: sequence_length
                #  -> sequence_length, batch_size=1
                y_data = y_train.unsqueeze(1).to(self.device).float()

                y_preds = tabpfn_model(
                    (x_data, y_data),
                    single_eval_pos=len(x_train),
                ).reshape(-1, 10)[:, :num_classes]

                y_query = y_query.long().flatten()
                loss = self.criterion(y_preds, y_query)

                epoch_loss += loss.item()

                loss.backward()
                self.optimizer.step()

                # Print batch progress
            training_metrics = {"epoch_loss": epoch_loss / num_batches}
            # Call validation function after each epoch
            with torch.no_grad():
                evaluation_metrics = self.model_validation(val_loader, tabpfn_model)

            metrics = {**training_metrics, **evaluation_metrics}
            if self.logger is not None:
                # log metrics to wandb
                self.logger.log(metrics, step=_epoch_i)

        tabpfn_model.eval()
        return self.tabpfn_classifier

    def model_validation(self, val_loader, tabpfn_model):
        metrics = []

        with torch.no_grad():
            for batch_i, (
                x_train_val,
                y_train_val,
                x_query_val,
                y_query_val,
            ) in enumerate(val_loader):
                self.tabpfn_classifier.model = (None, None, tabpfn_model)

                self.tabpfn_classifier.fit(x_train_val, y_train_val)
                y_preds = self.tabpfn_classifier.predict_proba(x_query_val)

                batch_metrics = self.compute_metrics(
                    y_query_val,
                    y_preds,
                    val_loader.dataset.name,
                )
                metrics.append(batch_metrics)

            # if needed aggregate over batches
            if len(metrics):
                metrics_temp = defaultdict(float)
                for metric in metrics:
                    for metric_key, metric_value in metric.items():
                        metrics_temp[metric_key] += metric_value

                metrics = dict(metrics_temp)
        return metrics

    def compute_metrics(self, y_true, y_preds_probs, dataset_name):
        metrics = {}

        y_preds_argmax = y_preds_probs.argmax(axis=-1)
        y_true_shape = np.unique(y_true).shape[0]
        prediction_shape = np.unique(y_preds_argmax).shape[0]

        # Accuracy
        accuracy = accuracy_score(y_true, y_preds_argmax)
        metrics[f"{dataset_name}/accuracy"] = accuracy

        is_binary = y_preds_probs.shape[-1] == 2

        if y_true_shape == prediction_shape:
            try:
                if is_binary:
                    auc = roc_auc_score(y_true, y_preds_argmax)
                else:
                    auc = roc_auc_score(
                        y_true,
                        y_preds_probs,
                        multi_class="ovr",
                        average="macro",
                    )
                metrics[f"{dataset_name}/auc"] = auc
            except ValueError as e:
                print(f"error: {e}")
                print(f"y_preds unique classes: {y_preds_probs.shape[-1]}")
                print(f"labels unique classes: {np.unique(y_true).shape[0]}")
        return metrics
