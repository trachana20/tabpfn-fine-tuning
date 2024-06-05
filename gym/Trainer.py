from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from evaluation.model_evaluation import classification_performance_metrics
from gym.utils import (
    to_numpy,
    update_epoch_metrics_with_batch_metrics,
    average_epoch_metrics,
)
from models.FineTuneTabPFNClassifier import FineTuneTabPFNClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from torch import nn


class Trainer:
    def __init__(
        self,
        logger=None,
    ):
        self.logger = logger

    def fine_tune_model(
        self,
        train_loader,
        val_loader,
        fine_tune_type,
        device,
        **kwargs,
    ):
        # This function materializes/instantiates the tabpfn classifier
        # either an existing model is loaded or a tabpfn is finetuned
        weights_path = kwargs.get("architectural", {}).get("weights_path", None)

        tabpfn_classifier = kwargs.get("architectural", {}).get(
            "tabpfn_classifier",
            None,
        )

        # 1. check if weights_path exists and if so load the fine_tune model
        if Path(weights_path).exists():
            return FineTuneTabPFNClassifier(
                tabpfn_classifier=tabpfn_classifier,
                weights_path=weights_path,
            )

        # 2. if weights_path does not exist, fine_tune the model
        # call the correct fine tuning function
        if fine_tune_type == "full_weight_fine_tuning":
            return self.full_weight_fine_tuning(
                tabpfn_classifier=tabpfn_classifier,
                train_loader=train_loader,
                val_loader=val_loader,
                weights_path=weights_path,
                device=device,
                **kwargs,
            )
        else:
            raise ValueError(f"Fine tune type {fine_tune_type} not supported")

    def _store_model_weights(self, weights, weights_path):
        torch.save(weights, weights_path)

    def full_weight_fine_tuning(
        self,
        tabpfn_classifier,
        train_loader,
        val_loader,
        weights_path,
        architectural,
        training,
        device,
    ):
        tabpfn_model = tabpfn_classifier.model[2]

        criterion = training["criterion"]()

        optimizer = training["optimizer"](
            params=tabpfn_model.parameters(),
            lr=training["learning_rate"],
        )
        tabpfn_model.train()
        tabpfn_model.to(device)

        num_classes = train_loader.dataset.num_classes

        for _epoch_i in range(training["epochs"]):
            epoch_metrics = {}
            for _batch_i, (x, y) in enumerate(train_loader):
                optimizer.zero_grad()
                x = x.transpose(0, 1)  # batch_first = False
                y = y.transpose(0, 1)  # batch_first = False

                single_eval_pos = int(x.shape[0] * 0.8)  # 80% for training 20% query
                y_train = y[:single_eval_pos]
                y_query = y[single_eval_pos:]

                # prepare x_data with padding with zeros up to 100 features
                x_data = nn.functional.pad(x, (0, 100 - x.shape[-1])).float()

                # y_train shape: sequence_length
                #  -> sequence_length, batch_size=1
                y_data = y_train.unsqueeze(1).to(device).float()

                y_preds = tabpfn_model(
                    (x_data, y_data),
                    single_eval_pos=single_eval_pos,
                ).reshape(-1, 10)[:, :num_classes]

                # y_preds are raw logits -> normalize with softmax
                # TODO find out what softmax temperature TabPFN uses
                y_preds = y_preds.softmax(dim=-1)

                y_query = y_query.long().flatten()
                loss = criterion(y_preds, y_query)

                # compute gradients and store in nn.Parameters
                loss.backward()
                # update weights with stored gradients
                optimizer.step()

                # compute batch-specific performance metrics
                batch_metrics = classification_performance_metrics(
                    y_preds=to_numpy(y_preds),
                    y_true=to_numpy(y_query),
                )
                # enrich batch metrics with additional information
                batch_metrics["train_loss"] = loss.item()
                batch_metrics["num_classes"] = y_preds.shape[-1]
                batch_metrics["num_features"] = x_data.shape[-1]
                batch_metrics["sequence_length"] = x.shape[0]

                # aggregate metrics over multiple batches
                epoch_metrics = update_epoch_metrics_with_batch_metrics(
                    epoch_metrics=epoch_metrics,
                    batch_metrics=batch_metrics,
                )

            # aggregate over batches and visualize metrics
            averaged_epoch_metrics = average_epoch_metrics(epoch_metrics=epoch_metrics)

            # TODO vall visualization

            # Call validation function after each epoch
            with torch.no_grad():
                self.model_validation(
                    tabpfn_classifier=tabpfn_classifier,
                    tabpfn_model=tabpfn_model,
                    val_loader=val_loader,
                )

            # metrics = {**training_metrics, **evaluation_metrics}
            # if self.logger is not None:
            #     # log metrics to wandb
            #     self.logger.update_traing_metrics(metrics, step=_epoch_i)

        self._store_model_weights(tabpfn_model.state_dict(), weights_path)
        tabpfn_model.eval()
        return tabpfn_classifier

    def model_validation(self, tabpfn_classifier, tabpfn_model, val_loader):
        metrics = []

        with torch.no_grad():
            for _batch_i, (x, y) in enumerate(val_loader):
                tabpfn_classifier.model = (None, None, tabpfn_model)

                single_eval_pos = x.shape[0] // 2
                x_train_val = x[:single_eval_pos]
                y_train_val = y[:single_eval_pos]
                x_query_val = x[single_eval_pos:]
                y_query_val = y[single_eval_pos:]

                tabpfn_classifier.fit(x_train_val, y_train_val)
                y_preds = tabpfn_classifier.predict_proba(x_query_val)

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
            except ValueError:
                pass
        return metrics
