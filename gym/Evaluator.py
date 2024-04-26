from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from tabpfn import TabPFNClassifier
from torch import nn
import time
from evaluation.model_evaluation import compute_classification_performance_metrics

if TYPE_CHECKING:
    from data.CustomDataloader import CustomDataLoader


class Evaluator:
    def __init__(
        self,
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
        if self.logger is not None:
            self.logger.register_incumbent(
                random_state=random_state,
                dataset_id=dataset_id,
                fold_i=fold_i,
                model=model,
            )

        model, performance_metrics = self.train_sklearn_model(
            model,
            train_loader,
            val_loader,
            **model_kwargs,
        )

        if self.logger is not None:
            step = 0
            self.logger.update_traing_metrics(
                performance_metrics=performance_metrics,
                step=step,
            )

    def train_sklearn_model(self, model, train_loader, val_loader, **training_kwargs):
        for _, (x, y) in enumerate(train_loader):
            start_time = time.time()
            model.fit(X=x, y=y)
            fitting_time = time.time() - start_time
        for _, (x_val, y_val) in enumerate(val_loader):
            start_time = time.time()
            y_preds = model.predict_proba(X=x_val)
            prediction_time = time.time() - start_time

        performance_metrics = compute_classification_performance_metrics(
            y_preds=y_preds,
            y_true=y_val,
        )
        performance_metrics["time_fit"] = fitting_time
        performance_metrics["time_predict"] = prediction_time

        return model, performance_metrics

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
