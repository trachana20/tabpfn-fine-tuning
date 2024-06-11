from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
import torch

from torch.nn import CrossEntropyLoss


# from tabpfn codebase (https://github.com/automl/TabPFN/blob/main/tabpfn/scripts/tabular_metrics.py)
def auc_metric(target, pred, multi_class="ovo", numpy=False):
    lib = np if numpy else torch
    try:
        if not numpy:
            target = torch.tensor(target) if not torch.is_tensor(target) else target
            pred = torch.tensor(pred) if not torch.is_tensor(pred) else pred
        if len(lib.unique(target)) > 2:
            if not numpy:
                return torch.tensor(
                    roc_auc_score(target, pred, multi_class=multi_class),
                )
            return roc_auc_score(target, pred, multi_class=multi_class)
        else:
            if len(pred.shape) == 2:
                pred = pred[:, 1]
            if not numpy:
                return torch.tensor(roc_auc_score(target, pred))
            return roc_auc_score(target, pred)
    except ValueError:
        #  filter later
        return np.nan if numpy else torch.tensor(np.nan)


def classification_performance_metrics(y_preds, y_true):
    # y_preds have shape (sequence_length, classes)
    # y_true has shape (sequence_length)

    metrics_dict = {}

    # Compute accuracy
    accuracy = accuracy_score(y_true, np.argmax(y_preds, axis=1))
    metrics_dict["accuracy"] = accuracy

    # Compute AUC (Area Under Curve)

    auc = auc_metric(target=y_true, pred=y_preds).item()
    metrics_dict["auc"] = auc

    # Compute F1 score
    f1 = f1_score(y_true, np.argmax(y_preds, axis=1), average="weighted")
    metrics_dict["f1"] = f1

    # Compute cross-entropy
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html
    log_loss_value = log_loss(y_pred=y_preds, y_true=y_true)
    metrics_dict["log_loss"] = log_loss_value

    # Return metrics as a dictionary
    return metrics_dict
