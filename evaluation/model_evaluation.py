import torch
from sklearn.metrics import accuracy_score
import torch.nn as nn

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.metrics import log_loss
import numpy as np


def classification_performance_metrics(y_preds, y_true):
    # y_preds have shape (sequence_length, classes)
    # y_true has shape (sequence_length)

    # Compute accuracy
    accuracy = accuracy_score(y_true, np.argmax(y_preds, axis=1))

    # Compute AUC (Area Under Curve)
    auc = roc_auc_score(y_true, y_preds[:, 1]) if y_preds.shape[1] > 1 else None

    # Compute F1 score
    f1 = f1_score(y_true, np.argmax(y_preds, axis=1), average="weighted")

    # Compute cross-entropy
    cross_entropy = log_loss(y_true, y_preds)

    # Return metrics as a dictionary
    return {"accuracy": accuracy, "auc": auc, "f1": f1, "cross_entropy": cross_entropy}
