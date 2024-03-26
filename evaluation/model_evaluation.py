import torch
from sklearn.metrics import accuracy_score
import torch.nn as nn

import numpy as np


def evaluate_accuracy(dataloader, model, device):
    with torch.no_grad():
        accuracies_over_batches = []
        for batch_i, (x_train, y_train, x_query, y_query) in enumerate(
            dataloader,
        ):
            model.fit(x_train, y_train)
            preds = model.predict(x_query)
            if len(x_query) > 10:
                # only if we have at least 10 elements in the query set
                # we can calculate the accuracy
                accuracy = accuracy_score(y_query, preds)
                accuracies_over_batches.append(accuracy)
    results = {"accuracy": np.mean(accuracies_over_batches)}
    return results
