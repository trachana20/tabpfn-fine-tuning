import torch
from sklearn.metrics import accuracy_score
import torch.nn as nn

import numpy as np


def evaluate_accuracy(dataloader, model):
    accuracies_over_batches = []
    for batch_i, (x, y) in enumerate(dataloader):
        single_eval_pos = 100

        # cast tensors to numpy arrays to suite the sklearn interface
        x_train = x[:single_eval_pos].numpy()
        y_train = y[:single_eval_pos].numpy()

        x_query = x[single_eval_pos:].numpy()
        y_query = y[single_eval_pos:].numpy()

        model.fit(x_train, y_train)
        preds = model.predict_proba(x_query)
        if x_query.shape[0] > 10:  # TODO Set this to a reasonable value
            # only if we have at least 10 elements in the query set
            # we can calculate the accuracy
            accuracy = accuracy_score(y_query, preds)
            accuracies_over_batches.append(accuracy)
    results = {"accuracy": np.mean(accuracies_over_batches)}
    return results
