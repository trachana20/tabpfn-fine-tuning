import torch
from sklearn.metrics import accuracy_score
import torch.nn as nn


def evaluate_accuracy(dataloader, model, device):
    with torch.no_grad():
        results = {"accuracy": []}
        for batch_i, (x_train, y_train, x_query, y_query) in enumerate(
            dataloader,
        ):
            model.fit(x_train, y_train)
            preds = model.predict(x_query)

            accuracy = accuracy_score(y_query, preds)
            results["accuracy"].append(accuracy)

    return results
