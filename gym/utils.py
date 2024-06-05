import numpy as np


def update_epoch_metrics_with_batch_metrics(epoch_metrics, batch_metrics):
    for key, value in batch_metrics.items():
        if key not in epoch_metrics:
            epoch_metrics[key] = {"summed_value": 0, "counts": 0}
        if np.isnan(value):
            # for the case that the performance metrics (e.g AUC if
            # classes in y_preds are not represented in y_true,...) computation
            # ran into an error we need to keep track how many
            # values we aggregate over.
            continue

        epoch_metrics[key]["summed_value"] += value
        epoch_metrics[key]["counts"] += 1

    # we do not necessarily need to return the epoch metrics
    # dict, as it only ever one instance, without copying
    # but this style makes unwanted sideeffects less likely
    return epoch_metrics


def average_epoch_metrics(epoch_metrics):
    averaged_metrics = {}

    for key, values_counts in epoch_metrics.items():
        if key not in averaged_metrics:
            # not necessary but cleaner
            # usually we iterate over each metric only once
            averaged_metrics[key] = 0
        averaged_metrics[key] = values_counts["summed_value"] / values_counts["counts"]

    return averaged_metrics


def to_numpy(tensor):
    # !only use this function after the parameters have been updated.
    # This function will break the computational graph and convert the
    # tensor to a numpy array for further computation.
    return tensor.cpu().detach().numpy()
