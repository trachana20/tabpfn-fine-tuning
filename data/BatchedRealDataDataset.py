from __future__ import annotations

import numpy as np
from torch.utils.data import Dataset, IterableDataset
import torch


class BatchedRealDataDataset(Dataset):
    def __init__(self, data, target, name):
        self.name = name
        self.number_rows = len(data)
        self.number_features = len(data.columns) - 1

        self.features = data.drop(target, axis=1).to_numpy()
        self.labels = data[target].values
        self.num_classes = np.unique(self.labels).shape[0]

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

    def _collate_fn(self, batch):
        xs = torch.cat(
            [torch.from_numpy(array[0]).unsqueeze(0) for array in batch], dim=0
        ).unsqueeze(0)
        ys = torch.Tensor([array[1] for array in batch]).unsqueeze(0)

        return xs, ys

    def __len__(self):
        return self.number_rows
