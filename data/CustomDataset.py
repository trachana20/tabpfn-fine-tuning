from __future__ import annotations

import numpy as np
from torch.utils.data import Dataset


class RealDataDataset(Dataset):
    def __init__(self, data, target, name):
        self.name = name
        self.number_rows = len(data)
        self.number_features = len(data.columns) - 1

        self.features = data.drop(target, axis=1).to_numpy()
        self.labels = data[target].values
        self.num_classes = np.unique(self.labels).shape[0]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # ignore idx
        # we want to shuffle the dataset
        perm_indices = np.random.permutation(self.number_rows)
        return self.features[perm_indices], self.labels[perm_indices]
