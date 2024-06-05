from __future__ import annotations

import numpy as np
from torch.utils.data import Dataset


class FullRealDataDataset(Dataset):
    def __init__(self, data, target, name):
        self.name = name
        self.number_rows = len(data)
        self.number_features = len(data.columns) - 1

        self.features = data.drop(target, axis=1).to_numpy()
        self.labels = data[target].values
        self.num_classes = np.unique(self.labels).shape[0]

    def __len__(self):
        # as we return the complete dataset, we
        # will only iterate once over the dataloader
        return 1

    def __getitem__(self, idx):
        # ignore idx
        # we want to shuffle the dataset and return all rows
        # and all features for finetuning
        perm_indices = np.random.permutation(self.number_rows)
        return self.features[perm_indices], self.labels[perm_indices]
