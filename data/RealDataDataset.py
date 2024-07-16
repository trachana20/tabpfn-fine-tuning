# from __future__ import annotations

# import numpy as np
# from torch.utils.data import Dataset
# import pandas as pd

# class RealDataDataset(Dataset):
#     def __init__(self, data, target, name):
#         if isinstance(data, np.ndarray):
#             data = pd.DataFrame(data)
#         self.name = name
#         self.number_rows = len(data)
#         self.number_features = len(data.columns) - 1

#         self.features = data.drop(target, axis=1).to_numpy()
#         self.labels = data[target].values
#         self.num_classes = np.unique(self.labels).shape[0]

#     def __len__(self):
#         return self.number_rows

#     def __getitem__(self, idx):
#         return self.features[idx], self.labels[idx]


import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class RealDataDataset(Dataset):
    def __init__(self, data, target, name):
        self.name = name

        # Ensure the data is a DataFrame
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"Expected data to be a DataFrame, but got {type(data)}")
        
        # Ensure the target column is present
        if target not in data.columns:
            raise KeyError(f"Target column '{target}' not found in data columns: {data.columns.tolist()}")

        self.number_rows = len(data)
        self.number_features = len(data.columns) - 1

        self.features = data.drop(target, axis=1)
        self.labels = data[target]
        self.num_classes = np.unique(self.labels).shape[0]

    def __len__(self):
        return self.number_rows

    def __getitem__(self, idx):
        return self.features.iloc[idx].to_numpy(), self.labels.iloc[idx]


    def _collate_fn(self, batch):
        xs = torch.cat(
            [torch.from_numpy(array[0]).unsqueeze(0) for array in batch], dim=0
        ).unsqueeze(0)
        ys = torch.Tensor([array[1] for array in batch]).unsqueeze(0)