import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, data, target):
        self.features = data.drop(target, axis=1).to_numpy()
        self.labels = data[target].values

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
