from __future__ import annotations

from torch.utils.data import DataLoader


class CustomDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle, num_workers):
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )
