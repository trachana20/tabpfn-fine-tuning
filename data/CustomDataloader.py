from torch.utils.data import DataLoader
import torch


class CustomDataLoader(DataLoader):
    def __init__(
        self,
        dataset,
        sequence_length,
        shuffle,
        num_workers,
    ):
        super().__init__(
            dataset,
            # Double the batch size to accommodate both train and query data
            batch_size=sequence_length,
            shuffle=shuffle,
            num_workers=num_workers,
        )

    def __iter__(self):
        for batch in super().__iter__():
            x, y = batch

            yield (x, y)
