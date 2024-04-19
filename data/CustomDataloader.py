from torch.utils.data import DataLoader
import torch


class CustomDataLoader(DataLoader):
    def __init__(
        self,
        dataset,
        sequence_length,
        min_single_eval_pos,
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
        self.sequence_length = sequence_length
        self.min_single_eval_pos = min_single_eval_pos

    def __iter__(self):
        for batch in super().__iter__():
            x, y = batch
            # single evaluation position can be at most the length of the sequence
            single_eval_pos = torch.randint(
                self.min_single_eval_pos,
                x.shape[0] - 1,
                size=(1,),
            )
            x_train, y_train = x[:single_eval_pos], y[:single_eval_pos]
            x_query, y_query = x[single_eval_pos:], y[single_eval_pos:]
            yield (x_train, y_train, x_query, y_query)
