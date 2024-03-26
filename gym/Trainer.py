from torch.nn import Module
from data.CustomDataloader import CustomDataLoader
from tabpfn import TabPFNClassifier
import torch
import torch.nn as nn


class Trainer:
    def __init__(
        self,
        tabpfn_classifier: TabPFNClassifier,
        learning_rate: float,
        criterion,
        optimizer,
        device,
    ):
        self.tabpfn_classifier = tabpfn_classifier

        self.learning_rate = learning_rate
        self.criterion = criterion
        self.optimizer = optimizer(
            self.tabpfn_classifier.model[2].parameters(),
            lr=self.learning_rate,
        )
        self.device = device

    def fine_tune_model(
        self,
        train_loader: CustomDataLoader,
        val_loader: CustomDataLoader,
        epochs: int,
    ):
        tabpfn_model = self.tabpfn_classifier.model[2]
        tabpfn_model.train()

        tabpfn_model.to(self.device)
        tabpfn_model.train()

        for _ in range(epochs):
            for _, (x_train, y_train, x_query, y_query) in enumerate(
                train_loader,
            ):
                self.optimizer.zero_grad()

                # x_data shape: sequence_length, num_features
                #  -> sequence_length, batch_size=1, num_features
                x_data = (
                    torch.cat([x_train, x_query], dim=0).unsqueeze(1).to(self.device)
                )

                # prepare x_data with padding with zeros up to 100 features
                x_data = nn.functional.pad(
                    x_data,
                    (0, 100 - x_data.shape[-1]),
                ).float()

                # y_train shape: sequence_length
                #  -> sequence_length, batch_size=1
                y_data = y_train.unsqueeze(1).to(self.device).float()

                output = tabpfn_model((x_data, y_data), single_eval_pos=len(x_train))

                loss = self.criterion(output.reshape(-1, 10), y_query.long().flatten())

                loss.backward()
                self.optimizer.step()
        tabpfn_model.eval()
        return self.tabpfn_classifier
