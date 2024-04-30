import torch.nn as nn
import torch


class MockTabPFN(nn.Module):
    def __init__(self):
        super(MockTabPFN, self).__init__()
        self.linear = nn.Linear(100, 10, False)

        # Initialize the weights with random values
        torch.nn.init.normal_(self.linear.weight, mean=0, std=0.1)

    def forward(self, X, single_eval_pos):
        out = self.linear(X[1])
        out = out[single_eval_pos:]
        return out
