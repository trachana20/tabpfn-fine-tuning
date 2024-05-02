import torch.nn as nn
import torch


class MockTabPFN(nn.Module):
    def __init__(self, tabpfn_classifier):
        super(MockTabPFN, self).__init__()
        self.linear = nn.Linear(100, 10, False)
        # Initialize the weights with random values
        torch.nn.init.normal_(self.linear.weight, mean=0, std=10)

        # store the weight in model_weights/MockTabPFN.pth
        weights_path = "model_weights/MockTabPFN.pth"
        self.save_weights(weights_path)

    def forward(self, X, single_eval_pos):
        out = self.linear(X[1])
        out = out[single_eval_pos:]
        return out

    def save_weights(self, file_path):
        # Ensure the path exists
        torch.save(self.linear.weight, file_path)
