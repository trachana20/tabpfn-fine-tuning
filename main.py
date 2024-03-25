from __future__ import annotations

import numpy as np
import torch
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNClassifier
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


classifier = TabPFNClassifier(device="cpu", N_ensemble_configurations=32)

classifier.fit(X_train, y_train)
y_eval, p_eval = classifier.predict(X_test, return_winning_probability=True)

print('Accuracy', accuracy_score(y_test, y_eval))


tabpfn_model = classifier.model[2]


# Step 4: Set up your optimizer and loss function
optimizer = Adam(tabpfn_model.parameters(), lr=0.01)  # Define your optimizer
criterion = nn.CrossEntropyLoss()  # Define your loss function


# Define the size of your dataset

num_features = 100
num_classes = 10

batch_size = 16
sequence_length = 500

# Create random data with a standard Gaussian distribution for features
x = np.random.randn( batch_size,sequence_length, num_features).astype(np.float32)

# Create random class labels
y = np.random.randint(num_classes, size=(batch_size,sequence_length))

# Convert numpy arrays to PyTorch tensors
x = torch.tensor(x)
y = torch.tensor(y).float()

# Define a custom Dataset class
class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Create an instance of your custom Dataset
dataset = CustomDataset(x, y)


# Create a DataLoader
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Step 5: Continue training the model
for _epoch in range(5):
    single_eval_pos = 400
    for _batch_idx, (data, target) in enumerate(train_loader):
        x_data = data.transpose(0,1)
        y_train = target[:single_eval_pos].transpose(0,1)
        y_test = target[:,single_eval_pos:]

        optimizer.zero_grad()

        output = tabpfn_model((x_data,y_train),single_eval_pos=single_eval_pos)
        output = output.reshape(-1, 10)

        y_test = y_test.long().flatten()

        loss = criterion(output, y_test)


        loss.backward()
        optimizer.step()


classifier.model = (0,0,tabpfn_model)# not sure if this is necessary: 


X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

classifier.fit(X_train, y_train)
y_eval, p_eval = classifier.predict(X_test, return_winning_probability=True)

print('Accuracy', accuracy_score(y_test, y_eval))


    # # Optionally, you might want to save checkpoints periodically
    # if epoch % 10 == 0:  # Save checkpoint every 10 epochs
    #     torch.save({
    #         "epoch": epoch,
    #         "model_state_dict": model.state_dict(),
    #         "optimizer_state_dict": optimizer.state_dict(),
    #         "loss": loss,
    #         # Add other relevant information you might want to save
    #     }, f"checkpoint_epoch_{epoch}.ckpt")