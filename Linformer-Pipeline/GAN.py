import torch
import torch.nn as nn
import torch.optim as optim

class GAN:
    def __init__(self, input_dim, output_dim, device):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device

        self.generator = self.build_generator().to(self.device)
        self.discriminator = self.build_discriminator().to(self.device)

        self.criterion = nn.BCELoss()
        self.optimizer_g = optim.Adam(self.generator.parameters(), lr=0.0002)
        self.optimizer_d = optim.Adam(self.discriminator.parameters(), lr=0.0002)

    def build_generator(self):
        return nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.output_dim),
            nn.Tanh()
        )

    def build_discriminator(self):
        return nn.Sequential(
            nn.Linear(self.output_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def train(self, data_loader, epochs=10000, batch_size=128):
        
        for epoch in range(epochs):
            for real_data in data_loader:
                real_data_input = real_data[0].to(self.device)
                current_batch_size = real_data_input.size(0)

                # Train Discriminator
                self.optimizer_d.zero_grad()

                # Train on real data
                real_labels = torch.ones((current_batch_size, 1)).to(self.device)
                real_output = self.discriminator(real_data_input)
                d_loss_real = self.criterion(real_output, real_labels)

                # Train on fake data
                noise = torch.randn((current_batch_size, self.input_dim)).to(self.device)
                fake_data = self.generator(noise)
                fake_labels = torch.zeros((current_batch_size, 1)).to(self.device)
                fake_output = self.discriminator(fake_data.detach())
                d_loss_fake = self.criterion(fake_output, fake_labels)

                d_loss = (d_loss_real + d_loss_fake) / 2
                d_loss.backward()
                self.optimizer_d.step()

                # Train Generator
                self.optimizer_g.zero_grad()

                valid_labels = torch.ones((current_batch_size, 1)).to(self.device)
                fake_output = self.discriminator(fake_data)
                g_loss = self.criterion(fake_output, valid_labels)

                g_loss.backward()
                self.optimizer_g.step()
                print(f"Epoch {epoch}/{epochs} [D loss: {d_loss.item()} | G loss: {g_loss.item()}]")