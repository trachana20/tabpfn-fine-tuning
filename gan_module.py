import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class GAN:
    def __init__(self, input_dim, condition_dim, output_dim, device):
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.output_dim = output_dim
        self.device = device

        self.generator = self.build_generator().to(self.device)
        self.discriminator = self.build_discriminator().to(self.device)

        self.criterion = nn.BCELoss()
        self.optimizer_g = optim.Adam(self.generator.parameters(), lr=0.0002)
        self.optimizer_d = optim.Adam(self.discriminator.parameters(), lr=0.0002)

    def build_generator(self):
        return nn.Sequential(
            nn.Linear(self.input_dim + self.condition_dim, 128),
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

    def train(self, data, epochs=10000, batch_size=128):
        half_batch = batch_size // 2
        data_loader = DataLoader(TensorDataset(torch.tensor(data, dtype=torch.float32)), batch_size=half_batch,
                                 shuffle=True)

        for epoch in range(epochs):
            for real_data in data_loader:
                real_data = real_data[0].to(self.device)
                batch_size = real_data.size(0)

                # Train Discriminator
                self.optimizer_d.zero_grad()

                # Train on real data
                real_labels = torch.ones((batch_size, 1)).to(self.device)
                real_output = self.discriminator(real_data)
                d_loss_real = self.criterion(real_output, real_labels)

                # Train on fake data
                noise = torch.randn((batch_size, self.input_dim)).to(self.device)
                conditions = real_data[:, -self.condition_dim:]
                gen_input = torch.cat((noise, conditions), dim=1)
                fake_data = self.generator(gen_input)
                fake_labels = torch.zeros((batch_size, 1)).to(self.device)
                fake_output = self.discriminator(fake_data.detach())
                d_loss_fake = self.criterion(fake_output, fake_labels)

                d_loss = (d_loss_real + d_loss_fake) / 2
                d_loss.backward()
                self.optimizer_d.step()

                # Train Generator
                self.optimizer_g.zero_grad()

                valid_labels = torch.ones((batch_size, 1)).to(self.device)
                fake_output = self.discriminator(fake_data)
                g_loss = self.criterion(fake_output, valid_labels)

                g_loss.backward()
                self.optimizer_g.step()

            if epoch % 1000 == 0:
                print(f"Epoch {epoch}/{epochs} [D loss: {d_loss.item()} | G loss: {g_loss.item()}]")

    def generate_synthetic_data(self, num_samples, scaler, encoder, continuous_features, categorical_features,
                                encoded_cat_columns):
        noise = torch.randn((num_samples, self.input_dim)).to(self.device)
        conditions = torch.randn((num_samples, self.condition_dim)).to(self.device)  # Modify based on actual conditions
        gen_input = torch.cat((noise, conditions), dim=1)
        synthetic_data = self.generator(gen_input).cpu().detach().numpy()

        synthetic_continuous = synthetic_data[:, :len(continuous_features)]
        if continuous_features:
            synthetic_continuous = scaler.inverse_transform(synthetic_continuous)

        synthetic_df = pd.DataFrame(synthetic_data, columns=continuous_features + list(encoded_cat_columns))
        if continuous_features:
            synthetic_df[continuous_features] = synthetic_continuous

        for feature in continuous_features:
            synthetic_df[feature] = synthetic_df[feature].round()

        for feature in categorical_features:
            encoded_columns = [col for col in synthetic_df.columns if feature in col]
            synthetic_df[feature] = synthetic_df[encoded_columns].idxmax(axis=1).apply(lambda x: x.split('_')[-1])
            synthetic_df = synthetic_df.drop(columns=encoded_columns)

        return synthetic_df

    def create_synthetic_data(df, categorical_indicator, input_dim=100, epochs=10000, batch_size=128, num_samples=1000):
        data, scaler, encoder, continuous_features, categorical_features, encoded_cat_columns = load_and_preprocess_data(
            df, categorical_indicator)
        output_dim = data.shape[1]
        condition_dim = len(encoded_cat_columns)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gan = GAN(input_dim, condition_dim, output_dim, device)
        gan.train(data, epochs, batch_size)

        synthetic_df = gan.generate_synthetic_data(num_samples, scaler, encoder, continuous_features,
                                                   categorical_features, encoded_cat_columns)
        synthetic_df = synthetic_df[df.columns]  # Ensure the columns match the original DataFrame

        return synthetic_df


def load_and_preprocess_data(df, categorical_indicator):
    continuous_features = []
    categorical_features = []

    for col, is_categorical in zip(df.columns, categorical_indicator):
        if is_categorical:
            categorical_features.append(col)
        else:
            continuous_features.append(col)

    for column in df.columns:
        if df[column].dtype == 'category' or len(df[column].unique()) < 10:
            df[column] = df[column].fillna(df[column].mode()[0])
        else:
            df[column] = df[column].fillna(df[column].median())

    scaler = StandardScaler()
    if continuous_features:
        df[continuous_features] = scaler.fit_transform(df[continuous_features])

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_cats = encoder.fit_transform(df[categorical_features].astype(str))
    encoded_cat_columns = encoder.get_feature_names_out(categorical_features)

    df = df.drop(columns=categorical_features)
    df[encoded_cat_columns] = encoded_cats

    data = df.values

    return data, scaler, encoder, continuous_features, categorical_features, encoded_cat_columns