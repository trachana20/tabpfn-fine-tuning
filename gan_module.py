import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import tensorflow as tf
from tensorflow.keras import layers, Model


class GANModule:
    def __init__(self, input_dim=100, epochs=10000, batch_size=128, num_samples=1000):
        self.input_dim = input_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_samples = num_samples

    def load_and_preprocess_data(self, df, categorical_indicator):
        self.column_info = df.columns
        self.continuous_features = []
        self.categorical_features = []

        for col, is_categorical in zip(self.column_info, categorical_indicator):
            if is_categorical:
                self.categorical_features.append(col)
            else:
                self.continuous_features.append(col)

        for column in df.columns:
            if df[column].dtype == 'category' or len(df[column].unique()) < 10:
                df[column] = df[column].fillna(df[column].mode()[0])
            else:
                df[column] = df[column].fillna(df[column].median())

        self.scaler = StandardScaler()
        if self.continuous_features:
            df[self.continuous_features] = self.scaler.fit_transform(df[self.continuous_features])

        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_cats = self.encoder.fit_transform(df[self.categorical_features].astype(str))
        self.encoded_cat_columns = self.encoder.get_feature_names_out(self.categorical_features)

        df = df.drop(columns=self.categorical_features)
        df[self.encoded_cat_columns] = encoded_cats

        self.data = df.values
        return self.data

    def build_generator(self):
        self.generator = tf.keras.Sequential([
            layers.InputLayer(input_shape=(self.input_dim + len(self.encoded_cat_columns),)),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            layers.Dense(self.data.shape[1], activation='tanh')
        ])
        return self.generator

    def build_discriminator(self):
        self.discriminator = tf.keras.Sequential([
            layers.InputLayer(input_shape=(self.data.shape[1],)),
            layers.Dense(256, activation='relu'),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ])
        return self.discriminator

    def compile_gan(self):
        self.discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.discriminator.trainable = False

        gan_input = layers.Input(shape=(self.generator.input_shape[1],))
        generated_data = self.generator(gan_input)
        gan_output = self.discriminator(generated_data)

        self.gan = Model(gan_input, gan_output)
        self.gan.compile(optimizer='adam', loss='binary_crossentropy')
        return self.gan

    def train_gan(self):
        half_batch = self.batch_size // 2

        for epoch in range(self.epochs):
            noise = np.random.normal(0, 1, (half_batch, self.input_dim))
            idx = np.random.randint(0, self.data.shape[0], half_batch)
            real_data = self.data[idx]
            conditions = real_data[:, -len(self.encoded_cat_columns):]

            gen_input = np.concatenate([noise, conditions], axis=1)
            generated_data = self.generator.predict(gen_input)

            real_data_with_cond = np.concatenate([real_data[:, :-len(self.encoded_cat_columns)], conditions], axis=1)

            d_loss_real = self.discriminator.train_on_batch(real_data_with_cond, np.ones((half_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch(generated_data, np.zeros((half_batch, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            noise = np.random.normal(0, 1, (self.batch_size, self.input_dim))
            idx = np.random.randint(0, self.data.shape[0], self.batch_size)
            conditions = self.data[idx, -len(self.encoded_cat_columns):]
            gen_input = np.concatenate([noise, conditions], axis=1)

            valid_y = np.array([1] * self.batch_size)
            g_loss = self.gan.train_on_batch(gen_input, valid_y)

            if epoch % 1000 == 0:
                print(f"Epoch {epoch} / {self.epochs} [D loss: {d_loss[0]} | D accuracy: {100 * d_loss[1]}] [G loss: {g_loss}]")

    def generate_synthetic_data(self):
        noise = np.random.normal(0, 1, (self.num_samples, self.generator.input_shape[1] - len(self.encoded_cat_columns)))

        conditions = np.tile(self.encoder.transform([[str(i) for i in range(len(self.categorical_features))]] * self.num_samples),
                             (self.num_samples // len(self.categorical_features) + 1, 1))[:self.num_samples]

        gen_input = np.concatenate([noise, conditions], axis=1)
        synthetic_data = self.generator.predict(gen_input)

        synthetic_continuous = synthetic_data[:, :len(self.continuous_features)]
        if self.continuous_features:
            synthetic_continuous = self.scaler.inverse_transform(synthetic_continuous)

        synthetic_df = pd.DataFrame(synthetic_data, columns=self.continuous_features + list(self.encoded_cat_columns))
        if self.continuous_features:
            synthetic_df[self.continuous_features] = synthetic_continuous

        for feature in self.continuous_features:
            synthetic_df[feature] = synthetic_df[feature].round()

        for feature in self.categorical_features:
            encoded_columns = [col for col in synthetic_df.columns if feature in col]
            synthetic_df[feature] = synthetic_df[encoded_columns].idxmax(axis=1).apply(lambda x: x.split('_')[-1])
            synthetic_df = synthetic_df.drop(columns=encoded_columns)

        return synthetic_df

    def create_data_frame_of_synthetic_data(self, data_frame, categorical_indicator):
        self.load_and_preprocess_data(data_frame, categorical_indicator)
        self.build_generator()
        self.build_discriminator()
        self.compile_gan()
        self.train_gan()
        synthetic_df = self.generate_synthetic_data()
        return synthetic_df
