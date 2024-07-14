import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model


def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    for column in df.columns:
        if df[column].dtype == 'object' or len(df[column].unique()) < 10:
            df[column] = df[column].fillna(df[column].mode()[0])
        else:
            df[column] = df[column].fillna(df[column].median())

    continuous_features = []
    categorical_features = []

    for column in df.columns:
        if df[column].dtype == 'object' or len(df[column].unique()) < 10:
            categorical_features.append(column)
        else:
            continuous_features.append(column)

    scaler = StandardScaler()
    df[continuous_features] = scaler.fit_transform(df[continuous_features])

    encoder = OneHotEncoder(sparse_output=False)
    encoded_cats = encoder.fit_transform(df[categorical_features])
    encoded_cat_columns = encoder.get_feature_names_out(categorical_features)

    df = df.drop(columns=categorical_features)
    df[encoded_cat_columns] = encoded_cats

    data = df.values

    return data, scaler, encoder, continuous_features, categorical_features, encoded_cat_columns, df, categorical_features


def build_generator(input_dim, output_dim):
    model = tf.keras.Sequential([
        layers.Dense(128, activation='relu', input_dim=input_dim),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
        layers.Dense(output_dim, activation='tanh')
    ])
    return model


def build_discriminator(input_dim):
    model = tf.keras.Sequential([
        layers.Dense(256, activation='relu', input_dim=input_dim),
        layers.LeakyReLU(alpha=0.2),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.LeakyReLU(alpha=0.2),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    return model


def compile_gan(generator, discriminator):
    discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    discriminator.trainable = False

    gan_input = layers.Input(shape=(generator.input_shape[1],))
    generated_data = generator(gan_input)
    gan_output = discriminator(generated_data)

    gan = Model(gan_input, gan_output)
    gan.compile(optimizer='adam', loss='binary_crossentropy')
    return gan


def train_gan(gan, generator, discriminator, data, input_dim, epochs=10000, batch_size=128):
    half_batch = batch_size // 2

    for epoch in range(epochs):
        noise = np.random.normal(0, 1, (half_batch, input_dim))
        generated_data = generator.predict(noise)

        idx = np.random.randint(0, data.shape[0], half_batch)
        real_data = data[idx]

        d_loss_real = discriminator.train_on_batch(real_data, np.ones((half_batch, 1)))
        d_loss_fake = discriminator.train_on_batch(generated_data, np.zeros((half_batch, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        noise = np.random.normal(0, 1, (batch_size, input_dim))
        valid_y = np.array([1] * batch_size)
        g_loss = gan.train_on_batch(noise, valid_y)

        if epoch % 1000 == 0:
            print(f"Epoch {epoch} / {epochs} [D loss: {d_loss[0]} | D accuracy: {100 * d_loss[1]}] [G loss: {g_loss}]")


def generate_synthetic_data(generator, scaler, encoder, continuous_features, categorical_features, encoded_cat_columns,
                            num_samples=1000):
    noise = np.random.normal(0, 1, (num_samples, generator.input_shape[1]))
    synthetic_data = generator.predict(noise)

    # Inverse transform the continuous features
    synthetic_continuous = synthetic_data[:, :len(continuous_features)]
    synthetic_continuous = scaler.inverse_transform(synthetic_continuous)

    # Create a DataFrame for easier manipulation
    synthetic_df = pd.DataFrame(synthetic_data, columns=continuous_features + list(encoded_cat_columns))

    # Replace continuous data with the inverse transformed values
    synthetic_df[continuous_features] = synthetic_continuous

    # Round continuous features
    for feature in continuous_features:
        synthetic_df[feature] = synthetic_df[feature].round()

    # Explicitly convert the 'age' column to integers
    if 'age' in synthetic_df.columns:
        synthetic_df['age'] = synthetic_df['age'].astype(int)

    # Round off the 'fare' column to 4 decimal points and ensure non-negative values
    if 'fare' in synthetic_df.columns:
        synthetic_df['fare'] = synthetic_df['fare'].round(4)
        synthetic_df['fare'] = synthetic_df['fare'].clip(lower=0)

    # Decode one-hot encoded categorical features manually
    for feature in categorical_features:
        encoded_columns = [col for col in synthetic_df.columns if feature in col]
        synthetic_df[feature] = synthetic_df[encoded_columns].idxmax(axis=1).apply(lambda x: x.split('_')[-1])
        synthetic_df = synthetic_df.drop(columns=encoded_columns)

    return synthetic_df


def main(file_path, input_dim=100, epochs=10000, batch_size=128, num_samples=1000):
    data, scaler, encoder, continuous_features, categorical_features, encoded_cat_columns, original_df, original_cat_features = load_and_preprocess_data(
        file_path)
    output_dim = data.shape[1]

    generator = build_generator(input_dim, output_dim)
    discriminator = build_discriminator(output_dim)
    gan = compile_gan(generator, discriminator)

    train_gan(gan, generator, discriminator, data, input_dim, epochs, batch_size)

    synthetic_df = generate_synthetic_data(generator, scaler, encoder, continuous_features, categorical_features,
                                           encoded_cat_columns, num_samples)
    synthetic_df.to_csv('synthetic_data.csv', index=False)
    print("Synthetic data saved to 'synthetic_data.csv'")


if __name__ == "__main__":
    file_path = '/Users/rachana/tanpfn/data/dataset/Titanic.csv'  # Update this with your file path
    main(file_path)
