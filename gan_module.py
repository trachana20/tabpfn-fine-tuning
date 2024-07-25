import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import tensorflow as tf
from tensorflow.keras import layers, Model

def load_and_preprocess_data(df, categorical_indicator):

    # Extract column names and their categories/types
    column_info = df.columns
    continuous_features = []
    categorical_features = []

    for col, is_categorical in zip(column_info, categorical_indicator):
        if is_categorical:
            categorical_features.append(col)
        else:
            continuous_features.append(col)

    for column in df.columns:
        if df[column].dtype == 'category' or len(df[column].unique()) < 10:
            df[column] = df[column].fillna(df[column].mode()[0])
        else:
            df[column] = df[column].fillna(df[column].median())

    scaler = StandardScaler()  # But, mostly all the columns in Titanic and dress sales are categorical
    if continuous_features:
        df[continuous_features] = scaler.fit_transform(df[continuous_features])

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_cats = encoder.fit_transform(
        df[categorical_features].astype(str))  # Ensure all categorical data is string type
    encoded_cat_columns = encoder.get_feature_names_out(categorical_features)

    df = df.drop(columns=categorical_features)
    df[encoded_cat_columns] = encoded_cats

    data = df.values

    return data, scaler, encoder, continuous_features, categorical_features, encoded_cat_columns


def build_generator(input_dim, condition_dim, output_dim):
    model = tf.keras.Sequential([
        layers.InputLayer(input_shape=(input_dim + condition_dim,)),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
        layers.Dense(output_dim, activation='tanh')
    ])
    return model


def build_discriminator(input_dim, condition_dim):
    model = tf.keras.Sequential([
        layers.InputLayer(input_shape=(input_dim + condition_dim,)),
        layers.Dense(256, activation='relu'),
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


def train_gan(gan, generator, discriminator, data, input_dim, condition_dim, epochs=10000, batch_size=128):
    half_batch = batch_size // 2
    output_dim = data.shape[1]

    for epoch in range(epochs):
        noise = np.random.normal(0, 1, (half_batch, input_dim))
        idx = np.random.randint(0, data.shape[0], half_batch)
        real_data = data[idx]
        conditions = real_data[:, -condition_dim:]

        gen_input = np.concatenate([noise, conditions], axis=1)
        generated_data = generator.predict(gen_input)

        real_data_with_cond = np.concatenate([real_data[:, :-condition_dim], conditions], axis=1)

        d_loss_real = discriminator.train_on_batch(real_data_with_cond, np.ones((half_batch, 1)))
        d_loss_fake = discriminator.train_on_batch(generated_data, np.zeros((half_batch, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        noise = np.random.normal(0, 1, (batch_size, input_dim))
        idx = np.random.randint(0, data.shape[0], batch_size)
        conditions = data[idx, -condition_dim:]
        gen_input = np.concatenate([noise, conditions], axis=1)

        valid_y = np.array([1] * batch_size)
        g_loss = gan.train_on_batch(gen_input, valid_y)

        if epoch % 1000 == 0:
            print(f"Epoch {epoch} / {epochs} [D loss: {d_loss[0]} | D accuracy: {100 * d_loss[1]}] [G loss: {g_loss}]")


def generate_synthetic_data(generator, scaler, encoder, continuous_features, categorical_features, encoded_cat_columns,
                            num_samples=1000):
    noise = np.random.normal(0, 1, (num_samples, generator.input_shape[1] - len(encoded_cat_columns)))

    conditions = np.tile(encoder.transform([[str(i) for i in range(len(categorical_features))]] * num_samples),
                         (num_samples // len(categorical_features) + 1, 1))[:num_samples]

    gen_input = np.concatenate([noise, conditions], axis=1)
    synthetic_data = generator.predict(gen_input)

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

    # let us also save the augmented data generated in CSV file for analysis
    synthetic_df.to_csv('synthetic_data.csv', index=False)
    print("Synthetic data saved to 'synthetic_data.csv'")

    return synthetic_df


def main(file_path, input_dim=100, epochs=10000, batch_size=128, num_samples=1000):
    data, scaler, encoder, continuous_features, categorical_features, encoded_cat_columns = load_and_preprocess_data(
        file_path)
    output_dim = data.shape[1]
    condition_dim = len(encoded_cat_columns)

    generator = build_generator(input_dim, condition_dim, output_dim)
    discriminator = build_discriminator(output_dim - condition_dim, condition_dim)
    gan = compile_gan(generator, discriminator)

    train_gan(gan, generator, discriminator, data, input_dim, condition_dim, epochs, batch_size)

    synthetic_df = generate_synthetic_data(generator, scaler, encoder, continuous_features, categorical_features,
                                           encoded_cat_columns, num_samples)
    synthetic_df.to_csv('synthetic_data.csv', index=False)
    print("Synthetic data saved to 'synthetic_data.csv'")


def createDataFrameOfSyntheticData(data_frame, categorical_indicator, input_dim=100, epochs=10000, batch_size=128, num_samples=1000):
    data, scaler, encoder, continuous_features, categorical_features, encoded_cat_columns = load_and_preprocess_data(
        data_frame, categorical_indicator)
    output_dim = data.shape[1]
    condition_dim = len(encoded_cat_columns)

    generator = build_generator(input_dim, condition_dim, output_dim)
    discriminator = build_discriminator(output_dim - condition_dim, condition_dim)
    gan = compile_gan(generator, discriminator)

    train_gan(gan, generator, discriminator, data, input_dim, condition_dim, epochs, batch_size)

    synthetic_df = generate_synthetic_data(generator, scaler, encoder, continuous_features, categorical_features,
                                           encoded_cat_columns, num_samples)
    return synthetic_df


if __name__ == "__main__":
    file_path = '/Users/rachana/tanpfn/data/dataset/dress_sales_openML.csv'  # This is for manual debugging and
    # not included in the pipeline
    main(file_path)
