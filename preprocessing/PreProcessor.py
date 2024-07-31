from __future__ import annotations

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import (
    LabelEncoder,
    PowerTransformer,
    QuantileTransformer,
    StandardScaler,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.impute import SimpleImputer

class PreProcessor:
    def __init__(
        self,
        categorical_encoder_type="ordinal",
        data_tranformer_type="power_transform",
        numerical_data_imputation="mean",
        categorical_data_imputation="most_frequent",
    ):
        self.encode_categorical_type = categorical_encoder_type
        self.data_tranformer_type = data_tranformer_type
        self.numerical_data_imputation = numerical_data_imputation
        self.categorical_data_imputation = categorical_data_imputation

    def preprocess(
        self,
        train_data,
        val_data,
        test_data,
        target: str,
        categorical_indicator: list,
        attribute_names: list,
    ):
        # Drop rows where target is missing, because we can't learn from them
        train_data, val_data, test_data = self.drop_row_where_taget_is_missing(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            target=target,
        )

        train_data, val_data, test_data = self.target_encoder(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            target=target,
        )

        # Get categorical and numerical features using list comprehension and zip
        categorical_features, numerical_features = (
            self.get_categorical_and_numerical_features(
                train_data=train_data,
                categorical_indicator=categorical_indicator,
                attribute_names=attribute_names,
                target=target,
            )
        )
        train_data, val_data, test_data = self.drop_constant_categorical_features(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            categorical_features=categorical_features,
        )

        train_data, val_data, test_data = self.drop_all_unique_categorical_features(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            categorical_features=categorical_features,
        )

        train_data, val_data, test_data = self.impute_missing_values(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            categorical_features=categorical_features,
            numerical_features=numerical_features,
        )
        # Preprocess data with missing values, encoding, outliers, and scaling
        train_data, val_data, test_data = self.encode_categorical(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            categorical_features=categorical_features,
        )

        train_data, val_data, test_data = self.handle_outliers(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            categorical_features=categorical_features,
            numerical_features=numerical_features,
        )

        train_data, val_data, test_data = self.scale(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            target=target,
        )

        # Apply data transformations
        train_data, val_data, test_data = self._data_transformations(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            numerical_features=numerical_features,
        )
        return train_data, val_data, test_data

    def target_encoder(self, train_data, val_data, test_data, target):
        # encodes the target column to a numerical datatype
        encoder = LabelEncoder()
        train_data[target] = encoder.fit_transform(train_data[target])
        # apply transformation on val_data and test_data
        val_data[target] = encoder.transform(val_data[target])
        test_data[target] = encoder.transform(test_data[target])
        return train_data, val_data, test_data

    def drop_row_where_taget_is_missing(self, train_data, val_data, test_data, target):
        train_data = train_data.dropna(subset=[target])
        val_data = val_data.dropna(subset=[target])
        test_data = test_data.dropna(subset=[target])
        return train_data, val_data, test_data

    def drop_constant_categorical_features(
        self,
        train_data,
        val_data,
        test_data,
        categorical_features,
    ):
        for cat_feature in categorical_features:
            # Check if the feature has more than one unique value in the training data
            num_unique = train_data[cat_feature].nunique()

            # If the feature is constant in the training data, drop it from all datasets
            if num_unique == 1:
                train_data = train_data.drop(columns=[cat_feature])
                val_data = val_data.drop(columns=[cat_feature])
                test_data = test_data.drop(columns=[cat_feature])

        return train_data, val_data, test_data

    def drop_all_unique_categorical_features(
        self,
        train_data,
        val_data,
        test_data,
        categorical_features,
    ):
        for cat_feature in categorical_features[:]:
            if cat_feature in train_data:
                num_unique_values = train_data[cat_feature].nunique()
                len_data = len(train_data)
                # drop if more than 95% are unique values
                if num_unique_values >= len_data * 0.95:
                    train_data = train_data.drop(columns=[cat_feature])
                    val_data = val_data.drop(columns=[cat_feature])
                    test_data = test_data.drop(columns=[cat_feature])
                    categorical_features.remove(cat_feature)

        return train_data, val_data, test_data

    def get_categorical_and_numerical_features(
        self,
        train_data,
        categorical_indicator,
        attribute_names,
        target,
    ):
        # Get categorical and numerical features using list comprehension and zip
        categorical_features = {
            name
            for indicator, name in zip(categorical_indicator, attribute_names)
            if indicator and name != target
        }
        numerical_features = {
            name
            for indicator, name in zip(categorical_indicator, attribute_names)
            if not indicator and name != target
        }
        # the OpenML categorical indicator is not very accuracte:
        # e.g: the "ticket" column in the titanic ds is marked as numerical
        # but contains both
        # get non-numerical datatypes because for some reason
        # the openml categorical_features are not including strings etc
        # append to categorical features

        # assume that train_data has the same columns as val_data and test_data
        for column in train_data.columns:
            if column == target:
                continue
            if not pd.api.types.is_numeric_dtype(train_data[column]):
                # Not Numeric
                categorical_features.add(column)
                numerical_features.discard(column)
            else:
                # Numeric
                numerical_features.add(column)
                categorical_features.discard(column)

        # convert list -> set -> list so that we eliminate duplicates
        return list(categorical_features), list(numerical_features)

    def impute_missing_values(
        self,
        train_data,
        val_data,
        test_data,
        categorical_features,
        numerical_features,
    ):
        # Handle missing values for numerical features with mean imputation

        for num_feature in numerical_features:
            train_data[num_feature] = pd.to_numeric(
                train_data[num_feature],
                errors="coerce",
            )

            val_data[num_feature] = pd.to_numeric(
                val_data[num_feature],
                errors="coerce",
            )
            test_data[num_feature] = pd.to_numeric(
                test_data[num_feature],
                errors="coerce",
            )

            if pd.api.types.is_float_dtype(train_data[num_feature]):
                # float -> mean
                fill_metric = train_data[num_feature].mean()
            else:
                # non floating point(int, etc..) -> mode
                fill_metric = train_data[num_feature].mode()

            train_data[num_feature] = train_data[num_feature].fillna(fill_metric)

        # Handle missing values for categorical features
        for cat_features in categorical_features:
            if cat_features in train_data:
                most_frequent = train_data[cat_features].mode().iloc[0]

                train_data[cat_features] = train_data[cat_features].fillna(most_frequent)
                val_data[cat_features] = val_data[cat_features].fillna(most_frequent)
                test_data[cat_features] = test_data[cat_features].fillna(most_frequent)

        return train_data, val_data, test_data

    def handle_outliers(
        self,
        train_data,
        val_data,
        test_data,
        categorical_features,
        numerical_features,
    ):
        # Calculate mean and std deviation for each numerical feature using train data
        means = train_data[numerical_features].mean()
        std_devs = train_data[numerical_features].std()

        def remove_outliers(df, means, std_devs, numerical_features):
            # Calculate z-scores using the training data statistics
            z_scores = (df[numerical_features] - means) / std_devs
            # Determine outliers (z-score > 3 or z-score < -3)
            outliers = np.abs(z_scores) > 3
            # Remove outliers
            non_outliers = ~outliers
            # Keep rows that are not outliers in any numerical feature
            return df[non_outliers.all(axis=1)]

        # Remove outliers from train, val, and test data
        train_data = remove_outliers(train_data, means, std_devs, numerical_features)
        val_data = remove_outliers(val_data, means, std_devs, numerical_features)
        test_data = remove_outliers(test_data, means, std_devs, numerical_features)

        return train_data, val_data, test_data

    def scale(self, train_data, val_data, test_data, target):
        # standard scaler (x-mean) / std
        columns = list(train_data.columns)
        columns.remove(target)
        for col in columns:
            mean = train_data[col].mean()
            std = train_data[col].std()
            # Reshape the data to 2D for StandardScaler
            train_data[col] = train_data[col].apply(lambda x: (x - mean) / std)
            val_data[col] = val_data[col].apply(lambda x: (x - mean) / std)
            test_data[col] = test_data[col].apply(lambda x: (x - mean) / std)

        return train_data, val_data, test_data

    def _ordinal_encode(
        self,
        train_data,
        val_data,
        test_data,
        categorical_features,
    ):
        # Function to handle unseen categories
        def transform_with_unseen_handling(data, column, le):
            known_categories = set(le.classes_)
            transformed = []

            for item in data[column]:
                if item in known_categories:
                    transformed.append(le.transform([item])[0])
                else:
                    # Handle unseen category, e.g., assigning -1 or next integer value
                    ## or len(le.classes_) if you want to use a new category
                    transformed.append(-1)

            return np.array(transformed)

        # Encoder initialization and fitting on training data
        encoder = LabelEncoder()

        for feature in categorical_features:
            # fit and apply encoder to the train data
            encoder.fit(train_data[feature])
            train_data.loc[:, feature] = encoder.transform(train_data.loc[:, feature])

            # apply the fitted encoder to val_data and test_data
            # handle case if category is not represented in train data but in test or val
            val_data.loc[:, feature] = transform_with_unseen_handling(
                data=val_data,
                column=feature,
                le=encoder,
            )
            test_data.loc[:, feature] = transform_with_unseen_handling(
                data=test_data,
                column=feature,
                le=encoder,
            )

        return train_data, val_data, test_data

    def _one_hot_encode(
        self,
        train_data,
        val_data,
        test_data,
        categorical_features,
    ):
        pass

    def encode_categorical(
        self,
        train_data,
        val_data,
        test_data,
        categorical_features,
    ):
        if self.encode_categorical_type == "ordinal":
            return self._ordinal_encode(
                train_data=train_data,
                val_data=val_data,
                test_data=test_data,
                categorical_features=categorical_features,
            )
        elif self.encode_categorical_type == "one-hot":
            return self._one_hot_encode(
                train_data=train_data,
                val_data=val_data,
                test_data=test_data,
                categorical_features=categorical_features,
            )
        else:
            raise ValueError("Invalid categorical encoder type")

    def _data_transformations(
        self,
        train_data,
        val_data,
        test_data,
        numerical_features,
    ):
        if self.data_tranformer_type == "power_transform":
            pt = PowerTransformer()

            train_data[numerical_features] = pt.fit_transform(
                train_data[numerical_features]
            )
            val_data[numerical_features] = pt.transform(val_data[numerical_features])
            test_data[numerical_features] = pt.transform(test_data[numerical_features])

        elif self.data_tranformer_type == "quantile_transform":
            qt = QuantileTransformer(output_distribution="normal")

            train_data[numerical_features] = qt.fit_transform(
                train_data[numerical_features]
            )
            val_data[numerical_features] = qt.transform(val_data[numerical_features])
            test_data[numerical_features] = qt.transform(test_data[numerical_features])

        else:
            raise ValueError("Invalid data transformer type")
        return train_data, val_data, test_data

    def calculate_cosine_similarity(self, X_train, X_test):
        # Normalize X_train and X_test
        X_train_normalized = X_train / np.linalg.norm(X_train, axis=1, keepdims=True)
        X_test_normalized = X_test / np.linalg.norm(X_test, axis=1, keepdims=True)

        # Compute cosine similarity between X_train and X_test
        cosine_sim = cosine_similarity(X_test_normalized, X_train_normalized)

        return cosine_sim

    # Function to augment X_train using KNN
    def augment_X_train_knn(self, X_train, X_test, target_instance, top_k=5, metric='minkowski', aggregation='median'):
        print("X Train", X_train)
        print("X Test", X_test)
        X_train_ = X_train
        X_train_.drop(columns=target_instance)
        X_test.drop(columns=target_instance)


        # Initialize the NearestNeighbors model
        knn = NearestNeighbors(n_neighbors=top_k, metric=metric)
        # Fit the model on the training data
        knn.fit(X_train_)
        # Find the top k neighbors for each instance in the test set
        distances, indices = knn.kneighbors(X_test)
        # Initialize an empty array to store the augmented rows
        augmented_rows = []
        
        for i in range(indices.shape[0]):
            top_indices = indices[i]
            top_distances = distances[i]
            if aggregation == 'weighted_average':
                weights = 1 / (top_distances + 1e-6)  # Adding a small constant to avoid division by zero
                mean_row = np.average(X_train.iloc[top_indices, :].values, axis=0, weights=weights)
            elif aggregation == 'median':
                mean_row = np.median(X_train.iloc[top_indices, :].values, axis=0)
            elif aggregation == 'mean':
                mean_row = np.mean(X_train.iloc[top_indices, :].values, axis=0)
            elif aggregation == 'max':
                mean_row = np.max(X_train.iloc[top_indices, :].values, axis=0)
            elif aggregation == 'min':
                mean_row = np.min(X_train.iloc[top_indices, :].values, axis=0)
            elif aggregation == 'random':
                mean_row = X_train.iloc[np.random.choice(top_indices), :].values
            elif aggregation == 'first':
                mean_row = X_train.iloc[top_indices[0], :].values

            else:
                raise ValueError("Unsupported aggregation method: choose from 'weighted_average' or 'median'")
            augmented_rows.append(mean_row)
        
        # Append augmented_rows to X_train
        X_train_augmented = np.vstack([X_train.values, augmented_rows])
        # Convert back to DataFrame with original columns
        X_train_augmented = pd.DataFrame(X_train_augmented, columns=X_train.columns)
        return X_train_augmented

    def augment_dataset(self, train_data, test_data, target_instance):
        # Convert to DataFrame if needed
        train_df = pd.DataFrame(train_data, columns=train_data.columns)
        test_df = pd.DataFrame(test_data, columns=test_data.columns)
        test_df = test_df[train_df.columns]

        # Impute missing values
        # Separate numeric and non-numeric columns
        numeric_cols = train_data.select_dtypes(include=['number']).columns
        non_numeric_cols = train_df.select_dtypes(exclude=['number']).columns
        
        # Impute missing values for numeric columns
        numeric_imputer = SimpleImputer(strategy='mean')
        imputed_train_numeric = numeric_imputer.fit_transform(train_df[numeric_cols])
        imputed_test_numeric = numeric_imputer.transform(test_df[numeric_cols])
        
        # Combine the imputed numeric and non-numeric data
        imputed_train_df = pd.DataFrame(imputed_train_numeric, columns=numeric_cols)
        imputed_test_df = pd.DataFrame(imputed_test_numeric, columns=numeric_cols)

        # Ensure no NaN values after imputation
        assert not imputed_train_df.isnull().values.any(), "Train DataFrame contains NaN values after imputation"
        assert not imputed_test_df.isnull().values.any(), "Test DataFrame contains NaN values after imputation"


        # Augment train data
        augmented_df = self.augment_X_train_knn(imputed_train_df, imputed_test_df, target_instance)

        # Check if augmented DataFrame contains NaN values
        if augmented_df.isnull().values.any():
            raise error("DataFrame contains NaN values after augmentation")

        if str(target_instance) in augmented_df.columns:
        # for every row in train data [data] change the survived column to 1 if the value is greater than 0.5
            augmented_df[target_instance] = augmented_df[target_instance].apply(lambda x: 1 if x > 0.5 else 0)
        train_data = augmented_df
        return train_data
