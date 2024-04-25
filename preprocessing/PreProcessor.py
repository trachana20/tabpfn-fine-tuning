from __future__ import annotations

import pandas as pd
from pandas import DataFrame
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    PowerTransformer,
    StandardScaler,
)


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
        data_df: DataFrame,
        target: str,
        categorical_indicator: list,
        attribute_names: list,
    ):
        # Drop rows where target is missing, because we can't learn from them
        data_df = self.drop_row_where_taget_is_missing(data_df, target)

        # Split data into features and target
        x_data = data_df.drop(columns=[target])
        target_data = data_df[target]
        target_data = self.target_encoder(target_data)

        # Get categorical and numerical features using list comprehension and zip
        categorical_features, numerical_features = (
            self.get_categorical_and_numerical_features(
                data=data_df,
                categorical_indicator=categorical_indicator,
                attribute_names=attribute_names,
                target=target,
            )
        )

        # Preprocess data with missing values, encoding, outliers, and scaling
        x_data = self.encode_categorical(x_data, categorical_features)

        x_data = self.impute_missing_values(
            data=x_data,
            categorical_features=categorical_features,
            numerical_features=numerical_features,
        )

        x_data = self.handle_outliers(
            x_data,
            categorical_features,
            numerical_features,
        )

        x_data = self.scale(x_data)

        # Apply data transformations
        x_data = self._data_transformations(
            x_data,
            numerical_features,
        )
        target_data = pd.DataFrame({target: target_data})
        return pd.concat([x_data, target_data], axis=1)

    def target_encoder(self, target):
        encoder = LabelEncoder()
        return encoder.fit_transform(target)

    def drop_row_where_taget_is_missing(self, data, target):
        return data.dropna(subset=[target])

    def get_categorical_and_numerical_features(
        self,
        data,
        categorical_indicator,
        attribute_names,
        target,
    ):
        # Get categorical and numerical features using list comprehension and zip
        categorical_features = [
            name
            for indicator, name in zip(categorical_indicator, attribute_names)
            if indicator and name != target
        ]
        numerical_features = [
            name
            for indicator, name in zip(categorical_indicator, attribute_names)
            if not indicator and name != target
        ]
        # get non-numerical datatypes because for some reason
        # the openml categorical_features are not including strings etc
        # append to categorical features

        for column in data.columns:
            if not pd.api.types.is_numeric_dtype(data[column]) and column != target:
                categorical_features.append(column)

        return categorical_features, numerical_features

    def impute_missing_values(self, data, categorical_features, numerical_features):
        """Impute missing values in the provided data.

        Parameters:
        - data: DataFrame, the dataset containing missing values.
        - categorical_features: list of str, names of categorical features.
        - numerical_features: list of str, names of numerical features.

        Returns:
        - DataFrame, dataset with missing values imputed.
        """
        # Handle missing values for numerical features
        numerical_data = data[numerical_features]
        imputer_num = SimpleImputer(strategy="mean")
        numerical_data_imputed = imputer_num.fit_transform(numerical_data)

        # Handle missing values for categorical features
        categorical_data = data[categorical_features]
        imputer_cat = SimpleImputer(strategy="most_frequent")
        categorical_data_imputed = imputer_cat.fit_transform(categorical_data)

        # Convert back to DataFrame
        numerical_df = pd.DataFrame(numerical_data_imputed, columns=numerical_features)
        categorical_df = pd.DataFrame(
            categorical_data_imputed,
            columns=categorical_features,
        )

        # Concatenate numerical and categorical DataFrames
        data_imputed = pd.concat([numerical_df, categorical_df], axis=1)

        return data_imputed

    def handle_outliers(
        self,
        data,
        categorical_features,
        numerical_features,
    ):
        # Handle outliers (you can implement your logic here)
        # TODO: Implement outlier handling
        return data

    def scale(self, data):
        # Scale data
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        return DataFrame(data_scaled, columns=data.columns)

    def _ordinal_encode(
        self,
        data,
        categorical_features,
    ):
        encoder = LabelEncoder()
        for feature in categorical_features:
            data[feature] = encoder.fit_transform(data[feature])
        return data

    def _one_hot_encode(
        self,
        data,
        categorical_features,
    ):
        encoder = OneHotEncoder(sparse=False)
        one_hot_encoded = encoder.fit_transform(data[categorical_features])
        data = data.drop(columns=categorical_features)
        one_hot_df = pd.DataFrame(
            one_hot_encoded,
            columns=encoder.get_feature_names_out(categorical_features),
            index=data.index,
        )
        return pd.concat([data, one_hot_df], axis=1)

    def encode_categorical(
        self,
        data,
        categorical_features,
    ):
        if self.encode_categorical_type == "ordinal":
            data = self._ordinal_encode(
                data,
                categorical_features,
            )
        elif self.encode_categorical_type == "one-hot":
            data = self._one_hot_encode(
                data,
                categorical_features,
            )
        else:
            raise ValueError("Invalid categorical encoder type")
        return data

    def _data_transformations(self, data, numerical_features):
        if self.data_tranformer_type == "power_transform":
            pt = PowerTransformer()
            data[numerical_features] = pt.fit_transform(data[numerical_features])
        elif self.data_tranformer_type == "quantile_transform":
            # Apply quantile transformation or other appropriate transformation
            pass
        else:
            raise ValueError("Invalid data transformer type")
        return data
