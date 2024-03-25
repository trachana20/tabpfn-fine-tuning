from pandas import DataFrame
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer


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

    def preprocess(self, data: DataFrame, target: str):
        # Drop rows where target is missing, because we can't learn from them
        data = self.drop_row_where_taget_is_missing(data, target)

        # Split data into features and target
        x_data = data.drop(columns=[target])
        target_data = data[target]

        # Get categorical and numerical features
        categorical_features, numerical_features = (
            self.get_categorical_and_numerical_features(x_data)
        )

        # Preprocess data with missing values, encoding, outliers, and scaling

        x_data = self.impute_missing_values(
            x_data,
            categorical_features,
            numerical_features,
        )

        x_data = self.encode_categorical(x_data, categorical_features)

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

        return pd.concat([x_data, target_data], axis=1)

    def drop_row_where_taget_is_missing(self, data, target):
        return data.dropna(subset=[target])

    def get_categorical_and_numerical_features(self, data):
        categorical_types = ["object", "category", "bool"]
        # Categorical features have dtype "object" or "category" or "bool"
        categorical_features = data.select_dtypes(
            include=categorical_types,
        ).columns.tolist()

        # Numerical features have dtype other than "object", "category", or "bool"
        numerical_features = data.select_dtypes(
            exclude=categorical_types,
        ).columns.tolist()
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
