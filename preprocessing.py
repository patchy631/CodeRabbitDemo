import pandas as pd
from sklearn.impute import SimpleImputer


def impute_missing_values(df, numerical_cols, categorical_cols):
    """
    Impute missing values in the DataFrame.
    """
    num_imputer = SimpleImputer(strategy='median')
    cat_imputer = SimpleImputer(strategy='most_frequent')

    df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])
    df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

    return df


def one_hot_encode_categorical_features(df, categorical_cols):
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    return df


def normalize_features(features):
    mean = sum(features) / len(features)
    variance = sum((x - mean) ** 2 for x in features) / len(features)
    std_dev = variance ** 0.5

    normalized_features = [(x - mean) / std_dev for x in features]
    return normalized_features