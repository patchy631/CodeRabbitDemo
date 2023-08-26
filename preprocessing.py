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
    """
    One-hot encode categorical features in the DataFrame.
    """
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    return df