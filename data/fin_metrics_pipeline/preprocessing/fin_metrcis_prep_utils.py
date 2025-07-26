import pandas as pd
from sklearn.model_selection import train_test_split


def read_csv(filepath):
    return pd.read_csv(filepath)


def save_csv(df, path):
    df.to_csv(path, index=False)


def stratified_split(df, stratify_col, train_size=0.8, val_size=0.1, test_size=0.1, random_state=42):
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, "Splits must sum to 1."

    df_train, df_temp = train_test_split(
        df,
        test_size=(1 - train_size),
        stratify=df[stratify_col],
        random_state=random_state
    )

    df_val, df_test = train_test_split(
        df_temp,
        test_size=test_size / (test_size + val_size),
        stratify=df_temp[stratify_col],
        random_state=random_state
    )

    return df_train.reset_index(drop=True), df_val.reset_index(drop=True), df_test.reset_index(drop=True)
