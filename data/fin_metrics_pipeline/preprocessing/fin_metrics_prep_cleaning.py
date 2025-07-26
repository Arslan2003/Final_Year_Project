import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from fin_metrics_prep_config import RAW_DATA_FILE, INDUSTRY_FILE, INDUSTRY_COL, DROP_COLS
from fin_metrcis_prep_utils import read_csv


def load_and_merge_data():
    """
    Loads raw financial and industry data, cleans company names,
    and merges them into a single DataFrame.
    """
    df = read_csv(RAW_DATA_FILE)
    industries = read_csv(INDUSTRY_FILE)

    # Standardize names for merging
    df["Name"] = df["Name"].str.strip().str.lower()
    industries["Company Name"] = industries["Company Name"].str.strip().str.lower()

    # Map industry labels
    name_to_industry = industries.set_index("Company Name")[INDUSTRY_COL].to_dict()
    df[INDUSTRY_COL] = df["Name"].map(name_to_industry)
    df[INDUSTRY_COL] = df[INDUSTRY_COL].fillna("Unknown")

    return df


def drop_irrelevant_columns(df):
    """
    Drops predefined irrelevant columns.
    """
    return df.drop(columns=DROP_COLS, errors='ignore')


def strip_white_spaces(df):
    str_cols = df.select_dtypes(include='object').columns
    for col in str_cols:
        df[col] = df[col].astype(str).str.strip()
    return df


def drop_sparse_columns(df, threshold=0.5):
    """
    Drops columns with more than `threshold` proportion of missing values.
    """
    min_count = len(df) * (1 - threshold)
    return df.dropna(axis=1, thresh=min_count)


def scale_large_columns(df):
    """
    Scales large numerical columns (e.g., Market Cap) down to millions
    and renames them to indicate the new scale.
    """
    scale_to_million_cols = ['Market Cap', 'Shares Outstanding']
    for col in scale_to_million_cols:
        if col in df.columns:
            df[col] = df[col] / 1e6
            df.rename(columns={col: f"{col} (Millions)"}, inplace=True)
    return df


def normalize_numerical(df, label_col="Valuation Label", scaler=None):
    from sklearn.preprocessing import MinMaxScaler
    import numpy as np

    df = df.copy()

    # Separate the label column
    y = df[label_col]
    features = df.drop(columns=[label_col])

    # Replace inf/-inf with NaN
    features.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Select numeric columns only
    numeric_cols = features.select_dtypes(include='number').columns

    if scaler is None:
        scaler = MinMaxScaler()
        features[numeric_cols] = scaler.fit_transform(features[numeric_cols])
    else:
        features[numeric_cols] = scaler.transform(features[numeric_cols])

    # Recombine features with target label
    df = pd.concat([features, y.reset_index(drop=True)], axis=1)

    return df, scaler

