from fin_metrics_prep_cleaning import (
    load_and_merge_data,
    drop_irrelevant_columns,
    drop_sparse_columns,
    normalize_numerical,
    scale_large_columns,
    strip_white_spaces
)
from fin_metrcis_prep_utils import save_csv, stratified_split
from fin_metrics_prep_valuation import apply_valuation
from fin_metrics_prep_config import TRAIN_FILE, VAL_FILE, TEST_FILE, INDUSTRY_COL

def main():
    df = load_and_merge_data()
    df = drop_irrelevant_columns(df)
    df = drop_sparse_columns(df)
    df = strip_white_spaces(df)
    df = scale_large_columns(df)

    df = apply_valuation(df)  # ← This should come BEFORE the split

    if INDUSTRY_COL not in df.columns:
        raise ValueError(f"'{INDUSTRY_COL}' column missing before stratified split.")

    # Split before normalization
    df_train, df_val, df_test = stratified_split(df, stratify_col=INDUSTRY_COL)

    # Normalize features (excluding Valuation_Label)
    df_train, scaler = normalize_numerical(df_train)
    df_val, _ = normalize_numerical(df_val, scaler=scaler)
    df_test, _ = normalize_numerical(df_test, scaler=scaler)

    save_csv(df_train, TRAIN_FILE)
    save_csv(df_val, VAL_FILE)
    save_csv(df_test, TEST_FILE)



if __name__ == "__main__":
    main()
