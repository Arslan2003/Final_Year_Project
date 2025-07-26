# File paths (relative to script location or working directory)
RAW_DATA_FILE = "../raw_financial_metrics.csv"
INDUSTRY_FILE = "../../readable-issuer_list_archive_2025-02-03.csv"

TRAIN_FILE = "../fin_metrics_train.csv"
VAL_FILE = "../fin_metrics_val.csv"
TEST_FILE = "../fin_metrics_test.csv"

# Industry column name
INDUSTRY_COL = "ICB Industry"

# Columns to drop
DROP_COLS = [
    'Ticker', 'Website', 'Address', 'City', 'Postcode', 'Exchange',
    'Country', 'Recommendation', 'Number of Analyst Opinions'
]