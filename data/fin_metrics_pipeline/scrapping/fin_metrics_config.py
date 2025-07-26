# fin_metrics_config.py

# Paths to the input and output files
TICKERS_CSV_PATH = "../../ticker_finder_pipeline/found_tickers.csv"  # Path to found_tickers.csv
ICB_INDUSTRY_CSV_PATH = "../../readable-issuer_list_archive_2025-02-03.csv"  # Path to the ICB Industry file
OUTPUT_CSV_PATH = "../raw_financial_metrics.csv"  # Path to save the financial metrics
OUTPUT_FAILED_CSV_PATH = '../failed_company_info.csv'

API_DELAY = 3
