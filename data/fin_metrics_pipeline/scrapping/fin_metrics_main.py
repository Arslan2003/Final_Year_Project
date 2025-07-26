import pandas as pd
from fin_metrics_downloader import CompanyInfoFetcher
from fin_metrics_config import OUTPUT_CSV_PATH, OUTPUT_FAILED_CSV_PATH


def main():
    # Initialize the CompanyInfoFetcher to retrieve company info for tickers
    fetcher = CompanyInfoFetcher()

    # Fetch company information for all tickers
    fetcher.fetch_all()

    # Save the successfully fetched data and failed tickers to CSV files
    fetcher.save_to_csv()
    

if __name__ == "__main__":
    main()
