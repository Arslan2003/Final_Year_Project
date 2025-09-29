import yfinance as yf
import pandas as pd
import time
from tqdm import tqdm

# Load company data
df = pd.read_csv('../../ticker_finder_pipeline/found_tickers.csv')

# Drop rows with missing 'Ticker' or 'Company Name'
df = df.dropna(subset=['Ticker', 'Company Name'])

# df = df.head()

# Initialize containers
stock_prices = {}
failed_tickers = []

# Date range
start_date = "2015-04-01"
end_date = "2025-03-31"

# Loop with progress bar
for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Extracting Stock Prices"):
    ticker = row['Ticker']
    company_name = row['Company Name']

    try:
        data = yf.Ticker(ticker).history(start=start_date, end=end_date)
        if data.empty:
            raise ValueError("No data returned")

        data['Close'] = data['Close'].ffill()
        stock_prices[company_name] = data['Close']
        time.sleep(3)

    except Exception as e:
        failed_tickers.append({
            'Company Name': company_name,
            'Ticker': ticker,
            'Reason': str(e)
        })

# Convert stock prices to DataFrame and transpose
stock_prices_df = pd.DataFrame(stock_prices).T

# Save results
stock_prices_df.to_csv('historical_stock_prices_2015_2025.csv')

# Save failed tickers
if failed_tickers:
    failed_df = pd.DataFrame(failed_tickers)
    failed_df.to_csv('stock_prices_failed.csv', index=False)

print("Stock price extraction complete.")
print("Saved: historical_stock_prices_2015_2025.csv")
if failed_tickers:
    print("⚠️ Some tickers failed. See: stock_prices_failed.csv")
