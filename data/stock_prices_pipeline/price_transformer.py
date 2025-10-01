import pandas as pd

# Load the dataset
df = pd.read_csv('historical_stock_prices_2015_2025.csv', index_col=0)  # Replace with actual path if needed

# Transpose so dates become rows
df_t = df.T

# Convert index to datetime and handle timezone
df_t.index = pd.to_datetime(df_t.index, utc=True)

# Resample to month-end frequency and compute average
monthly_avg_t = df_t.resample('ME').mean()

# Transpose back so companies are rows again
monthly_avg = monthly_avg_t.T


# Clean column names: remove timezone and convert to 'YYYY-MM'
monthly_avg.columns = monthly_avg.columns.tz_convert(None).to_period('M').astype(str)


# Save to CSV
monthly_avg.to_csv('monthly_avg_stock_prices.csv', index=True)

