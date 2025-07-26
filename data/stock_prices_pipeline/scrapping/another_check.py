import pandas as pd


df = pd.read_csv('monthly_avg_stock_prices.csv', index_col=0)
pd.set_option('display.max_rows', None)
print(df.head())
print(df.columns)
print(df.isna().sum())
