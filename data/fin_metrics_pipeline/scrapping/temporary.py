import pandas as pd
import matplotlib as plt


# df = pd.read_csv('../raw_financial_metrics.csv')
# print(df.columns)
# print(df.dtypes)
# print(f'{df.isna().sum()}')
# draw a histogram of the age column
# print(df.columns)
# df2 = pd.read_csv('../../readable-issuer_list_archive_2025-02-03.csv')
# print(df2.columns)
# print(df2.isna().sum())
# df3 = pd.read_csv('../fin_metrics_test.csv')
# print(df3.dtypes)
# print(df3.shape)
# print(df3.isna().sum())
# print(df3['ICB Industry'].head())


df = pd.read_csv('../fin_metrics_train.csv')
print(df.columns)
print(df.dtypes)
print(df.shape)
print(df['ICB Industry'].unique())

