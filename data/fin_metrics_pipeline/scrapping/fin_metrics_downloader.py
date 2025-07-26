import time
import yfinance as yf
from fin_metrics_config import *
from fin_metrics_utils import *


class CompanyInfoFetcher:
    def __init__(self):
        self.tickers = pd.read_csv(TICKERS_CSV_PATH)["Ticker"].dropna().tolist()  # [:5] - the limit for testing
        self.failed_tickers = []
        self.company_info_list = []

    def fetch_company_info(self, ticker):
        try:
            info = yf.Ticker(ticker).info
            return info
        except Exception as e:
            print(f"Failed to fetch data for {ticker}: {e}")
            self.failed_tickers.append(ticker)
            return None

    def fetch_all(self):
        total_tickers = len(self.tickers)

        for idx, ticker in enumerate(self.tickers, start=1):
            print(f"{idx}/{total_tickers}: Fetching data for {ticker}...", end=" ")

            info = self.fetch_company_info(ticker)

            if info:
                company_info = {
                    'Ticker': ticker,
                    'Name': info.get('longName', None),
                    'Website': info.get('website', None),
                    'Address': info.get('address1', None),
                    'City': info.get('city', None),
                    'Postcode': info.get('zip', None),
                    'Country': info.get('country', None),
                    'Exchange': info.get('exchange', None),
                    'Market Cap': info.get('marketCap', None),
                    'Shares Outstanding': info.get('sharesOutstanding', None),
                    'Float Shares': info.get('floatShares', None),
                    'Regular Market Price': info.get('regularMarketPrice', None),
                    'Regular Market Change': info.get('regularMarketChange', None),
                    'Regular Market Change (%)': info.get('regularMarketChangePercent', None),
                    'Enterprise to Revenue': info.get('enterpriseToRevenue', None),
                    'Enterprise to EBITDA': info.get('enterpriseToEbitda', None),
                    'Book Value': info.get('bookValue', None),
                    'Beta': info.get('beta', None),
                    'P/E Ratio': info.get('trailingPE', None),  # Extracting P/E Ratio
                    'P/B Ratio': info.get('priceToBook', None),  # Extracting P/B Ratio
                    'Payout Ratio': info.get('payoutRatio', None),
                    'Total Revenue': info.get('totalRevenue', None),
                    'Gross Profits': info.get('grossProfits', None),
                    'EBITDA': info.get('ebitda', None),
                    'Operating Margins': info.get('operatingMargins', None),
                    'Profit Margins': info.get('profitMargins', None),
                    'Price to Sales (TTM)': info.get('priceToSalesTrailing12Months', None),
                    'Forward EPS': info.get('forwardEps', None),
                    'Trailing EPS': info.get('trailingEps', None),
                    'PEG Ratio': info.get('pegRatio', None),
                    'Operating Cash Flow': info.get('operatingCashflow', None),
                    'Free Cash Flow': info.get('freeCashflow', None),
                    'Total Cash': info.get('totalCash', None),
                    'Total Debt': info.get('totalDebt', None),
                    'Current Total Assets': info.get('totalAssets', None),
                    'Total Cash Per Share': info.get('totalCashPerShare', None),
                    'Debt to Equity Ratio': info.get('debtToEquity', None),
                    'Quick Ratio': info.get('quickRatio', None),
                    'Current Ratio': info.get('currentRatio', None),
                    '50 Day Average': info.get('fiftyDayAverage', None),
                    '200 Day Average': info.get('twoHundredDayAverage', None),
                    '52 Week Change': info.get('52WeekChange', None),
                    '52 Week High': info.get('fiftyTwoWeekHigh', None),
                    '52 Week Low': info.get('fiftyTwoWeekLow', None),
                    'Dividend Rate': info.get('dividendRate', None),  # Should I use None or 0?
                    'Dividend Yield': info.get('dividendYield', None),  # Should I use None or 0?
                    'Earnings Quarterly Growth': info.get('earningsQuarterlyGrowth', None),
                    'Revenue Growth': info.get('revenueGrowth', None),
                    'Recommendation': info.get('recommendationKey', None),
                    'Recommendation Mean': info.get('recommendationMean', None),
                    'Number of Analyst Opinions': info.get('numberOfAnalystOpinions', None),
                }
                self.company_info_list.append(company_info)
                print("Success")
            else:
                print("Failed")

            # Wait to avoid hitting the API too quickly
            time.sleep(API_DELAY)

    def save_to_csv(self):
        # Save company info to CSV
        save_to_csv(pd.DataFrame(self.company_info_list), OUTPUT_CSV_PATH)

        # Save failed tickers to CSV
        save_to_csv(pd.DataFrame(self.failed_tickers, columns=["Failed Tickers"]), OUTPUT_FAILED_CSV_PATH)


if __name__ == "__main__":
    fetcher = CompanyInfoFetcher()
    fetcher.fetch_all()
    fetcher.save_to_csv()


