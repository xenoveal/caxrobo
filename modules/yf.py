import yfinance as yf
import pandas as pd

from datetime import datetime, timedelta
from modules.logger import setup_logger

logger = setup_logger('YahooFinanceLogger', 'logs/yahoofinance.log')

class YahooFinance:
    def __init__(self):
        """Initialize the YahooFinance class."""
        pass

    def fetch_hist_data(self, symbol: str, years: int = 1) -> pd.DataFrame:
        """
        Fetch historical OHLCV data for a cryptocurrency from Yahoo Finance.

        Args:
            symbol (str): The ticker symbol of the cryptocurrency (e.g., 'BTC-USD' for Bitcoin).
            years (int): The number of years of historical data to fetch (default is 10).

        Returns:
            pd.DataFrame: A DataFrame containing the OHLCV data.
        """
        # Define the time period (last 'years' years)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * years)

        # Ensure to reset the time components for consistency
        start_date = start_date.replace(microsecond=0)
        end_date = end_date.replace(microsecond=0)

        # Fetch the historical data
        crypto_data = yf.download(symbol, start=start_date, end=end_date)

        # Check if data is retrieved successfully
        if crypto_data.empty:
            logger.warning("No data retrieved from specified symbol in YahooFinance.")
            return None

        # Return the OHLCV data
        logger.info("Success to get data from Yahoofinance.")
        return crypto_data


# Example usage
if __name__ == "__main__":
    yahoo_finance = YahooFinance()
    symbol = "BTC-USD"  # Ticker symbol for Bitcoin
    data = yahoo_finance.fetch_hist_data(symbol)
    print(data)
