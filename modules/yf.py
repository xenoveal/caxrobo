import yfinance as yf
import pandas as pd

from datetime import datetime, timedelta
from modules.logger import setup_logger

logger = setup_logger('YahooFinanceLogger', 'logs/yahoofinance.log')

class YahooFinance:
    def __init__(self):
        """Initialize the YahooFinance class."""
        pass

    def fetch_hist_data(
            self, 
            symbol: str, 
            days: int = 365, 
            interval: str = "1d", 
            end_date: datetime = datetime.now()
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data for a cryptocurrency from Yahoo Finance.

        Args:
            symbol (str): The ticker symbol of the cryptocurrency (e.g., 'BTC-USD' for Bitcoin).
            days (int): The number of days of historical data to fetch (default is 365 days).
            interval (str): Valid intervals [1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo] (default is 1d).
            end_date (datetime): the end date to get the data (default is current time).

        Returns:
            pd.DataFrame: A DataFrame containing the OHLCV data.
        """

        start_date = (end_date - timedelta(days=days)).replace(microsecond=0, second=0, minute=0, hour=0)
        end_date = end_date.replace(microsecond=0, second=0, minute=0, hour=0)

        try:
            data = yf.download(symbol, start=start_date, end=end_date, interval=interval)

            if data.empty:
                logger.warning(f"{symbol} returned empty in YahooFinance.")
                return None

        except Exception as e:
            logger.error(f"Error when getting the data from YahooFinance. Error: {e}")
            return None

        # Return the OHLCV data
        logger.info("Success to get data from Yahoo Finance.")
        return data

# Example usage
if __name__ == "__main__":
    yahoo_finance = YahooFinance()
    symbol = "BTC-USD"  # Ticker symbol for Bitcoin
    data = yahoo_finance.fetch_hist_data(symbol)
    print(data)
