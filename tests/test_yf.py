import unittest
from unittest.mock import patch
import pandas as pd
from modules.yf import YahooFinance  # Adjust the import based on your actual module structure
from datetime import datetime, timedelta

class TestYahooFinance(unittest.TestCase):

    def setUp(self):
        """Set up the YahooFinance instance before each test."""
        self.yahoo_finance = YahooFinance()

    @patch('yfinance.download')  # Mock the yfinance.download method
    def test_fetch_hist_data_success(self, mock_download):
        """Test fetch_hist_data method when data is retrieved successfully."""
        # Create a mock DataFrame to simulate the downloaded data
        mock_data = {
            'Open': [100, 110, 105],
            'High': [110, 115, 107],
            'Low': [90, 105, 102],
            'Close': [105, 112, 104],
            'Volume': [1000, 1500, 1200],
        }
        mock_df = pd.DataFrame(mock_data, index=pd.date_range(start='2020-01-01', periods=3, freq='D'))
        mock_download.return_value = mock_df

        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)

        # Ensure to reset the time components for consistency
        start_date = start_date.replace(microsecond=0)
        end_date = end_date.replace(microsecond=0)

        # Call the fetch_hist_data method
        result = self.yahoo_finance.fetch_hist_data('BTC-USD', days=365)

        # Assert that the returned DataFrame matches the mock data
        pd.testing.assert_frame_equal(result, mock_df)
        mock_download.assert_called_once_with(
            'BTC-USD', 
            start=start_date,
            end=end_date,
            interval="1d"
        )

    @patch('yfinance.download')  # Mock the yfinance.download method
    def test_fetch_hist_data_no_data(self, mock_download):
        """Test fetch_hist_data method when no data is retrieved."""
        # Simulate an empty DataFrame
        mock_download.return_value = pd.DataFrame()

        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)

        # Ensure to reset the time components for consistency
        start_date = start_date.replace(microsecond=0)
        end_date = end_date.replace(microsecond=0)

        # Call the fetch_hist_data method
        result = self.yahoo_finance.fetch_hist_data('BTC-USD', days=365)

        # Assert that the result is None
        self.assertIsNone(result)
        mock_download.assert_called_once_with(
            'BTC-USD', 
            start=start_date,
            end=end_date,
            interval="1d"
        )

if __name__ == '__main__':
    unittest.main()