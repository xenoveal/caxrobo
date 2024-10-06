import unittest
from unittest.mock import patch
from modules.cg import CoinGecko

class TestCoinGecko(unittest.TestCase):
    
    def setUp(self):
        # Initialize the CoinGecko class with a dummy API key for testing
        self.coingecko = CoinGecko(api_key="dummy_api_key")

    @patch('requests.get')  # Mock the requests.get method
    def test_ping_success(self, mock_get):
        """Test ping method when the API is up."""
        # Mock the response to simulate a successful API ping
        mock_get.return_value.status_code = 200
        mock_get.return_value.text = "Pong"

        # Call the ping method
        result = self.coingecko.ping()

        # Assert the result and that the logger was called correctly
        self.assertEqual(result, "CoinGecko API is up!")
        mock_get.assert_called_once_with(self.coingecko.base_url + "/ping", headers=self.coingecko.headers)

    @patch('requests.get')  # Mock the requests.get method
    def test_ping_failure(self, mock_get):
        """Test ping method when the API is down."""
        # Mock the response to simulate a failed API ping
        mock_get.return_value.status_code = 500
        mock_get.return_value.text = "Internal Server Error"

        # Call the ping method
        result = self.coingecko.ping()

        # Assert the result and that the logger was called correctly
        self.assertIsNone(result)
        mock_get.assert_called_once_with(self.coingecko.base_url + "/ping", headers=self.coingecko.headers)

    @patch('requests.get')
    def test_get_price(self, mock_get):
        # Mock the API response
        mock_response = {
            "bitcoin": {
                "usd": 50000
            }
        }
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = mock_response

        # Test the get_price method
        price = self.coingecko.get_price("bitcoin")
        self.assertEqual(price, 50000)
    
    @patch('requests.get')
    def test_get_market_data(self, mock_get):
        # Mock the API response
        mock_response = [{
            "market_cap": 1000000000,
            "total_volume": 50000000,
            "price_change_percentage_24h": -2.5
        }]
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = mock_response

        # Test the get_market_data method
        market_data = self.coingecko.get_market_data("bitcoin")
        self.assertEqual(market_data['market_cap'], 1000000000)
        self.assertEqual(market_data['total_volume'], 50000000)
        self.assertEqual(market_data['price_change_percentage_24h'], -2.5)
    
    @patch('requests.get')
    def test_get_coin_info(self, mock_get):
        # Mock the API response
        mock_response = {
            "name": "Bitcoin",
            "symbol": "btc",
            "description": {"en": "Bitcoin is a decentralized cryptocurrency."},
            "links": {"homepage": ["https://bitcoin.org"]},
            "genesis_date": "2009-01-03"
        }
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = mock_response

        # Test the get_coin_info method
        coin_info = self.coingecko.get_coin_info("bitcoin")
        self.assertEqual(coin_info['name'], "Bitcoin")
        self.assertEqual(coin_info['symbol'], "btc")
        self.assertEqual(coin_info['description'], "Bitcoin is a decentralized cryptocurrency.")
        self.assertEqual(coin_info['homepage'], "https://bitcoin.org")
        self.assertEqual(coin_info['genesis_date'], "2009-01-03")
    
    @patch('requests.get')
    def test_get_supported_coins(self, mock_get):
        # Mock the API response
        mock_response = [
            {"id": "bitcoin", "name": "Bitcoin"},
            {"id": "ethereum", "name": "Ethereum"}
        ]
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = mock_response

        # Test the get_supported_coins method
        coins = self.coingecko.get_supported_coins()
        self.assertEqual(len(coins), 2)
        self.assertEqual(coins[0]['name'], "Bitcoin")
        self.assertEqual(coins[1]['name'], "Ethereum")

if __name__ == '__main__':
    unittest.main()
