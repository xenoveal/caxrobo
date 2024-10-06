import requests

from modules.logger import setup_logger
logger = setup_logger('CoinGeckoLogger', 'logs/coingecko.log')

class CoinGecko:
    def __init__(self, api_key):
        self.base_url = "https://api.coingecko.com/api/v3"
        self.headers = {
            "x-cg-demo-api-key": api_key  # Adding the API key to the headers
        }

    def ping(self):
        """
        Check the CoinGecko API server status.
        :return: "CoinGecko API is up!" if the server is online, otherwise None.
        """
        endpoint = "/ping"
        response = requests.get(self.base_url + endpoint, headers=self.headers)
        if response.status_code == 200:
            logger.info(f"Success ping CoinGecko! {response.text}")
            return "CoinGecko API is up!"
        else:
            logger.error(f"Failed ping CoinGecko! {response.text}")
            return None

    def get_price(self, coin_id, currency='usd'):
        """
        Get the current price of a cryptocurrency.
        :param coin_id: ID of the cryptocurrency (e.g., 'bitcoin')
        :param currency: The fiat currency to compare against (e.g., 'usd')
        :return: The current price of the cryptocurrency in the specified currency.
        """
        endpoint = f"/simple/price?ids={coin_id}&vs_currencies={currency}"
        response = requests.get(self.base_url + endpoint, headers=self.headers)
        if response.status_code == 200:
            data = response.json()
            return data.get(coin_id, {}).get(currency, None)
        else:
            return None
    
    def get_market_data(self, coin_id):
        """
        Get market data for a cryptocurrency, including market cap, 24h volume, and 24h change.
        :param coin_id: ID of the cryptocurrency (e.g., 'bitcoin')
        :return: A dictionary with market data like market cap, 24h volume, and 24h price change.
        """
        endpoint = f"/coins/markets?vs_currency=usd&ids={coin_id}"
        response = requests.get(self.base_url + endpoint, headers=self.headers)
        if response.status_code == 200:
            data = response.json()
            if len(data) > 0:
                return {
                    'market_cap': data[0]['market_cap'],
                    'total_volume': data[0]['total_volume'],
                    'price_change_percentage_24h': data[0]['price_change_percentage_24h']
                }
            else:
                return None
        else:
            return None
    
    def get_coin_info(self, coin_id):
        """
        Get detailed information about a cryptocurrency.
        :param coin_id: ID of the cryptocurrency (e.g., 'bitcoin')
        :return: A dictionary containing detailed information like the name, symbol, and description.
        """
        endpoint = f"/coins/{coin_id}"
        response = requests.get(self.base_url + endpoint, headers=self.headers)
        if response.status_code == 200:
            data = response.json()
            return {
                'name': data['name'],
                'symbol': data['symbol'],
                'description': data['description']['en'],
                'homepage': data['links']['homepage'][0],
                'genesis_date': data['genesis_date'],
            }
        else:
            return None
    
    def get_supported_coins(self):
        """
        Get a list of supported coins by CoinGecko.
        :return: A list of coin IDs and names.
        """
        endpoint = "/coins/list"
        response = requests.get(self.base_url + endpoint, headers=self.headers)
        if response.status_code == 200:
            data = response.json()
            return [{'id': coin['id'], 'name': coin['name']} for coin in data]
        else:
            return None

if __name__ == "__main__":

    from modules.utils import get_env

    api_key = get_env("CG_API_KEY")
    coingecko = CoinGecko(api_key)
    
    # Test ping to coin gecko
    ping = coingecko.ping()
    print(ping)
