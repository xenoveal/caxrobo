# global var
SYMBOL = "BTC-USD"

# get and save historical data
import pandas as pd
from modules.yf import YahooFinance

yf = YahooFinance()
csv_data: pd.DataFrame = yf.fetch_hist_data(SYMBOL)

# save to csv for model training
csv_data.to_csv(f"dataset/{SYMBOL}.csv")