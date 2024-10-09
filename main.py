# global var
SYMBOL = "BTC-USD"
INTERVAL = "1h"
DAYS = 720

import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations

from modules.hmm import HMMBuilder
from modules.logger import setup_logger
from modules.backtest import Backtest

logger = setup_logger('MainLogger', 'logs/main.log')


def get_save_dataset(dataset_path: str) -> None:
    logger.info("Start function to get & save data..")
    from modules.yf import YahooFinance

    yf = YahooFinance()
    csv_data: pd.DataFrame = yf.fetch_hist_data(SYMBOL, days=DAYS, interval=INTERVAL)

    csv_data.to_csv(dataset_path)

def analyze_states(data, states, model):
    logger.info("Analyzing states...")
    df_analysis = data.copy()
    df_analysis['State'] = states

    for state in range(model.n_components):
        state_data = df_analysis[df_analysis['State'] == state]
        logger.info(f"[State-Analysis] {state} - Number of periods: {len(state_data)}")
        logger.info(f"[State-Analysis] {state}: \n{state_data[['Returns', 'Volatility', 'VolumeChange']].describe()}")

def plot_results(data, states, model, portfolio_value):
    logger.info("Plotting results with portfolio overlay...")
    
    fig, ax1 = plt.subplots(figsize=(15, 7))
    
    # Plot BTC closing price
    ax1.plot(data.index, data['Close'], label='BTC Price', color='blue', lw=2)
    ax1.set_ylabel('BTC Price', color='blue')
    
    # Fill the states on the same BTC price chart
    for state in range(model.n_components):
        mask = (states == state)
        ax1.fill_between(
            data.index, data['Close'].min(), data['Close'].max(),
            where=mask, alpha=0.3, label=f'State {state}'
        )
    ax1.legend(loc='upper left')

    # Create a second y-axis to overlay portfolio value
    ax2 = ax1.twinx()
    ax2.plot(data.index, portfolio_value, label='Portfolio Value', color='green', lw=2)
    ax2.set_ylabel('Portfolio Value (USD)', color='green')
    ax2.legend(loc='upper right')
    
    plt.title('BTC Price and Portfolio Value Over Time')
    plt.tight_layout()

    plt.show()

def bruteforce_backtest(to_predict, states, initial_balance, model) -> dict:
    logger.info("Running bruteforce backtest...")

    best_result = {
        "buy_states": None,
        "sell_states": None,
        "final_portfolio_value": 0
    }

    # Test all possible combinations of buy and sell states
    state_indices = list(range(model.n_components))
    
    for buy_comb_len in range(1, model.n_components):  # Buy states can range from 1 to (n_components - 1)
        buy_combinations = combinations(state_indices, buy_comb_len)

        for buy_states in buy_combinations:
            # Test remaining states for selling
            for sell_comb_len in range(1, model.n_components):
                sell_combinations = combinations([s for s in state_indices if s not in buy_states], sell_comb_len)

                for sell_states in sell_combinations:
                    # Run backtest for each combination of buy/sell states
                    backtest = Backtest(data=to_predict, states=states, initial_balance=initial_balance,
                                        buy_state=list(buy_states), sell_state=list(sell_states))

                    result = backtest.strategy()
                    final_value = backtest.portfolio_value[-1]

                    if final_value > best_result["final_portfolio_value"]:
                        best_result["buy_states"] = list(buy_states)
                        best_result["sell_states"] = list(sell_states)
                        best_result["final_portfolio_value"] = final_value

                    logger.info(f"Tested buy={list(buy_states)}, sell={list(sell_states)} -> Final Portfolio Value: {final_value}")

    return best_result

def main():
    dataset_path = f"dataset/{SYMBOL}-{DAYS}-{INTERVAL}.csv"
    get_save_dataset(dataset_path)

    # Initialize and train the HMM model
    hmm = HMMBuilder(dataset_path=dataset_path,
                     features=['Returns', 'Volatility', 'VolumeChange'])

    hmm.train_hmm(n_components=7, covariance_type="full", n_iter=100, random_state=42)

    # Predict hidden states
    to_predict = hmm.df
    states = hmm.predict_states(to_predict)

    analyze_states(to_predict, states, hmm.model)

    logger.info(f"Transition Matrix: \n{hmm.model.transmat_}")
    logger.info("Printing means and covariances of each state...")
    for i in range(hmm.model.n_components):
        logger.info("State %s === Mean: \n%s", str(i), str(hmm.model.means_[i]))
        logger.info("State %s === Covariance: \n%s", str(i), str(hmm.model.covars_[i]))

    # Bruteforce backtest to find the best buy/sell states
    # best_result = bruteforce_backtest(to_predict, states, initial_balance=100, model=hmm.model)
    # logger.info(f"Best Buy States: {best_result['buy_states']}")
    # logger.info(f"Best Sell States: {best_result['sell_states']}")
    # logger.info(f"Backtesting - Final Portfolio Value: {best_result['final_portfolio_value']}")

    '''
    2024-10-08 21:25:34,670 - MainLogger - INFO - Best Buy States: [2, 3, 6]
    2024-10-08 21:25:34,670 - MainLogger - INFO - Best Sell States: [0, 4, 5]
    2024-10-08 21:25:34,671 - MainLogger - INFO - Backtesting - Final Portfolio Value: 737.1641624345604
    '''
    best_result = {
        'buy_states': [2, 3, 6],
        'sell_states': [0, 4, 5],
        'final_portfolio_value': 737.16
    }
    best_backtest = Backtest(data=to_predict, states=states, initial_balance=100,
                            buy_state=best_result['buy_states'], sell_state=best_result['sell_states'])
    best_backtest.strategy()
    # best_backtest.plot_performance()
    plot_results(to_predict, states, hmm.model, best_backtest.portfolio_value)

if __name__ == "__main__":
    main()
