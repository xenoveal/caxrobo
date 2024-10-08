# global var
SYMBOL = "BTC-USD"
INTERVAL = "1h"
DAYS = 730

import pandas as pd
import matplotlib.pyplot as plt 

from modules.hmm import HMMBuilder
from modules.logger import setup_logger

logger = setup_logger('MainLogger', 'logs/main.log')


def get_save_dataset(dataset_path: str) -> None:
    logger.info("Start function to get & save data..")
    from modules.yf import YahooFinance

    yf = YahooFinance()
    csv_data: pd.DataFrame = yf.fetch_hist_data(SYMBOL, days=DAYS, interval=INTERVAL)

    csv_data.to_csv(dataset_path)


def analyze_states (data, states, model):
    logger.info("Analyzing states...")
    df_analysis = data.copy()
    df_analysis['State'] = states

    for state in range(model.n_components) :
        state_data = df_analysis[df_analysis['State']==state]
        logger.info(f"[State-Analysis] {state} - Number of periods: {len(state_data)}")
        logger.info(f"[State-Analysis] {state}: \n{state_data[['Returns','Volatility', 'VolumeChange']].describe()}")


def plot_results (data, states, model):
    logger.info("Plotting results...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize= (15, 10), sharex=True)
    ax1.plot (data. index, data ['Close'])
    for state in range (model.n_components) :
        mask = (states == state)
        ax1.fill_between(
            data. index, data ['Close'].min(), data ['Close'].max(),
            where=mask, alpha=0.3, label=f'State {state}'
        )
    ax1.legend()
    ax2.plot (data.index, data['Returns'])
    ax2.set_title('Bitcoin Returns') 
    ax2.set_ylabel('Returns') 
    ax2.set_xlabel('Datetime')

    plt.tight_layout ()
    logger.info("Showing plot...")
    plt.show()


def main():
    dataset_path = f"dataset/{SYMBOL}-{DAYS}-{INTERVAL}.csv"
    get_save_dataset(dataset_path)
    
    hmm = HMMBuilder(dataset_path=dataset_path, 
                     features=['Returns', 'Volatility', 'VolumeChange'])

    hmm.train_hmm(n_components=7, covariance_type = "full", 
                  n_iter = 100, random_state = 42)

    to_predict = hmm.df
    states = hmm.predict_states(to_predict)

    analyze_states(to_predict, states, hmm.model)
    plot_results(to_predict, states, hmm.model)
    
    logger.info(f"Transition Matrix: \n{hmm.model.transmat_}")
    logger.info("Printing means and covariances of each state...")
    for i in range (hmm.model.n_components) :
        logger.info("State %s === Mean: \n%s", str(i), str(hmm.model.means_[i]))
        logger.info("State %s === Covariance: \n%s", str(i), str(hmm.model.covars_[i]))

    logger.info("Bitcoin HMM analysis completed.")


if __name__ == "__main__":
    main()