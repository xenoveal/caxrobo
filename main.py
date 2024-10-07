# global var
SYMBOL = "BTC-USD"
INTERVAL = "1h"
DAYS = 720

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
from modules.logger import setup_logger

logger = setup_logger('MainLogger', 'logs/main.log')


def get_save_dataset(dataset_path: str) -> None:
    logger.info("Start function to get & save data..")
    from modules.yf import YahooFinance

    yf = YahooFinance()
    csv_data: pd.DataFrame = yf.fetch_hist_data(SYMBOL, days=DAYS, interval=INTERVAL)

    csv_data.to_csv(dataset_path)


def data_processing(dataset_path: str) -> pd.DataFrame:
    logger.info("Start data processing..")
    df = pd.read_csv(dataset_path)

    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df.set_index('Datetime', inplace=True)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    
    df['Returns'] = df['Close'].pct_change()
    df['Volatility'] = df['Returns'].rolling(window=24).std()
    df['VolumeChange'] = df['Volume'].pct_change()

    logger.info(f"Dropping null & infinity rows. Initial data shape: {df.shape}")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(method='ffill', inplace=True)
    df.dropna(inplace=True)

    logger.info(f"Data processing completed. Data shape: {df.shape}")
    return df


def train_hmm(
    data: pd.DataFrame, n_components: int = 3, 
    features:list = ['Returns', 'Volatility', 'VolumeChange']
) :
    logger.info(f"Training HMM with {n_components} components...")
    X = data[features].values
    
    logger.info("Normalizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    logger.info("Fitting HMM model...")
    model = hmm.GaussianHMM(
        n_components=n_components, covariance_type="diag", 
        n_iter=2000, verbose=True, random_state=42
    )
    model.fit(X_scaled)

    logger.info("HMM training completed.")
    return model, scaler


def predict_states (
    model, data, scaler,
    features = ['Returns', 'Volatility', 'VolumeChange']
):
    logger.info("Predicting states...")
    X = data[features].values
    X_scaled = scaler.transform(X)
    states = model.predict(X_scaled)
    logger.info(f"States predicted. Unique states: {np.unique (states)}")
    return states


def analyze_states (data, states, model):
    logger.info("Analyzing states...")
    df_analysis = data.copy()
    df_analysis['State'] = states

    for state in range(model.n_components) :
        state_data = df_analysis[df_analysis['State']==state]
        logger.info(f"[State-Analysis] {state}: \n{state_data[['Returns','Volatility', 'VolumeChange']].describe()}")
        logger.info(f"[State-Analysis] {state} - Number of periods: {len(state_data)}")


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
    df = data_processing(dataset_path)

    features = ['Returns', 'Volatility', 'VolumeChange']
    
    model, scaler = train_hmm(df, 3, features)
    states = predict_states(model, df, scaler, features)

    analyze_states(df, states, model)
    plot_results(df, states, model)
    
    logger.info(f"Transition Matrix: \n{model.transmat_}")
    logger.info("Printing means and covariances of each state...")
    for i in range (model.n_components) :
        logger.info("State %s === Mean: \n%s", str(i), str(model.means_[i]))
        logger.info("State %s === Covariance: \n%s", str(i), str(model.covars_[i]))

    logger.info("Bitcoin HMM analysis completed.")


if __name__ == "__main__":
    main()