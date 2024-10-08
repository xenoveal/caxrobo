import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from modules.logger import setup_logger

logger = setup_logger('BacktestLogger', 'logs/backtest.log')

class Backtest:
    """
    Class for backtesting a trading strategy based on HMM hidden states.
    
    Attributes:
    ----------
    data : pd.DataFrame
        The dataset with features and predicted states.
    states : np.ndarray
        The predicted hidden states from the HMM model.
    initial_balance : float
        Starting balance for the backtest.
    """

    def __init__(self, data: pd.DataFrame, states: np.ndarray, initial_balance: float = 10000,
                 buy_state: list = [1], sell_state: list = [2]):
        """
        Initializes the Backtest class with data, states, and initial balance.

        Parameters:
        ----------
        data : pd.DataFrame
            Data containing market features.
        states : np.ndarray
            Array of predicted hidden states.
        initial_balance : float, optional
            Starting balance for backtesting (default is 10,000).
        buy_state : list, optional
            State which defines buy action (default is "1").
        sell_state : list, optional
            State which defines buy action (default is "2").
        """
        self.data = data
        self.data['State'] = states
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.position = 0  # Holds the number of BTC we own
        self.portfolio_value = []
        self.buy_state = buy_state
        self.sell_state = sell_state

    def strategy(self) -> pd.DataFrame:
        """
        Implements a basic strategy where we buy Bitcoin if the state is 0 (bullish)
        and sell if the state is 1 (bearish).

        Returns:
        -------
        pd.DataFrame
            DataFrame containing the portfolio value over time.
        """
        logger.info("Starting backtest strategy...")

        for i in range(len(self.data)):
            price = self.data['Close'].iloc[i]

            if self.data['State'].iloc[i] in self.buy_state and self.position == 0:
                # Buy signal
                self.position = self.balance / price  # Buy BTC with available balance
                self.balance = 0  # We spent all balance on buying
                logger.debug(f"Buying at {price:.2f}, Position: {self.position:.4f} BTC")
                
            elif self.data['State'].iloc[i] in self.sell_state and self.position > 0:
                # Sell signal
                self.balance = self.position * price  # Sell all BTC
                logger.debug(f"Selling at {price:.2f}, Balance: {self.balance:.2f} USD")
                self.position = 0  # No BTC left

            # Track portfolio value
            portfolio_value = self.balance + self.position * price  # Cash + BTC value
            self.portfolio_value.append(portfolio_value)

        logger.info("Backtest strategy completed.")
        return pd.DataFrame({
            'Datetime': self.data.index,
            'PortfolioValue': self.portfolio_value
        })

    def plot_performance(self) -> None:
        """
        Plots the portfolio performance over time.
        """
        logger.info("Plotting portfolio performance...")
        plt.figure(figsize=(12, 6))
        plt.plot(self.data.index, self.portfolio_value, label='Portfolio Value', color='b')
        plt.title('Portfolio Performance Over Time')
        plt.xlabel('Datetime')
        plt.ylabel('Portfolio Value (USD)')
        plt.legend()
        plt.grid(True)
        plt.show()

        logger.info("Portfolio performance plotted.")
