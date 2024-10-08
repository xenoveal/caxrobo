import unittest
import pandas as pd
import numpy as np
from modules.backtest import Backtest

class TestBacktest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Simulate a sample dataset with timestamps and close prices
        np.random.seed(42)
        
        # Generate a date range for the dataset
        dates = pd.date_range(start="2023-01-01", periods=100, freq='D')
        
        # Simulate close prices and hidden states
        close_prices = np.arange(101, 201)  # Gradually increasing prices
        states = np.random.choice([0, 1, 2], size=100)  # Random hidden states (buy:0, sell:1, hold:2)
        
        # Create DataFrame
        cls.df = pd.DataFrame({
            'Datetime': dates,
            'Close': close_prices
        }).set_index('Datetime')
        
        cls.states = states

    def test_backtest_with_default_states(self):
        """
        Test the Backtest class with default buy and sell states (buy=[1], sell=[2]).
        """
        backtest = Backtest(data=self.df, states=self.states, initial_balance=1000)
        result = backtest.strategy()
        
        # Check if the result is a DataFrame
        self.assertIsInstance(result, pd.DataFrame)
        
        # Check if portfolio values are calculated
        self.assertEqual(len(result), len(self.df))
        self.assertTrue(all(result['PortfolioValue'] > 0))

    def test_backtest_with_custom_states(self):
        """
        Test the Backtest class with custom buy/sell states (e.g., buy=[0, 2], sell=[1]).
        """
        backtest = Backtest(data=self.df, states=self.states, initial_balance=1000, buy_state=[0, 2], sell_state=[1])
        result = backtest.strategy()

        # Check if the result is a DataFrame
        self.assertIsInstance(result, pd.DataFrame)
        
        # Ensure the portfolio value changes over time
        self.assertEqual(len(result), len(self.df))
        self.assertTrue(all(result['PortfolioValue'] > 0))

    def test_portfolio_growth(self):
        """
        Test that the portfolio grows when there are favorable buy/sell signals.
        """
        # Create a deterministic state sequence where buy happens early, sell later
        states = np.array([0] * 50 + [1] * 50)  # Buy for 50 days, sell for the next 50 days
        backtest = Backtest(data=self.df, states=states, initial_balance=1000, buy_state=[0], sell_state=[1])
        result = backtest.strategy()
        
        # Check that the final portfolio value is higher than the initial balance
        self.assertGreater(result['PortfolioValue'].iloc[-1], 1000)

    def test_plot_performance(self):
        """
        Test the plot_performance function does not raise errors.
        """
        backtest = Backtest(data=self.df, states=self.states, initial_balance=1000)
        result = backtest.strategy()
        
        # Simply check if the plot function works without errors
        try:
            backtest.plot_performance()
            plot_success = True
        except Exception as e:
            plot_success = False
            print(f"Plotting failed with error: {e}")
        
        self.assertTrue(plot_success)

if __name__ == '__main__':
    unittest.main()
