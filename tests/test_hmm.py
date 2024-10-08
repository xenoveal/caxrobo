import unittest
import pandas as pd
import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from modules.hmm import HMMBuilder


class TestHMMBuilder(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Simulate a sample dataset
        np.random.seed(42)
        data = {
            'Datetime': pd.date_range(start='2020-01-01', periods=100, freq='h'),
            'Open': np.random.randint(100),
            'High': np.random.randint(100),
            'Low': np.random.randint(100),
            'Close': np.random.randint(100),
            'Volume': np.random.randint(1, 1000, 100)
        }
        cls.df = pd.DataFrame(data)
        cls.df.to_csv('dataset/test_dataset.csv', index=False)

    def test_gaussian_hmm_with_standard_scaler(self):
        """
        Test HMMBuilder with GaussianHMM and MinMaxScaler.
        """
        features = ['Returns', 'Volatility', 'VolumeChange']
        hmm_builder = HMMBuilder(dataset_path='dataset/test_dataset.csv', features=features, scaler=MinMaxScaler())
        model = hmm_builder.train_hmm(n_components=3, n_iter=100, random_state=42)
        
        self.assertIsInstance(model, hmm.GaussianHMM)
        self.assertEqual(model.n_components, 3)

    def test_gaussian_hmm_with_robust_scaler(self):
        """
        Test HMMBuilder with GaussianHMM and RobustScaler.
        """
        features = ['Returns', 'Volatility', 'VolumeChange']
        hmm_builder = HMMBuilder(dataset_path='dataset/test_dataset.csv', features=features, scaler=RobustScaler())
        model = hmm_builder.train_hmm(n_components=3, n_iter=100, random_state=42)
        
        self.assertIsInstance(model, hmm.GaussianHMM)
        self.assertEqual(model.n_components, 3)

    def test_categorical_hmm(self):
        """
        Test HMMBuilder with CategoricalHMM.
        """
        features = ['Open', 'High', 'Low', 'Close', 'Volume']
        categorical_model = hmm.CategoricalHMM(n_components=3, random_state=42, verbose=True)
        hmm_builder = HMMBuilder(dataset_path='dataset/test_dataset.csv', features=features, model=categorical_model)

        model = hmm_builder.train_hmm(n_iter=100)

        self.assertIsInstance(model, hmm.CategoricalHMM)
        self.assertEqual(model.n_components, 3)

    def test_predict_states(self):
        """
        Test the predict_states method on GaussianHMM with MinMaxScaler.
        """
        features = ['Returns', 'Volatility', 'VolumeChange']
        hmm_builder = HMMBuilder(dataset_path='dataset/test_dataset.csv', features=features)
        hmm_builder.train_hmm(n_components=3, n_iter=100, random_state=42)
        
        predicted_states = hmm_builder.predict_states(hmm_builder.df)
        
        self.assertIsInstance(predicted_states, np.ndarray)
        self.assertEqual(len(predicted_states), len(hmm_builder.df))

    @classmethod
    def tearDownClass(cls):
        import os
        os.remove('dataset/test_dataset.csv')

if __name__ == '__main__':
    unittest.main()
