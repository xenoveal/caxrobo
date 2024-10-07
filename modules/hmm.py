import pandas as pd
import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
from modules.logger import setup_logger

logger = setup_logger('HMMLogger', 'logs/hmm.log')

class HMMBuilder:
    """
    A class used to build and train a Hidden Markov Model (HMM) 
    on a given dataset with specific features.

    Attributes:
    ----------
    dataset_path : str
        Path to the CSV dataset containing time-series data.
    features : list
        List of feature names used for training the model.
    scaler : object, optional
        A scaler object to normalize the features, default is StandardScaler.
    model : object, optional
        The HMM model object from hmmlearn, default is GaussianHMM.
    """

    def __init__(self, dataset_path: str, features: list, 
                 scaler: any = StandardScaler(), model: any = hmm.GaussianHMM(verbose=True)):
        """
        Initializes the HMMBuilder with dataset path, features, scaler, and HMM model.
        
        Parameters:
        ----------
        dataset_path : str
            Path to the dataset.
        features : list
            Features to use for training the HMM model.
        scaler : object
            Scaler to normalize the data, default is StandardScaler.
        model : object
            Hidden Markov Model object from hmmlearn, default is GaussianHMM.
        """
        self.features = features
        self.df = self._data_processing(dataset_path)
        self.scaler = scaler
        self.model = model

    def _data_processing(self, dataset_path: str) -> pd.DataFrame:
        """
        Private method to load, clean, and process the dataset.
        
        Parameters:
        ----------
        dataset_path : str
            Path to the dataset CSV file.
            
        Returns:
        -------
        pd.DataFrame
            A processed DataFrame ready for model training.
        """
        logger.info("Start data processing..")
        df = pd.read_csv(dataset_path)

        df['Datetime'] = pd.to_datetime(df['Datetime'])
        df.set_index('Datetime', inplace=True)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

        # Feature Engineering
        df['Returns'] = df['Close'].pct_change()
        df['Volatility'] = df['Returns'].rolling(window=24).std()
        df['VolumeChange'] = df['Volume'].pct_change()

        # Handle missing values and infinities
        logger.info(f"Dropping null & infinity rows. Initial data shape: {df.shape}")
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.ffill(inplace=True)
        df.dropna(inplace=True)

        logger.info(f"Data processing completed. Data shape: {df.shape}")
        return df

    def train_hmm(self, **params) -> any:
        """
        Trains the Hidden Markov Model on the processed dataset.
        
        Parameters:
        ----------
        **params : keyword arguments
            Parameters to set in the HMM model.

        Returns:
        -------
        any
            The trained HMM model.
        """
        logger.info(f"Training HMM model...")
        
        # Extract feature data
        X = self.df[self.features].values

        # Scale the features
        logger.info("Normalizing features...")
        X_scaled = self.scaler.fit_transform(X)

        # Fit the HMM model
        logger.info("Fitting HMM model...")
        self.model.set_params(**params)
        self.model.fit(X_scaled)

        logger.info("HMM training completed.")
        return self.model

    def predict_states(self, to_predict: pd.DataFrame) -> np.ndarray:
        """
        Predicts the hidden states for the provided data.
        
        Parameters:
        ----------
        to_predict : pd.DataFrame
            DataFrame containing the features used for prediction.
        
        Returns:
        -------
        np.ndarray
            An array of predicted states.
        """
        logger.info("Predicting hidden states...")
        X = to_predict[self.features].values
        X_scaled = self.scaler.transform(X)
        
        states = self.model.predict(X_scaled)
        logger.info(f"States predicted. Unique states: {np.unique(states)}")
        return states
