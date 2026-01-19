"""
ARIMA model for stock price prediction
"""
import numpy as np
import pandas as pd
import logging
from typing import Tuple
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
import config

logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)


class ARIMAModel:
    """
    ARIMA (AutoRegressive Integrated Moving Average) model for time series forecasting
    """
    
    def __init__(self, order: Tuple[int, int, int] = None):
        """
        Initialize ARIMA model
        
        Args:
            order: ARIMA order (p, d, q)
        """
        self.order = order or config.ARIMA_ORDER
        self.model = None
        self.fitted_model = None
        self.history = None
        
    def train(self, data: pd.Series) -> None:
        """
        Train the ARIMA model
        
        Args:
            data: Time series data (closing prices)
        """
        logger.info(f"Training ARIMA model with order {self.order}")
        
        self.history = data.copy()
        
        try:
            # Fit ARIMA model
            self.model = ARIMA(data, order=self.order)
            self.fitted_model = self.model.fit()
            
            logger.info("ARIMA model trained successfully")
            logger.info(f"AIC: {self.fitted_model.aic:.2f}, BIC: {self.fitted_model.bic:.2f}")
            
        except Exception as e:
            logger.error(f"Error training ARIMA model: {str(e)}")
            raise
    
    def predict(self, steps: int = 1) -> np.ndarray:
        """
        Make predictions
        
        Args:
            steps: Number of steps to forecast
            
        Returns:
            Array of predictions
        """
        if self.fitted_model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        forecast = self.fitted_model.forecast(steps=steps)
        return forecast.values
    
    def get_confidence_intervals(self, steps: int = 1, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get confidence intervals for predictions
        
        Args:
            steps: Number of steps to forecast
            alpha: Significance level (default 0.05 for 95% CI)
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if self.fitted_model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        forecast_result = self.fitted_model.get_forecast(steps=steps)
        conf_int = forecast_result.conf_int(alpha=alpha)
        
        return conf_int.iloc[:, 0].values, conf_int.iloc[:, 1].values
    
    def save(self, filepath: str) -> None:
        """Save the model"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'order': self.order,
                'fitted_model': self.fitted_model,
                'history': self.history
            }, f)
        logger.info(f"ARIMA model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'ARIMAModel':
        """Load a saved model"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        model = cls(order=data['order'])
        model.fitted_model = data['fitted_model']
        model.history = data['history']
        
        logger.info(f"ARIMA model loaded from {filepath}")
        return model


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Evaluate model performance
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary of metrics
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape
    }
    
    return metrics


if __name__ == "__main__":
    # Example usage
    from data_fetcher import StockDataFetcher
    from feature_engineering import create_features
    
    # Fetch and prepare data
    fetcher = StockDataFetcher("AAPL")
    data = fetcher.fetch_data()
    data_with_features = create_features(data)
    
    # Test ARIMA
    print("\n=== Testing ARIMA Model ===")
    arima = ARIMAModel()
    arima.train(data_with_features['Close'])
    predictions = arima.predict(steps=7)
    print(f"7-day forecast: {predictions}")
