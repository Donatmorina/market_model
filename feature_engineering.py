"""
Feature engineering module for calculating technical indicators
"""
import pandas as pd
import numpy as np
import logging
from typing import Optional
import ta

import config

logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Creates technical indicators and features for stock price prediction
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize feature engineer with stock data
        
        Args:
            data: DataFrame with OHLCV data
        """
        self.data = data.copy()
        self.features_added = []
        
    def add_all_features(self) -> pd.DataFrame:
        """
        Add all technical indicators and features
        
        Returns:
            DataFrame with all features added
        """
        logger.info("Adding all technical indicators...")
        
        self.add_price_features()
        self.add_moving_averages()
        self.add_rsi()
        self.add_macd()
        self.add_bollinger_bands()
        self.add_volume_features()
        self.add_momentum_features()
        self.add_volatility_features()
        self.add_lag_features()
        
        # Remove any NaN values created by indicators
        self.data = self.data.dropna()
        
        logger.info(f"Added {len(self.features_added)} features: {', '.join(self.features_added)}")
        return self.data
    
    def add_price_features(self):
        """Add basic price-based features"""
        # Price changes
        self.data['Price_Change'] = self.data['Close'].diff()
        self.data['Price_Change_Pct'] = self.data['Close'].pct_change() * 100
        
        # Daily returns
        self.data['Daily_Return'] = self.data['Close'].pct_change()
        
        # High-Low range
        self.data['HL_Range'] = self.data['High'] - self.data['Low']
        self.data['HL_Range_Pct'] = (self.data['HL_Range'] / self.data['Close']) * 100
        
        # Open-Close difference
        self.data['OC_Diff'] = self.data['Close'] - self.data['Open']
        self.data['OC_Diff_Pct'] = (self.data['OC_Diff'] / self.data['Open']) * 100
        
        self.features_added.extend([
            'Price_Change', 'Price_Change_Pct', 'Daily_Return',
            'HL_Range', 'HL_Range_Pct', 'OC_Diff', 'OC_Diff_Pct'
        ])
    
    def add_moving_averages(self):
        """Add moving average indicators"""
        # Simple Moving Averages
        self.data['SMA_7'] = self.data['Close'].rolling(window=config.MA_SHORT_PERIOD).mean()
        self.data['SMA_30'] = self.data['Close'].rolling(window=config.MA_MEDIUM_PERIOD).mean()
        self.data['SMA_90'] = self.data['Close'].rolling(window=config.MA_LONG_PERIOD).mean()
        
        # Exponential Moving Averages
        self.data['EMA_7'] = self.data['Close'].ewm(span=config.MA_SHORT_PERIOD, adjust=False).mean()
        self.data['EMA_30'] = self.data['Close'].ewm(span=config.MA_MEDIUM_PERIOD, adjust=False).mean()
        self.data['EMA_90'] = self.data['Close'].ewm(span=config.MA_LONG_PERIOD, adjust=False).mean()
        
        # Moving average crossovers
        self.data['SMA_Cross_7_30'] = self.data['SMA_7'] - self.data['SMA_30']
        self.data['EMA_Cross_7_30'] = self.data['EMA_7'] - self.data['EMA_30']
        
        # Distance from moving averages
        self.data['Distance_SMA_30'] = ((self.data['Close'] - self.data['SMA_30']) / self.data['SMA_30']) * 100
        self.data['Distance_EMA_30'] = ((self.data['Close'] - self.data['EMA_30']) / self.data['EMA_30']) * 100
        
        self.features_added.extend([
            'SMA_7', 'SMA_30', 'SMA_90', 'EMA_7', 'EMA_30', 'EMA_90',
            'SMA_Cross_7_30', 'EMA_Cross_7_30', 'Distance_SMA_30', 'Distance_EMA_30'
        ])
    
    def add_rsi(self):
        """Add Relative Strength Index"""
        # Using ta library for RSI calculation
        self.data['RSI'] = ta.momentum.RSIIndicator(
            close=self.data['Close'],
            window=config.RSI_PERIOD
        ).rsi()
        
        # RSI overbought/oversold signals
        self.data['RSI_Overbought'] = (self.data['RSI'] > 70).astype(int)
        self.data['RSI_Oversold'] = (self.data['RSI'] < 30).astype(int)
        
        self.features_added.extend(['RSI', 'RSI_Overbought', 'RSI_Oversold'])
    
    def add_macd(self):
        """Add MACD indicator"""
        # Using ta library for MACD calculation
        macd_indicator = ta.trend.MACD(
            close=self.data['Close'],
            window_fast=config.MACD_FAST,
            window_slow=config.MACD_SLOW,
            window_sign=config.MACD_SIGNAL
        )
        
        self.data['MACD'] = macd_indicator.macd()
        self.data['MACD_Signal'] = macd_indicator.macd_signal()
        self.data['MACD_Diff'] = macd_indicator.macd_diff()
        
        # MACD crossover signal
        self.data['MACD_Bullish'] = (self.data['MACD_Diff'] > 0).astype(int)
        
        self.features_added.extend(['MACD', 'MACD_Signal', 'MACD_Diff', 'MACD_Bullish'])
    
    def add_bollinger_bands(self):
        """Add Bollinger Bands"""
        # Using ta library for Bollinger Bands
        bollinger = ta.volatility.BollingerBands(
            close=self.data['Close'],
            window=config.BOLLINGER_PERIOD,
            window_dev=config.BOLLINGER_STD
        )
        
        self.data['BB_High'] = bollinger.bollinger_hband()
        self.data['BB_Mid'] = bollinger.bollinger_mavg()
        self.data['BB_Low'] = bollinger.bollinger_lband()
        
        # Bollinger Band width (volatility measure)
        self.data['BB_Width'] = ((self.data['BB_High'] - self.data['BB_Low']) / self.data['BB_Mid']) * 100
        
        # Price position within bands
        self.data['BB_Position'] = ((self.data['Close'] - self.data['BB_Low']) / 
                                    (self.data['BB_High'] - self.data['BB_Low']))
        
        # Bollinger Band breakout signals
        self.data['BB_Upper_Break'] = (self.data['Close'] > self.data['BB_High']).astype(int)
        self.data['BB_Lower_Break'] = (self.data['Close'] < self.data['BB_Low']).astype(int)
        
        self.features_added.extend([
            'BB_High', 'BB_Mid', 'BB_Low', 'BB_Width', 'BB_Position',
            'BB_Upper_Break', 'BB_Lower_Break'
        ])
    
    def add_volume_features(self):
        """Add volume-based features"""
        # Volume moving average
        self.data['Volume_MA'] = self.data['Volume'].rolling(window=config.VOLUME_MA_PERIOD).mean()
        
        # Volume ratio
        self.data['Volume_Ratio'] = self.data['Volume'] / self.data['Volume_MA']
        
        # Volume change
        self.data['Volume_Change_Pct'] = self.data['Volume'].pct_change() * 100
        
        # On-Balance Volume (OBV)
        self.data['OBV'] = ta.volume.OnBalanceVolumeIndicator(
            close=self.data['Close'],
            volume=self.data['Volume']
        ).on_balance_volume()
        
        # Volume Price Trend
        self.data['VPT'] = ta.volume.VolumePriceTrendIndicator(
            close=self.data['Close'],
            volume=self.data['Volume']
        ).volume_price_trend()
        
        self.features_added.extend([
            'Volume_MA', 'Volume_Ratio', 'Volume_Change_Pct', 'OBV', 'VPT'
        ])
    
    def add_momentum_features(self):
        """Add momentum indicators"""
        # Rate of Change
        self.data['ROC'] = ta.momentum.ROCIndicator(
            close=self.data['Close'],
            window=12
        ).roc()
        
        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(
            high=self.data['High'],
            low=self.data['Low'],
            close=self.data['Close'],
            window=14,
            smooth_window=3
        )
        self.data['Stochastic'] = stoch.stoch()
        self.data['Stochastic_Signal'] = stoch.stoch_signal()
        
        # Williams %R
        self.data['Williams_R'] = ta.momentum.WilliamsRIndicator(
            high=self.data['High'],
            low=self.data['Low'],
            close=self.data['Close'],
            lbp=14
        ).williams_r()
        
        self.features_added.extend(['ROC', 'Stochastic', 'Stochastic_Signal', 'Williams_R'])
    
    def add_volatility_features(self):
        """Add volatility measures"""
        # Average True Range (ATR)
        self.data['ATR'] = ta.volatility.AverageTrueRange(
            high=self.data['High'],
            low=self.data['Low'],
            close=self.data['Close'],
            window=14
        ).average_true_range()
        
        # Historical volatility (rolling standard deviation of returns)
        self.data['Volatility_20'] = self.data['Daily_Return'].rolling(window=20).std() * np.sqrt(252) * 100
        
        self.features_added.extend(['ATR', 'Volatility_20'])
    
    def add_lag_features(self, lags: list = [1, 2, 3, 5, 7]):
        """
        Add lagged price features
        
        Args:
            lags: List of lag periods to create
        """
        for lag in lags:
            self.data[f'Close_Lag_{lag}'] = self.data['Close'].shift(lag)
            self.data[f'Return_Lag_{lag}'] = self.data['Daily_Return'].shift(lag)
            self.features_added.extend([f'Close_Lag_{lag}', f'Return_Lag_{lag}'])
    
    def get_feature_names(self) -> list:
        """Get list of all created features"""
        return self.features_added
    
    def get_data(self) -> pd.DataFrame:
        """Get the data with all features"""
        return self.data


def create_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function to create all features
    
    Args:
        data: Raw OHLCV data
        
    Returns:
        DataFrame with all features
    """
    engineer = FeatureEngineer(data)
    return engineer.add_all_features()


if __name__ == "__main__":
    # Example usage
    from data_fetcher import StockDataFetcher
    
    fetcher = StockDataFetcher("AAPL")
    data = fetcher.fetch_data()
    
    engineer = FeatureEngineer(data)
    data_with_features = engineer.add_all_features()
    
    print(f"\nOriginal columns: {list(data.columns)}")
    print(f"\nNew features added: {engineer.get_feature_names()}")
    print(f"\nTotal columns: {len(data_with_features.columns)}")
    print(f"\nData shape: {data_with_features.shape}")
    print(f"\nLast row:\n{data_with_features.iloc[-1]}")
