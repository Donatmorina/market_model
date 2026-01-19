"""
Utility functions for visualization and metrics
"""
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import os
from typing import Optional, Tuple
import logging
import config

logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

# Set plot style
try:
    plt.style.use(config.PLOT_STYLE)
except:
    plt.style.use('seaborn-v0_8')


def plot_stock_data(
    data: pd.DataFrame,
    ticker: str,
    save: bool = True,
    filename: Optional[str] = None
) -> None:
    """
    Plot historical stock data with technical indicators
    
    Args:
        data: DataFrame with stock data and features
        ticker: Stock ticker symbol
        save: Whether to save the plot
        filename: Custom filename for saving
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Plot 1: Price and Moving Averages
    axes[0].plot(data.index, data['Close'], label='Close Price', linewidth=2)
    if 'SMA_7' in data.columns:
        axes[0].plot(data.index, data['SMA_7'], label='SMA 7', alpha=0.7)
    if 'SMA_30' in data.columns:
        axes[0].plot(data.index, data['SMA_30'], label='SMA 30', alpha=0.7)
    if 'EMA_30' in data.columns:
        axes[0].plot(data.index, data['EMA_30'], label='EMA 30', alpha=0.7, linestyle='--')
    
    axes[0].set_title(f'{ticker} - Price and Moving Averages', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Price ($)', fontsize=12)
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Volume
    axes[1].bar(data.index, data['Volume'], alpha=0.5, color='skyblue')
    if 'Volume_MA' in data.columns:
        axes[1].plot(data.index, data['Volume_MA'], color='orange', label='Volume MA', linewidth=2)
    
    axes[1].set_title('Trading Volume', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Volume', fontsize=12)
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: RSI
    if 'RSI' in data.columns:
        axes[2].plot(data.index, data['RSI'], label='RSI', color='purple', linewidth=2)
        axes[2].axhline(y=70, color='r', linestyle='--', label='Overbought', alpha=0.7)
        axes[2].axhline(y=30, color='g', linestyle='--', label='Oversold', alpha=0.7)
        axes[2].fill_between(data.index, 30, 70, alpha=0.1)
        
        axes[2].set_title('Relative Strength Index (RSI)', fontsize=14, fontweight='bold')
        axes[2].set_ylabel('RSI', fontsize=12)
        axes[2].set_ylim(0, 100)
        axes[2].legend(loc='best')
        axes[2].grid(True, alpha=0.3)
    
    axes[2].set_xlabel('Date', fontsize=12)
    
    plt.tight_layout()
    
    if save:
        if filename is None:
            filename = f"{ticker}_technical_analysis.png"
        filepath = os.path.join(config.PLOTS_DIR, filename)
        plt.savefig(filepath, dpi=config.DPI, bbox_inches='tight')
        logger.info(f"Plot saved to {filepath}")
    
    plt.show()


def plot_forecast(
    historical_dates: pd.DatetimeIndex,
    historical_prices: np.ndarray,
    forecast_dates: pd.DatetimeIndex,
    forecast_prices: np.ndarray,
    ticker: str,
    model_name: str = "Model",
    confidence_intervals: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    save: bool = True,
    filename: Optional[str] = None
) -> None:
    """
    Plot historical data with future forecast
    
    Args:
        historical_dates: Historical dates
        historical_prices: Historical prices
        forecast_dates: Forecast dates
        forecast_prices: Forecasted prices
        ticker: Stock ticker symbol
        model_name: Name of the model
        confidence_intervals: Tuple of (lower, upper) confidence bounds
        save: Whether to save the plot
        filename: Custom filename for saving
    """
    plt.figure(figsize=config.FIGURE_SIZE)
    
    # Plot historical data
    plt.plot(historical_dates, historical_prices, 
            label='Historical Price', color=config.COLOR_ACTUAL, linewidth=2)
    
    # Plot forecast
    plt.plot(forecast_dates, forecast_prices, 
            label=f'{model_name} Forecast', color=config.COLOR_PREDICTED, 
            linewidth=2, linestyle='--', marker='o')
    
    # Add confidence intervals if provided
    if confidence_intervals is not None:
        lower, upper = confidence_intervals
        plt.fill_between(forecast_dates, lower, upper, 
                        alpha=0.2, color=config.COLOR_PREDICTED,
                        label='Confidence Interval')
    
    # Connect historical to forecast
    plt.plot([historical_dates[-1], forecast_dates[0]], 
            [historical_prices[-1], forecast_prices[0]],
            color=config.COLOR_PREDICTED, linewidth=2, linestyle=':')
    
    plt.title(f'{ticker} - {model_name} Forecast', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price ($)', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Format x-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()
    
    plt.tight_layout()
    
    if save:
        if filename is None:
            filename = f"{ticker}_{model_name.lower()}_forecast.png"
        filepath = os.path.join(config.PLOTS_DIR, filename)
        plt.savefig(filepath, dpi=config.DPI, bbox_inches='tight')
        logger.info(f"Plot saved to {filepath}")
    
    plt.show()


def print_metrics(metrics: dict, model_name: str = "Model") -> None:
    """
    Print model evaluation metrics
    
    Args:
        metrics: Dictionary of metrics
        model_name: Name of the model
    """
    print(f"\n{model_name} Performance Metrics:")
    print("-" * 40)
    for key, value in metrics.items():
        print(f"  {key:<10}: {value:.4f}")
    print()
