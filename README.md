# Stock Market Price Prediction with ARIMA

A minimal stock price prediction system using historical data and ARIMA time series modeling.

## Features

- **Automatic Data Fetching**: Uses yfinance to retrieve stock data from Yahoo Finance (no API key needed)
- **Feature Engineering**: Calculates technical indicators (MA, RSI, MACD, Bollinger Bands)
- **ARIMA Model**: Statistical time series forecasting with confidence intervals
- **Visualization**: Plots technical analysis and forecast predictions

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Quick Prediction
```bash
python main.py --ticker AAPL
```

### Custom Parameters
```bash
python main.py --ticker MSFT --days 1000 --forecast 14
```

### Visualize Only (No Training)
```bash
python main.py --ticker GOOGL --visualize-only
```

### Skip Plots
```bash
python main.py --ticker AAPL --no-plots
```

## Project Structure

```
├── main.py                  # Main script - runs complete pipeline
├── config.py                # Configuration settings
├── data_fetcher.py          # Yahoo Finance data retrieval
├── feature_engineering.py   # Technical indicators calculation
├── models.py                # ARIMA model implementation
├── utils.py                 # Visualization functions
├── requirements.txt         # Dependencies
├── data/                    # Cached data storage
├── models/                  # Saved models
└── plots/                   # Generated visualizations
```

## Model

**ARIMA (AutoRegressive Integrated Moving Average)**
- Statistical time series model
- Captures linear trends and patterns
- Fast training and prediction
- Provides confidence intervals for uncertainty estimation
- Default order: (5, 1, 2)

## Data Source

- **Yahoo Finance** via yfinance library
- Free access, no API key required
- Historical data: Open, High, Low, Close, Volume

## Technical Indicators

- Simple Moving Average (SMA) - 7, 30, 90 days
- Exponential Moving Average (EMA) - 7, 30 days
- Relative Strength Index (RSI) - 14 day
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Volume indicators

## Example Output

```
ARIMA Forecast (next 7 days):
Date         Predicted    Lower CI     Upper CI    
--------------------------------------------------
2026-01-20   $255.42     $253.87     $256.97
2026-01-21   $255.68     $253.12     $258.24
...
```

## License

MIT License
