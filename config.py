"""
Configuration file for stock market prediction model
"""
import datetime
import os

# ====================
# Data Configuration
# ====================

# Default stock ticker symbols
DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

# Default ticker for quick testing
DEFAULT_TICKER = "AAPL"

# Date ranges for data fetching
DEFAULT_START_DATE = datetime.datetime(2020, 1, 1)
DEFAULT_END_DATE = datetime.datetime.now()

# Minimum data points required for model training
MIN_DATA_POINTS = 100

# ====================
# Feature Engineering
# ====================

# Moving Average periods
MA_SHORT_PERIOD = 7  # 1 week
MA_MEDIUM_PERIOD = 30  # 1 month
MA_LONG_PERIOD = 90  # 3 months

# RSI configuration
RSI_PERIOD = 14

# MACD configuration
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Bollinger Bands configuration
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2

# Volume indicators
VOLUME_MA_PERIOD = 20

# ====================
# Model Configuration
# ====================

# ARIMA Model
ARIMA_ORDER = (5, 1, 2)  # (p, d, q)

# ====================
# Training Configuration
# ====================

# Train/Test split ratio
TRAIN_SPLIT = 0.8

# ====================
# Prediction Configuration
# ====================

# Number of days to forecast
FORECAST_DAYS = 30

# Confidence interval (for uncertainty estimation)
CONFIDENCE_INTERVAL = 0.95

# ====================
# Storage Configuration
# ====================

# Directories
DATA_DIR = "data"
MODELS_DIR = "models"
PLOTS_DIR = "plots"

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# File paths
CACHE_FILE = os.path.join(DATA_DIR, "stock_data_cache.pkl")

# ====================
# Visualization Configuration
# ====================

# Plot style
PLOT_STYLE = "seaborn-v0_8-darkgrid"
FIGURE_SIZE = (14, 8)
DPI = 100

# Colors
COLOR_ACTUAL = "#1f77b4"
COLOR_PREDICTED = "#ff7f0e"
COLOR_TRAIN = "#2ca02c"
COLOR_VALIDATION = "#d62728"

# ====================
# Logging Configuration
# ====================

LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
