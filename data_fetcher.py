"""
Data fetcher module for retrieving historical stock data
"""
import yfinance as yf
import pandas as pd
import datetime
import logging
from typing import Optional, Tuple
import pickle
import os

import config

logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)


class StockDataFetcher:
    """
    Fetches and manages historical stock data from Yahoo Finance using yfinance
    """
    
    def __init__(self, ticker: str, use_cache: bool = True):
        """
        Initialize the data fetcher
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
            use_cache: Whether to use cached data if available
        """
        self.ticker = ticker.upper()
        self.use_cache = use_cache
        self.data = None
        self.yf_ticker = None
        
    def fetch_data(
        self, 
        start_date: Optional[datetime.datetime] = None,
        end_date: Optional[datetime.datetime] = None,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch historical stock data from Yahoo Finance
        
        Args:
            start_date: Start date for historical data
            end_date: End date for historical data
            interval: Data interval (1d, 1h, 1wk, 1mo)
            
        Returns:
            DataFrame with historical stock data
        """
        start_date = start_date or config.DEFAULT_START_DATE
        end_date = end_date or config.DEFAULT_END_DATE
        
        logger.info(f"Fetching data for {self.ticker} from {start_date.date()} to {end_date.date()}")
        
        # Check cache first
        if self.use_cache:
            cached_data = self._load_from_cache(start_date, end_date)
            if cached_data is not None:
                logger.info(f"Loaded {len(cached_data)} records from cache")
                self.data = cached_data
                return self.data
        
        try:
            # Fetch data from Yahoo Finance
            self.yf_ticker = yf.Ticker(self.ticker)
            self.data = self.yf_ticker.history(
                start=start_date,
                end=end_date,
                interval=interval
            )
            
            if self.data.empty:
                raise ValueError(f"No data found for ticker {self.ticker}")
            
            logger.info(f"Fetched {len(self.data)} records for {self.ticker}")
            
            # Clean the data
            self.data = self._clean_data(self.data)
            
            # Save to cache
            if self.use_cache:
                self._save_to_cache(start_date, end_date)
            
            return self.data
            
        except Exception as e:
            logger.error(f"Error fetching data for {self.ticker}: {str(e)}")
            raise
    
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and prepare the data
        
        Args:
            data: Raw data from yfinance
            
        Returns:
            Cleaned DataFrame
        """
        # Remove any rows with missing values
        data = data.dropna()
        
        # Ensure index is datetime
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)
        
        # Sort by date
        data = data.sort_index()
        
        # Remove timezone info if present
        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)
        
        return data
    
    def get_latest_price(self) -> float:
        """
        Get the most recent closing price
        
        Returns:
            Latest closing price
        """
        if self.data is None or self.data.empty:
            raise ValueError("No data available. Call fetch_data() first.")
        
        return self.data['Close'].iloc[-1]
    
    def get_price_range(self) -> Tuple[float, float]:
        """
        Get the min and max prices in the dataset
        
        Returns:
            Tuple of (min_price, max_price)
        """
        if self.data is None or self.data.empty:
            raise ValueError("No data available. Call fetch_data() first.")
        
        return self.data['Close'].min(), self.data['Close'].max()
    
    def get_info(self) -> dict:
        """
        Get stock information from yfinance
        
        Returns:
            Dictionary with stock information
        """
        if self.yf_ticker is None:
            self.yf_ticker = yf.Ticker(self.ticker)
        
        try:
            info = self.yf_ticker.info
            return {
                'symbol': self.ticker,
                'name': info.get('longName', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 'N/A'),
                'currency': info.get('currency', 'USD')
            }
        except Exception as e:
            logger.warning(f"Could not fetch info for {self.ticker}: {str(e)}")
            return {'symbol': self.ticker}
    
    def _get_cache_filename(self, start_date: datetime.datetime, end_date: datetime.datetime) -> str:
        """Generate cache filename based on ticker and date range"""
        start_str = start_date.strftime('%Y%m%d')
        end_str = end_date.strftime('%Y%m%d')
        return os.path.join(config.DATA_DIR, f"{self.ticker}_{start_str}_{end_str}.pkl")
    
    def _load_from_cache(self, start_date: datetime.datetime, end_date: datetime.datetime) -> Optional[pd.DataFrame]:
        """Load data from cache if available"""
        cache_file = self._get_cache_filename(start_date, end_date)
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                logger.info(f"Loaded data from cache: {cache_file}")
                return data
            except Exception as e:
                logger.warning(f"Error loading cache: {str(e)}")
                return None
        
        return None
    
    def _save_to_cache(self, start_date: datetime.datetime, end_date: datetime.datetime):
        """Save data to cache"""
        cache_file = self._get_cache_filename(start_date, end_date)
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(self.data, f)
            logger.info(f"Saved data to cache: {cache_file}")
        except Exception as e:
            logger.warning(f"Error saving cache: {str(e)}")


def fetch_multiple_tickers(
    tickers: list,
    start_date: Optional[datetime.datetime] = None,
    end_date: Optional[datetime.datetime] = None
) -> dict:
    """
    Fetch data for multiple tickers
    
    Args:
        tickers: List of ticker symbols
        start_date: Start date for historical data
        end_date: End date for historical data
        
    Returns:
        Dictionary mapping tickers to their DataFrames
    """
    results = {}
    
    for ticker in tickers:
        try:
            fetcher = StockDataFetcher(ticker)
            data = fetcher.fetch_data(start_date, end_date)
            results[ticker] = data
        except Exception as e:
            logger.error(f"Failed to fetch {ticker}: {str(e)}")
            continue
    
    return results


if __name__ == "__main__":
    # Example usage
    fetcher = StockDataFetcher("AAPL")
    data = fetcher.fetch_data()
    
    print(f"\nData shape: {data.shape}")
    print(f"\nFirst few rows:\n{data.head()}")
    print(f"\nLatest price: ${fetcher.get_latest_price():.2f}")
    print(f"\nStock info: {fetcher.get_info()}")
