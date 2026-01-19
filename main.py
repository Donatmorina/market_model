"""
Main script for stock price prediction using ARIMA
"""
import argparse
import datetime
import logging

from data_fetcher import StockDataFetcher
from feature_engineering import create_features
from models import ARIMAModel, evaluate_model
from utils import plot_stock_data, plot_forecast, print_metrics
import config

logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)


def main():
    """
    Main function to run the complete stock price prediction pipeline
    """
    parser = argparse.ArgumentParser(
        description='Stock Price Prediction using ARIMA',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick prediction for Apple stock
  python main.py --ticker AAPL
  
  # Train and predict with custom parameters
  python main.py --ticker MSFT --days 1000 --forecast 14
  
  # Just visualize stock data
  python main.py --ticker GOOGL --visualize-only
        """
    )
    
    parser.add_argument('--ticker', type=str, default=config.DEFAULT_TICKER,
                        help='Stock ticker symbol (default: AAPL)')
    parser.add_argument('--days', type=int, default=730,
                        help='Number of days of historical data (default: 730)')
    parser.add_argument('--forecast', type=int, default=config.FORECAST_DAYS,
                        help='Number of days to forecast (default: 30)')
    parser.add_argument('--visualize-only', action='store_true',
                        help='Only fetch data and create visualizations (no training/prediction)')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip generating plots')
    
    args = parser.parse_args()
    
    # Print header
    print("\n" + "="*70)
    print(" " * 20 + "STOCK PRICE PREDICTION")
    print("="*70)
    print(f"\nTicker Symbol:        {args.ticker}")
    print(f"Historical Data:      {args.days} days")
    print(f"Forecast Period:      {args.forecast} days")
    print(f"Visualize Only:       {args.visualize_only}")
    print("="*70 + "\n")
    
    # Calculate date range
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=args.days)
    
    # Step 1: Fetch stock data
    logger.info("Step 1: Fetching stock data...")
    logger.info("-" * 70)
    
    fetcher = StockDataFetcher(args.ticker)
    data = fetcher.fetch_data(start_date=start_date, end_date=end_date)
    
    # Display stock information
    stock_info = fetcher.get_info()
    print(f"\nStock Information:")
    print(f"  Name:          {stock_info.get('name', 'N/A')}")
    print(f"  Symbol:        {stock_info.get('symbol', 'N/A')}")
    print(f"  Sector:        {stock_info.get('sector', 'N/A')}")
    print(f"  Industry:      {stock_info.get('industry', 'N/A')}")
    print(f"  Latest Price:  ${fetcher.get_latest_price():.2f}")
    
    min_price, max_price = fetcher.get_price_range()
    print(f"  Price Range:   ${min_price:.2f} - ${max_price:.2f}")
    print(f"  Data Points:   {len(data)}")
    print()
    
    # Step 2: Create features
    logger.info("Step 2: Engineering features...")
    logger.info("-" * 70)
    
    data_with_features = create_features(data)
    logger.info(f"Created {len(data_with_features.columns)} features")
    logger.info(f"Dataset shape: {data_with_features.shape}")
    print()
    
    # Step 3: Visualize data
    if not args.no_plots:
        logger.info("Step 3: Creating visualizations...")
        logger.info("-" * 70)
        plot_stock_data(data_with_features, args.ticker)
        print()
    
    # If visualize-only mode, stop here
    if args.visualize_only:
        logger.info("Visualization complete! Exiting (visualize-only mode).")
        return
    
    # Step 4: Train model and make predictions
    train_size = int(len(data_with_features) * config.TRAIN_SPLIT)
    
    logger.info("Step 4: Training ARIMA model...")
    logger.info("-" * 70)
    
    # Prepare data
    close_prices = data_with_features['Close']
    train_data = close_prices[:train_size]
    test_data = close_prices[train_size:]
    
    # Train model
    arima_model = ARIMAModel()
    arima_model.train(train_data)
    
    # Make predictions on test set
    import numpy as np
    test_predictions = arima_model.predict(steps=len(test_data))
    test_predictions = np.array(test_predictions)
    
    # Evaluate
    arima_metrics = evaluate_model(test_data.values, test_predictions)
    print_metrics(arima_metrics, "ARIMA")
    
    # Forecast future
    logger.info(f"Forecasting {args.forecast} days into the future...")
    
    # Retrain on full dataset
    arima_model.train(close_prices)
    future_predictions = arima_model.predict(steps=args.forecast)
    lower_ci, upper_ci = arima_model.get_confidence_intervals(
        steps=args.forecast,
        alpha=1 - config.CONFIDENCE_INTERVAL
    )
    
    # Generate forecast dates
    import pandas as pd
    last_date = data_with_features.index[-1]
    forecast_dates = pd.date_range(
        start=last_date + datetime.timedelta(days=1),
        periods=args.forecast,
        freq='D'
    )
    
    print(f"\nARIMA Forecast (next {min(7, args.forecast)} days):")
    print(f"{'Date':<12} {'Predicted':<12} {'Lower CI':<12} {'Upper CI':<12}")
    print("-" * 50)
    for i in range(min(7, args.forecast)):
        print(
            f"{forecast_dates[i].strftime('%Y-%m-%d'):<12} "
            f"${future_predictions[i]:<11.2f} "
            f"${lower_ci[i]:<11.2f} "
            f"${upper_ci[i]:<11.2f}"
        )
    print()
    
    # Plot forecast
    if not args.no_plots:
        plot_forecast(
            data_with_features.index[-90:],
            close_prices.values[-90:],
            forecast_dates,
            future_predictions,
            args.ticker,
            model_name="ARIMA",
            confidence_intervals=(lower_ci, upper_ci)
        )
    
    # Final summary
    print("\n" + "="*70)
    print(" " * 23 + "PREDICTION COMPLETE!")
    print("="*70)
    print(f"\nModel trained and tested on {len(data_with_features)} data points")
    print(f"Forecast generated for the next {args.forecast} days")
    if not args.no_plots:
        print(f"Visualizations saved to: {config.PLOTS_DIR}/")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    import numpy as np
    main()
