"""
Market Data Loader

Fetches stock price data using yfinance.
Always includes SPY as a market baseline for factor modeling.
"""

import pandas as pd
import yfinance as yf
from typing import List, Optional


def get_price_data(
    tickers: List[str],
    start_date: str,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Fetch adjusted close prices for a list of tickers.
    
    Args:
        tickers: List of stock ticker symbols (e.g., ['AAPL', 'MSFT'])
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: Optional end date in 'YYYY-MM-DD' format
        
    Returns:
        DataFrame with Adjusted Close prices, forward-filled for holidays.
        SPY is always included as the market baseline.
    """
    # Ensure SPY is always in the ticker list as market baseline
    tickers = [t.strip().upper() for t in tickers]
    if 'SPY' not in tickers:
        tickers.append('SPY')
    
    # Remove duplicates while preserving order
    tickers = list(dict.fromkeys(tickers))
    
    # Fetch data from yfinance
    try:
        data = yf.download(
            tickers=tickers,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=True  # Use adjusted prices
        )
        
        # Handle single ticker case (returns Series for single ticker)
        if len(tickers) == 1:
            prices_df = pd.DataFrame(data['Close'])
            prices_df.columns = tickers
        else:
            # Multi-ticker returns MultiIndex columns
            prices_df = data['Close']
        
        # Forward-fill missing values (holidays, etc.)
        prices_df = prices_df.ffill()
        
        # Drop any remaining NaN rows (at the start)
        prices_df = prices_df.dropna()
        
        return prices_df
        
    except Exception as e:
        raise ValueError(f"Error fetching price data: {e}")


def get_returns(prices_df: pd.DataFrame, period: str = 'M') -> pd.DataFrame:
    """
    Calculate returns from a price DataFrame.
    
    Args:
        prices_df: DataFrame of prices with DatetimeIndex
        period: Resampling period ('D' for daily, 'M' for monthly)
        
    Returns:
        DataFrame of percentage returns
    """
    if period != 'D':
        # Resample to the specified period (use last price)
        prices_df = prices_df.resample(period).last()
    
    returns = prices_df.pct_change().dropna()
    return returns
