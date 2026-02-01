"""
FRED Macro Data Loader

Fetches key economic indicators from the Federal Reserve Economic Data (FRED) API.
All series are resampled to Monthly End (M) frequency for alignment.
"""

import pandas as pd
from fredapi import Fred
from typing import Optional


# FRED Series IDs for key macro indicators
MACRO_SERIES = {
    'UNRATE': 'Unemployment Rate',        # Monthly, %
    'CPIAUCSL': 'CPI (Inflation)',        # Monthly, Index
    'FEDFUNDS': 'Fed Funds Rate',         # Monthly, %
    'DGS10': '10-Year Treasury Yield',    # Daily -> Monthly
    'BAMLH0A0HYM2': 'High Yield Spread',  # Daily -> Monthly (Credit Risk Proxy)
}


def get_macro_series(
    start_date: str,
    api_key: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Fetch macro economic series from FRED and align to monthly frequency.
    
    Args:
        start_date: Start date in 'YYYY-MM-DD' format
        api_key: FRED API key (or set via FRED_API_KEY env var)
        end_date: Optional end date in 'YYYY-MM-DD' format
        
    Returns:
        DataFrame with monthly-aligned macro series columns:
        - UNRATE: Unemployment Rate (%)
        - CPIAUCSL: Consumer Price Index
        - FEDFUNDS: Federal Funds Rate (%)
        - DGS10: 10-Year Treasury Yield (%)
        - BAMLH0A0HYM2: High Yield Credit Spread (%)
    """
    # Initialize FRED client
    if api_key:
        fred = Fred(api_key=api_key)
    else:
        # Will use FRED_API_KEY environment variable
        import os
        from dotenv import load_dotenv
        load_dotenv()
        fred = Fred(api_key=os.getenv('FRED_API_KEY'))
    
    # Fetch each series
    series_data = {}
    for series_id in MACRO_SERIES.keys():
        try:
            data = fred.get_series(
                series_id,
                observation_start=start_date,
                observation_end=end_date
            )
            # Resample to Monthly End frequency for alignment
            # Use last available observation for each month
            data = data.resample('ME').last()
            series_data[series_id] = data
        except Exception as e:
            print(f"Warning: Could not fetch {series_id}: {e}")
            continue
    
    # Combine all series into a single DataFrame
    macro_df = pd.DataFrame(series_data)
    
    # Drop any rows with missing values
    macro_df = macro_df.dropna()
    
    return macro_df


def get_series_descriptions() -> dict:
    """Return descriptions of available macro series."""
    return MACRO_SERIES.copy()
