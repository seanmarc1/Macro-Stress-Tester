"""
Feature Engineering Module

Creates aligned risk factors from macro and market data.
Ensures monthly alignment between all data sources.
"""

import pandas as pd
import numpy as np
from typing import List


def make_features(
    macro_df: pd.DataFrame,
    prices_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Create aligned risk factor DataFrame from macro and price data.
    
    Args:
        macro_df: DataFrame with macro series (UNRATE, CPIAUCSL, FEDFUNDS, DGS10, BAMLH0A0HYM2)
                  Should already be at Monthly End frequency
        prices_df: DataFrame of daily adjusted close prices
                   Will be resampled to Monthly End
    
    Returns:
        risk_factors DataFrame with aligned monthly data:
        - Equity returns (percent change for each ticker + SPY)
        - CPI_chg: CPI percentage change (inflation)
        - DGS10_chg: 10Y yield basis point change
        - FEDFUNDS_chg: Fed Funds basis point change
        - UNRATE_chg: Unemployment rate basis point change
        - BAMLH0A0HYM2: High Yield spread level (credit risk)
    """
    # Resample prices to Monthly End (last price of each month)
    monthly_prices = prices_df.resample('ME').last()
    
    # Calculate monthly equity returns (percent change)
    equity_returns = monthly_prices.pct_change()
    
    # Rename columns to indicate they are returns
    equity_returns.columns = [f'{col}_ret' for col in equity_returns.columns]
    
    # Calculate changes for macro variables
    macro_changes = pd.DataFrame(index=macro_df.index)
    
    # CPI - Percent change (inflation rate)
    if 'CPIAUCSL' in macro_df.columns:
        macro_changes['CPI_chg'] = macro_df['CPIAUCSL'].pct_change() * 100  # As percentage
    
    # Rates - Basis point changes (diff)
    if 'DGS10' in macro_df.columns:
        macro_changes['DGS10_chg'] = macro_df['DGS10'].diff()  # Percentage point change
    
    if 'FEDFUNDS' in macro_df.columns:
        macro_changes['FEDFUNDS_chg'] = macro_df['FEDFUNDS'].diff()  # Percentage point change
    
    if 'UNRATE' in macro_df.columns:
        macro_changes['UNRATE_chg'] = macro_df['UNRATE'].diff()  # Percentage point change
    
    # High Yield Spread - Level (credit risk proxy)
    if 'BAMLH0A0HYM2' in macro_df.columns:
        macro_changes['HY_Spread'] = macro_df['BAMLH0A0HYM2']
        macro_changes['HY_Spread_chg'] = macro_df['BAMLH0A0HYM2'].diff()  # Also include change
    
    # Align dates between equity returns and macro changes
    # Use intersection of indices
    common_index = equity_returns.index.intersection(macro_changes.index)
    
    # Combine into single risk_factors DataFrame
    risk_factors = pd.concat([
        equity_returns.loc[common_index],
        macro_changes.loc[common_index]
    ], axis=1)
    
    # Drop any rows with NaN (first row will have NaN from pct_change/diff)
    risk_factors = risk_factors.dropna()
    
    return risk_factors


def get_factor_columns() -> List[str]:
    """Return list of macro factor column names used in regression."""
    return [
        'SPY_ret',       # Market return
        'DGS10_chg',     # 10Y Treasury yield change
        'FEDFUNDS_chg',  # Fed Funds rate change
        'UNRATE_chg',    # Unemployment rate change
        'HY_Spread_chg', # High Yield spread change
        'CPI_chg',       # Inflation change
    ]
