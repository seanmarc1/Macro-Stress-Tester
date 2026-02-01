"""
Factor Beta Model

OLS regression to calculate stock sensitivities (betas) to macro factors.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from typing import List, Optional, Tuple


def fit_factor_model(
    risk_factors: pd.DataFrame,
    min_observations: int = 24
) -> pd.DataFrame:
    """
    Fit OLS factor model for each ticker in the risk_factors DataFrame.
    
    Model: R_ticker = α + β₁·R_SPY + β₂·Δ10Y + β₃·ΔFedFunds + β₄·ΔUnemploy + β₅·ΔHYSpread + ε
    
    Args:
        risk_factors: DataFrame from make_features() with equity returns and macro changes
        min_observations: Minimum months of data required (default 24)
        
    Returns:
        DataFrame with columns:
        - ticker: Stock ticker symbol
        - alpha: Intercept
        - beta_SPY: Market beta
        - beta_DGS10: 10Y Treasury sensitivity
        - beta_FEDFUNDS: Fed Funds rate sensitivity
        - beta_UNRATE: Unemployment sensitivity
        - beta_HY_Spread: Credit spread sensitivity
        - beta_CPI: Inflation sensitivity
        - R_squared: Model R² (goodness of fit)
        - t_stat_*: T-statistics for each coefficient
    """
    # Identify ticker return columns (ending in '_ret' but not SPY)
    ticker_cols = [col for col in risk_factors.columns 
                   if col.endswith('_ret') and col != 'SPY_ret']
    
    # Factor columns for X matrix
    factor_cols = ['SPY_ret', 'DGS10_chg', 'FEDFUNDS_chg', 'UNRATE_chg', 'HY_Spread_chg', 'CPI_chg']
    
    # Filter to available factor columns
    available_factors = [f for f in factor_cols if f in risk_factors.columns]
    
    if not available_factors:
        raise ValueError("No factor columns found in risk_factors DataFrame")
    
    # Prepare X matrix (factors + constant)
    X = risk_factors[available_factors].copy()
    X = sm.add_constant(X)
    
    results = []
    
    for ticker_col in ticker_cols:
        ticker = ticker_col.replace('_ret', '')
        
        # Get Y vector (ticker returns)
        y = risk_factors[ticker_col]
        
        # Create aligned data (drop any NaN)
        model_data = pd.concat([y, X], axis=1).dropna()
        
        # Check minimum observations
        if len(model_data) < min_observations:
            print(f"Skipping {ticker}: only {len(model_data)} observations (min: {min_observations})")
            continue
        
        y_clean = model_data[ticker_col]
        X_clean = model_data.drop(columns=[ticker_col])
        
        # Fit OLS regression
        try:
            model = sm.OLS(y_clean, X_clean).fit()
            
            # Extract results
            result = {
                'ticker': ticker,
                'alpha': model.params.get('const', 0),
                'n_observations': len(y_clean),
                'R_squared': model.rsquared,
                'Adj_R_squared': model.rsquared_adj,
            }
            
            # Add betas and t-stats for each factor
            for factor in available_factors:
                factor_name = factor.replace('_ret', '').replace('_chg', '')
                result[f'beta_{factor_name}'] = model.params.get(factor, 0)
                result[f'tstat_{factor_name}'] = model.tvalues.get(factor, 0)
                result[f'pval_{factor_name}'] = model.pvalues.get(factor, 1)
            
            results.append(result)
            
        except Exception as e:
            print(f"Error fitting model for {ticker}: {e}")
            continue
    
    if not results:
        raise ValueError("No tickers had sufficient data for model fitting")
    
    betas_df = pd.DataFrame(results)
    
    return betas_df


def get_significant_factors(
    betas_df: pd.DataFrame,
    ticker: str,
    p_threshold: float = 0.05
) -> List[str]:
    """
    Get list of statistically significant factors for a given ticker.
    
    Args:
        betas_df: DataFrame from fit_factor_model()
        ticker: Stock ticker to analyze
        p_threshold: P-value threshold for significance (default 0.05)
        
    Returns:
        List of significant factor names
    """
    row = betas_df[betas_df['ticker'] == ticker]
    if row.empty:
        return []
    
    pval_cols = [col for col in betas_df.columns if col.startswith('pval_')]
    significant = []
    
    for pval_col in pval_cols:
        if row[pval_col].values[0] < p_threshold:
            factor_name = pval_col.replace('pval_', '')
            significant.append(factor_name)
    
    return significant


def model_summary(betas_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a summary view of the factor model results.
    
    Returns DataFrame with ticker, key betas, R², and significance indicators.
    """
    summary_cols = ['ticker', 'beta_SPY', 'beta_DGS10', 'beta_FEDFUNDS', 
                    'beta_UNRATE', 'beta_HY_Spread', 'R_squared', 'n_observations']
    
    available_cols = [c for c in summary_cols if c in betas_df.columns]
    
    return betas_df[available_cols].round(4)
