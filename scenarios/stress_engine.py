"""
Stress Testing Engine

Defines Fed economic scenarios and calculates portfolio stress impacts.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List


# 2026 Federal Reserve Stress Scenarios
# Each scenario defines monthly magnitude shocks to macro factors
SCENARIOS = {
    'Severe Recession': {
        'description': 'Major economic downturn with flight to safety',
        'SPY': -0.15,           # -15% market drop
        'UNRATE': 1.5,          # +1.5 percentage points unemployment
        'HY_Spread': 2.5,       # +2.5 percentage points credit spread widening
        'DGS10': 0.0,           # No change (flight to safety may lower yields)
        'FEDFUNDS': 0.0,
        'CPI': 0.0,
    },
    'Inflation Shock': {
        'description': 'Sudden inflation spike forcing Fed tightening',
        'CPI': 1.0,             # +1.0% monthly CPI (annualized ~12%)
        'DGS10': 0.75,          # +75 bps 10Y yield
        'FEDFUNDS': 0.50,       # +50 bps Fed Funds rate
        'SPY': 0.0,
        'UNRATE': 0.0,
        'HY_Spread': 0.0,
    },
    'Rates Up / Risk Off': {
        'description': 'Rising rates trigger risk-off sentiment',
        'DGS10': 1.00,          # +100 bps 10Y yield
        'SPY': -0.05,           # -5% market drop
        'FEDFUNDS': 0.0,
        'UNRATE': 0.0,
        'HY_Spread': 0.0,
        'CPI': 0.0,
    },
}


# Mapping from scenario shock keys to beta column names
FACTOR_MAPPING = {
    'SPY': 'beta_SPY',
    'DGS10': 'beta_DGS10',
    'FEDFUNDS': 'beta_FEDFUNDS',
    'UNRATE': 'beta_UNRATE',
    'HY_Spread': 'beta_HY_Spread',
    'CPI': 'beta_CPI',
}


def stress_portfolio(
    betas_df: pd.DataFrame,
    scenario_name: str,
    portfolio_weights: Optional[Dict[str, float]] = None,
    notional_value: float = 1_000_000
) -> pd.DataFrame:
    """
    Calculate stress test impact on a portfolio.
    
    Args:
        betas_df: DataFrame from fit_factor_model() with betas for each ticker
        scenario_name: Name of scenario from SCENARIOS dict
        portfolio_weights: Dict mapping ticker -> weight (should sum to 1.0)
                          If None, equal weights are assumed
        notional_value: Total portfolio value in dollars (default $1M)
        
    Returns:
        DataFrame with columns:
        - ticker: Stock ticker
        - weight: Portfolio weight
        - predicted_return: Expected return under stress
        - dollar_impact: P&L impact in dollars
        - factor contributions: Breakdown by factor
    """
    if scenario_name not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario_name}. Available: {list(SCENARIOS.keys())}")
    
    scenario = SCENARIOS[scenario_name]
    
    # Get list of tickers
    tickers = betas_df['ticker'].tolist()
    
    # Default to equal weights if not provided
    if portfolio_weights is None:
        n_tickers = len(tickers)
        portfolio_weights = {t: 1.0 / n_tickers for t in tickers}
    
    # Validate weights are for tickers in betas_df
    for ticker in portfolio_weights.keys():
        if ticker not in tickers:
            print(f"Warning: {ticker} not in betas_df, skipping")
    
    results = []
    
    for _, row in betas_df.iterrows():
        ticker = row['ticker']
        weight = portfolio_weights.get(ticker, 0)
        
        if weight == 0:
            continue
        
        # Calculate predicted return from each factor
        predicted_return = 0
        factor_contributions = {}
        
        for shock_key, shock_value in scenario.items():
            if shock_key == 'description':
                continue
            
            beta_col = FACTOR_MAPPING.get(shock_key)
            if beta_col and beta_col in row.index:
                beta = row[beta_col]
                contribution = beta * shock_value
                predicted_return += contribution
                factor_contributions[f'{shock_key}_contribution'] = contribution
        
        # Calculate dollar impact
        dollar_position = notional_value * weight
        dollar_impact = dollar_position * predicted_return
        
        result = {
            'ticker': ticker,
            'weight': weight,
            'position_value': dollar_position,
            'predicted_return': predicted_return,
            'dollar_impact': dollar_impact,
            **factor_contributions
        }
        results.append(result)
    
    results_df = pd.DataFrame(results)
    
    return results_df


def get_portfolio_summary(
    stress_results: pd.DataFrame,
    scenario_name: str
) -> Dict:
    """
    Get summary statistics for portfolio stress test.
    
    Returns dict with:
    - total_loss: Total portfolio dollar loss
    - portfolio_var_pct: Portfolio VaR as percentage
    - worst_performer: Ticker with worst return
    - best_performer: Ticker with best return
    """
    total_loss = stress_results['dollar_impact'].sum()
    total_value = stress_results['position_value'].sum()
    portfolio_var_pct = (total_loss / total_value) * 100 if total_value > 0 else 0
    
    worst_idx = stress_results['predicted_return'].idxmin()
    best_idx = stress_results['predicted_return'].idxmax()
    
    return {
        'scenario': scenario_name,
        'total_loss': total_loss,
        'portfolio_var_pct': portfolio_var_pct,
        'worst_performer': stress_results.loc[worst_idx, 'ticker'],
        'worst_return': stress_results.loc[worst_idx, 'predicted_return'],
        'best_performer': stress_results.loc[best_idx, 'ticker'],
        'best_return': stress_results.loc[best_idx, 'predicted_return'],
    }


def get_factor_contributions(
    stress_results: pd.DataFrame
) -> pd.DataFrame:
    """
    Aggregate factor contributions across portfolio for waterfall chart.
    
    Returns DataFrame with factor name and total contribution.
    """
    contribution_cols = [c for c in stress_results.columns if c.endswith('_contribution')]
    
    contributions = {}
    for col in contribution_cols:
        factor_name = col.replace('_contribution', '')
        # Weight contributions by position value
        weighted_contrib = (stress_results[col] * stress_results['position_value']).sum()
        contributions[factor_name] = weighted_contrib
    
    contrib_df = pd.DataFrame([
        {'factor': k, 'dollar_contribution': v}
        for k, v in contributions.items()
    ])
    
    return contrib_df.sort_values('dollar_contribution')


def list_scenarios() -> List[str]:
    """Return list of available scenario names."""
    return list(SCENARIOS.keys())


def get_scenario_description(scenario_name: str) -> str:
    """Get description for a scenario."""
    if scenario_name in SCENARIOS:
        return SCENARIOS[scenario_name].get('description', '')
    return ''
