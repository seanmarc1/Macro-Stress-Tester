"""
Macro-Driven Stress Tester - Streamlit Dashboard

Interactive UI for stress testing portfolios against Fed economic scenarios.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from dotenv import load_dotenv
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.fred_loader import get_macro_series
from data.market_loader import get_price_data
from models.features import make_features
from models.beta_model import fit_factor_model, model_summary
from scenarios.stress_engine import (
    stress_portfolio, 
    get_portfolio_summary, 
    get_factor_contributions,
    list_scenarios,
    get_scenario_description,
    SCENARIOS
)

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Macro Stress Tester",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
    }
    .stress-header {
        color: #1f77b4;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("📊 Macro-Driven Stress Tester")
st.markdown("*Stress-test your portfolio against Federal Reserve economic scenarios*")

# ================== SIDEBAR ==================
st.sidebar.header("⚙️ Configuration")

# FRED API Key
default_api_key = os.getenv('FRED_API_KEY', '')
fred_api_key = st.sidebar.text_input(
    "FRED API Key",
    value=default_api_key,
    type="password",
    help="Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html"
)

# Ticker Input
st.sidebar.subheader("Portfolio Tickers")
ticker_input = st.sidebar.text_area(
    "Enter tickers (comma-separated)",
    value="AAPL, MSFT, GOOGL, AMZN, JPM",
    help="Enter stock tickers separated by commas"
)

# Parse tickers
tickers = [t.strip().upper() for t in ticker_input.split(',') if t.strip()]

# Notional Value
notional_value = st.sidebar.number_input(
    "Notional Value ($)",
    min_value=1000,
    max_value=100_000_000,
    value=1_000_000,
    step=10000,
    format="%d"
)

# Date range
st.sidebar.subheader("Data Range")
start_date = st.sidebar.date_input(
    "Start Date",
    value=pd.to_datetime("2020-01-01")
)

# Run button
run_analysis = st.sidebar.button("🚀 Run Stress Test", type="primary", use_container_width=True)

# ================== MAIN PANEL ==================

# Scenario Selector
st.header("📈 Scenario Selection")
col1, col2 = st.columns([1, 2])

with col1:
    scenario_name = st.selectbox(
        "Select Fed Scenario",
        options=list_scenarios(),
        help="Choose an economic scenario to stress test against"
    )

with col2:
    st.info(f"**{scenario_name}**: {get_scenario_description(scenario_name)}")
    
    # Show scenario shocks
    scenario_shocks = SCENARIOS[scenario_name]
    shock_display = []
    for k, v in scenario_shocks.items():
        if k != 'description' and v != 0:
            if k == 'SPY':
                shock_display.append(f"SPY: {v*100:+.0f}%")
            else:
                shock_display.append(f"{k}: {v:+.2f}")
    st.caption("Shocks: " + " | ".join(shock_display))

# ================== RESULTS ==================

if run_analysis:
    if not fred_api_key:
        st.error("⚠️ Please enter your FRED API Key in the sidebar")
    elif not tickers:
        st.error("⚠️ Please enter at least one ticker")
    else:
        with st.spinner("Fetching data and running analysis..."):
            try:
                # Step 1: Fetch macro data
                st.toast("Fetching FRED macro data...")
                macro_df = get_macro_series(
                    start_date=start_date.strftime("%Y-%m-%d"),
                    api_key=fred_api_key
                )
                
                # Step 2: Fetch market data
                st.toast("Fetching market prices...")
                prices_df = get_price_data(
                    tickers=tickers,
                    start_date=start_date.strftime("%Y-%m-%d")
                )
                
                # Step 3: Create features
                st.toast("Building risk factors...")
                risk_factors = make_features(macro_df, prices_df)
                
                # Step 4: Fit factor model
                st.toast("Fitting factor model...")
                betas_df = fit_factor_model(risk_factors)
                
                # Step 5: Run stress test
                st.toast("Running stress test...")
                # Equal weights for simplicity
                stress_results = stress_portfolio(
                    betas_df=betas_df,
                    scenario_name=scenario_name,
                    notional_value=notional_value
                )
                
                # Get summary
                summary = get_portfolio_summary(stress_results, scenario_name)
                
                # ================== TOP METRICS ==================
                st.header("📊 Stress Test Results")
                
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                
                with metric_col1:
                    # Calculate P&L and Percentage Return
                    pnl = summary['total_loss']  # This is already the P&L (negative = loss)
                    stressed_return = (pnl / notional_value) * 100
                    
                    # Determine color logic: Normal means Positive=Green, Negative=Red
                    # Label as "Projected P&L" so negative numbers make sense
                    st.metric(
                        label="Projected P&L",
                        value=f"${pnl:,.0f}",          # This will show "-$111,010" for losses
                        delta=f"{stressed_return:.2f}%",  # This will show "-11.1%"
                        delta_color="normal"           # FORCE: Green for up, Red for down
                    )
                
                with metric_col2:
                    st.metric(
                        "Portfolio VaR",
                        f"{abs(summary['portfolio_var_pct']):.2f}%",
                        delta="At Risk"
                    )
                
                with metric_col3:
                    st.metric(
                        "Worst Performer",
                        summary['worst_performer'],
                        delta=f"{summary['worst_return']*100:.1f}%",
                        delta_color="normal"  # Red for negative, Green for positive
                    )
                
                # ================== WATERFALL CHART ==================
                st.subheader("🌊 Factor Contribution Analysis")
                
                factor_contrib = get_factor_contributions(stress_results)
                
                # Create waterfall chart
                fig = go.Figure(go.Waterfall(
                    name="Factor Contributions",
                    orientation="v",
                    measure=["relative"] * len(factor_contrib) + ["total"],
                    x=factor_contrib['factor'].tolist() + ["Total"],
                    y=factor_contrib['dollar_contribution'].tolist() + [summary['total_loss']],
                    connector={"line": {"color": "rgb(63, 63, 63)"}},
                    decreasing={"marker": {"color": "#ef4444"}},
                    increasing={"marker": {"color": "#22c55e"}},
                    totals={"marker": {"color": "#3b82f6"}},
                    text=[f"${v:,.0f}" for v in factor_contrib['dollar_contribution']] + [f"${summary['total_loss']:,.0f}"],
                    textposition="outside"
                ))
                
                fig.update_layout(
                    title="Which Macro Factors Are Driving Your Portfolio Loss?",
                    showlegend=False,
                    height=400,
                    yaxis_title="Dollar Impact ($)",
                    xaxis_title="Macro Factor"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # ================== BETAS TABLE ==================
                st.subheader("📈 Factor Betas by Stock")
                st.caption("These coefficients show each stock's sensitivity to macro factors")
                
                # Format betas table
                display_df = model_summary(betas_df).copy()
                
                # Rename columns for display
                column_rename = {
                    'ticker': 'Ticker',
                    'beta_SPY': 'Market β',
                    'beta_DGS10': '10Y Yield β',
                    'beta_FEDFUNDS': 'Fed Funds β',
                    'beta_UNRATE': 'Unemployment β',
                    'beta_HY_Spread': 'Credit Spread β',
                    'R_squared': 'R²',
                    'n_observations': '# Months'
                }
                display_df = display_df.rename(columns=column_rename)
                
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True
                )
                
                # ================== DETAILED RESULTS ==================
                with st.expander("📋 Detailed Stress Results"):
                    st.dataframe(
                        stress_results.round(4),
                        use_container_width=True,
                        hide_index=True
                    )
                
                st.success("✅ Stress test completed successfully!")
                
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")
                st.exception(e)

else:
    # Show instructions when not running
    st.info("""
    👈 **To get started:**
    1. Enter your FRED API Key in the sidebar
    2. Enter your portfolio tickers (comma-separated)
    3. Set your notional portfolio value
    4. Select a stress scenario
    5. Click **Run Stress Test**
    """)
    
    # Show scenario descriptions
    st.subheader("📚 Available Scenarios")
    
    for scenario, details in SCENARIOS.items():
        with st.expander(scenario):
            st.write(f"**Description:** {details.get('description', 'N/A')}")
            st.write("**Macro Shocks:**")
            for k, v in details.items():
                if k != 'description' and v != 0:
                    if k == 'SPY':
                        st.write(f"- {k}: {v*100:+.0f}% market return")
                    else:
                        st.write(f"- {k}: {v:+.2f} percentage points")

# Footer
st.markdown("---")
st.caption("Built with Streamlit | Data from FRED & Yahoo Finance | © 2026 Macro Stress Tester")
