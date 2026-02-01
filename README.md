# Macro-Driven Stress Tester

A Python tool to stress-test a portfolio of stocks against Federal Reserve economic scenarios. Built with factor models, real-time macro data from FRED, and an interactive Streamlit dashboard.

## Features

- **Macro Data Pipeline**: Fetches unemployment, inflation, interest rates, and credit spreads from FRED
- **Market Data Integration**: Real-time stock prices via yfinance with SPY as market baseline
- **Factor Model**: OLS regression to calculate stock sensitivities (betas) to macro factors
- **Scenario Engine**: Predefined 2026 Fed stress scenarios (Severe Recession, Inflation Shock, etc.)
- **Interactive Dashboard**: Streamlit UI with waterfall charts and detailed analytics

## Project Structure

```
Macro-Stress-Tester/
├── data/
│   ├── fred_loader.py      # FRED macro data fetching
│   └── market_loader.py    # yfinance price data
├── models/
│   ├── features.py         # Feature engineering
│   └── beta_model.py       # OLS factor model
├── scenarios/
│   └── stress_engine.py    # Fed scenarios & stress testing
├── app/
│   └── streamlit_app.py    # Dashboard UI
├── reports/                # Generated reports
├── requirements.txt
├── .env.example
└── README.md
```

## Setup Instructions

### 1. Clone and Navigate
```bash
cd Macro-Stress-Tester
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure FRED API Key
1. Get a free API key from [FRED](https://fred.stlouisfed.org/docs/api/api_key.html)
2. Copy `.env.example` to `.env`
3. Replace `your_api_key_here` with your actual API key

```bash
cp .env.example .env
# Edit .env and add your FRED_API_KEY
```

### 5. Run the Dashboard
```bash
streamlit run app/streamlit_app.py
```

## Usage

1. **Enter your FRED API Key** in the sidebar (or set it in `.env`)
2. **Input tickers** as a comma-separated list (e.g., `AAPL, MSFT, TSLA, JPM`)
3. **Set your notional value** (default: $1,000,000)
4. **Select a scenario** from the dropdown:
   - Severe Recession
   - Inflation Shock
   - Rates Up / Risk Off
5. **Analyze results**:
   - View projected portfolio loss
   - Examine waterfall chart for factor contributions
   - Review individual stock betas

## Scenarios

| Scenario | SPY | Unemployment | CPI | 10Y Yield | Fed Funds | HY Spread |
|----------|-----|--------------|-----|-----------|-----------|-----------|
| Severe Recession | -15% | +1.5% | - | - | - | +2.5% |
| Inflation Shock | - | - | +1.0% | +0.75% | +0.50% | - |
| Rates Up / Risk Off | -5% | - | - | +1.00% | - | - |

## License

MIT License
