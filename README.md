# Stock Market Analytics Dashboard

https://st-stock-analysis.streamlit.app/

## Project Overview
This Streamlit dashboard transforms raw financial data into easily accessible actionable investment insights—built by an active investor for efficient nightly portfolio reviews. I use it as my personal decision engine, combining real-time stock ticker scanning with my own self-built machine learning algorithm to quickly evaluate:

1. Short-Term Opportunities: Technical charts and AI price forecasts for scalp or swing trades on options

2. Long-Term Holdings: Fundamental ratio benchmarking and valuation trends for long term investments

3. Portfolio Health Checks: Consolidated metrics across all positions

Designed for investors who manage their own capital, like myself, it automates what used to take hours of manual spreadsheet work and internet research into a 10-minute ritual every few nights. The tool emerged from my frustration with juggling multiple data sources (Yahoo Finance, SEC filings, technical scanners) and now serves as a unified system to rapidly screen for overbought/oversold conditions, compare growth metrics against sector peers, and validate gut feelings with quantitative models.

## Technical Implementation
The dashbobard leverages a modern Python data stack built on pandas for data transformation, yfinance for market data acquisition, and TensorFlow for time-series forecasting. The data pipeline handles automatic timezone normalization, missing value imputation, daily batch updates, and other minor issues that occurred with API integration during deployment. Visualizations are rendered with Plotly for interactive technical analysis, while Streamlit provides the framework for web application deployment.

## Analytical Features
The technical analysis module calculates moving averages (SMA/EMA) with configurable lookback windows, identifying key support/resistance levels and trend reversals. Fundamental metrics including P/E ratios, beta values, and dividend yields are computed daily from cleaned API data. The predictive modeling system uses an LSTM architecture trained on five years of daily prices, achieving a mean absolute error of $2.34 on holdout test data through walk-forward validation.

## Model Architecture
```python
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(100, 1)),
    Dropout(0.2),
    LSTM(units=50, return_sequences=True),
    Dropout(0.2),
    LSTM(units=50),
    Dense(units=1)
])
```
## How I Use This

Every Sunday evening—and a few weeknights when time allows—I kick off my market review by firing up the dashboard. My routine starts with a scan of my high-conviction stocks (DKNG, HIMS, PLTR), the Magnificent 7 mega-caps, and key indices like SPY, QQQ, and IWM to gauge broader market health. This ritual helps me reset for the week ahead, separating noise from actionable trends.

For technical analysis, I adjust my approach based on timeframe. For short-term trades (options/swing positions), I rely on the **8EMA and 21EMA** for momentum. A stock trading above these moving averages suggests bullish momentum, but I pair this with volume trends and RSI to avoid false breakouts. For long-term investments, I use the **200SMA** as a baseline filter—if a stock can't hold this level, I'm unlikely to add to my position. I cross-reference this with the Fundamentals Tab to check valuation (P/E, PEG ratios) against sector peers.

The dashboard's Keras price-forecasting model acts as a "second opinion" tool. While I don't take its predictions at face value, I use its projected price bands (5-10 days out) to identify potential reversals when they align with technical levels, and to flag overextended moves where the model diverges sharply from current momentum. That said, the model is still a work in progress—I'm actively backtesting its accuracy under different market conditions and plan to refine its confidence thresholds over time.

**Key Features I Rely On:**
- EMA crossovers for short-term momentum signals
- 200SMA as a long-term trend filter
- Fundamental ratios for valuation context
- AI price bands as a confirmation tool

**Why This Works:** The dashboard's real value is in consolidation—it replaces a patchwork of spreadsheets, brokerage charts, and news scans with a single interface. What used to take hours now takes 15-30 minutes, letting me focus on execution rather than data wrangling. I've also learned to document my process (saving annotated charts and model predictions) to review my hits and misses. As any trader knows, the hardest part isn't finding opportunities—it's sticking to a system. This tool enforces that discipline.

Future Improvements:
- Display the accuracy of the model against the given stock under the price predictions chart
- Include more indicators such as volume, RSI, and more EMAs (to show EMA crossovers)
- Add a way to compare 2 different stocks side by side
- Add more customizations as needed
