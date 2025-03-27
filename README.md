# Stock Market Analytics Dashboard

https://st-stock-analysis.streamlit.app/

## Project Overview
This production-grade analytics platform demonstrates how to transform raw financial data into actionable investment insights. Designed by a data analyst for investment professionals, the dashboard combines real-time market data processing with machine learning to deliver three core analytical workflows: technical chart analysis, fundamental ratio benchmarking, and AI-powered price forecasting.

## Technical Implementation
The application leverages a modern Python data stack built on pandas for data transformation, yfinance for market data acquisition, and TensorFlow for time-series forecasting. The data pipeline handles automatic timezone normalization, missing value imputation, and daily batch updates. Visualizations are rendered with Plotly for interactive technical analysis, while Streamlit provides the framework for operational deployment.

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

Future Improvements:
- Display the accuracy of the model against the given stock under the price predictions chart
- Include more indicators such as volume, RSI, and more EMAs (to show EMA crossovers)
- Add a way to compare 2 different stocks side by side
- Add more customizations as needed
