import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# Page configuration
st.set_page_config(
    page_title="Stock Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache data fetches
@st.cache_data(ttl=60*60)  # Cache for 1 hour
def get_stock_data(ticker, start_date, end_date):
    try:
        data = yf.download(
            ticker, 
            start=start_date, 
            end=end_date,
            progress=False,
            timeout=10  # Add timeout
        )
        if data.empty:
            st.error(f"No data found for {ticker}")
            return None
        return data
    except Exception as e:
        st.error(f"Failed to fetch data: {str(e)}")
        return None

@st.cache_resource
def load_model_from_github():
    """Load Keras model from GitHub"""
    model_url = "https://raw.githubusercontent.com/siddp95/st-stock-analysis/main/stock_price_prediction.keras"
    return load_model(model_url)

# Sidebar controls
with st.sidebar:
    st.title("Stock Analysis")
    
    # Ticker input
    ticker = st.text_input("Enter stock ticker:", "AAPL").upper()
    
    # Date range
    start_date = st.date_input("Start date", datetime.now() - timedelta(days=365))
    end_date = st.date_input("End date", datetime.now())
    
    # Technical indicators
    st.subheader("Technical Indicators")
    sma_window = st.slider("SMA Window (days)", 10, 200, 50)
    ema_window = st.slider("EMA Window (days)", 10, 200, 20)
    
    st.markdown("---")
    st.markdown("Created with Streamlit")

@st.cache_data
def safe_convert_to_series(data):
    """Ensure we have a proper pandas Series with caching"""
    if isinstance(data, pd.DataFrame):
        return data['Close'].copy() if 'Close' in data.columns else data.iloc[:, 0].copy()
    return data.copy() if isinstance(data, pd.Series) else pd.Series(data)

# Load model (cached after first load)
try:
    model = load_model_from_github()
except Exception as e:
    st.error(f"Could not load model: {str(e)}")
    model = None

# Main app
def main():
    st.title("Stock Analysis Dashboard")
    
    # Download data with error handling
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
            st.error("No data available for this ticker/date range. Try a different ticker or date range.")
            return
        
        # Convert to proper numeric format
        data = data.apply(pd.to_numeric, errors='coerce')
        close_prices = safe_convert_to_series(data['Close']).dropna()
        
        # Debug info
        st.sidebar.write(f"Data from {data.index[0].date()} to {data.index[-1].date()}")
        st.sidebar.write(f"Data points: {len(close_prices)}")
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return
    
    # Tab layout
    tab1, tab2, tab3 = st.tabs(["Market Overview", "Technical Analysis", "Fundamental Analysis"])
    
    with tab1:
        st.header(f"{ticker} Market Overview")
        
        # Get real-time data (1-minute intervals for current trading day)
        current_data = yf.Ticker(ticker).history(period='1d', interval='1m')
        regular_data = data  # Your existing daily data
        
        # Determine which data to use (prefer real-time)
        use_realtime = len(current_data) > 0 and current_data.index[-1].date() == datetime.now().date()
        
        if use_realtime:
            # REAL-TIME MODE (MARKET OPEN)
            current_price = float(current_data['Close'].iloc[-1])
            today_open = float(current_data['Open'].iloc[0])
            today_high = float(current_data['High'].max())
            today_low = float(current_data['Low'].min())
            prev_close = float(regular_data['Close'].iloc[-1]) if len(regular_data) > 0 else today_open
            
            # Calculate today's performance
            price_change = current_price - prev_close
            percent_change = (price_change / prev_close) * 100
            open_change = current_price - today_open
            
            # --- Current Session ---
            st.subheader("Live Trading Data")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Price", 
                        f"${current_price:.2f}",
                        help="Most recent traded price")
            with col2:
                st.metric("Today's Range", 
                        f"${today_low:.2f} - ${today_high:.2f}",
                        help="Daily price range")
            with col3:
                st.metric("Change from Open", 
                        f"{(open_change/today_open)*100:.2f}%",
                        delta=f"{open_change:.2f}")
            
            # --- Daily Performance ---
            st.subheader("Daily Performance")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Change from Previous Close", 
                        f"{percent_change:.2f}%",
                        delta=f"{price_change:.2f}")
            with col2:
                st.metric("Previous Close", 
                        f"${prev_close:.2f}")
            
        else:
            # REGULAR HOURS MODE (USE DAILY DATA)
            if len(regular_data) > 0:
                latest = regular_data.iloc[-1]
                current_price = float(latest['Close'])
                prev_close = float(regular_data['Close'].iloc[-2]) if len(regular_data) > 1 else current_price
                
                price_change = current_price - prev_close
                percent_change = (price_change / prev_close) * 100
                
                st.subheader("Latest Available Data")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Closing Price", f"${current_price:.2f}")
                with col2:
                    st.metric("Daily Change", 
                            f"{percent_change:.2f}%", 
                            delta=f"{price_change:.2f}")
                with col3:
                    st.metric("Date", 
                            f"{latest.name.strftime('%Y-%m-%d')}")
                
                if not use_realtime:
                    st.info("Market may be closed - showing most recent close")
            else:
                st.warning("No market data available")
        
        # --- Performance Summary ---
        st.subheader("Performance Summary")
        period = st.radio("Time Period:", ["1W", "1M", "3M", "YTD"], 
                        horizontal=True, label_visibility="collapsed")
        
        # Calculate period returns
        if len(regular_data) > 1:
            if period == "1W":
                subset = regular_data.tail(5)
            elif period == "1M":
                subset = regular_data.tail(21)
            elif period == "3M":
                subset = regular_data.tail(63)
            else:  # YTD
                subset = regular_data[regular_data.index.year == datetime.now().year]
            
            if len(subset) > 1:
                start_price = float(subset['Close'].iloc[0])
                end_price = float(subset['Close'].iloc[-1])
                period_return = (end_price - start_price) / start_price * 100
                st.metric(f"{period} Return", f"{period_return:.2f}%")
        
        # --- News Section ---
        try:
            news = yf.Ticker(ticker).news
            if news:
                st.subheader("Latest News")
                with st.expander(f"📰 {news[0]['title']}"):
                    st.write(f"**Source:** {news[0]['publisher']}")
                    st.write(f"**Time:** {pd.to_datetime(news[0]['providerPublishTime'], unit='s').strftime('%Y-%m-%d %H:%M')}")
                    if 'summary' in news[0]:
                        st.write(news[0]['summary'])
                    st.caption(f"[Read more]({news[0]['link']})")
        except Exception as e:
            st.info("News feed temporarily unavailable")
    
    with tab2:
        st.header(f"{ticker} Technical Analysis")
        
        # Create fresh figure
        fig = go.Figure()
        
        # 1. Price Line (main series)
        fig.add_trace(go.Scatter(
            x=close_prices.index,
            y=close_prices,
            name='Price',
            line=dict(color='#2ECC71', width=3),
            opacity=0.9
        ))
        
        # 2. SMA Line
        if sma_window > 0:
            sma = close_prices.rolling(window=sma_window).mean().dropna()
            fig.add_trace(go.Scatter(
                x=sma.index,
                y=sma,
                name=f'SMA {sma_window}',
                line=dict(color='orange', width=1.5)
            ))
        
        # 3. EMA Line
        if ema_window > 0:
            ema = close_prices.ewm(span=ema_window, adjust=False).mean().dropna()
            fig.add_trace(go.Scatter(
                x=ema.index,
                y=ema,
                name=f'EMA {ema_window}',
                line=dict(color='blue', width=1.5)
            ))
        
        # Configure layout
        fig.update_layout(
            height=500,
            title=f"{ticker} Price with Indicators",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            hovermode='x unified',
            showlegend=True,
            plot_bgcolor='white',
            margin=dict(l=40, r=40, t=40, b=40),
            xaxis=dict(
                showgrid=True,
                gridcolor='lightgrey',
                range=[close_prices.index.min(), close_prices.index.max()]
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='lightgrey',
                range=[close_prices.min() * 0.95, close_prices.max() * 1.05]
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Prediction Chart (unchanged working version)
        st.subheader("Price Predictions")
        try:
            if len(close_prices) < 100:
                st.warning("Not enough data for predictions (need at least 100 days)")
            else:
                split_idx = int(len(close_prices) * 0.8)
                data_train = pd.DataFrame(close_prices.iloc[:split_idx])
                data_test = pd.DataFrame(close_prices.iloc[split_idx:])
                
                scaler = MinMaxScaler(feature_range=(0, 1))
                pas_100_days = data_train.tail(100)
                data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
                data_test_scale = scaler.fit_transform(data_test)
                
                x, y = [], []
                for i in range(100, len(data_test_scale)):
                    x.append(data_test_scale[i-100:i])
                    y.append(data_test_scale[i, 0])
                
                if len(x) > 0:
                    x, y = np.array(x), np.array(y)
                    predict = model.predict(x)
                    
                    scale = 1/scaler.scale_
                    predict = predict * scale
                    y = y * scale
                    
                    pred_fig = go.Figure()
                    pred_fig.add_trace(go.Scatter(
                        x=close_prices.index[-len(y):],
                        y=y.flatten(),
                        name='Actual Price',
                        line=dict(color='green', width=2)
                    ))
                    pred_fig.add_trace(go.Scatter(
                        x=close_prices.index[-len(predict):],
                        y=predict.flatten(),
                        name='Predicted Price',
                        line=dict(color='red', width=2, dash='dot')
                    ))
                    pred_fig.update_layout(
                        title='Actual vs Predicted Prices',
                        xaxis_title='Date',
                        yaxis_title='Price ($)',
                        height=400,
                        plot_bgcolor='white'
                    )
                    st.plotly_chart(pred_fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error generating predictions: {str(e)}")
    
    with tab3:
        st.header(f"{ticker} Fundamental Analysis")
        
        try:
            ticker_info = yf.Ticker(ticker).info
            
            if not ticker_info:
                st.warning("No fundamental data available")
            else:
                cols = st.columns(3)
                metrics = [
                    ("Current Price", ticker_info.get('currentPrice', 'N/A')),
                    ("Market Cap", f"${ticker_info.get('marketCap', 0)/1e9:.2f}B" if ticker_info.get('marketCap') else 'N/A'),
                    ("P/E Ratio", ticker_info.get('trailingPE', 'N/A')),
                    ("PEG Ratio", ticker_info.get('pegRatio', 'N/A')),
                    ("Dividend Yield", f"{ticker_info.get('dividendYield', 0)*100:.2f}%" if ticker_info.get('dividendYield') else 'N/A'),
                    ("Beta", ticker_info.get('beta', 'N/A'))
                ]
                
                for i, (label, value) in enumerate(metrics):
                    with cols[i % 3]:
                        st.metric(label, value)
                
                with st.expander("Detailed Fundamental Data"):
                    st.write("**Valuation Measures**")
                    st.write(f"Forward P/E: {ticker_info.get('forwardPE', 'N/A')}")
                    st.write(f"Price/Sales: {ticker_info.get('priceToSalesTrailing12Months', 'N/A')}")
                    st.write(f"Price/Book: {ticker_info.get('priceToBook', 'N/A')}")
                    
                    st.write("**Financial Health**")
                    st.write(f"Debt/Equity: {ticker_info.get('debtToEquity', 'N/A')}")
                    st.write(f"Current Ratio: {ticker_info.get('currentRatio', 'N/A')}")
        except Exception as e:
            st.error(f"Error loading fundamental data: {str(e)}")

if __name__ == "__main__":
    main()

