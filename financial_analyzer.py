import streamlit as st
import streamlit_authenticator as stauth
import yaml
import finnhub
import ccxt
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pmdarima as pm
import sqlite3
import threading
import queue
import time
from datetime import datetime
from ta.trend import SMAIndicator
from prophet import Prophet  # Updated import for Prophet

# --- Secure Credential Management ---
# Load credentials from config.yaml (excluded from Git)
try:
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
except FileNotFoundError:
    st.error("config.yaml not found. Please create it with your credentials.")
    st.stop()

authenticator = stauth.Authenticate(
    credentials=config['credentials'],
    cookie_name="financial_tool",
    key="abcdef",
    cookie_expiry_days=30
)

# --- WebSocket and Queue for Real-Time Data ---
price_queue = queue.Queue()

# --- Database Setup for Portfolios ---
conn = sqlite3.connect("user_data.db")
c = conn.cursor()
c.execute("CREATE TABLE IF NOT EXISTS portfolios (username TEXT, portfolio TEXT)")
conn.commit()

# --- Finnhub and CCXT Setup ---
finnhub_client = finnhub.Client(api_key="YOUR_FINNHUB_API_KEY")  # Replace with your Finnhub API key
crypto_exchange = ccxt.binance()

# --- WebSocket Functions ---
def stock_websocket_thread(ticker):
    ws = finnhub_client.websocket
    ws.on('message', lambda ws, msg: price_queue.put(msg['data'][0]['p']) if 'data' in msg else None)
    ws.connect()
    ws.send({'type': 'subscribe', 'symbol': ticker})
    ws.run_forever()

def crypto_websocket_thread(ticker):
    def on_message(message):
        if 'lastPrice' in message:
            price_queue.put(float(message['lastPrice']))
    crypto_exchange.load_markets()
    symbol = ticker.replace('/', '')  # e.g., "BTC/USDT" -> "BTCUSDT"
    websocket = crypto_exchange.websocket
    websocket.on_message = on_message
    websocket.connect()
    websocket.subscribe_ticker(symbol)
    while True:
        time.sleep(1)

# --- Streamlit App ---
st.title("Financial Data Analysis Tool")

# Sidebar for Settings and Login
with st.sidebar:
    st.title("Settings")
    analysis_type = st.selectbox("Analysis Type", ["Single Asset", "Portfolio"])
    if analysis_type == "Single Asset":
        asset_type = st.selectbox("Asset Type", ["Stock", "Crypto"])
        ticker = st.text_input("Ticker Symbol (e.g., AAPL or BTC/USDT)").upper()
        data_source = st.selectbox("Select Data Source", ["Finnhub", "CCXT"])
    period = st.selectbox("Time Period", ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"])
    analyses = st.multiselect("Select Analyses", ["Stats", "Moving Averages", "RSI", "MACD"])
    prediction_model = st.selectbox("Prediction Model", ["ARIMA", "Prophet"])

    # Login (Fixed)
    authentication_status, username = authenticator.login(location='main')

# Handle Authentication Status
if authentication_status:
    user_name = config['credentials']['usernames'][username]['name']
    st.write(f"Welcome, {user_name}!")

    # Portfolio Management
    if analysis_type == "Portfolio":
        st.subheader("Portfolio Management")
        with st.form("portfolio_form"):
            ticker = st.text_input("Ticker")
            weight = st.number_input("Weight (%)", min_value=0.0, max_value=100.0)
            submit = st.form_submit_button("Add to Portfolio")
            if submit:
                c.execute("INSERT OR REPLACE INTO portfolios (username, portfolio) VALUES (?, ?)", 
                          (username, f"{ticker}:{weight}"))
                conn.commit()
                st.success(f"Added {ticker} with weight {weight}%")
        
        # Load saved portfolio
        c.execute("SELECT portfolio FROM portfolios WHERE username = ?", (username,))
        saved_portfolio = c.fetchone()
        if saved_portfolio:
            st.write(f"Your saved portfolio: {saved_portfolio[0]}")

    # Main Content
    col1, col2 = st.columns(2)
    with col1:
        # Real-Time Price Updates
        if analysis_type == "Single Asset" and ticker:
            st.write("### Real-Time Price Updates")
            price_placeholder = st.empty()
            if asset_type == "Stock":
                threading.Thread(target=stock_websocket_thread, args=(ticker,), daemon=True).start()
            elif asset_type == "Crypto":
                threading.Thread(target=crypto_websocket_thread, args=(ticker,), daemon=True).start()
            for _ in range(100):  # Limited iterations for demo
                try:
                    latest_price = price_queue.get_nowait()
                    price_placeholder.write(f"Latest Price: {latest_price}")
                except queue.Empty:
                    price_placeholder.write("Waiting for price update...")
                time.sleep(1)

    with col2:
        # Fetch Historical Data
        @st.cache_data
        def fetch_data(asset_type, ticker, period, data_source):
            try:
                if data_source == "Finnhub":
                    data = finnhub_client.stock_candles(ticker, 'D', int(time.time() - 86400 * 365), int(time.time()))
                    data = pd.DataFrame(data, columns=["c", "h", "l", "o", "t", "v"])
                    data.columns = ["close", "high", "low", "open", "timestamp", "volume"]
                    data["timestamp"] = pd.to_datetime(data["timestamp"], unit="s")
                    data.set_index("timestamp", inplace=True)
                elif data_source == "CCXT":
                    ohlcv = crypto_exchange.fetch_ohlcv(ticker, timeframe="1d", limit=365)
                    data = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
                    data["timestamp"] = pd.to_datetime(data["timestamp"], unit="ms")
                    data.set_index("timestamp", inplace=True)
                else:
                    return None
                if data.empty:
                    return None
                return data
            except Exception as e:
                st.error(f"Error fetching data: {e}")
                return None

        if analysis_type == "Single Asset" and ticker:
            with st.spinner("Fetching data..."):
                data = fetch_data(asset_type, ticker, period, data_source)
            
            if data is not None:
                # Compute Analyses
                if "Moving Averages" in analyses:
                    short_window = st.slider("Short MA Window", min_value=5, max_value=50, value=20)
                    long_window = st.slider("Long MA Window", min_value=20, max_value=100, value=50)
                    data["short_ma"] = data["close"].rolling(window=short_window).mean()
                    data["long_ma"] = data["close"].rolling(window=long_window).mean()
                
                if "RSI" in analyses:
                    rsi_window = st.slider("RSI Window", min_value=5, max_value=30, value=14)
                    delta = data["close"].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_window).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_window).mean()
                    rs = gain / loss
                    data["rsi"] = 100 - (100 / (1 + rs))
                
                if "MACD" in analyses:
                    ema_fast = data["close"].ewm(span=12, adjust=False).mean()
                    ema_slow = data["close"].ewm(span=26, adjust=False).mean()
                    data["macd"] = ema_fast - ema_slow
                    data["macd_signal"] = data["macd"].ewm(span=9, adjust=False).mean()
                
                # Display Statistics
                if "Stats" in analyses:
                    stats = {
                        "Mean": data["close"].mean(),
                        "Median": data["close"].median(),
                        "Std Dev": data["close"].std(),
                        "Min": data["close"].min(),
                        "Max": data["close"].max()
                    }
                    st.write("### Basic Statistics")
                    for key, value in stats.items():
                        st.write(f"{key}: {value:.2f}")
                
                # Plot Data
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, 
                                    subplot_titles=(f"{ticker} Price", "Indicators"))
                fig.add_trace(go.Scatter(x=data.index, y=data["close"], name="Close Price"), row=1, col=1)
                if "Moving Averages" in analyses:
                    fig.add_trace(go.Scatter(x=data.index, y=data["short_ma"], name=f"Short MA ({short_window})"), 
                                  row=1, col=1)
                    fig.add_trace(go.Scatter(x=data.index, y=data["long_ma"], name=f"Long MA ({long_window})"), 
                                  row=1, col=1)
                if "RSI" in analyses:
                    fig.add_trace(go.Scatter(x=data.index, y=data["rsi"], name="RSI"), row=2, col=1)
                    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
                if "MACD" in analyses:
                    fig.add_trace(go.Scatter(x=data.index, y=data["macd"], name="MACD"), row=2, col=1)
                    fig.add_trace(go.Scatter(x=data.index, y=data["macd_signal"], name="MACD Signal"), row=2, col=1)
                fig.update_layout(title=f"{ticker} Financial Analysis", height=800)
                st.plotly_chart(fig)
                
                # Price Prediction
                if st.checkbox("Include Price Prediction"):
                    if prediction_model == "ARIMA":
                        model = pm.auto_arima(data["close"], seasonal=False, stepwise=True)
                        forecast = model.predict(n_periods=5)
                        future_dates = pd.date_range(start=data.index[-1], periods=6, freq="B")[1:]
                    elif prediction_model == "Prophet":
                        df = data.reset_index().rename(columns={'timestamp': 'ds', 'close': 'y'})
                        model = Prophet()
                        model.fit(df)
                        future = model.make_future_dataframe(periods=30)
                        forecast = model.predict(future)
                        forecast = forecast['yhat'][-30:]
                        future_dates = pd.date_range(start=data.index[-1], periods=31, freq="B")[1:]
                    fig.add_trace(go.Scatter(x=future_dates[:len(forecast)], y=forecast, 
                                           name=f"{prediction_model} Prediction", mode="lines", 
                                           line=dict(dash="dash")), row=1, col=1)
                    st.plotly_chart(fig)
                
                # Export Data
                csv = data.to_csv(index=True)
                st.download_button(label="Export Data as CSV", data=csv, file_name=f"{ticker}_analysis.csv", 
                                  mime="text/csv")

elif authentication_status is False:
    st.error("Username/password is incorrect")
elif authentication_status is None:
    st.warning("Please enter your username and password")
