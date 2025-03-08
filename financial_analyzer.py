import streamlit as st
import yfinance as yf
import ccxt
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots

st.title("Financial Data Analysis Tool")
st.write("Select the analysis type, enter ticker(s), choose the time period, and select analyses.")

analysis_type = st.radio("Analysis Type", ["Single Asset", "Portfolio"])
period = st.selectbox("Time Period", ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"])
analyses = st.multiselect("Select Analyses", ["Stats", "Moving Averages", "RSI", "MACD"])

@st.cache_data
def fetch_data(asset_type, ticker, period):
    try:
        if asset_type.lower() == "stock":
            data = yf.download(ticker, period=period)
        elif asset_type.lower() == "crypto":
            exchange = ccxt.binance()
            ohlcv = exchange.fetch_ohlcv(ticker, timeframe="1d", limit=365)
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

if analysis_type == "Single Asset":
    asset_type = st.selectbox("Asset Type", ["Stock", "Crypto"])
    ticker = st.text_input("Ticker Symbol", placeholder="e.g., AAPL for stock, BTC/USDT for crypto")
    
    with st.spinner("Fetching data..."):
        data = fetch_data(asset_type, ticker, period)
    
    if data is not None:
        if "Moving Averages" in analyses:
            data["short_ma"] = data["close"].rolling(window=20).mean()
            data["long_ma"] = data["close"].rolling(window=50).mean()
        
        if "RSI" in analyses:
            delta = data["close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data["rsi"] = 100 - (100 / (1 + rs))
        
        if "MACD" in analyses:
            ema_fast = data["close"].ewm(span=12, adjust=False).mean()
            ema_slow = data["close"].ewm(span=26, adjust=False).mean()
            data["macd"] = ema_fast - ema_slow
            data["macd_signal"] = data["macd"].ewm(span=9, adjust=False).mean()
        
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
        
        # Alerts
        if "RSI" in analyses and data["rsi"].iloc[-1] > 70:
            st.warning("RSI is above 70, indicating overbought conditions.")
        elif "RSI" in analyses and data["rsi"].iloc[-1] < 30:
            st.warning("RSI is below 30, indicating oversold conditions.")
        if "MACD" in analyses and data["macd"].iloc[-1] > data["macd_signal"].iloc[-1] and data["macd"].iloc[-2] <= data["macd_signal"].iloc[-2]:
            st.info("MACD crossed above signal line, potential buy signal.")
        
        num_indicator_rows = sum(1 for analysis in analyses if analysis in ["RSI", "MACD"])
        subplot_titles = [f"{ticker} Price"]
        if "RSI" in analyses:
            subplot_titles.append("RSI")
        if "MACD" in analyses:
            subplot_titles.append("MACD")
        
        fig = make_subplots(rows=1 + num_indicator_rows, cols=1, shared_xaxes=True,
                            vertical_spacing=0.1, subplot_titles=subplot_titles)
        
        fig.add_trace(go.Scatter(x=data.index, y=data["close"], name="Close Price"), row=1, col=1)
        if "Moving Averages" in analyses:
            fig.add_trace(go.Scatter(x=data.index, y=data["short_ma"], name="Short MA (20)"), row=1, col=1)
            fig.add_trace(go.Scatter(x=data.index, y=data["long_ma"], name="Long MA (50)"), row=1, col=1)
        
        row = 2
        if "RSI" in analyses:
            fig.add_trace(go.Scatter(x=data.index, y=data["rsi"], name="RSI"), row=row, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=row, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=row, col=1)
            row += 1
        
        if "MACD" in analyses:
            fig.add_trace(go.Scatter(x=data.index, y=data["macd"], name="MACD"), row=row, col=1)
            fig.add_trace(go.Scatter(x=data.index, y=data["macd_signal"], name="MACD Signal"), row=row, col=1)
        
        fig.update_layout(title=f"{ticker} Financial Analysis", height=800)
        st.plotly_chart(fig)
        
        if st.checkbox("Include Price Prediction"):
            from sklearn.linear_model import LinearRegression
            
            X = np.arange(len(data)).reshape(-1, 1)
            y = data["close"].values
            model = LinearRegression()
            model.fit(X, y)
            future_days = 5
            future_X = np.arange(len(data), len(data) + future_days).reshape(-1, 1)
            predictions = model.predict(future_X)
            future_dates = pd.date_range(start=data.index[-1], periods=future_days + 1, freq="B")[1:]
            
            fig.add_trace(go.Scatter(x=future_dates, y=predictions, name="Prediction", mode="lines", line=dict(dash="dash")), row=1, col=1)
            st.plotly_chart(fig)
        
        csv = data.to_csv(index=True)
        st.download_button(label="Export Data as CSV", data=csv, file_name=f"{ticker}_analysis.csv", mime="text/csv")
        
        st.write("### Latest Price")
        if asset_type == "Stock":
            if st.button("Get Latest Price (delayed)"):
                stock = yf.Ticker(ticker)
                latest_price = stock.info.get("regularMarketPrice", "N/A")
                st.write(f"Latest Price (delayed): {latest_price}")
        elif asset_type == "Crypto":
            if st.button("Get Latest Price"):
                exchange = ccxt.binance()
                ticker_data = exchange.fetch_ticker(ticker)
                latest_price = ticker_data["last"]
                st.write(f"Latest Price: {latest_price}")
    else:
        st.error("Failed to fetch data. Please check the ticker symbol and try again.")

else:  # Portfolio
    st.write("Enter tickers and weights (e.g., AAPL:0.5,MSFT:0.5)")
    portfolio_input = st.text_area("Portfolio", "AAPL:0.5,MSFT:0.5")
    if portfolio_input:
        try:
            portfolio = {ticker.strip().upper(): float(weight) for ticker, weight in [item.split(":") for item in portfolio_input.split(",")]}
            total_weight = sum(portfolio.values())
            if abs(total_weight - 1.0) > 0.01:
                st.error("Weights must sum to approximately 1.0")
            else:
                with st.spinner("Fetching portfolio data..."):
                    data_dict = {}
                    for ticker in portfolio:
                        data = fetch_data("stock", ticker, period)
                        if data is not None:
                            data_dict[ticker] = data["close"]
                        else:
                            st.warning(f"Could not fetch data for {ticker}")
                    
                    if data_dict:
                        portfolio_df = pd.concat(data_dict, axis=1).dropna()
                        daily_returns = portfolio_df.pct_change()
                        portfolio_returns = daily_returns.dot(pd.Series(portfolio))
                        cumulative_return = (1 + portfolio_returns).cumprod() - 1
                        total_return = cumulative_return.iloc[-1]
                        volatility = portfolio_returns.std() * np.sqrt(252)
                        
                        st.write(f"### Portfolio Metrics")
                        st.write(f"Total Return: {total_return:.2%}")
                        st.write(f"Annualized Volatility: {volatility:.2%}")
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=cumulative_return.index, y=cumulative_return, name="Portfolio Cumulative Return"))
                        fig.update_layout(title="Portfolio Performance", height=500)
                        st.plotly_chart(fig)
                        
                        csv = pd.concat([portfolio_df, cumulative_return.rename("Cumulative Return")], axis=1).to_csv(index=True)
                        st.download_button(label="Export Portfolio Data as CSV", data=csv, file_name="portfolio_analysis.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Error in portfolio analysis: {e}")