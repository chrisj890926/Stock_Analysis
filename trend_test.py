import yfinance as yf
import pandas as pd
import numpy as np
import backtrader as bt
from keras.models import Sequential
from keras.layers import LSTM, Dense
from tqdm import tqdm

# Function to preprocess data
def preprocess_data(data):
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]  # Flatten MultiIndex
    if data.isnull().any().any():
        print("Warning: Data contains NaN values. Filling with forward fill.")
        data = data.fillna(method='ffill').fillna(method='bfill')
    if np.isinf(data).any().any():
        print("Warning: Data contains Inf values. Replacing with max value.")
        data = data.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(method='bfill')
    data = data.rename(columns=str)  # Ensure column names are strings
    return data

# Function to download stock data
def download_stock_data(ticker, start_date, end_date):
    print(f"Downloading data for {ticker}...")
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if data.empty:
        raise ValueError(f"No data available for {ticker}")
    return preprocess_data(data)

# Define Backtrader data feed
class PandasData(bt.feeds.PandasData):
    params = (
        ('datetime', None),
        ('open', 'Open'),
        ('high', 'High'),
        ('low', 'Low'),
        ('close', 'Close'),
        ('volume', 'Volume'),
    )

# LSTM model definition
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Backtesting with Backtrader
def backtest_lstm_model(ticker, start_date, end_date, model, time_steps):
    # Download and preprocess data
    data = download_stock_data(ticker, start_date, end_date)
    data_feed = PandasData(dataname=data)
    
    # Backtrader setup
    cerebro = bt.Cerebro()
    cerebro.adddata(data_feed)
    cerebro.addstrategy(bt.Strategy)
    
    # Run backtest
    try:
        cerebro.run()
        cerebro.plot()
    except ValueError as ve:
        print(f"Plotting failed for {ticker}: {ve}")
    except Exception as e:
        print(f"Unexpected error during backtest for {ticker}: {e}")

# Main function
def main():
    tickers = ['AAPL', 'MSFT', 'TSLA', 'AMZN']
    start_date = '2022-01-01'
    end_date = '2024-12-31'
    time_steps = 60

    # Create LSTM model
    model = create_lstm_model((time_steps, 1))

    # Process each ticker
    for ticker in tqdm(tickers, desc="Processing Stocks"):
        try:
            backtest_lstm_model(ticker, start_date, end_date, model, time_steps)
        except Exception as e:
            print(f"Error processing {ticker}: {e}")

if __name__ == "__main__":
    main()
