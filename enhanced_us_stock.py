import yfinance as yf
import pandas as pd
import talib
import matplotlib.pyplot as plt
import os
import numpy as np

# ETF and Stock Lists
def get_us_etf_tickers():
    return [
        "QQQ",  "QQQM", "VOO", "VTI", "SPY", "IVV", "DIA",
        "VT", "VXUS", "VUG", "VO", "XLF", "BND", "VWO",  "DIAU", "QQQX"
    ]

def get_us_stock_tickers():
    return [
        "TSLA", "AAPL", "AMZN", "AMD", "NKE", "V", "TSM", "INTC",
        "MSFT", "ADBE", "GOOG", "MU", "NVDA", "MCD", "SMCI", "BE",
        "PLUG", "APD", "FCEL", "BLDP", "KO", "PLTR", "SOUN", "META",
        "CFLT", "AVGO", "QCOM", "IBKR", "LULU", "DDOG", "ZS", "MDB",
        "NFLX", "ORLY", "BKNG", "ASML", "INTU", "TPL", "URI", "UNH",
        "ASX",
        # US_industry
        "DJCO", "PLBY", "ORCL", "PFE", "MRK", "TPR", "RH", "MSTR",
        "PGR", "TER", "MRVL", "ANET", "LW", "CLS", "CIEN", "INTU",
        "IBM", "BRK/B", "MELI", "COIN", "CRWD", "FICO", "NOW",
        "HUBS", "IT", "MA", "ROP", "MPWR", "KLAC", "TYL", "MTD",
        "LLY", "REGN", "GHC", "MUSA", "CVCO", "MSCI", "GS", "AMP",
        "LNG", "PNRG", "TRGP", "PH", "LII", "SPOT", "CHTR", "MLM",
        "NEU", "LIN", "COST", "CASY", "SAM", "EQIX", "PSA", "ESS",
        "CEG", "VST", "ATO", "QUBT", "TNXP", "PG", "JNJ", "ABBV"

    ]

# Ensure Directory Exists
def ensure_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Format and Validate Data
def format_data(df):
    try:
        required_columns = ['Close', 'High', 'Low', 'Volume']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        df = df.astype({
            'Close': 'float64',
            'High': 'float64',
            'Low': 'float64',
            'Volume': 'float64',
        })

        if df.isnull().any().any():
            raise ValueError("Data contains missing values.")
        return df
    except Exception as e:
        print(f"Data formatting error: {e}")
        return None

# Calculate Advanced Indicators
def calculate_advanced_indicators(df):
    try:
        close = df['Close'].values.astype(np.float64).flatten()
        high = df['High'].values.astype(np.float64).flatten()
        low = df['Low'].values.astype(np.float64).flatten()
        volume = df['Volume'].values.astype(np.float64).flatten()

        print(f"Data Shapes - Close: {close.shape}, High: {high.shape}, Low: {low.shape}, Volume: {volume.shape}")

        if any(arr.ndim != 1 for arr in [close, high, low, volume]):
            raise ValueError("Input arrays must be 1-dimensional.")

        df['STOCH_K'], df['STOCH_D'] = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)
        df['CCI'] = talib.CCI(high, low, close, timeperiod=20)
        df['WILLIAMS_R'] = talib.WILLR(high, low, close, timeperiod=14)
        df['MFI'] = talib.MFI(high, low, close, volume, timeperiod=14)
        df['ROC'] = talib.ROC(close, timeperiod=10)

        return df
    except Exception as e:
        print(f"Indicator calculation error: {e}")
        return None

# Plot Advanced Indicators
def plot_advanced_indicators(ticker, df, category):
    output_dir = f'results_stock_analysis/{category}/{ticker}'
    ensure_directory(output_dir)

    fig, ax = plt.subplots(5, 1, figsize=(14, 20), sharex=True)

    df[['STOCH_K', 'STOCH_D']].plot(ax=ax[0], title=f'{ticker} - Stochastic Oscillator', color=['blue', 'orange'])
    ax[0].set_ylabel('STOCH')

    df[['CCI']].plot(ax=ax[1], title='CCI (Commodity Channel Index)', color='purple')
    ax[1].axhline(100, color='red', linestyle='--', label='Overbought (100)')
    ax[1].axhline(-100, color='green', linestyle='--', label='Oversold (-100)')
    ax[1].legend(loc='upper left')
    ax[1].set_ylabel('CCI')

    df[['WILLIAMS_R']].plot(ax=ax[2], title='Williams %R', color='brown')
    ax[2].axhline(-20, color='red', linestyle='--', label='Overbought (-20)')
    ax[2].axhline(-80, color='green', linestyle='--', label='Oversold (-80)')
    ax[2].legend(loc='upper left')
    ax[2].set_ylabel('WILLR')

    df[['MFI']].plot(ax=ax[3], title='MFI (Money Flow Index)', color='green')
    ax[3].axhline(80, color='red', linestyle='--', label='Overbought (80)')
    ax[3].axhline(20, color='green', linestyle='--', label='Oversold (20)')
    ax[3].legend(loc='upper left')
    ax[3].set_ylabel('MFI')

    df[['ROC']].plot(ax=ax[4], title='ROC (Rate of Change)', color='darkcyan')
    ax[4].set_ylabel('ROC')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/{ticker}_advanced_indicators.png')
    plt.close()

# Analyze Individual Stock or ETF
def analyze_ticker(ticker, category):
    try:
        print(f"Downloading data for: {ticker}...")
        data = yf.download(ticker, start='2024-01-01', end=pd.Timestamp('today'), threads=False)

        if data.empty:
            print(f"No available data for {ticker}.")
            return

        data = data.ffill().bfill()
        data.index = pd.to_datetime(data.index, errors='coerce')

        if data.index.isnull().any() or len(data) < 20:
            print(f"Insufficient data or incorrect date index for {ticker}.")
            return

        print(f"Processing {ticker} data, {len(data)} rows found.")
        df = format_data(data)

        if df is None:
            print(f"Data formatting failed for {ticker}.")
            return

        df = calculate_advanced_indicators(df)

        if df is None:
            print(f"Indicator calculation failed for {ticker}.")
            return

        output_file = f'results_stock_analysis/{category}/{ticker}/{ticker}_advanced_indicators.csv'
        ensure_directory(f'results_stock_analysis/{category}/{ticker}')
        df.to_csv(output_file, index=True)
        print(f"Analysis complete for {ticker}, results saved to {output_file}")

        plot_advanced_indicators(ticker, df, category)
        print(f"Chart generated for {ticker}.")

    except Exception as e:
        print(f"Error analyzing {ticker}: {e}")

# Main Function
def main():
    etfs = get_us_etf_tickers()
    stocks = get_us_stock_tickers()

    print(f"Number of ETFs to analyze: {len(etfs)}")
    for etf in etfs:
        analyze_ticker(etf, "ETF")

    print(f"Number of stocks to analyze: {len(stocks)}")
    for stock in stocks:
        analyze_ticker(stock, "Stocks")

if __name__ == "__main__":
    main()
