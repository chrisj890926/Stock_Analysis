import yfinance as yf
import pandas as pd
import mplfinance as mpf
import os

# 美股 ETF 清單
def get_us_etf_tickers():
    return [
        "QQQ", "QQQM", "VOO", "VTI", "SPY", "IVV", "DIA",
        "VT", "VXUS", "VUG", "VO", "XLF", "BND", "VWO", "DIAU", "QQQX"
    ]

# 美股個股清單
def get_us_stock_tickers():
    return [
        "TSLA", "AAPL", "AMZN", "AMD", "NKE", "V", "TSM", "INTC",
        "MSFT", "ADBE", "GOOG", "MU", "NVDA", "MCD", "SMCI", "BE",
        "PLUG", "APD", "FCEL", "BLDP", "KO", "PLTR", "SOUN", "META",
        "CFLT", "AVGO", "QCOM", "IBKR", "LULU", "DDOG", "ZS", "MDB",
        "NFLX", "ORLY", "BKNG", "ASML", "INTU", "TPL", "URI", "UNH",
        "ASX", "DJCO", "PLBY", "ORCL", "PFE", "MRK", "TPR", "RH",
        "MSTR", "PGR", "TER", "MRVL", "ANET", "LW", "CLS", "CIEN",
        "INTU", "IBM", "BRK/B", "MELI", "COIN", "CRWD", "FICO", "NOW",
        "HUBS", "IT", "MA", "ROP", "MPWR", "KLAC", "TYL", "MTD", "LLY",
        "REGN", "GHC", "MUSA", "CVCO", "MSCI", "GS", "AMP", "LNG", "PNRG",
        "TRGP", "URI", "PH", "LII", "SPOT", "CHTR", "MLM", "NEU", "LIN",
        "COST", "CASY", "SAM", "EQIX", "PSA", "ESS", "CEG", "VST", "ATO",
        "QUBT", "TNXP", "PG", "JNJ", "ABBV"
    ]

# Function to clean and prepare stock data
def clean_stock_data(df):
    # Flatten MultiIndex columns if necessary
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    
    # Ensure all relevant columns are numeric
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert non-numeric to NaN
            if df[col].isnull().any():
                print(f"Warning: Non-numeric data found in column '{col}'. Dropping rows with NaN.")
            df = df.dropna(subset=[col])  # Drop rows with NaN in these columns
        else:
            raise KeyError(f"Missing expected column: {col}")
    return df

# Function to download and prepare stock data
def download_and_prepare_stock_data(ticker, start_date='2024-10-01'):
    print(f"Downloading data for {ticker}...")
    df = yf.download(ticker, start=start_date, end=pd.Timestamp('today'), interval='1d')
    if df.empty:
        raise ValueError(f"No data available for {ticker}")
    df = df.ffill().bfill()  # Fill missing values
    print(f"{ticker} data downloaded with columns: {list(df.columns)}")
    return clean_stock_data(df)

# Function to plot candlestick chart
def plot_candlestick_chart(df, ticker, category):
    print(f"Plotting candlestick chart for {ticker}...")
    # Define custom market colors
    mc = mpf.make_marketcolors(up='r', down='g', inherit=True)
    s = mpf.make_mpf_style(base_mpf_style='yahoo', marketcolors=mc)
    
    # Plotting configuration
    kwargs = dict(
        type='candle',
        mav=(5, 20, 60),  # Moving averages
        volume=True,
        figratio=(10, 8),
        figscale=0.75,
        title=ticker,
        style=s
    )
    
    # Plot and save the chart
    output_dir = f"results_stock_analysis/{category}/{ticker}"
    os.makedirs(output_dir, exist_ok=True)
    chart_path = os.path.join(output_dir, f"{ticker}_candlestick.png")
    mpf.plot(df, **kwargs, savefig=chart_path)
    print(f"Chart for {ticker} saved to {chart_path}")

# Main function
def main():
    etf_tickers = get_us_etf_tickers()
    stock_tickers = get_us_stock_tickers()
    
    # Process ETFs
    for ticker in etf_tickers:
        try:
            df = download_and_prepare_stock_data(ticker)
            plot_candlestick_chart(df, ticker, "ETF")
        except Exception as e:
            print(f"Error processing ETF {ticker}: {e}")
    
    # Process Stocks
    for ticker in stock_tickers:
        try:
            df = download_and_prepare_stock_data(ticker)
            plot_candlestick_chart(df, ticker, "Stocks")
        except Exception as e:
            print(f"Error processing Stock {ticker}: {e}")

if __name__ == "__main__":
    main()
