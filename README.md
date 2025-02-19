# 主題目錄 - 台股
1. [Enhance TW Stock Analysis](#enhance-tw-stock-analysis)
2. [Trend TW Stock Analysis](#trend-tw-stock-analysis)
3. [TW Stock Indicators](#tw-stock-indicators)
4. [Stock Analysis with News Sentiment](#stock-analysis-with-news-sentiment)
5. [Stock Price Prediction](#stock-price-prediction)
   
# 主題目錄 - 美股
1. [Enhanced US Stock Analysis](#enhanced-us-stock-analysis)
2. [Trend Analysis](#trend-analysis)
3. [US Stock Indicators](#us-stock-indicators)


## Enhance TW Stock Analysis
此主題負責提供台股 ETF 與個股的技術分析工具，包括 Stochastic Oscillator, CCI, Williams %R, Money Flow Index, 和 Rate of Change 指標計算與圖表繪製。

### 功能模組
- **技術指標計算**
- **數據清理與格式化**
- **自動化報表生成**

### 使用的函數
- `get_tw_etf_tickers()`：取得台股 ETF 清單。
- `get_tw_stock_tickers()`：取得台股個股清單。
- `calculate_advanced_indicators(df)`：計算技術指標。
- `plot_advanced_indicators(ticker, df, category)`：繪製技術指標圖表。

```python
import yfinance as yf
import pandas as pd
import talib
import matplotlib.pyplot as plt
import os
import numpy as np

### ETF and Stock Lists
### 台股 ETF 清單
def get_tw_etf_tickers():
    return [
        "0050.TW", "0051.TW", "00830.TW", "00642U.TW", "00646.TW", "00637L.TW", 
        "00633L.TW", "00637R.TW", "00715L.TW", "00712.TW", "00650L.TW", "2882.TW", 
        "00679B.TW", "2881.TW"
    ]

### 台股個股清單
def get_tw_stock_tickers():
    return [
        "2330.TW", "2317.TW", "2382.TW", "2412.TW", "1216.TW", "3711.TW", "3231.TW", 
        "3045.TW", "2542.TW", "2449.TW", "2360.TW", "5388.TW", "2376.TW", "2603.TW", 
        "3035.TW", "2454.TW", "2357.TW", "2383.TW", "3017.TW", "8046.TW", "3037.TW", 
        "2891.TW", "1513.TW", "2308.TW", "8996.TW", "1795.TW", "2388.TW", "2301.TW", 
        "6757.TW", "4938.TW", "2609.TW", "6446.TW", "2345.TW"
    ]

### Ensure Directory Exists
def ensure_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

### Format and Validate Data
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

### Calculate Advanced Indicators
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

### Plot Advanced Indicators
def plot_advanced_indicators(ticker, df, category):
    output_dir = f'tw_results_stock_analysis/{category}/{ticker}'
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

### Analyze Individual Stock or ETF
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

        output_file = f'tw_results_stock_analysis/{category}/{ticker}/{ticker}_advanced_indicators.csv'
        ensure_directory(f'tw_results_stock_analysis/{category}/{ticker}')
        df.to_csv(output_file, index=True)
        print(f"Analysis complete for {ticker}, results saved to {output_file}")

        plot_advanced_indicators(ticker, df, category)
        print(f"Chart generated for {ticker}.")

    except Exception as e:
        print(f"Error analyzing {ticker}: {e}")

### Main Function
def main():
    etfs = get_tw_etf_tickers()
    stocks = get_tw_stock_tickers()

    print(f"Number of ETFs to analyze: {len(etfs)}")
    for etf in etfs:
        analyze_ticker(etf, "ETF")

    print(f"Number of stocks to analyze: {len(stocks)}")
    for stock in stocks:
        analyze_ticker(stock, "Stocks")

if __name__ == "__main__":
    main()
```

---

## Trend TW Stock Analysis
此主題專注於台股 ETF 與個股的趨勢分析，提供資料清洗與蠟燭圖表繪製功能。可以快速生成移動平均線與成交量的視覺化報表。

### 功能模組
- **蠟燭圖生成**
- **移動平均線計算**
- **成交量視覺化**

### 使用的函數
- `get_tw_etf_tickers()`：取得台股 ETF 清單。
- `download_and_prepare_stock_data(ticker, start_date)`：下載並清理股價數據。
- `plot_candlestick_chart(df, ticker, category)`：繪製蠟燭圖。

---
```python
import yfinance as yf
import pandas as pd
import mplfinance as mpf
import os

### 台股 ETF 清單
def get_tw_etf_tickers():
    return [
        "0050.TW", "0051.TW", "00830.TW", "00642U.TW", "00646.TW", "00637L.TW", 
        "00633L.TW", "00637R.TW", "00715L.TW", "00712.TW", "00650L.TW", "2882.TW", 
        "00679B.TW", "2881.TW"
    ]

### 台股個股清單
def get_tw_stock_tickers():
    return [
        "2330.TW", "2317.TW", "2382.TW", "2412.TW", "1216.TW", "3711.TW", "3231.TW", 
        "3045.TW", "2542.TW", "2449.TW", "2360.TW", "5388.TW", "2376.TW", "2603.TW", 
        "3035.TW", "2454.TW", "2357.TW", "2383.TW", "3017.TW", "8046.TW", "3037.TW", 
        "2891.TW", "1513.TW", "2308.TW", "8996.TW", "1795.TW", "2388.TW", "2301.TW", 
        "6757.TW", "4938.TW", "2609.TW", "6446.TW", "2345.TW"
    ]

### Function to clean and prepare stock data
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

### Function to download and prepare stock data
def download_and_prepare_stock_data(ticker, start_date='2024-10-01'):
    print(f"Downloading data for {ticker}...")
    df = yf.download(ticker, start=start_date, end=pd.Timestamp('today'), interval='1d')
    if df.empty:
        raise ValueError(f"No data available for {ticker}")
    df = df.ffill().bfill()  # Fill missing values
    print(f"{ticker} data downloaded with columns: {list(df.columns)}")
    return clean_stock_data(df)

### Function to plot candlestick chart
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
    output_dir = f"tw_results_stock_analysis/{category}/{ticker}"
    os.makedirs(output_dir, exist_ok=True)
    chart_path = os.path.join(output_dir, f"{ticker}_candlestick.png")
    mpf.plot(df, **kwargs, savefig=chart_path)
    print(f"Chart for {ticker} saved to {chart_path}")

### Main function
def main():
    etf_tickers = get_tw_etf_tickers()
    stock_tickers = get_tw_stock_tickers()
    
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
```

## TW Stock Indicators
此主題主要針對台股的技術指標進行計算與分析，支持多種指標如 RSI、MACD、布林通道和移動平均線等，並自動生成對應的圖表以輔助決策。

### 功能模組
- **多重技術指標計算**
- **圖表視覺化**
- **數據導出與存儲**

### 使用的函數
- `calculate_technical_indicators(df)`：計算技術指標（RSI、MACD 等）。
- `plot_technical_indicators(ticker, df, category)`：繪製技術指標的綜合圖表。
- `analyze_stock(ticker, category)`：綜合分析股票。

---
```python
import yfinance as yf
import pandas as pd
import talib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

### 忽略 FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

### 設置 Seaborn 樣式
sns.set(style='whitegrid')

### 台股 ETF 清單
def get_tw_etf_tickers():
    return [
        "0050", "0051", "00830", "00642U", "00646", "00637L", "00633L", "00637R",
        "00715L", "00712", "00650L", "2882", "006208", "00679B", "2881"
    ]

### 台股個股清單
def get_tw_stock_tickers():
    return [
        "2330", "2317", "2382", "2412", "1216", "3711", "3231", "3045",
        "2542", "2449", "2360", "5388", "2376", "2603", "3035", "2454",
        "2357", "2383", "3017", "8046", "3037", "2891", "1513", "2308",
        "8996", "1795", "2388", "2301", "6757", "4938", "2609", "6446",
        "2345"
    ]

### 自動加 .TW 後綴
def get_tw_etf_tickers_with_tw():
    return [ticker + ".TW" for ticker in get_tw_etf_tickers()]

def get_tw_stock_tickers_with_tw():
    return [ticker + ".TW" for ticker in get_tw_stock_tickers()]

### 計算技術指標
def calculate_technical_indicators(df):
    try:
        # 提取數據為一維 numpy.ndarray
        close = df['Close'].values.flatten()  # 確保為一維
        high = df['High'].values.flatten()
        low = df['Low'].values.flatten()

        # 打印調試信息
        print(f"Close shape: {close.shape}, High shape: {high.shape}, Low shape: {low.shape}")

        # 檢查數據是否足夠
        if len(close) < 20 or len(high) < 20 or len(low) < 20:
            raise ValueError("Not enough data to calculate indicators")

        # 初始化新列
        df['MA10'] = talib.SMA(close, timeperiod=10) if len(close) >= 10 else None
        df['MA20'] = talib.SMA(close, timeperiod=20) if len(close) >= 20 else None
        df['RSI'] = talib.RSI(close, timeperiod=14) if len(close) >= 14 else None
        df['MACD'], df['MACD_signal'], df['MACD_hist'] = (
            talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
            if len(close) >= 26
            else (None, None, None)
        )
        df['Upper_BB'], df['Middle_BB'], df['Lower_BB'] = (
            talib.BBANDS(close, timeperiod=20) if len(close) >= 20 else (None, None, None)
        )
        df['ATR'] = talib.ATR(high, low, close, timeperiod=14) if len(close) >= 14 else None
        df['ADX'] = talib.ADX(high, low, close, timeperiod=14) if len(close) >= 14 else None
        return df
    except Exception as e:
        print(f"Error calculating indicators: {e}")
        return df
### 繪製技術指標圖表 (改進版)
def plot_technical_indicators(ticker, df, category):
    fig, ax = plt.subplots(5, 1, figsize=(14, 20), sharex=True)

    # 第一部分：收盤價與均線
    df[['Close', 'MA10', 'MA20', 'Upper_BB', 'Middle_BB', 'Lower_BB']].plot(
        ax=ax[0], title=f'{ticker} - Price and Moving Averages'
    )
    ax[0].set_ylabel('Price')
    ax[0].legend(loc='upper left')

    # 第二部分：RSI
    df[['RSI']].plot(ax=ax[1], title='RSI', color='purple', legend=True)
    ax[1].axhline(y=70, color='red', linestyle='--', label='Overbought (70)')
    ax[1].axhline(y=30, color='green', linestyle='--', label='Oversold (30)')
    ax[1].set_ylabel('RSI')
    ax[1].legend(loc='upper left')

    # 第三部分：MACD
    df[['MACD', 'MACD_signal']].plot(ax=ax[2], title='MACD')
    ax[2].fill_between(df.index, df['MACD_hist'], 0, color='gray', alpha=0.3, label='MACD Histogram')
    ax[2].set_ylabel('MACD')
    ax[2].legend(loc='upper left')

    # 第四部分：成交量
    df[['Volume']].plot(ax=ax[3], title='Volume', color='blue', legend=True)
    ax[3].set_ylabel('Volume')

    # 第五部分：ATR 和 ADX
    df[['ATR', 'ADX']].plot(ax=ax[4], title='ATR and ADX', legend=True)
    ax[4].set_ylabel('ATR and ADX')
    ax[4].legend(loc='upper left')

    # 調整佈局，確保不重疊
    plt.tight_layout()

    # 儲存圖表
    output_dir = f'results_stock_analysis/{category}/{ticker}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(f'{output_dir}/{ticker}_technical_indicators.png')
    plt.close()

### 分析單個股票或 ETF
def analyze_stock(ticker, category):
    try:
        # 加載資料
        data = yf.download(ticker, start='2024-01-01', end=pd.Timestamp('today'), threads=False)
        if data.empty:
            print(f"No data for {ticker}")
            return None

        # 填充缺失值
        data = data.ffill().bfill()

        # 確保編正是 DatetimeIndex
        data.index = pd.to_datetime(data.index, errors='coerce')
        if data.index.isnull().any():
            print(f"Invalid index detected for {ticker}")
            return None

        # 檢查資料完整性
        if len(data) < 20:
            print(f"Not enough data for {ticker}")
            return None

        # 打印資料頭部信息以調試
        print(f"Processing {ticker} with data shape: {data.shape}")
        print(data.head())

        # 計算技術指標
        df = calculate_technical_indicators(data)

        # 檢查是否所有列都存在
        required_columns = ['Close', 'Volume', 'MA10', 'MA20', 'RSI', 'MACD', 'MACD_signal',
                            'Upper_BB', 'Middle_BB', 'Lower_BB', 'ATR', 'ADX']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing columns for {ticker}: {missing_columns}")
            return None

        # 提取相關特徵用於分析
        features = df[required_columns]

        # 繪製技術指標圖表
        plot_technical_indicators(ticker, df, category)

        # 儲存每個股票的技術指標和成交量資料
        features.to_csv(f'tw_results_stock_analysis/{category}/{ticker}/technical_indicators_and_volume.csv')

        return features
    except Exception as e:
        print(f"Error processing {ticker}: {e}")
        return None
### 主程序
def main():
    # 確認清單數量
    etf_tickers = get_tw_etf_tickers_with_tw()
    stock_tickers = get_tw_stock_tickers_with_tw()
```

## Stock Analysis with News Sentiment
此主題結合技術指標與新聞情感分析，提供模型準確度評估與綜合報告生成功能。包括 RSI、SMA 和 Sharpe Ratio 等技術指標計算，並結合隨機森林模型進行回報率預測。

### 功能模組
- **新聞情感分析**
- **技術指標與數據建模**
- **自動化報告生成**

### 使用的函數
- `get_news(ticker)`：提取相關股票新聞並進行情感分析。
- `analyze_news(news_list)`：計算新聞的平均情感分數。
- `analyze_stock(ticker, start_date, end_date, industry_keywords)`：綜合技術指標、情感分析和模型的股票評估。
- `generate_report(results, start_date, end_date)`：生成綜合報告。

---
```python
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import mplfinance as mpf
import logging
import datetime
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
import time

def get_news(ticker):
    # Simulated news fetching function
    url = f"https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, features='xml')
    news_items = soup.findAll('item')
    
    news_list = []
    for item in news_items[:50]:  # Get top 20 news
        title = item.title.text
        description = item.description.text
        sentiment = TextBlob(title + " " + description).sentiment.polarity
        news_list.append({
            'title': title,
            'description': description,
            'sentiment': sentiment
        })
    
    return news_list

def analyze_news(news_list):
    if not news_list:
        logging.warning("No news found for this stock.")
        return 0  # Return 0 or other default value
    avg_sentiment = sum(news['sentiment'] for news in news_list) / len(news_list)
    return avg_sentiment

def get_stock_data(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date)
    return df

def calculate_technical_indicators(df):
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = calculate_rsi(df['Close'], 14)
    return df

def calculate_rsi(prices, period):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    excess_returns = returns - risk_free_rate / 252
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

def calculate_max_drawdown(df):
    roll_max = df['Close'].rolling(window=252, min_periods=1).max()
    daily_drawdown = df['Close'] / roll_max - 1.0
    max_daily_drawdown = daily_drawdown.rolling(window=252, min_periods=1).min()
    return max_daily_drawdown.min()

def prepare_features(df):
    df = df.copy()  # Create a distinct copy
    df['Returns'] = df['Close'].pct_change()
    df['Target'] = np.where(df['Returns'].shift(-1) > 0, 1, 0)
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'SMA_50', 'RSI']
    X = df[features]
    y = df['Target']
    return X, y

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, scaler, accuracy, classification_report(y_test, y_pred)

def analyze_stock(ticker, start_date, end_date, industry_keywords):
    try:
        df = get_stock_data(ticker, start_date, end_date)
        df = calculate_technical_indicators(df)
        
        returns = df['Close'].pct_change().dropna()
        sharpe_ratio = calculate_sharpe_ratio(returns)
        max_drawdown = calculate_max_drawdown(df)
        
        X, y = prepare_features(df.dropna())
        model, scaler, accuracy, report = train_model(X, y)
        
        technical_score = 0
        if df['Close'].iloc[-1] > df['SMA_20'].iloc[-1]:
            technical_score += 1
        if df['Close'].iloc[-1] > df['SMA_50'].iloc[-1]:
            technical_score += 1
        if df['RSI'].iloc[-1] > 50:
            technical_score += 1
        if df['RSI'].iloc[-1] < 30:
            technical_score -= 1
        if df['RSI'].iloc[-1] > 70:
            technical_score -= 1
        
        stock_news = get_news(ticker)
        industry_news = []
        for keyword in industry_keywords:
            industry_news.extend(get_news(keyword))
        
        stock_sentiment = analyze_news(stock_news)
        industry_sentiment = analyze_news(industry_news)
        
        avg_sentiment = (stock_sentiment + industry_sentiment) / 2
        
        return {
            'ticker': ticker,
            'technical_score': technical_score,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'model_accuracy': accuracy,
            'classification_report': report,
            'news_sentiment': avg_sentiment,
            'news': stock_news + industry_news
        }
    except Exception as e:
        logging.error(f"Error analyzing stock {ticker}: {str(e)}")
        return None

def generate_report(results, start_date, end_date):
    report = f"\u80a1\u5e02\u5206\u6790\u7d9c\u5408\u5831\u544a ({start_date} \u5230 {end_date})\n\n"
    
    # 1. \u6574\u9ad4\u5e02\u5834\u6982\u6cc1
    avg_sentiment = np.mean([result['news_sentiment'] for result in results])
    market_trend = "\u4e0a\u6f32" if avg_sentiment > 0 else "\u4e0b\u8dcc"
    report += f"1. \u6574\u9ad4\u5e02\u5834\u6982\u6cc1\n"
    report += f"   - \u5e02\u5834\u8da8\u52e2: {market_trend}\n"
    report += f"   - \u5e73\u5747\u60c5\u611f\u5f97\u5206: {avg_sentiment:.2f}\n\n"
    
    # 2. \u884c\u696d\u5206\u6790
    industries = {
        '\u534a\u5b8f\u9ad4': ['2330.TW', '2303.TW', '2454.TW'],
        '\u96fb\u5b50\u96f6\u4ef6': ['2317.TW', '2354.TW', '2382.TW'],
        '\u91d1\u878d': ['2882.TW', '2881.TW', '2891.TW'],
        '\u901a\u8a0a\u7db2\u8def': ['2412.TW', '3045.TW', '4904.TW']
    }
    
    report += "2. \u884c\u696d\u5206\u6790\n"
    for industry, tickers in industries.items():
        industry_results = [r for r in results if r['ticker'] in tickers]
        avg_tech_score = np.mean([r['technical_score'] for r in industry_results])
        avg_sharpe = np.mean([r['sharpe_ratio'] for r in industry_results])
        best_stock = max(industry_results, key=lambda x: x['sharpe_ratio'])
        
        report += f"   {industry}\u884c\u696d:\n"
        report += f"   - \u5e73\u5747\u6280\u8853\u5f97\u5206: {avg_tech_score:.2f}\n"
        report += f"   - \u5e73\u5747\u590f\u666e\u6bd4\u7387: {avg_sharpe:.2f}\n"
        report += f"   - \u8868\u73fe\u6700\u4f73\u80a1\u7968: {best_stock['ticker']}\n\n"
    
    # 3. \u8868\u73fe\u6700\u4f73\u7684\u4e94\u652f\u80a1\u7968
    top_stocks = sorted(results, key=lambda x: x['sharpe_ratio'], reverse=True)[:10]
    report += "3. \u8868\u73fe\u6700\u4f73\u7684\u4e94\u652f\u80a1\u7968\n"
    for i, stock in enumerate(top_stocks, 1):
        report += f"   {i}. {stock['ticker']}: \u590f\u666e\u6bd4\u7387 {stock['sharpe_ratio']:.2f}, \u6280\u8853\u5f97\u5206 {stock['technical_score']}, \u60c5\u611f\u5f97\u5206 {stock['news_sentiment']:.2f}\n"
    report += "\n"
    
    # 4. ETF\u5206\u6790
    etfs = ['0050.TW', '0056.TW', '00878.TW', '00881.TW']
    report += "4. ETF\u5206\u6790\n"
    for etf in etfs:
        etf_data = next((r for r in results if r['ticker'] == etf), None)
        if etf_data:
            report += f"   - {etf}: \u6280\u8853\u5f97\u5206 {etf_data['technical_score']}, \u590f\u666e\u6bd4\u7387 {etf_data['sharpe_ratio']:.2f}, \u60c5\u611f\u5f97\u5206 {etf_data['news_sentiment']:.2f}\n"
    report += "\n"
    
     # 5. \u500b\u80a1\u5206\u6790 (\u524d10\u652f)
    report += "5. \u500b\u80a1\u5206\u6790 (\u524d10\u652f)\n"
    for stock in results[:10]:
        report += f"   - {stock['ticker']}: \u6280\u8853\u5f97\u5206 {stock['technical_score']}, \u590f\u666e\u6bd4\u7387 {stock['sharpe_ratio']:.2f}, \u6700\u5927\u56de\u64a4 {stock['max_drawdown']*100:.2f}%, \u60c5\u611f\u5f97\u5206 {stock['news_sentiment']:.2f}\n"
    report += "\n"
    
    # 6. \u6a5f\u5668\u5b78\u7fd2\u6a21\u578b\u8868\u73fe
    accuracies = [r['model_accuracy'] for r in results]
    report += "6. \u6a5f\u5668\u5b78\u7fd2\u6a21\u578b\u8868\u73fe\n"
    report += f"   - \u5e73\u5747\u6e96\u78ba\u7387: {np.mean(accuracies):.2f}\n"
    report += f"   - \u6700\u9ad8\u6e96\u78ba\u7387: {max(accuracies):.2f}\n"
    report += f"   - \u6700\u4f4e\u6e96\u78ba\u7387: {min(accuracies):.2f}\n\n"
    
    # 7. \u7d50\u8ad6\u8207\u5efa\u8b70
    report += "7. \u7d50\u8ad6\u8207\u5efa\u8b70\n"
    report += f"   - \u6574\u9ad4\u5e02\u5834\u5c55\u793a{market_trend}\u8da8\u52e2\n"
    report += "   - \u591a\u6578\u80a1\u7968\u7684\u590f\u666e\u6bd4\u7387\u70ba\u8ca0\uff0c\u8868\u660e\u76f8\u5c0d\u65bc\u7121\u98a8\u96aa\u5229\u7387\uff0c\u5b83\u5011\u7684\u8868\u73fe\u4e0d\u4f73\n"
    report += f"   - {top_st
```

# Stock Price Prediction
此主題專注於股票價格的預測與技術指標的生成，使用隨機森林模型進行未來股價預測，並生成實際與預測價格的對比圖表。

### 功能模組
- **股票價格下載與清洗**
- **技術指標生成**
- **隨機森林模型訓練與評估**
- **結果儲存與圖表生成**

### 使用的函數
- `download_stock_data(ticker, start_date, end_date)`：下載股票數據。
- `add_features(data)`：添加技術指標作為特徵。
- `prepare_model_data(data)`：準備數據以進行模型訓練。
- `train_model(X_train, y_train, X_test, y_test)`：訓練隨機森林回歸模型。
- `save_results(ticker, model, X_test, y_test, y_pred)`：儲存模型結果與生成對比圖表。
  
```python
import yfinance as yf
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sns.set(style="whitegrid")

# Function to download stock data
def download_stock_data(ticker, start_date, end_date):
    print(f"Downloading data for {ticker}...")
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    if stock_data.empty:
        print(f"No data found for {ticker}")
        return None
    stock_data.fillna(method="ffill", inplace=True)
    stock_data.fillna(method="bfill", inplace=True)
    return stock_data

# Function to calculate additional features
def add_features(data):
    print("Adding technical features...")
    data['Daily Return'] = data['Close'].pct_change()
    data['5-Day SMA'] = data['Close'].rolling(window=5).mean()
    data['20-Day SMA'] = data['Close'].rolling(window=20).mean()
    data['50-Day SMA'] = data['Close'].rolling(window=50).mean()
    data['200-Day SMA'] = data['Close'].rolling(window=200).mean()
    return data

# Function to prepare data for modeling
def prepare_model_data(data):
    print("Preparing data for modeling...")
    features = ['Open', 'High', 'Low', 'Close', 'Volume', '5-Day SMA', '20-Day SMA', '50-Day SMA', '200-Day SMA']
    data = data.dropna()
    X = data[features]
    y = data['Close'].shift(-1)  # Predicting next day's close price
    X = X[:-1]
    y = y[:-1]
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Function to train and evaluate model
def train_model(X_train, y_train, X_test, y_test):
    print("Training Random Forest Regressor...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Model Evaluation: MSE = {mse:.4f}, R2 = {r2:.4f}")
    return model, scaler

# Function to save model results
def save_results(ticker, model, X_test, y_test, y_pred):
    print("Saving model predictions...")
    results_dir = f"results/{ticker}"
    os.makedirs(results_dir, exist_ok=True)

    test_data = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred
    })
    test_data.to_csv(f"{results_dir}/predictions.csv", index=False)

    plt.figure(figsize=(10, 6))
    plt.plot(y_test.reset_index(drop=True), label="Actual", color="blue")
    plt.plot(pd.Series(y_pred), label="Predicted", color="orange")
    plt.title(f"{ticker} - Actual vs Predicted Prices")
    plt.legend()
    plt.savefig(f"{results_dir}/predictions_plot.png")
    plt.close()

# Main analysis function
def analyze_stock(ticker, start_date, end_date):
    data = download_stock_data(ticker, start_date, end_date)
    if data is None:
        return

    data = add_features(data)
    X_train, X_test, y_train, y_test = prepare_model_data(data)
    model, scaler = train_model(X_train, y_train, X_test, y_test)

    y_pred = model.predict(scaler.transform(X_test))
    save_results(ticker, model, X_test, y_test, y_pred)

# Main script
def main():
    tickers = ["AAPL", "MSFT", "TSLA"]
    start_date = "2020-01-01"
    end_date = "2023-01-01"

    for ticker in tickers:
        print(f"Analyzing {ticker}...")
        analyze_stock(ticker, start_date, end_date)

if __name__ == "__main__":
    main()
```

---

## Enhanced US Stock Analysis
此主題專注於美股的技術指標分析，提供多種指標的計算與可視化功能，適合進行中短期趨勢的判斷。

### 功能模組
- **技術指標計算**：包含 SMA、RSI、MACD 和布林通道。
- **股價數據下載與清洗**：基於 `yfinance` 提供可靠的股價資料。
- **技術圖表生成**：支持關鍵指標的整合圖表視覺化。

### 使用的函數
- `ensure_directory(path)`：確保目標目錄存在。
- `calculate_technical_indicators(df)`：計算技術指標。
- `download_and_process_data(ticker, start_date, end_date)`：下載並處理股票數據。
- `plot_technical_indicators(df, ticker)`：繪製指標圖表。
- `analyze_ticker(ticker, start_date, end_date)`：執行完整分析流程。

---
```python
import yfinance as yf
import pandas as pd
import talib
import matplotlib.pyplot as plt
import os

# Function to ensure directory exists
def ensure_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Function to calculate technical indicators
def calculate_technical_indicators(df):
    df['SMA_20'] = talib.SMA(df['Close'], timeperiod=20)
    df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
    df['MACD'], df['MACD_signal'], _ = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['Upper_BB'], df['Middle_BB'], df['Lower_BB'] = talib.BBANDS(df['Close'], timeperiod=20)
    return df

# Function to download and process stock data
def download_and_process_data(ticker, start_date, end_date):
    print(f"Downloading data for {ticker}...")
    df = yf.download(ticker, start=start_date, end=end_date)
    if df.empty:
        raise ValueError(f"No data available for {ticker}")
    df = calculate_technical_indicators(df)
    return df

# Function to plot technical indicators
def plot_technical_indicators(df, ticker):
    ensure_directory(f"results/{ticker}")
    plt.figure(figsize=(14, 10))
    plt.plot(df['Close'], label="Close Price", color="blue")
    plt.plot(df['SMA_20'], label="20-Day SMA", color="orange")
    plt.fill_between(df.index, df['Upper_BB'], df['Lower_BB'], color="gray", alpha=0.2, label="Bollinger Bands")
    plt.title(f"{ticker} Technical Indicators")
    plt.legend()
    plt.savefig(f"results/{ticker}/technical_indicators.png")
    plt.close()

# Main function to analyze a single ticker
def analyze_ticker(ticker, start_date, end_date):
    try:
        df = download_and_process_data(ticker, start_date, end_date)
        df.to_csv(f"results/{ticker}/{ticker}_data.csv")
        plot_technical_indicators(df, ticker)
        print(f"Analysis complete for {ticker}")
    except Exception as e:
        print(f"Error analyzing {ticker}: {e}")

# Main function
def main():
    tickers = ["AAPL", "MSFT", "GOOGL"]
    start_date = "2020-01-01"
    end_date = "2023-01-01"

    for ticker in tickers:
        analyze_ticker(ticker, start_date, end_date)

if __name__ == "__main__":
    main()
```

---

## Trend Analysis
此主題聚焦於美股的短期趨勢分析，通過生成蠟燭圖幫助理解市場趨勢和價格波動。

### 功能模組
- **蠟燭圖生成**：快速視覺化股票的開盤、收盤、高低價格。
- **趨勢分析**：適合短期交易決策參考。

### 使用的函數
- `ensure_directory(path)`：確保目標目錄存在。
- `download_stock_data(ticker, start_date, end_date)`：下載股票數據。
- `plot_candlestick_chart(df, ticker)`：生成蠟燭圖。
- `analyze_trend(ticker, start_date, end_date)`：執行完整趨勢分析流程。

---
```python
import yfinance as yf
import pandas as pd
import mplfinance as mpf
import os

# Function to ensure the output directory exists
def ensure_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Function to download stock data
def download_stock_data(ticker, start_date, end_date):
    print(f"Downloading data for {ticker}...")
    df = yf.download(ticker, start=start_date, end=end_date)
    if df.empty:
        raise ValueError(f"No data available for {ticker}")
    return df

# Function to plot candlestick chart
def plot_candlestick_chart(df, ticker):
    output_dir = f"results/{ticker}/"
    ensure_directory(output_dir)

    mc = mpf.make_marketcolors(up='g', down='r', inherit=True)
    s = mpf.make_mpf_style(base_mpf_style='yahoo', marketcolors=mc)

    mpf.plot(
        df,
        type='candle',
        mav=(5, 20),
        volume=True,
        style=s,
        title=f"Candlestick Chart for {ticker}",
        savefig=dict(fname=f"{output_dir}{ticker}_candlestick.png", dpi=100, pad_inches=0.25)
    )
    print(f"Candlestick chart saved for {ticker}")

# Main function for trend analysis
def analyze_trend(ticker, start_date, end_date):
    try:
        df = download_stock_data(ticker, start_date, end_date)
        plot_candlestick_chart(df, ticker)
        print(f"Trend analysis complete for {ticker}")
    except Exception as e:
        print(f"Error in trend analysis for {ticker}: {e}")

# Main script
def main():
    tickers = ["TSLA", "AMZN", "FB"]
    start_date = "2020-01-01"
    end_date = "2023-01-01"

    for ticker in tickers:
        analyze_trend(ticker, start_date, end_date)

if __name__ == "__main__":
    main()
```
## US Stock Indicators
此主題提供長期技術指標分析，如 50 天與 200 天移動平均線，幫助評估長期投資價值。

### 功能模組
- **長期技術指標計算**：支持 SMA、RSI、MACD 的深度分析。
- **結果儲存與可視化**：保存分析數據並生成圖表。

### 使用的函數
- `ensure_directory(path)`：確保目標目錄存在。
- `calculate_technical_indicators(df)`：計算長期技術指標。
- `download_and_process_data(ticker, start_date, end_date)`：下載並處理美股數據。
- `plot_indicators(df, ticker)`：繪製技術指標圖表。
- `analyze_us_stock(ticker, start_date, end_date)`：執行完整美股分析流程。

```python
import yfinance as yf
import pandas as pd
import talib
import matplotlib.pyplot as plt
import os

# Function to ensure the output directory exists
def ensure_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Function to calculate technical indicators
def calculate_technical_indicators(df):
    df['SMA_50'] = talib.SMA(df['Close'], timeperiod=50)
    df['SMA_200'] = talib.SMA(df['Close'], timeperiod=200)
    df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
    df['MACD'], df['MACD_signal'], _ = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    return df

# Function to download and process stock data
def download_and_process_data(ticker, start_date, end_date):
    print(f"Downloading data for {ticker}...")
    df = yf.download(ticker, start=start_date, end=end_date)
    if df.empty:
        raise ValueError(f"No data available for {ticker}")
    df = calculate_technical_indicators(df)
    return df

# Function to plot indicators
def plot_indicators(df, ticker):
    output_dir = f"results/{ticker}/"
    ensure_directory(output_dir)

    plt.figure(figsize=(14, 10))
    plt.plot(df['Close'], label="Close Price", color="blue")
    plt.plot(df['SMA_50'], label="50-Day SMA", color="orange")
    plt.plot(df['SMA_200'], label="200-Day SMA", color="green")
    plt.title(f"{ticker} Technical Indicators")
    plt.legend()
    plt.savefig(f"{output_dir}{ticker}_indicators.png")
    plt.close()

# Main function for analysis
def analyze_us_stock(ticker, start_date, end_date):
    try:
        df = download_and_process_data(ticker, start_date, end_date)
        df.to_csv(f"results/{ticker}/{ticker}_data.csv")
        plot_indicators(df, ticker)
        print(f"Analysis complete for {ticker}")
    except Exception as e:
        print(f"Error analyzing {ticker}: {e}")

# Main script
def main():
    tickers = ["GOOGL", "NFLX", "NVDA"]
    start_date = "2020-01-01"
    end_date = "2023-01-01"

    for ticker in tickers:
        analyze_us_stock(ticker, start_date, end_date)

if __name__ == "__main__":
    main()
```
