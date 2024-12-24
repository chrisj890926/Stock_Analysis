import yfinance as yf
import pandas as pd
import talib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

# 忽略 FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

# 設置 Seaborn 樣式
sns.set(style='whitegrid')

# 美股 ETF 清單
def get_us_etf_tickers():
    return [
        "QQQ",  "QQQM", "VOO", "VTI", "SPY", "IVV", "DIA",
        "VT", "VXUS", "VUG", "VO", "XLF", "BND", "VWO",  "DIAU", "QQQX"
    ]

# 美股個股清單
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

# 計算技術指標
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
# 繪製技術指標圖表 (改進版)
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

# 分析單個股票或 ETF
def analyze_stock(ticker, category):
    try:
        # 加载数据
        data = yf.download(ticker, start='2024-01-01', end=pd.Timestamp('today'), threads=False)
        if data.empty:
            print(f"No data for {ticker}")
            return None

        # 填充缺失值
        data = data.ffill().bfill()

        # 确保索引是 DatetimeIndex
        data.index = pd.to_datetime(data.index, errors='coerce')
        if data.index.isnull().any():
            print(f"Invalid index detected for {ticker}")
            return None

        # 检查数据完整性
        if len(data) < 20:
            print(f"Not enough data for {ticker}")
            return None

        # 打印数据头部信息以调试
        print(f"Processing {ticker} with data shape: {data.shape}")
        print(data.head())

        # 计算技术指标
        df = calculate_technical_indicators(data)

        # 检查是否所有列都存在
        required_columns = ['Close', 'Volume', 'MA10', 'MA20', 'RSI', 'MACD', 'MACD_signal',
                            'Upper_BB', 'Middle_BB', 'Lower_BB', 'ATR', 'ADX']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing columns for {ticker}: {missing_columns}")
            return None

        # 提取相关特征用于分析
        features = df[required_columns]

        # 绘制技术指标图表
        plot_technical_indicators(ticker, df, category)

        # 保存每个股票的技术指标和成交量数据
        features.to_csv(f'results_stock_analysis/{category}/{ticker}/technical_indicators_and_volume.csv')

        return features
    except Exception as e:
        print(f"Error processing {ticker}: {e}")
        return None
# 主程序
def main():
    # 確認清單數量
    etf_tickers = get_us_etf_tickers()
    stock_tickers = get_us_stock_tickers()

    print(f"美股 ETF 清單數量: {len(etf_tickers)}")  # 應為 18
    print(f"美股個股清單數量: {len(stock_tickers)}")  # 應為 45

    # 分析 US ETF
    for ticker in etf_tickers:
        df = analyze_stock(ticker, "ETF")  # 獲取分析結果
        if df is not None:
            check_missing_columns(ticker, df)

    # 分析 US 個股
    for ticker in stock_tickers:
        df = analyze_stock(ticker, "Stocks")  # 獲取分析結果
        if df is not None:
            check_missing_columns(ticker, df)

def check_missing_columns(ticker, df):
    """檢查缺失的技術指標列"""
    required_columns = ['Close', 'Volume', 'MA10', 'MA20', 'RSI', 'MACD', 'MACD_signal',
                        'Upper_BB', 'Middle_BB', 'Lower_BB', 'ATR', 'ADX']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Missing columns for {ticker}: {missing_columns}")

if __name__ == "__main__":
    main()
