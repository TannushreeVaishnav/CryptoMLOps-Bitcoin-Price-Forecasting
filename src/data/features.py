import pandas as pd
import numpy as np
import ta
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input:   raw OHLCV DataFrame (from investing.com CSV or yfinance)
    Output:  DataFrame with engineered features
    Critical: all features use .shift(1) where needed — zero look-ahead
    """

    # ── Trend
    df['SMA_7'] = SMAIndicator(close=df['Close'], window=7).sma_indicator()
    df['SMA_21'] = SMAIndicator(close=df['Close'], window=21).sma_indicator()
    df['EMA_12'] = EMAIndicator(close=df['Close'], window=12).ema_indicator()
    df['EMA_26'] = EMAIndicator(close=df['Close'], window=26).ema_indicator()
    
    macd = MACD(close=df['Close'], window_slow=26, window_fast=12, window_sign=9)
    df['MACD_12_26_9'] = macd.macd()
    df['MACDh_12_26_9'] = macd.macd_diff()
    df['MACDs_12_26_9'] = macd.macd_signal()

    # ── Momentum
    df['RSI_14'] = RSIIndicator(close=df['Close'], window=14).rsi()
    
    stoch = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'], window=14, smooth_window=3)
    df['STOCHk_14_3_3'] = stoch.stoch()
    df['STOCHd_14_3_3'] = stoch.stoch_signal()
    
    df['ROC_10'] = ROCIndicator(close=df['Close'], window=10).roc()

    # ── Volatility
    bb = BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['BBL_20_2.0'] = bb.bollinger_lband()
    df['BBM_20_2.0'] = bb.bollinger_mavg()
    df['BBU_20_2.0'] = bb.bollinger_hband()
    df['BBB_20_2.0'] = bb.bollinger_wband()
    df['BBP_20_2.0'] = bb.bollinger_pband()
    
    df['ATRr_14'] = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14).average_true_range()

    # ── Volume
    df['OBV'] = OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume']).on_balance_volume()

    # ── Price-derived
    df['daily_return']  = df['Close'].pct_change()
    df['log_return']    = np.log(df['Close'] / df['Close'].shift(1))
    df['price_range']   = (df['High'] - df['Low']) / df['Close']
    df['body_size']     = abs(df['Close'] - df['Open']) / df['Close']

    # ── Lag features — NO look-ahead bias
    for lag in [1, 2, 3, 5, 7, 14]:
        df[f'close_lag_{lag}']  = df['Close'].shift(lag)
        df[f'return_lag_{lag}'] = df['daily_return'].shift(lag)

    # ── Calendar
    df['day_of_week'] = df.index.dayofweek
    df['is_weekend']  = (df['day_of_week'] >= 5).astype(int)

    df.dropna(inplace=True)
    return df