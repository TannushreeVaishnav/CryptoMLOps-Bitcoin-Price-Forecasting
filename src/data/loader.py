import pandas as pd
import numpy as np
import yfinance as yf

def parse_volume(s: str) -> float:
    """'32.45K' → 32450.0 | '1.2M' → 1200000.0 | '900B' → 900000000000.0"""
    s = str(s).strip().replace(',', '')
    if s.endswith('K'): return float(s[:-1]) * 1_000
    if s.endswith('M'): return float(s[:-1]) * 1_000_000
    if s.endswith('B'): return float(s[:-1]) * 1_000_000_000
    try: return float(s)
    except: return np.nan

def parse_change(s: str) -> float:
    """'+2.34%' → 0.0234 | '-1.02%' → -0.0102"""
    return float(str(s).strip().replace('%', '').replace(',', '')) / 100

def load_bitcoin_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    df['Date']  = pd.to_datetime(df['Date'], format='mixed', dayfirst=True)
    df          = df.sort_values('Date').set_index('Date')

    for col in ['Price', 'Open', 'High', 'Low']:
        df[col] = df[col].astype(str).str.replace(',', '').astype(float)

    df['Volume'] = df['Vol.'].apply(parse_volume)
    df['Change'] = df['Change %'].apply(parse_change)
    df = df.rename(columns={'Price': 'Close'})
    df = df.drop(columns=['Vol.', 'Change %'])

    # Sanity assertions — fail loud if data is wrong
    assert (df['High'] >= df['Low']).all(),   "High < Low found - check your CSV"
    assert (df['High'] >= df['Close']).all(), "High < Close found - check your CSV"
    assert df[['Open','High','Low','Close']].isnull().sum().sum() == 0, "Nulls in OHLC"

    print(f"Loaded {len(df)} rows | {df.index.min()} to {df.index.max()}")
    return df

def fetch_live_btc(days: int = 60) -> pd.DataFrame:
    """
    Fetch last `days` daily BTC candles from Yahoo Finance.
    Free. No account. No API key. Always current.
    Called on every /predict request.
    """
    df = yf.download("BTC-USD", period=f"{days}d", interval="1d", progress=False)
    
    # Newer yfinance versions sometimes return a MultiIndex (Price, Ticker). We flatten it if needed.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = 'Date'

    assert len(df) >= 35, f"Need ≥35 rows for feature engineering, got {len(df)}"
    return df