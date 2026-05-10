import pandas as pd
import yfinance as yf
from prefect import flow, task
import subprocess
import os

@task(name="Fetch Yesterday Candle")
def fetch_yesterday_candle() -> pd.DataFrame:
    """Fetch last 2 days; take the completed yesterday candle."""
    df = yf.download("BTC-USD", period="2d", interval="1d", progress=False)
    
    # Newer yfinance versions sometimes return a MultiIndex (Price, Ticker). We flatten it if needed.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    
    # second-to-last row = yesterday (complete)
    yesterday_candle = df.iloc[[-2]].copy()   
    yesterday_candle.index = pd.to_datetime(yesterday_candle.index).tz_localize(None)
    yesterday_candle.index.name = 'Date'
    
    # Map YFinance columns to Investing.com schema
    yesterday_candle = yesterday_candle.rename(columns={"Close": "Price", "Volume": "Vol."})
    yesterday_candle["Change %"] = "0.0%"
    
    return yesterday_candle


@task(name="Append to Raw CSV")
def append_to_raw_csv(new_row: pd.DataFrame, path: str = "data/raw/bitcoin_investing.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    if os.path.exists(path):
        df = pd.read_csv(path, index_col='Date', parse_dates=True)
        # Check if the date is already in the dataframe to avoid duplicates
        if new_row.index[0] not in df.index:
            # pd.concat is the modern pandas way vs df.append
            df = pd.concat([df, new_row])
            df.to_csv(path)
            print(f"[SUCCESS] Appended: {new_row.index[0].date()}")
        else:
            print(f"[INFO] Already exists: {new_row.index[0].date()}, skipping")
    else:
        new_row.to_csv(path)
        print(f"[SUCCESS] Created new raw CSV with: {new_row.index[0].date()}")


@task(name="Rerun Feature Engineering (DVC)")
def rerun_feature_engineering():
    """Trigger DVC to re-run the pipeline steps depending on the raw data."""
    try:
        # Running 'dvc repro' automatically detects that data/raw/bitcoin_investing.csv 
        # has changed and triggers features.py and splitter.py
        print("[INFO] Running `dvc repro` to update features...")
        subprocess.run(["dvc", "repro"], check=True)
        print("[SUCCESS] Feature engineering and splits updated via DVC.")
    except Exception as e:
        print(f"[ERROR] Failed to run DVC repro: {e}")


@flow(name="daily-btc-update")
def daily_update():
    row = fetch_yesterday_candle()
    append_to_raw_csv(row, path="data/raw/bitcoin_investing.csv")
    rerun_feature_engineering()


if __name__ == "__main__":
    # Schedule: every day at 00:05 UTC
    # To run this scheduling server, you run `python src/pipelines/daily_update_flow.py`
    daily_update.serve(name="daily-btc-update", cron="5 0 * * *")