from prefect import flow, task
import os
import yaml
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data.loader import load_bitcoin_csv
from src.data.features import engineer_features as eng_features
from src.data.splitter import split_train_val_test

@task(name="load_and_validate")
def load_and_validate(raw_path: str):
    print(f"Loading raw data from {raw_path}...")
    df = load_bitcoin_csv(raw_path)
    return df

@task(name="engineer_features")
def engineer_features(df):
    print("Engineering features...")
    df_features = eng_features(df)
    return df_features

@task(name="split")
def split(df_features):
    print("Splitting data into Train/Val/Test...")
    train, val, test = split_train_val_test(df_features, date_column='Date')
    return train, val, test

@task(name="fit_scaler")
def fit_scaler(train, val, test, splits_dir: str, scaler_path: str):
    print("Applying StandardScaler...")
    scaler = StandardScaler()
    
    # Fit the scaler ONLY on the training data to prevent data leakage!
    train_scaled = pd.DataFrame(scaler.fit_transform(train), index=train.index, columns=train.columns)
    val_scaled = pd.DataFrame(scaler.transform(val), index=val.index, columns=val.columns)
    test_scaled = pd.DataFrame(scaler.transform(test), index=test.index, columns=test.columns)
    
    # Save the splits
    train_scaled.to_csv(os.path.join(splits_dir, "train.csv"))
    val_scaled.to_csv(os.path.join(splits_dir, "val.csv"))
    test_scaled.to_csv(os.path.join(splits_dir, "test.csv"))
    print(f"Splits saved to {splits_dir}")
    
    # Save the scaler for inference
    joblib.dump(scaler, scaler_path)
    print(f"Scaler artifact saved to {scaler_path}")
    return True

@flow(name="btc-preprocess")
def preprocess():
    # Paths
    raw_path = "data/raw/bitcoin_investing.csv"
    features_path = "data/processed/features.csv"
    splits_dir = "data/processed/splits"
    scaler_path = "artifacts/scaler.pkl"
    
    # Ensure directories exist
    os.makedirs(os.path.dirname(features_path), exist_ok=True)
    os.makedirs(splits_dir, exist_ok=True)
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    
    df = load_and_validate(raw_path)
    df_features = engineer_features(df)
    
    # Save features before splitting
    df_features.to_csv(features_path)
    print(f"Features saved to {features_path}")
    
    train, val, test = split(df_features)
    fit_scaler(train, val, test, splits_dir, scaler_path)
    
    print("[SUCCESS] Preprocessing complete!")

if __name__ == "__main__":
    preprocess()
