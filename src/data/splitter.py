import pandas as pd
import logging
from typing import Tuple, Generator
import argparse
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def split_train_val_test(df: pd.DataFrame, date_column: str = 'date') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the time-series data into Train, Validation, and Test sets based on exact dates.
    
    Train set:      Apr 2016 -> Dec 2022
    Validation set: Jan 2023 -> Dec 2023
    Test set:       Jan 2024 -> May 2026
    """
    df_split = df.copy()
    
    if date_column in df_split.columns:
        if not pd.api.types.is_datetime64_any_dtype(df_split[date_column]):
            df_split[date_column] = pd.to_datetime(df_split[date_column])
        
        train = df_split[(df_split[date_column] >= '2016-04-01') & (df_split[date_column] <= '2022-12-31')]
        val = df_split[(df_split[date_column] >= '2023-01-01') & (df_split[date_column] <= '2023-12-31')]
        test = df_split[(df_split[date_column] >= '2024-01-01') & (df_split[date_column] <= '2026-05-31')]
    elif pd.api.types.is_datetime64_any_dtype(df_split.index):
        train = df_split['2016-04-01':'2022-12-31']
        val = df_split['2023-01-01':'2023-12-31']
        test = df_split['2024-01-01':'2026-05-31']
    else:
        raise ValueError(f"Could not find a valid datetime column named '{date_column}' or a DatetimeIndex.")
    
    logger.info(f"Train set (2016-2022): {len(train)} rows")
    logger.info(f"Validation set (2023): {len(val)} rows")
    logger.info(f"Test set (2024-2026): {len(test)} rows")
    
    return train, val, test

def get_walk_forward_splits(df: pd.DataFrame, date_column: str = 'date') -> Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None]:
    """
    Generates Walk-Forward CV folds for hyperparameter tuning.
    
    Fold 1:  Train 2016-2018  |  Val 2019
    Fold 2:  Train 2016-2019  |  Val 2020   <- COVID crash stress test
    Fold 3:  Train 2016-2020  |  Val 2021   <- Bull run stress test
    Fold 4:  Train 2016-2021  |  Val 2022   <- Bear market stress test
    Fold 5:  Train 2016-2022  |  Val 2023
    """
    df_split = df.copy()
    has_date_col = date_column in df_split.columns
    
    if has_date_col:
        if not pd.api.types.is_datetime64_any_dtype(df_split[date_column]):
            df_split[date_column] = pd.to_datetime(df_split[date_column])
    elif not pd.api.types.is_datetime64_any_dtype(df_split.index):
         raise ValueError(f"Could not find a valid datetime column named '{date_column}' or a DatetimeIndex.")
         
    folds = [
        ('2016-04-01', '2018-12-31', '2019-01-01', '2019-12-31'), # Fold 1
        ('2016-04-01', '2019-12-31', '2020-01-01', '2020-12-31'), # Fold 2 (COVID)
        ('2016-04-01', '2020-12-31', '2021-01-01', '2021-12-31'), # Fold 3 (Bull run)
        ('2016-04-01', '2021-12-31', '2022-01-01', '2022-12-31'), # Fold 4 (Bear market)
        ('2016-04-01', '2022-12-31', '2023-01-01', '2023-12-31'), # Fold 5
    ]
    
    for i, (train_start, train_end, val_start, val_end) in enumerate(folds, 1):
        if has_date_col:
            train_fold = df_split[(df_split[date_column] >= train_start) & (df_split[date_column] <= train_end)]
            val_fold = df_split[(df_split[date_column] >= val_start) & (df_split[date_column] <= val_end)]
        else:
            # Slicing via DatetimeIndex
            train_fold = df_split[train_start:train_end]
            val_fold = df_split[val_start:val_end]
            
        logger.info(f"Fold {i} -> Train: {train_start} to {train_end} ({len(train_fold)} rows) | Val: {val_start} to {val_end} ({len(val_fold)} rows)")
        yield train_fold, val_fold


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Split time-series data using strict Walk-Forward dates.")
    parser.add_argument("--input", type=str, required=True, help="Path to input raw dataset (CSV or Parquet)")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory to save the resulting split datasets")
    parser.add_argument("--date_col", type=str, default="date", help="Name of the datetime column")
    
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    logger.info(f"Loading data from {args.input}")
    if args.input.endswith('.parquet'):
        df = pd.read_parquet(args.input)
    else:
        df = pd.read_csv(args.input)
        
    train, val, test = split_train_val_test(df, date_column=args.date_col)
    
    train_path = os.path.join(args.out_dir, "train.csv")
    val_path = os.path.join(args.out_dir, "val.csv")
    test_path = os.path.join(args.out_dir, "test.csv")
    
    train.to_csv(train_path, index=False)
    val.to_csv(val_path, index=False)
    test.to_csv(test_path, index=False)
    
    logger.info(f"Splits saved to {args.out_dir}/: train.csv, val.csv, test.csv")
