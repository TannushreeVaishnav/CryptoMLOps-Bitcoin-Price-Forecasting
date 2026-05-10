import torch
from torch.utils.data import Dataset
import pandas as pd

class BTCDataset(Dataset):
    def __init__(self, df: pd.DataFrame, seq_length: int = 30, target_col: str = 'Close'):
        self.seq_length = seq_length
        self.features = df.values
        self.target_idx = df.columns.get_loc(target_col)
        
    def __len__(self):
        # We need seq_length rows for input, and the next day as the target
        return len(self.features) - self.seq_length
        
    def __getitem__(self, idx):
        # x is the sequence of 30 days
        x = self.features[idx : idx + self.seq_length]
        # y is the target column of the 31st day
        y = self.features[idx + self.seq_length, self.target_idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
