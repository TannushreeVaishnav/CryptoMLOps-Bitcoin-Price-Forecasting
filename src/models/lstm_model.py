import torch
import torch.nn as nn
import pytorch_lightning as pl

class BTCLSTMModel(pl.LightningModule):
    """
    Sequence-to-one LSTM.
    Input:  (batch, 30, num_features)  — 30 days of daily features
    Output: (batch,)                   — next day's closing price
    """

    def __init__(self, input_size: int, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.2, lr: float = 1e-3, weight_decay: float = 1e-4):
        super().__init__()
        self.save_hyperparameters()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        out = self.dropout(lstm_out[:, -1, :])   # last timestep only
        return self.fc(out).squeeze(-1)

    def training_step(self, batch, _):
        x, y = batch
        loss = nn.functional.mse_loss(self(x), y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        y_hat = self(x)
        self.log('val_loss',  nn.functional.mse_loss(y_hat, y), prog_bar=True)
        self.log('val_mae',   nn.functional.l1_loss(y_hat, y),  prog_bar=True)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, patience=5, factor=0.5)
        return {'optimizer': opt, 'lr_scheduler': sched, 'monitor': 'val_loss'}