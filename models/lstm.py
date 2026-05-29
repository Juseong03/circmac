import torch
import torch.nn as nn
from .cnn import CNN1D

class LSTM(nn.Module):
    def __init__(
        self, 
        d_model=128, 
        n_layer=4, 
        dropout=0.1, 
        bidirectional=True, 
        cnn=False, 
        rc=False
    ):
        """
        BiLSTM-based Model

        Args:
            d_model (int): Dimension of input features.
            n_layer (int): Number of LSTM layers.
            dropout (float): Dropout probability.
        """
        super(LSTM, self).__init__()

        self.rc = rc
        if cnn:
            n_layer = n_layer // 2
            n_cnn_layer = n_layer // 2
            self.cnn = CNN1D(d_model, n_cnn_layer, dropout)
        else:
            self.cnn = nn.Identity()

        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_layer,
            dropout=dropout if n_layer > 1 else 0.0,  # Dropout only if n_layer > 1
            batch_first=True,  # Input/output shape: (batch_size, seq_len, d_model)
            bidirectional=bidirectional  # Enable bidirectional LSTM
        )

        # Dropout Layer
        self.dropout = nn.Dropout(dropout)

        # Output projection layer to map back to (batch_size, seq_len, d_model)
        if bidirectional:
            self.output_projection = nn.Linear(d_model * 2, d_model)
        else:
            self.output_projection = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None, x_rc=None, mask_rc=None):
        """
        Forward pass of the BiLSTM model.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        x = self.cnn(x)
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        x = self.output_projection(lstm_out)
        if self.rc:
            x_rc = self.cnn(x_rc)
            lstm_out_rc, _ = self.lstm(x_rc)
            lstm_out_rc = self.dropout(lstm_out_rc)
            x_rc = self.output_projection(lstm_out_rc)
        return x, x_rc
