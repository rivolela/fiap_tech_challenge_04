import torch
import torch.nn as nn
from typing import Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class EnhancedLSTM(nn.Module):
    """Enhanced LSTM model for time series forecasting."""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int, dropout: float = 0.2):
        super(EnhancedLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]  # Get last time step output
        return self.fc(out)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model configuration information."""
        return {
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'output_size': self.output_size,
            'total_params': sum(p.numel() for p in self.parameters())
        }