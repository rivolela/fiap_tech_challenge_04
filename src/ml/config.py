"""Simple configuration for LSTM training."""

DEFAULT_CONFIG = {
    'sequence_length': 24,
    'horizon': 6,
    'train_split': 0.8,
    'hidden_size': 32,
    'num_layers': 1,
    'dropout': 0.2,
    'learning_rate': 0.001,
    'weight_decay': 0.01,
    'scheduler_patience': 5,
    'epochs': 200
}

def get_config():
    """Get default configuration."""
    return DEFAULT_CONFIG.copy()