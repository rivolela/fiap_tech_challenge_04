import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from typing import Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)

class DataHandler:
    """Handles data loading, preprocessing, and sequence creation for LSTM training."""
    
    def __init__(self, feature_columns: Optional[List[str]] = None):
        self.feature_columns = feature_columns or self._get_default_feature_columns()
        self.scaler = StandardScaler()
        self.data: Optional[pd.DataFrame] = None
        
    def _get_default_feature_columns(self) -> List[str]:
        """Returns default feature columns for the model."""
        return [
            "preco_medio_close",
            "lag_1_mes_preco_medio_close",
            "lag_2_mes_preco_medio_close",
            "lag_3_mes_preco_medio_close",
            "lag_4_mes_preco_medio_close",
            "lag_5_mes_preco_medio_close",
            "lag_6_mes_preco_medio_close",
            "media_movel_6_meses_preco_medio_close",
            "desvio_padrao_movel_6_meses_preco_medio_close",
            "valor_minimo_6_meses_preco_medio_close",
            "valor_maximo_6_meses_preco_medio_close"
        ]
    
    def load_latest_parquet(self, base_dirs: Optional[List[str]] = None) -> pd.DataFrame:
        """Load the latest parquet file from specified directories."""
        if base_dirs is None:
            base_dirs = ["./data/transformed", "../data/transformed", "../../data/transformed"]
        
        latest_file = self._find_latest_parquet_file(base_dirs)
        if latest_file is None:
            raise FileNotFoundError("No parquet files found in any of the expected locations")
        
        logger.info(f"Loading latest parquet file: {latest_file}")
        self.data = self._load_and_process_data(latest_file)
        return self.data
    
    def _find_latest_parquet_file(self, base_dirs: List[str]) -> Optional[str]:
        """Find the most recently modified parquet file."""
        latest_file = None
        latest_time = 0
        
        for dir_path in base_dirs:
            if not os.path.exists(dir_path):
                continue
                
            for file in os.listdir(dir_path):
                if file.endswith('.parquet'):
                    file_path = os.path.join(dir_path, file)
                    file_time = os.path.getmtime(file_path)
                    
                    if file_time > latest_time:
                        latest_time = file_time
                        latest_file = file_path
        
        return latest_file
    
    def _load_and_process_data(self, file_path: str) -> pd.DataFrame:
        """Load and preprocess the parquet file."""
        df = pd.read_parquet(file_path)
        df = df.sort_values(by=["ano", "mes"], ascending=[True, True])
        
        # Filter existing columns
        existing_columns = [col for col in self.feature_columns if col in df.columns]
        df = df[existing_columns]
        
        self._log_data_statistics(df)
        return df
    
    def _log_data_statistics(self, df: pd.DataFrame) -> None:
        """Log basic statistics about the loaded data."""
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        logger.info(f"Target variable range: {df['preco_medio_close'].min():.2f} to {df['preco_medio_close'].max():.2f}")
        logger.info(f"NaN values: {df.isnull().sum().sum()}")
    
    def create_sequences(self, sequence_length: int, horizon: int) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
        """Create sequences for LSTM training."""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_latest_parquet() first.")
        
        data_scaled = self.scaler.fit_transform(self.data)
        
        X, y = [], []
        for i in range(len(data_scaled) - sequence_length - horizon + 1):
            X.append(data_scaled[i:(i + sequence_length)])
            y.append(data_scaled[i + sequence_length:i + sequence_length + horizon, 0])
        
        return np.array(X), np.array(y), self.scaler
    
    def inverse_transform_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """Inverse transform predictions back to original scale."""
        dummy_array = np.zeros((predictions.shape[0], self.scaler.n_features_in_))
        dummy_array[:, 0] = predictions
        return self.scaler.inverse_transform(dummy_array)[:, 0]