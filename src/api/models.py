"""LSTM Model Service Module with enhanced error handling and monitoring"""
import os
import sys
import pickle
import json
import torch
import torch.nn as nn
from ml.models.lstm_model import EnhancedLSTM
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any, Union
from .config import APIConfig


# Add the project root to Python path to resolve 'ml' module imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import monitoring
try:
    from ..monitoring import performance_monitor
    from ..monitoring.middleware import monitor_model_prediction
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False
    def monitor_model_prediction(func):
        return func  # No-op decorator if monitoring not available

logger = logging.getLogger(__name__)

# Add safe globals for secure model loading
torch.serialization.add_safe_globals([
    EnhancedLSTM,
    torch.nn.modules.rnn.LSTM,
    torch.nn.modules.linear.Linear,
    torch.nn.modules.dropout.Dropout
])

class LSTMModelService:
    """Service to handle LSTM model loading and predictions"""
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path or APIConfig.find_model_path()
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.config = None
        self.is_loaded = False
        
        # Required features for validation
        self.required_features = [
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
    
    def load_model(self) -> bool:
        """Load all model components with enhanced error handling"""
        try:
            logger.info(f"Loading LSTM model from: {self.model_path}")
            
            # Check if model directory exists
            if not os.path.exists(self.model_path):
                logger.error(f"Model directory not found: {self.model_path}")
                # Try alternative paths
                alt_path = APIConfig.find_model_path()
                if alt_path != self.model_path and os.path.exists(alt_path):
                    logger.info(f"Trying alternative path: {alt_path}")
                    self.model_path = alt_path
                else:
                    self._log_available_directories()
                    return False
            
            # List available files for debugging
            files_in_dir = os.listdir(self.model_path)
            logger.info(f"Files in model directory: {files_in_dir}")
            
            # Try to import ml module if needed
            try:
                import ml
                logger.info("✅ ml module imported successfully")
            except ImportError:
                logger.warning("⚠️ ml module not found, trying to add to path...")
                # Try to find and add ml module path
                possible_ml_paths = [
                    os.path.join(project_root, 'src'),
                    os.path.join(project_root, 'src', 'ml'),
                    project_root
                ]
                for ml_path in possible_ml_paths:
                    if ml_path not in sys.path:
                        sys.path.insert(0, ml_path)
                
                try:
                    import ml
                    logger.info("✅ ml module imported after path adjustment")
                except ImportError as e:
                    logger.warning(f"⚠️ Still can't import ml module: {e}")
                    # Continue anyway, might work with weights_only=True
            
            # Load PyTorch model with different strategies
            model_file = self._find_model_file()
            if not model_file:
                return False
                
            try:
                # Try loading with weights_only=True first (safer)
                self.model = torch.load(model_file, weights_only=True)
                logger.info("✅ Model loaded with weights_only=True")
            except Exception as e1:
                logger.warning(f"Failed to load with weights_only=True: {e1}")
                try:
                    # Try loading with weights_only=False
                    self.model = torch.load(model_file, weights_only=False)
                    logger.info("✅ Model loaded with weights_only=False")
                except Exception as e2:
                    logger.warning(f"Failed to load with weights_only=False: {e2}")
                    try:
                        # Try loading with map_location
                        self.model = torch.load(model_file, map_location='cpu', weights_only=False)
                        logger.info("✅ Model loaded with map_location=cpu")
                    except Exception as e3:
                        logger.error(f"All model loading strategies failed: {e3}")
                        return False
            
            self.model.eval()
            logger.info("✅ Model set to evaluation mode")
            
            # Load other components
            if not self._load_scaler():
                logger.warning("⚠️ Scaler loading failed, predictions may not be denormalized")
            
            if not self._load_feature_columns():
                logger.warning("⚠️ Using default feature columns")
            
            if not self._load_config():
                logger.warning("⚠️ Using default config")
            
            self.is_loaded = True
            logger.info("🎉 Model service initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Critical error loading model: {e}")
            logger.error(f"Current working directory: {os.getcwd()}")
            logger.error(f"Python path: {sys.path[:3]}...")  # Show first 3 entries
            self.is_loaded = False
            return False
    
    def _find_model_file(self) -> str:
        """Find the model file, trying different names"""
        possible_names = [
            'complete_model.pth',
            'model.pth',
            'lstm_model.pth',
            'best_model.pth',
            'final_model.pth'
        ]
        
        for name in possible_names:
            path = os.path.join(self.model_path, name)
            if os.path.exists(path):
                logger.info(f"Found model file: {name}")
                return path
        
        logger.error(f"No model file found. Checked: {possible_names}")
        return None
    
    def _load_scaler(self) -> bool:
        """Load the scaler file"""
        try:
            scaler_file = os.path.join(self.model_path, 'scaler.pkl')
            if os.path.exists(scaler_file):
                with open(scaler_file, 'rb') as f:
                    self.scaler = pickle.load(f)
                logger.info("✅ Scaler loaded")
                return True
            else:
                logger.warning(f"Scaler file not found: {scaler_file}")
                return False
        except Exception as e:
            logger.error(f"Error loading scaler: {e}")
            return False
    
    def _load_feature_columns(self) -> bool:
        """Load feature columns from JSON or pickle"""
        try:
            # Try JSON first
            json_file = os.path.join(self.model_path, 'feature_columns.json')
            if os.path.exists(json_file):
                with open(json_file, 'r') as f:
                    self.feature_columns = json.load(f)
                logger.info("✅ Feature columns loaded from JSON")
                return True
            
            # Try pickle
            pkl_file = os.path.join(self.model_path, 'feature_columns.pkl')
            if os.path.exists(pkl_file):
                with open(pkl_file, 'rb') as f:
                    self.feature_columns = pickle.load(f)
                logger.info("✅ Feature columns loaded from pickle")
                return True
            
            # Use default
            self.feature_columns = self.required_features
            logger.warning("Using default feature columns")
            return True
            
        except Exception as e:
            logger.error(f"Error loading feature columns: {e}")
            self.feature_columns = self.required_features
            return True
    
    def _load_config(self) -> bool:
        """Load model configuration"""
        try:
            # Try JSON first
            json_file = os.path.join(self.model_path, 'model_config.json')
            if os.path.exists(json_file):
                with open(json_file, 'r') as f:
                    self.config = json.load(f)
                logger.info("✅ Config loaded from JSON")
                return True
            
            # Try pickle
            pkl_file = os.path.join(self.model_path, 'model_config.pkl')
            if os.path.exists(pkl_file):
                with open(pkl_file, 'rb') as f:
                    self.config = pickle.load(f)
                logger.info("✅ Config loaded from pickle")
                return True
            
            # Use default
            self.config = {'sequence_length': 24, 'forecast_horizon': 6}
            logger.warning("Using default config")
            return True
            
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            self.config = {'sequence_length': 24, 'forecast_horizon': 6}
            return True
    
    def _log_available_directories(self):
        """Log available directories for debugging"""
        logger.info("Available directories with 'model' in name:")
        for root, dirs, files in os.walk('.'):
            if 'model' in root.lower() or any('model' in f.lower() for f in files):
                logger.info(f"  {root}: {files}")
    
    @monitor_model_prediction
    def predict(self, data: Union[List[Dict], Dict], forecast_horizon: int = 6) -> List[float]:
        """Make LSTM predictions with enhanced error handling"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            # Preprocess data
            processed_data = self._preprocess_data(data)
            
            # Make prediction
            with torch.no_grad():
                predictions = self.model(processed_data)
            
            # Convert to numpy
            pred_array = predictions.cpu().numpy()
            
            # Extract predictions for the requested horizon
            if len(pred_array.shape) > 1:
                result = pred_array[0, :forecast_horizon].tolist()
            else:
                result = pred_array.tolist()[:forecast_horizon]
            
            # Denormalize if scaler is available
            if self.scaler is not None:
                result = self._denormalize_predictions(result)
            else:
                logger.warning("⚠️ No scaler available, returning raw predictions")
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise RuntimeError(f"Prediction failed: {str(e)}")
    
    def _preprocess_data(self, data: Union[List[Dict], Dict]) -> torch.Tensor:
        """Preprocess data for LSTM prediction"""
        try:
            # Convert to DataFrame
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                df = pd.DataFrame([data])
            else:
                raise ValueError("Data must be list of dicts or single dict")
            
            # Ensure we have required features
            available_features = [col for col in self.feature_columns if col in df.columns]
            if not available_features:
                raise ValueError(f"No matching features found. Available: {df.columns.tolist()}")
            
            # Use available features
            data_filtered = df[available_features]
            
            # Check sequence length
            sequence_length = self.config.get('sequence_length', 24)
            if len(data_filtered) < sequence_length:
                logger.warning(f"Not enough data points ({len(data_filtered)} < {sequence_length}), using all available")
                sequence_length = len(data_filtered)
            
            # Scale data if scaler is available
            if self.scaler is not None:
                data_scaled = self.scaler.transform(data_filtered)
            else:
                # Normalize manually if no scaler
                data_scaled = (data_filtered - data_filtered.mean()) / (data_filtered.std() + 1e-8)
                data_scaled = data_scaled.values
            
            # Create sequence
            sequence = data_scaled[-sequence_length:]
            
            # Convert to tensor
            return torch.FloatTensor(sequence).unsqueeze(0)
            
        except Exception as e:
            logger.error(f"Preprocessing error: {e}")
            raise
    
    def _denormalize_predictions(self, predictions: List[float]) -> List[float]:
        """Denormalize predictions using the scaler"""
        try:
            if self.scaler is None:
                return predictions
            
            # Create dummy array for inverse transform
            dummy = np.zeros((len(predictions), self.scaler.n_features_in_))
            dummy[:, 0] = predictions
            
            # Apply inverse transform
            denormalized = self.scaler.inverse_transform(dummy)[:, 0]
            return denormalized.tolist()
            
        except Exception as e:
            logger.error(f"Denormalization error: {e}")
            return predictions
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information"""
        return {
            "model_status": {
                "is_loaded": self.is_loaded,
                "model_path": self.model_path,
                "has_scaler": self.scaler is not None
            },
            "model_config": self.config,
            "features": {
                "feature_columns": self.feature_columns,
                "num_features": len(self.feature_columns) if self.feature_columns else 0,
                "required_features": self.required_features
            },
            "capabilities": {
                "max_forecast_horizon": 12,
                "min_sequence_length": self.config.get('sequence_length', 24) if self.config else 24,
                "model_type": "LSTM Neural Network"
            }
        }
    
    def validate_input_data(self, data: Union[List[Dict], Dict]) -> tuple[bool, str]:
        """Validate input data for prediction"""
        try:
            if isinstance(data, list):
                if not data:
                    return False, "Empty data array provided"
                sample = data[0]
            elif isinstance(data, dict):
                sample = data
                data = [data]
            else:
                return False, "Data must be list of dicts or single dict"
            
            # Check minimum data length
            sequence_length = self.config.get('sequence_length', 24) if self.config else 24
            if len(data) < 6:  # Minimum reasonable amount
                return False, f"Need at least 6 records, got {len(data)}"
            
            # Check if we have some required features
            available_features = [f for f in self.required_features if f in sample]
            if len(available_features) < 5:  # At least 5 features
                return False, f"Need at least 5 matching features. Found: {available_features}"
            
            return True, "Data validation successful"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"

