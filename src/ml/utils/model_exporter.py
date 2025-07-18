import torch
import pickle
import json
import os
import numpy as np
import pandas as pd
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class ModelExporter:
    """Handle model export and loading for deployment."""
    
    @staticmethod
    def export_complete_model(model, scaler, feature_columns: List[str], 
                            model_path: str = "./outputs/model_export/"):
        """Export complete model package for deployment."""
        os.makedirs(model_path, exist_ok=True)
        
        # 1. Save PyTorch model state dict
        torch.save(model.state_dict(), os.path.join(model_path, 'lstm_model.pth'))
        logger.info("✓ Model state dict saved")
        
        # 2. Save complete model (for direct loading)
        torch.save(model, os.path.join(model_path, 'complete_model.pth'))
        logger.info("✓ Complete model saved")
        
        # 3. Save model configuration
        model_config = {
            'input_size': model.lstm.input_size,
            'hidden_size': model.lstm.hidden_size,
            'num_layers': model.lstm.num_layers,
            'output_size': model.fc.out_features,
            'dropout': model.lstm.dropout if hasattr(model.lstm, 'dropout') else 0.0
        }
        
        with open(os.path.join(model_path, 'model_config.json'), 'w') as f:
            json.dump(model_config, f, indent=2)
        logger.info("✓ Model configuration saved")
        
        # 4. Save scaler
        with open(os.path.join(model_path, 'scaler.pkl'), 'wb') as f:
            pickle.dump(scaler, f)
        logger.info("✓ Scaler saved")
        
        # 5. Save feature columns
        with open(os.path.join(model_path, 'feature_columns.json'), 'w') as f:
            json.dump(feature_columns, f, indent=2)
        logger.info("✓ Feature columns saved")
        
        # 6. Export to ONNX (for broader compatibility)
        try:
            ModelExporter._export_to_onnx(model, model_path)
        except Exception as e:
            logger.warning(f"ONNX export failed: {e}")
        
        # 7. Create deployment info
        deployment_info = {
            'model_type': 'LSTM',
            'framework': 'PyTorch',
            'input_shape': [1, 24, model_config['input_size']],  # [batch, sequence, features]
            'output_shape': [1, model_config['output_size']],
            'preprocessing_required': True,
            'files': {
                'model_state': 'lstm_model.pth',
                'complete_model': 'complete_model.pth',
                'config': 'model_config.json',
                'scaler': 'scaler.pkl',
                'features': 'feature_columns.json',
                'onnx': 'model.onnx'
            }
        }
        
        with open(os.path.join(model_path, 'deployment_info.json'), 'w') as f:
            json.dump(deployment_info, f, indent=2)
        
        logger.info(f"🎉 Complete model package exported to: {os.path.abspath(model_path)}")
        return model_path
    
    @staticmethod
    def _export_to_onnx(model, model_path: str):
        """Export model to ONNX format."""
        try:
            # Create dummy input
            dummy_input = torch.randn(1, 24, model.lstm.input_size)
            
            # Export to ONNX
            torch.onnx.export(
                model,
                dummy_input,
                os.path.join(model_path, 'model.onnx'),
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
            )
            logger.info("✓ ONNX model exported")
        except ImportError:
            logger.warning("ONNX not available. Install with: pip install onnx")
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")

class ModelLoader:
    """Load exported models for inference."""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.config = None
    
    def load_complete_package(self):
        """Load the complete model package."""
        # Load configuration
        with open(os.path.join(self.model_path, 'model_config.json'), 'r') as f:
            self.config = json.load(f)
        
        # Load complete model (with weights_only=False for backward compatibility)
        self.model = torch.load(
            os.path.join(self.model_path, 'complete_model.pth'),
            weights_only=False
        )
        self.model.eval()
        
        # Load scaler
        with open(os.path.join(self.model_path, 'scaler.pkl'), 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Load feature columns
        with open(os.path.join(self.model_path, 'feature_columns.json'), 'r') as f:
            self.feature_columns = json.load(f)
        
        logger.info("✓ Complete model package loaded successfully")
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_complete_package() first.")
        
        # Preprocess data
        processed_data = self._preprocess_data(data)
        
        # Make prediction
        with torch.no_grad():
            prediction = self.model(processed_data)
        
        # Convert to numpy and handle different shapes
        prediction_array = prediction.numpy()
        
        # For multi-output predictions (like 6-month forecasts)
        if prediction_array.ndim == 2:
            # Shape: (batch_size, forecast_horizon)
            return prediction_array
        else:
            # Single output case
            return prediction_array.reshape(1, -1)
    
    def _preprocess_data(self, data: pd.DataFrame) -> torch.Tensor:
        """Preprocess data for prediction."""
        # Select required columns
        data_filtered = data[self.feature_columns]
        
        # Scale data
        data_scaled = self.scaler.transform(data_filtered)
        
        # Create sequence (assuming single prediction)
        sequence = data_scaled[-24:]  # Last 24 time steps
        
        # Convert to tensor and add batch dimension
        return torch.FloatTensor(sequence).unsqueeze(0)
    
    def _inverse_transform(self, predictions: np.ndarray) -> np.ndarray:
        """Inverse transform predictions."""
        # Handle different prediction shapes
        if predictions.ndim == 1:
            # Single prediction
            dummy_array = np.zeros((1, len(self.feature_columns)))
            dummy_array[0, 0] = predictions[0]
            return self.scaler.inverse_transform(dummy_array)[0, 0:len(predictions)]
        else:
            # Multiple predictions
            batch_size, pred_size = predictions.shape
            dummy_array = np.zeros((batch_size, len(self.feature_columns)))
            dummy_array[:, 0:pred_size] = predictions
            return self.scaler.inverse_transform(dummy_array)[:, 0:pred_size]
