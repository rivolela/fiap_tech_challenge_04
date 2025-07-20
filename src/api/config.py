"""API Configuration Module"""
import os
import logging
from typing import Dict, Any

class APIConfig:
    """API Configuration Class"""
    
    # Server Configuration
    HOST = os.environ.get('HOST', '0.0.0.0')
    PORT = int(os.environ.get('PORT', 8081))  # Changed default port to avoid conflicts
    DEBUG = os.environ.get('DEBUG', 'false').lower() == 'true'
    
    # Model Configuration
    MODEL_PATH = os.environ.get('MODEL_PATH', './outputs/model_export/')
    PREDICTIONS_PATH = os.environ.get('PREDICTIONS_PATH', './outputs/predictions.csv')
    
    # Alternative model paths to try
    ALTERNATIVE_MODEL_PATHS = [
        './outputs/model_export/',
        './model_export/',
        './models/',
        './outputs/models/',
        os.path.join(os.path.dirname(__file__), '../../outputs/model_export/'),
        os.path.join(os.path.dirname(__file__), '../../outputs/models/'),
    ]
    
    # Logging Configuration
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FILE = os.environ.get('LOG_FILE', 'lstm_api.log')
    
    # API Metadata
    API_VERSION = "1.0.0"
    API_TITLE = "FIAP Tech Challenge 04 - LSTM Stock Prediction API"
    API_DESCRIPTION = "Production API for BBAS3 stock price prediction using LSTM"
    
    @classmethod
    def find_model_path(cls):
        """Find available model path from alternatives"""
        for path in cls.ALTERNATIVE_MODEL_PATHS:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path):
                # Check if it contains model files
                files = os.listdir(abs_path)
                model_files = [f for f in files if f.endswith('.pth') or f.endswith('.pkl')]
                if model_files:
                    return abs_path
        return cls.MODEL_PATH  # Return default if none found
    
    @classmethod
    def setup_logging(cls):
        """Setup application logging"""
        logging.basicConfig(
            level=getattr(logging, cls.LOG_LEVEL),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(cls.LOG_FILE),
                logging.StreamHandler()
            ]
        )
        
    @classmethod
    def get_config_dict(cls) -> Dict[str, Any]:
        """Get configuration as dictionary"""
        return {
            'host': cls.HOST,
            'port': cls.PORT,
            'debug': cls.DEBUG,
            'model_path': cls.find_model_path(),
            'predictions_path': cls.PREDICTIONS_PATH,
            'api_version': cls.API_VERSION,
            'api_title': cls.API_TITLE
        }