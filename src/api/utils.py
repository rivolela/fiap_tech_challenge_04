"""API Utility Functions"""
import os
import pandas as pd
import logging
from flask import jsonify
from datetime import datetime
from typing import Any, Dict, Optional, Tuple
from .config import APIConfig

logger = logging.getLogger(__name__)

def create_success_response(data: Any = None, message: str = "Success", status_code: int = 200) -> Tuple[Dict, int]:
    """Create standardized success response"""
    response = {
        'status': 'success',
        'message': message,
        'timestamp': datetime.now().isoformat()
    }
    
    if data is not None:
        response['data'] = data
    
    return jsonify(response), status_code

def create_error_response(message: str, status_code: int = 400, error_type: str = None) -> Tuple[Dict, int]:
    """Create standardized error response"""
    response = {
        'status': 'error',
        'message': message,
        'timestamp': datetime.now().isoformat()
    }
    
    if error_type:
        response['error_type'] = error_type
    
    return jsonify(response), status_code

def load_predictions_csv(file_path: str = None) -> Optional[pd.DataFrame]:
    """Load predictions from CSV file"""
    file_path = file_path or APIConfig.PREDICTIONS_PATH
    
    if not os.path.exists(file_path):
        logger.warning(f"Predictions file not found: {file_path}")
        return None
    
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} prediction records from {file_path}")
        return df
    except Exception as e:
        logger.error(f"Error reading predictions CSV: {e}")
        return None

def parse_prediction_request(raw_data: Any) -> Tuple[Optional[list], Optional[int], Optional[str]]:
    """Parse and validate prediction request data"""
    try:
        # Handle different input formats
        if isinstance(raw_data, list):
            # Direct array format
            historical_data = raw_data
            forecast_horizon = 6  # Default
        elif isinstance(raw_data, dict):
            if 'data' in raw_data:
                # Structured format: {"data": [...], "forecast_horizon": 6}
                historical_data = raw_data.get('data', [])
                forecast_horizon = raw_data.get('forecast_horizon', 6)
            else:
                # Single record with optional forecast_horizon
                forecast_horizon = raw_data.get('forecast_horizon', 6)
                # Remove forecast_horizon from data if present
                data_copy = raw_data.copy()
                data_copy.pop('forecast_horizon', None)
                historical_data = [data_copy]
        else:
            return None, None, "Invalid data format. Expected array or object."
        
        # Validate forecast horizon
        if not isinstance(forecast_horizon, int) or forecast_horizon < 1 or forecast_horizon > 12:
            return None, None, "Forecast horizon must be integer between 1 and 12"
        
        # Validate data presence
        if not historical_data:
            return None, None, "No historical data provided"
        
        return historical_data, forecast_horizon, None
        
    except Exception as e:
        return None, None, f"Error parsing request: {str(e)}"

def get_api_info(model_service) -> Dict[str, Any]:
    """Get comprehensive API information"""
    return {
        "service": APIConfig.API_TITLE,
        "version": APIConfig.API_VERSION,
        "description": APIConfig.API_DESCRIPTION,
        "status": "running",
        "model_loaded": model_service.is_loaded if model_service else False,
        "endpoints": {
            "/": "GET - API information",
            "/health": "GET - Health check",
            "/predict": "POST - Make stock price predictions",
            "/model/info": "GET - Model configuration details",
            "/predictions": "GET - Get saved predictions"
        },
        "usage": {
            "predict_endpoint": "/predict",
            "method": "POST",
            "content_type": "application/json",
            "required_features": model_service.required_features if model_service else [],
            "example_request": {
                "forecast_horizon": 6,
                "data": [
                    {
                        "preco_medio_close": 29.86,
                        "lag_1_mes_preco_medio_close": 31.98,
                        "# ... other features": "..."
                    }
                ]
            }
        },
        "configuration": APIConfig.get_config_dict()
    }