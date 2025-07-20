#!/usr/bin/env python3
"""
Flask API for LSTM Stock Price Prediction
"""

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import sys
import os
import pickle
import json
import torch
import logging
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('lstm_api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class ModelService:
    """Service to handle model loading and predictions"""
    
    def __init__(self, model_path="./outputs/model_export/"):
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.config = None
        self.model_loaded = False
    
    def load_model(self):
        """Load all model components"""
        try:
            # Load model
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
            
            # Load config
            with open(os.path.join(self.model_path, 'model_config.json'), 'r') as f:
                self.config = json.load(f)
            
            self.model_loaded = True
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def predict(self, data):
        """Make predictions"""
        if not self.model_loaded:
            raise RuntimeError("Model not loaded")
        
        try:
            # Preprocess data
            processed_data = self._preprocess_data(data)
            
            # Make prediction
            with torch.no_grad():
                prediction = self.model(processed_data)
            
            return prediction.numpy().tolist()
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise
    
    def _preprocess_data(self, data):
        """Preprocess data for prediction"""
        # Select required columns
        data_filtered = data[self.feature_columns]
        
        # Scale data
        data_scaled = self.scaler.transform(data_filtered)
        
        # Create sequence (last 24 time steps)
        sequence = data_scaled[-24:]
        
        # Convert to tensor and add batch dimension
        return torch.FloatTensor(sequence).unsqueeze(0)

# Initialize model service
model_service = ModelService()

@app.route('/', methods=['GET'])
def home():
    """Home endpoint"""
    return jsonify({
        'service': 'FIAP Tech Challenge 04 - LSTM Stock Prediction API',
        'version': '1.0.0',
        'status': 'running',
        'model_loaded': model_service.model_loaded,
        'endpoints': {
            '/': 'Home',
            '/health': 'Health check',
            '/predict': 'Make predictions',
            '/model/info': 'Model information'
        }
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model_service.model_loaded
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    try:
        if not model_service.model_loaded:
            try:
                model_service.load_model()
            except Exception as e:
                return jsonify({
                    'error': f'Model not available: {str(e)}'
                }), 503
        
        # Get data from request
        if request.is_json:
            data = request.json
        else:
            return jsonify({'error': 'Content-Type must be application/json'}), 400
        
        # Convert to DataFrame if needed
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            return jsonify({'error': 'Invalid data format'}), 400
        
        # Make prediction
        predictions = model_service.predict(df)
        
        return jsonify({
            'predictions': predictions,
            'timestamp': datetime.now().isoformat(),
            'input_shape': df.shape
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/model/info', methods=['GET'])
def model_info():
    """Model information endpoint"""
    if not model_service.model_loaded:
        return jsonify({'error': 'Model not loaded'}), 503
    
    return jsonify({
        'config': model_service.config,
        'feature_columns': model_service.feature_columns,
        'model_path': model_service.model_path,
        'loaded': model_service.model_loaded
    })

if __name__ == '__main__':
    # Try to load model on startup
    try:
        logger.info("Loading model on startup...")
        model_service.load_model()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.warning(f"Could not load model on startup: {e}")
        logger.warning("Model will be loaded on first request")
    
    # Run Flask app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
