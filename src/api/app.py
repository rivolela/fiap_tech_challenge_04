"""Main Flask Application with Monitoring"""
import os
import sys
import logging
from flask import Flask, request
from src.api.config import APIConfig
from src.api.models import LSTMModelService
from src.api.utils import (
    create_success_response, 
    create_error_response, 
    load_predictions_csv,
    parse_prediction_request,
    get_api_info
)
from .monitoring_routes import monitoring_bp, record_prediction, record_prediction_for_drift

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

# Import monitoring components
try:
    from ..monitoring.middleware import MonitoringMiddleware, get_monitoring_blueprint, monitor_endpoint
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False
    def monitor_endpoint(name=None):
        def decorator(func):
            return func
        return decorator

# Setup logging
APIConfig.setup_logging()
logger = logging.getLogger(__name__)

def create_app() -> Flask:
    """Create and configure Flask application with monitoring"""
    app = Flask(__name__)
    
    # Initialize monitoring middleware
    if MONITORING_AVAILABLE:
        monitoring = MonitoringMiddleware(app)
        logger.info("Sistema de monitoramento ativado")
    else:
        logger.warning("Sistema de monitoramento não disponível")
    
    # Register monitoring blueprint
    app.register_blueprint(monitoring_bp)
    
    # Initialize model service
    model_service = LSTMModelService()
    
    @app.route('/', methods=['GET'])
    @monitor_endpoint('home')
    def home():
        """API information endpoint"""
        try:
            info = get_api_info(model_service)
            return create_success_response(info, "API information retrieved")
        except Exception as e:
            logger.error(f"Error in home endpoint: {e}")
            return create_error_response(f"Failed to get API info: {str(e)}", 500)

    @app.route('/health', methods=['GET'])
    @monitor_endpoint('health')
    def health():
        """Health check endpoint"""
        try:
            health_data = {
                "status": "healthy",
                "model_loaded": model_service.is_loaded,
                "service": APIConfig.API_TITLE,
                "version": APIConfig.API_VERSION
            }
            return create_success_response(health_data, "API is healthy")
        except Exception as e:
            logger.error(f"Error in health endpoint: {e}")
            return create_error_response(f"Health check failed: {str(e)}", 500)

    @app.route('/predict', methods=['POST'])
    @monitor_endpoint('predict')
    def predict():
        """Make LSTM predictions"""
        import time
        start_time = time.time()
        
        try:
            # Load model if not loaded
            if not model_service.is_loaded:
                logger.info("Loading model on first prediction request...")
                if not model_service.load_model():
                    return create_error_response(
                        "Model could not be loaded", 
                        503, 
                        "model_unavailable"
                    )
            
            # Validate request format
            if not request.is_json:
                return create_error_response(
                    "Content-Type must be application/json", 
                    400,
                    "invalid_content_type"
                )
            
            raw_data = request.get_json()
            
            # Parse request data
            historical_data, forecast_horizon, error_msg = parse_prediction_request(raw_data)
            if error_msg:
                return create_error_response(error_msg, 400, "invalid_request_format")
            
            # Check if parsing was successful
            if historical_data is None or forecast_horizon is None:
                return create_error_response("Invalid request data format", 400, "invalid_data")
            
            # Validate input data
            is_valid, validation_msg = model_service.validate_input_data(historical_data)
            if not is_valid:
                return create_error_response(validation_msg, 400, "validation_error")
            
            logger.info(f"Processing prediction: {len(historical_data)} records, horizon: {forecast_horizon}")
            
            # Make prediction
            predictions = model_service.predict(historical_data, forecast_horizon)
            
            # Calculate processing time
            processing_time_ms = (time.time() - start_time) * 1000
            
            logger.info(f"Prediction successful: {len(predictions)} predictions generated in {processing_time_ms:.2f}ms")
            
            # Record prediction for monitoring
            try:
                # Record basic prediction metrics
                record_prediction(processing_time_ms)
                
                # Record for drift monitoring - use the last historical record and first prediction
                if historical_data and predictions:
                    record_prediction_for_drift(
                        historical_data[-1], 
                        predictions[0] if isinstance(predictions, list) else predictions
                    )
            except Exception as monitoring_error:
                logger.warning(f"Failed to record prediction metrics: {monitoring_error}")
            
            # Get model configuration safely
            sequence_length = 24  # default
            features_used = 0
            if hasattr(model_service, 'config') and model_service.config:
                sequence_length = model_service.config.get('sequence_length', 24)
            if hasattr(model_service, 'feature_columns') and model_service.feature_columns:
                features_used = len(model_service.feature_columns)
            
            prediction_data = {
                'predictions': predictions,
                'forecast_horizon': forecast_horizon,
                'input_records': len(historical_data),
                'processing_time_ms': round(processing_time_ms, 2),
                'model_info': {
                    'sequence_length': sequence_length,
                    'features_used': features_used
                }
            }
            
            return create_success_response(
                prediction_data, 
                "Predictions generated successfully"
            )
            
        except RuntimeError as e:
            logger.error(f"Runtime error in predict: {e}")
            return create_error_response(str(e), 500, "prediction_error")
        except Exception as e:
            logger.error(f"Unexpected error in predict: {e}")
            return create_error_response(
                "Internal server error during prediction", 
                500, 
                "server_error"
            )

    @app.route('/model/info', methods=['GET'])
    def model_info():
        """Get detailed model information"""
        try:
            # Try to load model if not loaded
            if not model_service.is_loaded:
                logger.info("Loading model for info request...")
                model_service.load_model()
            
            if not model_service.is_loaded:
                return create_error_response(
                    "Model not available", 
                    503, 
                    "model_unavailable"
                )
            
            info = model_service.get_model_info()
            return create_success_response(info, "Model information retrieved")
            
        except Exception as e:
            logger.error(f"Error in model info endpoint: {e}")
            return create_error_response(
                f"Failed to get model info: {str(e)}", 
                500, 
                "server_error"
            )

    @app.route('/predictions', methods=['GET'])
    def get_saved_predictions():
        """Get saved predictions from CSV file"""
        try:
            df = load_predictions_csv()
            
            if df is None:
                return create_error_response(
                    "No predictions file found", 
                    404, 
                    "file_not_found"
                )
            
            # Convert to records and add metadata
            predictions_data = {
                'predictions': df.to_dict(orient='records'),
                'count': len(df),
                'columns': df.columns.tolist(),
                'file_path': APIConfig.PREDICTIONS_PATH
            }
            
            return create_success_response(
                predictions_data,
                f"Retrieved {len(df)} prediction records"
            )
            
        except Exception as e:
            logger.error(f"Error in predictions endpoint: {e}")
            return create_error_response(
                f"Failed to load predictions: {str(e)}", 
                500,
                "file_read_error"
            )

    @app.errorhandler(404)
    def not_found(error):
        """Handle 404 errors"""
        return create_error_response(
            "Endpoint not found. Check API documentation at GET /",
            404,
            "endpoint_not_found"
        )

    @app.errorhandler(500)
    def internal_error(error):
        """Handle 500 errors"""
        logger.error(f"Internal server error: {error}")
        return create_error_response(
            "Internal server error. Check server logs.",
            500,
            "internal_server_error"
        )
    
    # Try to load model on startup
    try:
        logger.info("Attempting to load model on startup...")
        if model_service.load_model():
            logger.info("✅ Model loaded successfully on startup")
        else:
            logger.warning("⚠️ Model loading failed on startup - will retry on first request")
    except Exception as e:
        logger.warning(f"⚠️ Could not load model on startup: {e}")
        logger.info("Model will be loaded on first request")
    
    return app

def run_app():
    """Run the Flask application"""
    app = create_app()
    
    print("🚀 Starting FIAP Tech Challenge 04 - LSTM API")
    print("=" * 60)
    print(f"🌐 Server: http://{APIConfig.HOST}:{APIConfig.PORT}")
    print(f"🔧 Debug: {APIConfig.DEBUG}")
    print(f"📊 Model: {APIConfig.MODEL_PATH}")
    print(f"📄 Logs: {APIConfig.LOG_FILE}")
    print("📡 Available endpoints:")
    print("  GET  /           - API information")
    print("  GET  /health     - Health check")  
    print("  POST /predict    - Make predictions")
    print("  GET  /model/info - Model details")
    print("  GET  /predictions - Get saved predictions")
    if MONITORING_AVAILABLE:
        print("  GET  /monitoring - Monitoring dashboard")
    else:
        print("  Monitoring not available")
    print("📂 Predictions file: ", APIConfig.PREDICTIONS_PATH)
    print("=" * 60)
    
    try:
        app.run(
            host=APIConfig.HOST,
            port=APIConfig.PORT,
            debug=APIConfig.DEBUG,
            threaded=True
        )
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"❌ Port {APIConfig.PORT} is already in use.")
            print(f"💡 Try: PORT=8081 python -m src.api.app")
        raise

if __name__ == '__main__':
    run_app()