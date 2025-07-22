#!/usr/bin/env python3
"""
Render Platform Entrypoint for FIAP LSTM API

This script initializes the Flask application with Render-specific configurations.
It handles environment variables, model loading, and starts the server on the correct port.
"""

import os
import sys
import logging
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Configure logging
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def main():
    """Main entrypoint for Render deployment."""
    try:
        logger.info("🚀 Starting FIAP LSTM API on Render...")
        
        # Import Flask app
        from api.main import create_app
        
        # Create Flask app
        app = create_app()
        
        # Get port from environment (Render uses PORT env var)
        port = int(os.getenv('PORT', 10000))
        
        # Render deployment info
        logger.info(f"📊 Model Path: {os.getenv('MODEL_PATH', './outputs/model_export/')}")
        logger.info(f"🌐 Starting server on port {port}")
        logger.info(f"🏥 Health check available at: /health")
        logger.info(f"📈 Prediction endpoint available at: /predict")
        
        # Run the app
        app.run(
            host='0.0.0.0',
            port=port,
            debug=False,  # Always False in production
            threaded=True
        )
        
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        logger.error("Make sure all dependencies are installed and PYTHONPATH is correct")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Failed to start server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
