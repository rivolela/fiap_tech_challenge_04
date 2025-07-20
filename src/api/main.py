#!/usr/bin/env python3
"""
API Server Runner
"""
import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)


from flask import Flask
app = Flask(__name__)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8081))
    print(f"🚀 Starting LSTM API on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)