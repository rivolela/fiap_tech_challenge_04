#!/usr/bin/env python3
"""
Test client for LSTM Stock Prediction API
"""

import requests
import json
import pandas as pd
from datetime import datetime

class LSTMAPIClient:
    """Client to interact with LSTM API"""
    
    def __init__(self, base_url="http://192.168.1.200:8081"):
        self.base_url = base_url
    
    def test_connection(self):
        """Test if API is running"""
        try:
            response = requests.get(f"{self.base_url}/")
            return response.status_code == 200
        except:
            return False
    
    def get_api_info(self):
        """Get API information"""
        response = requests.get(f"{self.base_url}/")
        return response.json()
    
    def health_check(self):
        """Check API health"""
        response = requests.get(f"{self.base_url}/health")
        return response.json()
    
    def get_model_info(self):
        """Get model information"""
        response = requests.get(f"{self.base_url}/model/info")
        return response.json()
    
    def predict(self, data, forecast_horizon=6):
        """Make prediction"""
        payload = {
            "forecast_horizon": forecast_horizon,
            "data": data
        }
        
        response = requests.post(
            f"{self.base_url}/predict",
            json=payload,
            headers={'Content-Type': 'application/json'}
        )
        
        return response.json(), response.status_code

def test_api():
    """Test the LSTM API"""
    print("🧪 Testing LSTM Stock Prediction API")
    print("=" * 50)
    
    client = LSTMAPIClient()
    
    # Test connection
    print("1. Testing connection...")
    if client.test_connection():
        print("   ✅ API is running")
    else:
        print("   ❌ API is not accessible")
        print("   💡 Make sure to run: python api_server.py")
        return
    
    # Get API info
    print("\n2. Getting API info...")
    try:
        info = client.get_api_info()
        print(f"   ✅ Service: {info.get('service', 'Unknown')}")
        print(f"   ✅ Version: {info.get('version', 'Unknown')}")
        print(f"   ✅ Model loaded: {info.get('model_loaded', False)}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Health check
    print("\n3. Health check...")
    try:
        health = client.health_check()
        print(f"   ✅ Status: {health.get('status', 'unknown')}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Get model info
    print("\n4. Getting model info...")
    try:
        model_info = client.get_model_info()
        if 'model_info' in model_info:
            features = model_info['model_info'].get('feature_columns', [])
            print(f"   ✅ Required features: {len(features)}")
            print(f"   ✅ Features: {features[:3]}..." if len(features) > 3 else f"   ✅ Features: {features}")
        else:
            print(f"   ⚠️ Model info: {model_info}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test prediction with sample data
    print("\n5. Testing prediction...")
    try:
        # Create sample data (you'd replace this with real data)
        sample_data = []
        
        # Try to load real data from the project
        try:
            import glob
            data_path = "./data/transformed/"
            parquet_files = glob.glob(f"{data_path}*.parquet")
            
            if parquet_files:
                df = pd.read_parquet(parquet_files[0])
                # Get last 30 rows and convert to dict format
                sample_data = df.tail(30).to_dict('records')
                print(f"   📊 Using real data: {len(sample_data)} records")
            else:
                print("   ⚠️ No real data found, using mock data")
                # Mock data structure (replace with actual feature names)
                sample_data = [
                    {
                        "preco_medio_close": 25.5,
                        "lag_1_mes_preco_medio_close": 25.2,
                        "lag_2_mes_preco_medio_close": 24.8,
                        "media_movel_6_meses_preco_medio_close": 25.0
                        # Add other required features...
                    }
                    for _ in range(30)  # Create 30 mock records
                ]
        except Exception as e:
            print(f"   ⚠️ Could not load real data: {e}")
            sample_data = []
        
        if sample_data:
            result, status_code = client.predict(sample_data, forecast_horizon=6)
            
            if status_code == 200:
                predictions = result.get('predictions', [])
                print(f"   ✅ Prediction successful!")
                print(f"   ✅ Forecast horizon: {result.get('forecast_horizon', 'N/A')}")
                print(f"   ✅ Predictions: {predictions}")
                print(f"   📝 Note: {result.get('note', '')}")
            else:
                print(f"   ❌ Prediction failed: {result}")
        else:
            print("   ⚠️ No sample data available for prediction test")
            
    except Exception as e:
        print(f"   ❌ Prediction test error: {e}")
    
    print("\n" + "=" * 50)
    print("✅ API testing completed!")

def show_usage():
    """Show API usage examples"""
    print("\n🔧 API Usage Examples")
    print("=" * 30)
    
    print("\n1. Using curl:")
    print("```bash")
    print("# Health check")
    print("curl http://localhost:5000/health")
    print("")
    print("# Get model info")
    print("curl http://localhost:5000/model/info")
    print("")
    print("# Make prediction")
    print("curl -X POST http://localhost:5000/predict \\")
    print("  -H 'Content-Type: application/json' \\")
    print("  -d '{")
    print('    "forecast_horizon": 6,')
    print('    "data": [')
    print('      {"feature1": 25.5, "feature2": 24.8, ...},')
    print('      {"feature1": 25.2, "feature2": 24.9, ...}')
    print('    ]')
    print("  }'")
    print("```")
    
    print("\n2. Using Python requests:")
    print("```python")
    print("import requests")
    print("")
    print("data = [")
    print("  {'preco_medio_close': 25.5, 'lag_1_mes_preco_medio_close': 25.2},")
    print("  # ... more historical data")
    print("]")
    print("")
    print("response = requests.post('http://localhost:5000/predict', json={")
    print("  'forecast_horizon': 6,")
    print("  'data': data")
    print("})")
    print("")
    print("result = response.json()")
    print("predictions = result['predictions']")
    print("```")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--usage':
        show_usage()
    else:
        test_api()
        show_usage()
