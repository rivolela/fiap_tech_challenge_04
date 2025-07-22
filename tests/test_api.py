#!/usr/bin/env python3
"""
Simple test for the LSTM API
"""

import requests
import json

def test_api(base_url="http://localhost:8081"):
    """Test the API endpoints"""
    print("🧪 Testing LSTM API")
    print("=" * 30)
    
    # Test home endpoint
    try:
        response = requests.get(f"{base_url}/")
        print(f"✅ Home endpoint: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Service: {data.get('service', 'Unknown')}")
    except Exception as e:
        print(f"❌ Home endpoint failed: {e}")
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health")
        print(f"✅ Health endpoint: {response.status_code}")
    except Exception as e:
        print(f"❌ Health endpoint failed: {e}")
    
    # Test model info
    try:
        response = requests.get(f"{base_url}/model/info")
        print(f"✅ Model info endpoint: {response.status_code}")
    except Exception as e:
        print(f"❌ Model info endpoint failed: {e}")

if __name__ == "__main__":
    test_api()
