# 🚀 LSTM Model Production API Guide

## 📋 Overview

This guide shows you how to productize your LSTM model using a Flask API that can serve predictions over HTTP.

## 🛠️ Setup & Installation

### 1. Install Dependencies
```bash
pip install flask requests pandas numpy torch scikit-learn
```

### 2. Ensure Model is Exported
Make sure you have a trained model in `outputs/model_export/`:
```bash
ls outputs/model_export/
# Should show: complete_model.pth, scaler.pkl, feature_columns.json, model_config.json
```

## 🚀 Running the API

### Start the API Server
```bash
python api_server.py
```

The API will start on `http://localhost:5000`

### Different Port
```bash
PORT=8081 python api_server.py
```

## 📡 API Endpoints

### 1. Home / API Info
**GET** `http://localhost:5000/`
```bash
curl http://localhost:5000/
```

### 2. Health Check
**GET** `http://localhost:5000/health`
```bash
curl http://localhost:5000/health
```

### 3. Model Information
**GET** `http://localhost:5000/model/info`
```bash
curl http://localhost:5000/model/info
```

### 4. Make Predictions (Main Endpoint)
**POST** `http://localhost:5000/predict`

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "forecast_horizon": 6,
    "data": [
      {
        "preco_medio_close": 25.5,
        "lag_1_mes_preco_medio_close": 25.2,
        "lag_2_mes_preco_medio_close": 24.8,
        "lag_3_mes_preco_medio_close": 24.5,
        "lag_4_mes_preco_medio_close": 24.2,
        "lag_5_mes_preco_medio_close": 24.0,
        "lag_6_mes_preco_medio_close": 23.8,
        "media_movel_6_meses_preco_medio_close": 24.5,
        "desvio_padrao_movel_6_meses_preco_medio_close": 0.8,
        "valor_minimo_6_meses_preco_medio_close": 23.5,
        "valor_maximo_6_meses_preco_medio_close": 25.8
      }
    ]
  }'
```

## 🧪 Testing the API

### Automated Test
```bash
python test_api_client.py
```

### Manual Test
```bash
# 1. Start API server (in one terminal)
python api_server.py

# 2. Test (in another terminal)
python test_api_client.py
```

## 📊 Request/Response Format

### Request Format
```json
{
  "forecast_horizon": 6,
  "data": [
    {
      "preco_medio_close": 25.5,
      "lag_1_mes_preco_medio_close": 25.2,
      "lag_2_mes_preco_medio_close": 24.8,
      "lag_3_mes_preco_medio_close": 24.5,
      "lag_4_mes_preco_medio_close": 24.2,
      "lag_5_mes_preco_medio_close": 24.0,
      "lag_6_mes_preco_medio_close": 23.8,
      "media_movel_6_meses_preco_medio_close": 24.5,
      "desvio_padrao_movel_6_meses_preco_medio_close": 0.8,
      "valor_minimo_6_meses_preco_medio_close": 23.5,
      "valor_maximo_6_meses_preco_medio_close": 25.8
    }
  ]
}
```

### Response Format
```json
{
  "status": "success",
  "predictions": [0.1234, 0.1567, 0.1890, 0.2123, 0.2456, 0.2789],
  "forecast_horizon": 6,
  "input_records": 30,
  "timestamp": "2025-07-20T12:34:56.789",
  "note": "Predictions are in scaled format. Use inverse transform for actual values."
}
```

## 🐍 Python Client Example

```python
import requests

# API client
class LSTMClient:
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
    
    def predict(self, data, forecast_horizon=6):
        response = requests.post(
            f"{self.base_url}/predict",
            json={
                "forecast_horizon": forecast_horizon,
                "data": data
            }
        )
        return response.json()

# Usage
client = LSTMClient()
result = client.predict(your_data, forecast_horizon=6)
predictions = result['predictions']
```

## 🔧 Integration with Existing Code

### Use with production_inference.py
```python
# In production_inference.py, you can call the API instead of direct inference:
import requests

def call_api_prediction(data):
    response = requests.post('http://localhost:5000/predict', 
                           json={'data': data, 'forecast_horizon': 6})
    return response.json()
```

## 🚀 Production Deployment

### 1. Using Gunicorn (Recommended)
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 api_server:app
```

### 2. Using Docker
```dockerfile
FROM python:3.11
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 5000
CMD ["python", "api_server.py"]
```

### 3. Environment Variables
```bash
export PORT=8000
export DEBUG=false
python api_server.py
```

## 📈 Monitoring & Logging

- Logs are written to `lstm_api.log`
- Monitor health at `/health` endpoint
- Check model status at `/model/info`

## 🛠️ Troubleshooting

### Common Issues

1. **Port already in use:**
   ```bash
   PORT=8081 python api_server.py
   ```

2. **Model not found:**
   - Ensure model is exported: `python src/ml/main.py`
   - Check `outputs/model_export/` directory

3. **Missing features:**
   - Check required features at `/model/info`
   - Ensure your data has all required columns

4. **Memory issues:**
   - Use smaller batch sizes
   - Consider model quantization

## ✅ Complete Workflow

1. **Train Model:** `python src/ml/main.py`
2. **Start API:** `python api_server.py`
3. **Test API:** `python test_api_client.py`
4. **Make Predictions:** Use curl/Python/any HTTP client
5. **Monitor:** Check logs and `/health` endpoint

Your LSTM model is now production-ready! 🎉
