# 🚀 Render Deployment Guide - FIAP LSTM API

Complete guide for deploying your LSTM stock prediction API on Render platform.

## 📋 Prerequisites

1. **Git Repository**: Your code must be in a GitHub repository
2. **Render Account**: Sign up at [render.com](https://render.com)
3. **Model Files**: Ensure `outputs/model_export/` contains trained model files

## 🛠️ Deployment Steps

### Step 1: Prepare Your Repository

```bash
# Ensure all files are committed
git add .
git commit -m "feat: Add Render deployment configuration"
git push origin main
```

### Step 2: Deploy on Render

#### Option A: Using Render Dashboard (Recommended)

1. **Create New Web Service**:
   - Go to [Render Dashboard](https://dashboard.render.com)
   - Click "New +" → "Web Service"
   - Connect your GitHub repository

2. **Configure Service**:
   ```
   Name: fiap-lstm-api
   Runtime: Python 3
   Build Command: pip install -r requirements.txt
   Start Command: python render_start.py
   ```

3. **Environment Variables**:
   ```
   PORT=10000
   DEBUG=false
   MODEL_PATH=./outputs/model_export/
   PREDICTIONS_PATH=./outputs/predictions.csv
   LOG_LEVEL=INFO
   PYTHONPATH=/opt/render/project/src
   ```

#### Option B: Using Blueprint File

1. **Deploy with render.yaml**:
   - Upload `render.yaml` to your repository
   - In Render Dashboard, select "Deploy from Blueprint"
   - Choose your repository and the `render.yaml` file

### Step 3: Health Check

After deployment, verify your API:

```bash
# Replace YOUR-APP-URL with your Render app URL
curl https://YOUR-APP-URL.onrender.com/health

# Expected response:
# {"status": "healthy", "timestamp": "...", "model_loaded": true}
```

## 🧪 Testing Your Deployed API

### Health Check
```bash
curl https://your-app.onrender.com/health
```

### Model Information
```bash
curl https://your-app.onrender.com/model/info
```

### Make Prediction
```bash
curl -X POST https://your-app.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "days": 30,
    "features": {
      "volume": 1000000,
      "high": 25.50,
      "low": 24.80,
      "close": 25.20
    }
  }'
```

## 📁 Required Files for Render

Your repository should contain:

```
├── render.yaml              # Render service configuration
├── render_start.py          # Render-specific entrypoint
├── Dockerfile.render        # Optional: Docker configuration
├── requirements.txt         # Python dependencies
├── src/
│   └── api/
│       ├── main.py         # Flask application
│       ├── models.py       # LSTM model service
│       ├── config.py       # Configuration
│       └── utils.py        # Utilities
├── outputs/
│   └── model_export/       # Trained model files
└── tests/
    └── test_api.py         # API tests
```

## ⚙️ Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 10000 | Server port (Render automatically sets this) |
| `DEBUG` | false | Flask debug mode |
| `MODEL_PATH` | ./outputs/model_export/ | Path to trained model files |
| `PREDICTIONS_PATH` | ./outputs/predictions.csv | Path to predictions file |
| `LOG_LEVEL` | INFO | Logging level |
| `PYTHONPATH` | /opt/render/project/src | Python module search path |

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors**:
   - Check PYTHONPATH environment variable
   - Ensure all dependencies in requirements.txt

2. **Model Loading Failed**:
   - Verify model files exist in `outputs/model_export/`
   - Check file permissions and paths

3. **Port Issues**:
   - Render automatically assigns PORT environment variable
   - Don't hardcode port numbers

4. **Memory Issues**:
   - Consider upgrading to Render's paid plan
   - Optimize model loading and caching

### Debugging Commands

```bash
# Check logs in Render dashboard or use CLI
render logs -s your-service-name

# Test locally before deploying
python render_start.py

# Validate API endpoints
python tests/test_api.py
```

## 📊 Monitoring

### Built-in Metrics
- Health check endpoint: `/health`
- Model info endpoint: `/model/info`
- Request logging and error tracking

### Render Dashboard
- CPU and memory usage
- Request metrics
- Error logs
- Deployment history

## 🔒 Security Considerations

1. **Environment Variables**: Store sensitive data as environment variables
2. **HTTPS**: Render provides automatic HTTPS
3. **API Keys**: Implement API key authentication if needed
4. **Rate Limiting**: Consider implementing rate limiting for production

## 💰 Cost Optimization

### Free Tier Limitations
- Service sleeps after 15 minutes of inactivity
- 750 hours/month limit
- Limited CPU and memory

### Upgrading Options
- **Starter ($7/month)**: No sleep, dedicated resources
- **Standard ($25/month)**: More CPU, memory, and features

## 🚀 Production Checklist

- [ ] All model files committed to repository
- [ ] Environment variables configured
- [ ] Health check endpoint working
- [ ] API endpoints tested
- [ ] Error handling implemented
- [ ] Logging configured
- [ ] Monitoring setup
- [ ] Documentation updated

## 📞 Support

- **Render Documentation**: [docs.render.com](https://docs.render.com)
- **API Issues**: Check application logs in Render dashboard
- **Model Questions**: Review `MODEL_EXPORT_GUIDE.md`

---

🎯 **Ready to Deploy!** Your FIAP LSTM API is now configured for Render deployment.

Follow the steps above to get your stock prediction API running in the cloud! 🌟
