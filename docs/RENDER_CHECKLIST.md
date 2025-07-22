# ✅ Render Deployment Checklist - FIAP LSTM API

## 📋 Pre-Deployment Preparation

### Repository Setup
- [ ] All code committed to GitHub repository
- [ ] `main` branch is up to date
- [ ] Repository is public or Render has access

### Required Files Present
- [ ] `render.yaml` - Service configuration
- [ ] `render_start.py` - Platform entrypoint
- [ ] `Dockerfile.render` - Docker configuration  
- [ ] `requirements.txt` - Python dependencies
- [ ] `src/api/main.py` - Flask application
- [ ] `src/api/models.py` - LSTM model service
- [ ] `outputs/model_export/` - Trained model files

### Model Files Verification
```bash
# Check if model files exist
ls -la outputs/model_export/

# Expected files:
# - model.pth (PyTorch model)
# - scaler_features.pkl (feature scaler)
# - scaler_target.pkl (target scaler)
# - training_config.json (model metadata)
```

## 🚀 Render Platform Setup

### Account & Service Creation
- [ ] Render account created at [render.com](https://render.com)
- [ ] GitHub account connected to Render
- [ ] New Web Service created from repository

### Service Configuration
- [ ] **Name**: `fiap-lstm-api`
- [ ] **Runtime**: Python 3
- [ ] **Build Command**: `pip install -r requirements.txt`
- [ ] **Start Command**: `python render_start.py`
- [ ] **Branch**: `main`
- [ ] **Plan**: Free tier (or upgrade as needed)

### Environment Variables
```bash
# Required environment variables in Render dashboard:
PORT=10000                              # Auto-set by Render
DEBUG=false                             # Production mode
MODEL_PATH=./outputs/model_export/      # Model files location
PREDICTIONS_PATH=./outputs/predictions.csv  # Predictions file
LOG_LEVEL=INFO                          # Logging level
PYTHONPATH=/opt/render/project/src      # Python module path
```

## 🧪 Post-Deployment Testing

### Health Check
```bash
# Replace YOUR-APP-URL with your actual Render URL
curl https://YOUR-APP-URL.onrender.com/health

# Expected Response:
# {
#   "status": "healthy",
#   "timestamp": "2024-01-XX T XX:XX:XX",
#   "model_loaded": true,
#   "uptime": "XX seconds"
# }
```

### Model Information
```bash
curl https://YOUR-APP-URL.onrender.com/model/info

# Expected Response:
# {
#   "model_type": "LSTM",
#   "input_features": 4,
#   "sequence_length": 60,
#   "prediction_horizon": "6 months"
# }
```

### Prediction Test
```bash
curl -X POST https://YOUR-APP-URL.onrender.com/predict \
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

# Expected Response:
# {
#   "predictions": [24.85, 24.92, 25.01, ...],
#   "dates": ["2024-02-01", "2024-02-02", ...],
#   "confidence_interval": {...},
#   "model_version": "v1.0",
#   "generated_at": "2024-01-XX T XX:XX:XX"
# }
```

## 📊 Monitoring & Verification

### Render Dashboard Checks
- [ ] Service status shows "Live"
- [ ] Build logs show successful deployment
- [ ] No error messages in application logs
- [ ] Health check endpoint responding
- [ ] CPU and memory usage within limits

### Application Logs Review
```bash
# Check for these log messages in Render dashboard:
✅ "🚀 Starting FIAP LSTM API on Render..."
✅ "📊 Model loaded successfully"  
✅ "🌐 Starting server on port 10000"
✅ "🏥 Health check available at: /health"
```

### Performance Testing
- [ ] Response time < 5 seconds for predictions
- [ ] Health check responds in < 1 second
- [ ] No memory leaks after multiple requests
- [ ] Model predictions are consistent

## 🔧 Troubleshooting Checklist

### Common Issues & Solutions

**Import Errors:**
- [ ] Check PYTHONPATH environment variable
- [ ] Verify all dependencies in requirements.txt
- [ ] Ensure proper module structure in src/

**Model Loading Errors:**
- [ ] Confirm model files exist in outputs/model_export/
- [ ] Check file permissions and sizes
- [ ] Verify model format compatibility

**Connection Issues:**
- [ ] Check if service is in "Live" status
- [ ] Verify health check endpoint works
- [ ] Test with simple GET request first

**Performance Issues:**
- [ ] Monitor memory usage in dashboard
- [ ] Consider upgrading to paid plan
- [ ] Implement model caching if needed

## 🎯 Production Readiness

### Final Validation
- [ ] All endpoints tested and working
- [ ] Error handling working properly  
- [ ] Logging configured and visible
- [ ] Environment variables properly set
- [ ] Security considerations addressed
- [ ] Documentation updated

### Scaling Considerations
- [ ] Evaluate free tier limitations
- [ ] Plan for paid tier if needed
- [ ] Consider API rate limiting
- [ ] Monitor resource usage

## 📚 Documentation Updates

### Files to Update
- [ ] README.md with deployment URL
- [ ] API documentation with endpoints
- [ ] Update any hardcoded localhost URLs
- [ ] Add production environment notes

### Team Communication
- [ ] Share deployment URL with team
- [ ] Document API endpoints and usage
- [ ] Provide troubleshooting guide
- [ ] Schedule monitoring review

---

## 🎉 Deployment Complete!

✅ **Congratulations!** Your FIAP LSTM API is now live on Render!

**Your API is accessible at:** https://your-app-name.onrender.com

**Next Steps:**
1. Test all endpoints thoroughly
2. Monitor performance and logs
3. Share with team/stakeholders
4. Plan for production scaling if needed

---

**Support Resources:**
- 📖 [RENDER_DEPLOYMENT_GUIDE.md](./RENDER_DEPLOYMENT_GUIDE.md)
- 🔧 [HOW_TO_RUN.md](./HOW_TO_RUN.md)
- 📊 [Render Dashboard](https://dashboard.render.com)
