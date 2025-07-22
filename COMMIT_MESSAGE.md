feat: Add Render platform deployment configuration for FIAP LSTM API

🚀 Complete Render deployment setup with production-ready configurations

## New Files Added:
- `render.yaml` - Render service blueprint with environment variables
- `render_start.py` - Platform-optimized entrypoint script  
- `Dockerfile.render` - Multi-stage Docker build for Render platform
- `RENDER_DEPLOYMENT_GUIDE.md` - Comprehensive deployment documentation
- `RENDER_CHECKLIST.md` - Step-by-step deployment validation guide

## Key Features:
✅ Production Flask API with modular architecture
✅ Environment-based configuration management
✅ Health monitoring endpoints (/health, /model/info)
✅ Docker containerization with security best practices
✅ Automatic HTTPS and scaling via Render platform
✅ Comprehensive error handling and logging
✅ Port 8081 default (avoiding macOS AirPlay conflicts)

## Environment Configuration:
- PORT: 10000 (Render auto-assigned)
- PYTHONPATH: /opt/render/project/src
- MODEL_PATH: ./outputs/model_export/
- Production-ready logging and debugging disabled

## Updated Documentation:
- `HOW_TO_RUN.md` - Added Render deployment section
- Complete troubleshooting guides and API testing examples
- Production deployment checklist for team reference

## API Endpoints Ready:
- GET /health - Service health check
- GET /model/info - LSTM model information  
- POST /predict - Stock price predictions for BBAS3

Ready for cloud deployment with scalable infrastructure! 🌟

Co-authored-by: GitHub Copilot <copilot@github.com>
