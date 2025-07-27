import os
from flask import send_file
import psutil
import time
import json
from datetime import datetime
from flask import Blueprint, jsonify, request, render_template_string, redirect

# Create Blueprint
monitoring_bp = Blueprint('monitoring', __name__, url_prefix='/monitoring')
# Endpoint para acessar o relatório HTML de drift
@monitoring_bp.route('/drift-report', methods=['GET'])
def get_drift_report():
    """Serve o último relatório HTML de drift gerado pelo Evidently"""
    # Caminho do diretório de relatórios
    reports_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "outputs", "drift_reports")
    if not os.path.exists(reports_dir):
        return {"status": "error", "message": "Nenhum relatório de drift encontrado."}, 404

    # Procurar o arquivo mais recente
    report_files = [f for f in os.listdir(reports_dir) if f.endswith('.html')]
    if not report_files:
        return {"status": "error", "message": "Nenhum relatório de drift encontrado."}, 404

    latest_report = max(report_files, key=lambda f: os.path.getmtime(os.path.join(reports_dir, f)))
    report_path = os.path.join(reports_dir, latest_report)
    return send_file(report_path, mimetype='text/html')

# In-memory storage for metrics
_metrics = {
    "predictions": [],
    "system_stats": {
        "cpu": [],
        "memory": []
    },
    "start_time": datetime.now().isoformat(),
    "total_predictions": 0
}

@monitoring_bp.route('/', methods=['GET'])
def monitoring_home():
    """Redirect to dashboard"""
    return redirect('/monitoring/dashboard')

@monitoring_bp.route('/health', methods=['GET'])
def health():
    """Return system health status"""
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory_percent = psutil.virtual_memory().percent
    
    # Define thresholds
    cpu_warning = 70
    cpu_critical = 90
    memory_warning = 80
    memory_critical = 90
    
    # Check status
    if cpu_percent >= cpu_critical or memory_percent >= memory_critical:
        status = "critical"
    elif cpu_percent >= cpu_warning or memory_percent >= memory_warning:
        status = "degraded"
    else:
        status = "healthy"
    
    return jsonify({
        "status": status,
        "timestamp": datetime.now().isoformat(),
        "checks": {
            "cpu": {
                "status": "healthy" if cpu_percent < cpu_warning else 
                         ("warning" if cpu_percent < cpu_critical else "critical"),
                "value": cpu_percent,
                "unit": "%"
            },
            "memory": {
                "status": "healthy" if memory_percent < memory_warning else 
                         ("warning" if memory_percent < memory_critical else "critical"),
                "value": memory_percent,
                "unit": "%"
            },
            "prediction_service": {
                "status": "healthy",
                "total_predictions": _metrics["total_predictions"]
            }
        }
    })

@monitoring_bp.route('/stats', methods=['GET'])
def stats():
    """Return performance statistics"""
    prediction_stats = {}
    if _metrics["predictions"]:
        durations = [p["duration_ms"] for p in _metrics["predictions"]]
        prediction_stats = {
            "avg_duration_ms": sum(durations) / len(durations),
            "min_duration_ms": min(durations),
            "max_duration_ms": max(durations),
            "p95_duration_ms": sorted(durations)[int(len(durations) * 0.95)] if len(durations) >= 20 else max(durations),
            "total_count": len(durations)
        }
    
    system_stats = {}
    if _metrics["system_stats"]["cpu"]:
        system_stats = {
            "avg_cpu_percent": sum(_metrics["system_stats"]["cpu"]) / len(_metrics["system_stats"]["cpu"]),
            "avg_memory_percent": sum(_metrics["system_stats"]["memory"]) / len(_metrics["system_stats"]["memory"]),
        }
    
    return jsonify({
        "timestamp": datetime.now().isoformat(),
        "total_predictions": _metrics["total_predictions"],
        "prediction_stats": prediction_stats,
        "system_stats": system_stats
    })

@monitoring_bp.route('/metrics', methods=['GET'])
def metrics():
    """Return Prometheus format metrics"""
    metrics_text = f"""# HELP lstm_api_predictions_total Total number of predictions made
# TYPE lstm_api_predictions_total counter
lstm_api_predictions_total {_metrics["total_predictions"]}
# HELP lstm_api_cpu_usage CPU usage percentage
# TYPE lstm_api_cpu_usage gauge
lstm_api_cpu_usage {psutil.cpu_percent()}
# HELP lstm_api_memory_usage Memory usage percentage
# TYPE lstm_api_memory_usage gauge
lstm_api_memory_usage {psutil.virtual_memory().percent}
"""
    return metrics_text, 200, {'Content-Type': 'text/plain'}

@monitoring_bp.route('/dashboard', methods=['GET'])
def dashboard():
    """Return interactive HTML dashboard with clickable endpoints"""
    html = """<!DOCTYPE html>
<html>
<head>
    <title>LSTM API Monitoring Dashboard</title>
    <meta http-equiv="refresh" content="10">
    <style>
        body { 
            font-family: 'Arial', sans-serif; 
            margin: 0; 
            padding: 20px; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { 
            background: white; 
            padding: 20px; 
            border-radius: 10px; 
            margin-bottom: 20px; 
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .card { 
            background: white; 
            padding: 20px; 
            margin: 10px 0; 
            border-radius: 10px; 
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .metric { 
            display: inline-block; 
            margin: 15px; 
            text-align: center; 
            padding: 10px;
            background: #f8f9fa;
            border-radius: 8px;
            min-width: 120px;
        }
        .value { 
            font-size: 28px; 
            font-weight: bold; 
            color: #333;
        }
        .endpoints-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }
        .endpoint-card {
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 15px;
            transition: all 0.3s ease;
        }
        .endpoint-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            border-color: #007bff;
        }
        .endpoint-link {
            text-decoration: none;
            color: #007bff;
            font-weight: bold;
            font-size: 16px;
            display: block;
            margin-bottom: 8px;
        }
        .endpoint-link:hover {
            color: #0056b3;
            text-decoration: underline;
        }
        .endpoint-method {
            display: inline-block;
            background: #28a745;
            color: white;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 12px;
            margin-right: 8px;
        }
        .endpoint-method.post { background: #007bff; }
        .endpoint-description {
            color: #666;
            font-size: 14px;
            margin-top: 5px;
        }
        .status-healthy { color: #28a745; }
        .status-degraded { color: #ffc107; }
        .status-critical { color: #dc3545; }
        .refresh-info {
            text-align: center;
            color: #666;
            font-size: 12px;
            margin-top: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 LSTM API Monitoring Dashboard</h1>
            <p>Real-time monitoring and performance metrics for your LSTM Stock Prediction API</p>
        </div>
        
        <div class="card">
            <h2>📊 System Health</h2>
            <div id="health">Loading...</div>
        </div>
        
        <div class="card">
            <h2>📈 Prediction Statistics</h2>
            <div id="predictions">Loading...</div>
        </div>
        
        <div class="card">
            <h2>🔗 API Endpoints</h2>
            <div class="endpoints-grid">
                <!-- Main API Endpoints -->
                <div class="endpoint-card">
                    <a href="/" target="_blank" class="endpoint-link">
                        <span class="endpoint-method">GET</span> /
                    </a>
                    <div class="endpoint-description">API information and documentation</div>
                </div>
                
                <div class="endpoint-card">
                    <a href="/health" target="_blank" class="endpoint-link">
                        <span class="endpoint-method">GET</span> /health
                    </a>
                    <div class="endpoint-description">Basic health check</div>
                </div>
                
                <div class="endpoint-card">
                    <a href="/model/info" target="_blank" class="endpoint-link">
                        <span class="endpoint-method">GET</span> /model/info
                    </a>
                    <div class="endpoint-description">Model configuration details</div>
                </div>
                
                <div class="endpoint-card">
                    <a href="/predictions" target="_blank" class="endpoint-link">
                        <span class="endpoint-method">GET</span> /predictions
                    </a>
                    <div class="endpoint-description">Get saved predictions</div>
                </div>
                
                <div class="endpoint-card">
                    <span class="endpoint-method post">POST</span> /predict
                    <div class="endpoint-description">Make stock price predictions (requires JSON payload)</div>
                </div>
                
                <!-- Monitoring Endpoints -->
                <div class="endpoint-card">
                    <a href="/monitoring/health" target="_blank" class="endpoint-link">
                        <span class="endpoint-method">GET</span> /monitoring/health
                    </a>
                    <div class="endpoint-description">Detailed system health status</div>
                </div>
                
                <div class="endpoint-card">
                    <a href="/monitoring/stats" target="_blank" class="endpoint-link">
                        <span class="endpoint-method">GET</span> /monitoring/stats
                    </a>
                    <div class="endpoint-description">Performance statistics JSON</div>
                </div>
                
                <div class="endpoint-card">
                    <a href="/monitoring/metrics" target="_blank" class="endpoint-link">
                        <span class="endpoint-method">GET</span> /monitoring/metrics
                    </a>
                    <div class="endpoint-description">Prometheus metrics format</div>
                </div>
                
                <div class="endpoint-card">
                    <a href="/monitoring/drift-report" target="_blank" class="endpoint-link">
                        <span class="endpoint-method">GET</span> /monitoring/drift-report
                    </a>
                    <div class="endpoint-description">Data drift analysis report</div>
                </div>
            </div>
        </div>
        
        <div class="refresh-info">
            🔄 Dashboard auto-refreshes every 10 seconds | Last updated: <span id="lastUpdate"></span>
        </div>
    </div>

    <script>
        // Update dashboard data
        async function updateStats() {
            try {
                const healthRes = await fetch('/monitoring/health');
                const health = await healthRes.json();
                
                const statsRes = await fetch('/monitoring/stats');
                const stats = await statsRes.json();
                
                // Update health status
                const statusClass = health.status === 'healthy' ? 'status-healthy' : 
                                  health.status === 'degraded' ? 'status-degraded' : 'status-critical';
                
                document.getElementById('health').innerHTML = `
                    <div class="metric">
                        <div class="value ${statusClass}">${health.status.toUpperCase()}</div>
                        <div>System Status</div>
                    </div>
                    <div class="metric">
                        <div class="value">${health.checks.cpu.value.toFixed(1)}%</div>
                        <div>CPU Usage</div>
                    </div>
                    <div class="metric">
                        <div class="value">${health.checks.memory.value.toFixed(1)}%</div>
                        <div>Memory Usage</div>
                    </div>
                `;
                
                // Update prediction stats
                document.getElementById('predictions').innerHTML = `
                    <div class="metric">
                        <div class="value">${stats.total_predictions}</div>
                        <div>Total Predictions</div>
                    </div>
                    <div class="metric">
                        <div class="value">${stats.prediction_stats?.avg_duration_ms?.toFixed(2) || 'N/A'}</div>
                        <div>Avg Duration (ms)</div>
                    </div>
                    <div class="metric">
                        <div class="value">${stats.prediction_stats?.max_duration_ms?.toFixed(2) || 'N/A'}</div>
                        <div>Max Duration (ms)</div>
                    </div>
                    <div class="metric">
                        <div class="value">${stats.prediction_stats?.min_duration_ms?.toFixed(2) || 'N/A'}</div>
                        <div>Min Duration (ms)</div>
                    </div>
                `;
                
                // Update last refresh time
                document.getElementById('lastUpdate').textContent = new Date().toLocaleTimeString();
                
            } catch (e) {
                console.error('Error updating stats:', e);
                document.getElementById('health').innerHTML = '<div style="color: red;">Error loading data</div>';
                document.getElementById('predictions').innerHTML = '<div style="color: red;">Error loading data</div>';
            }
        }
        
        // Update immediately and then every 10 seconds
        updateStats();
        setInterval(updateStats, 10000);
    </script>
</body>
</html>
"""
    return render_template_string(html)

# Function to record prediction metrics - to be called from the prediction endpoint
def record_prediction(duration_ms):
    """Record a prediction for monitoring"""
    timestamp = datetime.now().isoformat()
    
    # Record system stats
    _metrics["system_stats"]["cpu"].append(psutil.cpu_percent(interval=0))
    _metrics["system_stats"]["memory"].append(psutil.virtual_memory().percent)
    
    # Record prediction
    _metrics["predictions"].append({
        "timestamp": timestamp,
        "duration_ms": duration_ms
    })
    
    # Keep only the last 1000 records
    if len(_metrics["predictions"]) > 1000:
        _metrics["predictions"] = _metrics["predictions"][-1000:]
    
    # Keep only the last 1000 system stats
    if len(_metrics["system_stats"]["cpu"]) > 1000:
        _metrics["system_stats"]["cpu"] = _metrics["system_stats"]["cpu"][-1000:]
        _metrics["system_stats"]["memory"] = _metrics["system_stats"]["memory"][-1000:]
    
    # Increment counter
    _metrics["total_predictions"] += 1

# Function for drift monitoring (alias for compatibility)
def record_prediction_for_drift(features, prediction, actual=None):
    """Record prediction for drift monitoring (currently just records basic metrics)"""
    # For now, just record a basic prediction metric
    # You can extend this later to work with the drift monitor
    record_prediction(0)  # Use 0 for drift predictions since we don't have timing here

@monitoring_bp.route('/predict', methods=['POST'])
def predict():
    req = request.get_json()
    # Try both keys for compatibility
    records = req.get("historical_data") or req.get("data")
    if not records or not isinstance(records, list):
        return jsonify({
            "error_type": "validation_error",
            "message": "Missing or invalid 'historical_data' (or 'data') key",
            "status": "error",
            "timestamp": datetime.now().isoformat()
        }), 400

    if len(records) < 6:
        return jsonify({
            "error_type": "validation_error",
            "message": f"Need at least 6 records, got {len(records)}",
            "status": "error",
            "timestamp": datetime.now().isoformat()
        }, 400)

    # Simulate prediction and timing
    start = time.time()
    # Dummy prediction: just return the input values
    prediction = [r.get("preco_medio_close", 0) for r in records[:6]]
    duration_ms = (time.time() - start) * 1000

    record_prediction(duration_ms)

    return jsonify({
        "status": "ok",
        "predictions": prediction,
        "duration_ms": duration_ms,
        "timestamp": datetime.now().isoformat()
    })

@monitoring_bp.route('/drift-report', methods=['GET'])
def drift_report():
    """Simple drift report endpoint that doesn't use Prometheus"""
    try:
        # Simple response without using the middleware directly
        return jsonify({
            "status": "ok",
            "drift_detected": False,
            "message": "Drift monitoring is available",
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error generating drift report: {str(e)}"
        }), 500

