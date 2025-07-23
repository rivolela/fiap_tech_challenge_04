import psutil
import time
import json
from datetime import datetime
from flask import Blueprint, jsonify, request, render_template_string

# Create Blueprint
monitoring_bp = Blueprint('monitoring', __name__, url_prefix='/monitoring')

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
    """Return simple HTML dashboard"""
    html = """<!DOCTYPE html>
<html>
<head>
    <title>LSTM API Monitoring</title>
    <meta http-equiv="refresh" content="5">
    <style>
        body { font-family: Arial; margin: 20px; }
        .card { background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }
        .metric { display: inline-block; margin: 10px; text-align: center; }
        .value { font-size: 24px; font-weight: bold; }
    </style>
</head>
<body>
    <h1>LSTM API Monitoring Dashboard</h1>
    <div class="card">
        <h2>System Health</h2>
        <div id="health">Loading...</div>
    </div>
    <div class="card">
        <h2>Prediction Stats</h2>
        <div id="predictions">Loading...</div>
    </div>
    <script>
        // Simple dashboard that refreshes data
        async function updateStats() {
            try {
                const healthRes = await fetch('/monitoring/health');
                const health = await healthRes.json();
                
                const statsRes = await fetch('/monitoring/stats');
                const stats = await statsRes.json();
                
                document.getElementById('health').innerHTML = `
                    <div class="metric">
                        <div class="value">${health.status}</div>
                        <div>Status</div>
                    </div>
                    <div class="metric">
                        <div class="value">${health.checks.cpu.value.toFixed(1)}%</div>
                        <div>CPU</div>
                    </div>
                    <div class="metric">
                        <div class="value">${health.checks.memory.value.toFixed(1)}%</div>
                        <div>Memory</div>
                    </div>
                `;
                
                document.getElementById('predictions').innerHTML = `
                    <div class="metric">
                        <div class="value">${stats.total_predictions}</div>
                        <div>Total Predictions</div>
                    </div>
                    <div class="metric">
                        <div class="value">${stats.prediction_stats?.avg_duration_ms?.toFixed(2) || 'N/A'}</div>
                        <div>Avg Duration (ms)</div>
                    </div>
                `;
            } catch (e) {
                console.error('Error updating stats:', e);
            }
        }
        
        // Update immediately and then every 5 seconds
        updateStats();
        setInterval(updateStats, 5000);
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
        }), 400

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

