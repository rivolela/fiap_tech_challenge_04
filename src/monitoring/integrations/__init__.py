"""
Integrações com ferramentas externas para monitoramento.

Este pacote contém integrações com:
- Dashboard (Dash/Plotly)
- Prometheus para métricas
- Flask Middleware para integração com API Flask
"""

from src.monitoring.integrations.dashboard import setup_monitoring_dashboard
from src.monitoring.integrations.prometheus import (
    setup_prometheus_metrics,
    start_prometheus_server,
    update_model_info
)

from src.monitoring.integrations.flask_middleware import (
    MonitoringMiddleware,
    monitor_endpoint,
    monitor_model_prediction,
    get_monitoring_blueprint
)

__all__ = [
    'setup_monitoring_dashboard',
    'setup_prometheus_metrics',
    'start_prometheus_server',
    'update_model_info',
    'MonitoringMiddleware',
    'monitor_endpoint',
    'monitor_model_prediction',
    'get_monitoring_blueprint',
]
