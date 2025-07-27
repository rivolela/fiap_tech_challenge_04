"""
Dashboard para Visualização de Métricas
======================================

Funções para criar dashboards interativos de métricas.
"""

import logging
import threading
from typing import Optional, Dict, Any

# Configuração do logger
logger = logging.getLogger(__name__)


def setup_monitoring_dashboard(port: int = 8050) -> Optional[Dict[str, Any]]:
    """
    Configura um dashboard web para visualização de métricas do modelo
    
    Args:
        port: Porta para o servidor web
    
    Returns:
        Optional[Dict[str, Any]]: Informações da aplicação do dashboard ou None se não foi possível criar
    """
    try:
        import dash
        from dash import dcc, html
        import plotly.graph_objs as go
        import numpy as np
    except ImportError:
        logger.error("Para usar o dashboard, instale dash e plotly: pip install dash plotly")
        return None
    
    # Importar aqui para evitar importação circular
    from src.monitoring.core.monitor import ModelPerformanceMonitor
    from src.monitoring.core.system import SystemMonitor
    
    # Criar um monitor de sistema para o dashboard
    system_monitor = SystemMonitor(log_file="dashboard_system.log")
    
    # Obter instância do monitor de performance
    # Assumindo que temos uma instância global ou forma de acessá-la
    try:
        # Tenta importar a instância do monitor de performance
        from src.monitoring import performance_monitor
    except ImportError:
        # Se não conseguir, cria um novo
        logger.warning("Não foi possível importar o monitor de performance existente. Usando um monitor vazio.")
        performance_monitor = ModelPerformanceMonitor()
    
    # Iniciar app Dash
    app = dash.Dash(__name__)
    
    # Layout do dashboard
    app.layout = html.Div([
        html.H1("LSTM Model Monitoring Dashboard"),
        
        html.Div([
            html.H3("Prediction Performance"),
            dcc.Graph(id='prediction-duration-graph'),
            dcc.Interval(
                id='prediction-interval',
                interval=10*1000,  # 10 segundos
                n_intervals=0
            )
        ]),
        
        html.Div([
            html.H3("System Resources"),
            dcc.Graph(id='system-resources-graph'),
            dcc.Interval(
                id='system-interval',
                interval=10*1000,  # 10 segundos
                n_intervals=0
            )
        ])
    ])
    
    @app.callback(
        dash.Output('prediction-duration-graph', 'figure'),
        [dash.Input('prediction-interval', 'n_intervals')]
    )
    def update_prediction_graph(_):
        """Atualiza o gráfico de duração das predições"""
        predictions = performance_monitor.get_recent_metrics(60)
        
        if not predictions:
            # Criar gráfico vazio se não houver dados
            return {
                'data': [
                    {'x': [], 'y': [], 'type': 'line', 'name': 'Duration (ms)'}
                ],
                'layout': {
                    'title': 'No prediction data available',
                    'xaxis': {'title': 'Time'},
                    'yaxis': {'title': 'Duration (ms)'}
                }
            }
            
        timestamps = [p.timestamp for p in predictions]
        durations = [p.duration_ms for p in predictions]
        memory_usage = [p.memory_usage_mb for p in predictions]
        
        return {
            'data': [
                {'x': timestamps, 'y': durations, 'type': 'line', 'name': 'Duration (ms)'},
                {'x': timestamps, 'y': memory_usage, 'type': 'line', 'name': 'Memory (MB)', 'yaxis': 'y2'}
            ],
            'layout': {
                'title': 'Prediction Performance Over Time',
                'xaxis': {'title': 'Time'},
                'yaxis': {'title': 'Duration (ms)'},
                'yaxis2': {
                    'title': 'Memory (MB)',
                    'overlaying': 'y',
                    'side': 'right'
                }
            }
        }
        
    @app.callback(
        dash.Output('system-resources-graph', 'figure'),
        [dash.Input('system-interval', 'n_intervals')]
    )
    def update_system_graph(_):
        """Atualiza o gráfico de recursos do sistema"""
        # Coletar métricas atuais
        system_monitor.collect_metrics()
        
        metrics = system_monitor.get_recent_metrics(60)
        
        if not metrics:
            # Criar gráfico vazio se não houver dados
            return {
                'data': [
                    {'x': [], 'y': [], 'type': 'line', 'name': 'CPU (%)'}
                ],
                'layout': {
                    'title': 'No system data available',
                    'xaxis': {'title': 'Time'},
                    'yaxis': {'title': 'Usage (%)'}
                }
            }
            
        timestamps = [m.timestamp for m in metrics]
        cpu = [m.cpu_percent for m in metrics]
        memory = [m.memory_percent for m in metrics]
        
        return {
            'data': [
                {'x': timestamps, 'y': cpu, 'type': 'line', 'name': 'CPU (%)'},
                {'x': timestamps, 'y': memory, 'type': 'line', 'name': 'Memory (%)'}
            ],
            'layout': {
                'title': 'System Resource Usage',
                'xaxis': {'title': 'Time'},
                'yaxis': {'title': 'Usage (%)'}
            }
        }
    
    # Função para iniciar o servidor
    def start_dashboard():
        system_monitor.collect_metrics()  # Coletar uma vez para ter dados iniciais
        app.run_server(debug=False, host='0.0.0.0', port=port)
        
    # Iniciar em uma thread separada
    dashboard_thread = threading.Thread(target=start_dashboard, daemon=True)
    dashboard_thread.start()
    
    logger.info(f"Monitoring dashboard started on http://localhost:{port}")
    
    # Retornar informações da aplicação
    return {
        "app": app,
        "thread": dashboard_thread,
        "system_monitor": system_monitor,
        "url": f"http://localhost:{port}"
    }
