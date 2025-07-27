"""
Sistema de Monitoramento e Métricas para LSTM API
==================================================

Este módulo fornece monitoramento abrangente para o modelo LSTM em produção,
incluindo métricas de performance, recursos do sistema e qualidade do modelo.

NOTA: Este arquivo contém o código original do módulo de monitoramento antes da refatoração.
Foi preservado para referência e compatibilidade durante a transição.
"""

import time
import psutil
import logging
import threading
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from functools import wraps
import json
import os

try:
    from memory_profiler import memory_usage
    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False

logger = logging.getLogger(__name__)

# Definições das classes de dados para métricas
@dataclass
class PredictionMetrics:
    """Métricas de uma predição individual"""
    timestamp: str
    duration_ms: float
    input_size: int
    output_size: int
    memory_usage_mb: float
    cpu_percent: float
    gpu_memory_mb: Optional[float] = None
    model_confidence: Optional[float] = None
    prediction_variance: Optional[float] = None


@dataclass
class SystemMetrics:
    """Métricas do sistema"""
    timestamp: str
    cpu_percent: float
    memory_percent: float
    memory_available_mb: float
    disk_usage_percent: float
    gpu_utilization: Optional[float] = None
    gpu_memory_used_mb: Optional[float] = None
    gpu_memory_total_mb: Optional[float] = None


class ModelPerformanceMonitor:
    """Monitor para rastrear o desempenho do modelo LSTM em produção"""
    
    def __init__(self, max_history: int = 1000):
        """Inicializa o monitor de desempenho"""
        self.start_time = datetime.now()
        self.prediction_history = deque(maxlen=max_history)
        self.system_metrics = deque(maxlen=max_history)
        self.monitoring_enabled = False
        self._monitoring_thread = None
        self._stop_monitoring = threading.Event()
        
        # Configuração de logging
        self.logger = logging.getLogger("model_monitoring")
        if not self.logger.handlers:
            handler = logging.FileHandler("monitoring.log")
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        self.logger.info("Model Performance Monitor initialized")
        
    def record_prediction(self, input_data, prediction, duration_ms, metadata=None):
        """
        Registra uma predição para monitoramento
        
        Args:
            input_data: Dados de entrada para o modelo
            prediction: Saída da predição
            duration_ms: Tempo de execução em milissegundos
            metadata: Dados adicionais para registro
        """
        if metadata is None:
            metadata = {}
            
        # Determinar tamanho de entrada e saída
        input_size = getattr(input_data, 'size', 1)
        if hasattr(prediction, '__len__'):
            output_size = len(prediction)
        else:
            output_size = 1
            
        # Obter uso de memória e CPU
        memory_usage_mb = psutil.Process().memory_info().rss / (1024 * 1024)
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Capturar timestamp
        timestamp = datetime.now().isoformat()
        
        # Criar métrica de predição
        metrics = PredictionMetrics(
            timestamp=timestamp,
            duration_ms=duration_ms,
            input_size=input_size,
            output_size=output_size,
            memory_usage_mb=memory_usage_mb,
            cpu_percent=cpu_percent,
            model_confidence=metadata.get('confidence')
        )
        
        # Adicionar à lista de histórico
        self.prediction_history.append(metrics)
        
        # Registrar log
        self.logger.info(
            f"Prediction recorded: duration={duration_ms:.2f}ms, "
            f"memory={memory_usage_mb:.2f}MB, confidence={metrics.model_confidence}"
        )
        
        return metrics
    
    def get_recent_predictions(self, n=10):
        """Retorna as n predições mais recentes"""
        return list(self.prediction_history)[-n:]
    
    def get_performance_metrics(self):
        """Calcula métricas de desempenho agregadas"""
        if not self.prediction_history:
            return {
                "prediction_count": 0,
                "avg_duration_ms": 0,
                "avg_memory_usage_mb": 0,
                "avg_confidence": 0
            }
        
        predictions = list(self.prediction_history)
        
        return {
            "prediction_count": len(predictions),
            "avg_duration_ms": sum(p.duration_ms for p in predictions) / len(predictions),
            "avg_memory_usage_mb": sum(p.memory_usage_mb for p in predictions) / len(predictions),
            "avg_confidence": sum(p.model_confidence or 0 for p in predictions) / len(predictions)
        }
    
    def start_monitoring(self, interval=60):
        """
        Inicia monitoramento contínuo do sistema em segundo plano
        
        Args:
            interval: Intervalo em segundos entre medições
        """
        if self.monitoring_enabled:
            logger.warning("Monitoring is already running")
            return
            
        self.monitoring_enabled = True
        self._stop_monitoring.clear()
        
        self._monitoring_thread = threading.Thread(
            target=self._monitor_system,
            args=(interval,),
            daemon=True
        )
        self._monitoring_thread.start()
        
        self.logger.info(f"System monitoring started with interval of {interval}s")
        
    def stop_monitoring(self):
        """Para o monitoramento contínuo do sistema"""
        if not self.monitoring_enabled:
            return
            
        self._stop_monitoring.set()
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5)
            
        self.monitoring_enabled = False
        self.logger.info("System monitoring stopped")
        
    def _monitor_system(self, interval):
        """Coleta métricas do sistema periodicamente"""
        while not self._stop_monitoring.is_set():
            try:
                # Coletar métricas do sistema
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                memory_available = memory.available / (1024 * 1024)  # MB
                disk = psutil.disk_usage('/')
                disk_percent = disk.percent
                
                # Capturar timestamp
                timestamp = datetime.now().isoformat()
                
                # Criar métrica do sistema
                metrics = SystemMetrics(
                    timestamp=timestamp,
                    cpu_percent=cpu_percent,
                    memory_percent=memory_percent,
                    memory_available_mb=memory_available,
                    disk_usage_percent=disk_percent
                )
                
                # Adicionar ao histórico
                self.system_metrics.append(metrics)
                
                # Registrar log periódico (a cada 10 medições)
                if len(self.system_metrics) % 10 == 0:
                    self.logger.info(
                        f"System metrics: CPU={cpu_percent:.1f}%, "
                        f"Memory={memory_percent:.1f}%, "
                        f"Disk={disk_percent:.1f}%"
                    )
                    
            except Exception as e:
                self.logger.error(f"Error monitoring system: {str(e)}")
                
            # Aguardar até o próximo intervalo
            time.sleep(interval)
            
    def get_recent_system_metrics(self, n=60):
        """Retorna as n métricas de sistema mais recentes"""
        return list(self.system_metrics)[-n:]
        
    def get_uptime(self):
        """Retorna o tempo de atividade do monitor"""
        return datetime.now() - self.start_time


def track_prediction(func):
    """
    Decorador para rastrear automaticamente o desempenho de uma função de predição
    
    Uso:
        @track_prediction
        def predict(model, input_data):
            return model.predict(input_data)
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        try:
            # Capturar uso de memória antes
            if MEMORY_PROFILER_AVAILABLE:
                mem_before = memory_usage(-1, interval=0.001, timeout=1, max_usage=True)
            
            # Executar a função
            result = func(*args, **kwargs)
            
            # Calcular duração
            duration_ms = (time.time() - start_time) * 1000
            
            # Capturar uso de memória depois
            if MEMORY_PROFILER_AVAILABLE:
                mem_after = memory_usage(-1, interval=0.001, timeout=1, max_usage=True)
                mem_used = mem_after - mem_before
            else:
                mem_used = 0
                
            # Identificar entrada e saída
            input_data = args[1] if len(args) > 1 else None
            
            # Registrar predição
            performance_monitor.record_prediction(
                input_data=input_data,
                prediction=result,
                duration_ms=duration_ms,
                metadata={"memory_increase": mem_used}
            )
            
            return result
            
        except Exception as e:
            # Registrar falha
            performance_monitor.record_prediction(
                input_data=None,
                prediction=None,
                duration_ms=(time.time() - start_time) * 1000,
                metadata={"error": str(e)}
            )
            raise
            
    return wrapper


# Função para criar dashboards para visualização de métricas
def setup_monitoring_dashboard(port=8050):
    """
    Configura um dashboard web para visualização de métricas do modelo
    
    Args:
        port: Porta para o servidor web
    
    Returns:
        app: Aplicação do dashboard
    """
    try:
        import dash
        from dash import dcc, html
        import plotly.graph_objs as go
        import numpy as np
    except ImportError:
        logger.error("Para usar o dashboard, instale dash e plotly: pip install dash plotly")
        return None
        
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
        predictions = performance_monitor.get_recent_predictions(60)
        
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
        metrics = performance_monitor.get_recent_system_metrics(60)
        
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
        
    # Iniciar monitoramento do sistema
    performance_monitor.start_monitoring()
    
    # Função para iniciar o servidor
    def start_dashboard():
        app.run_server(debug=False, host='0.0.0.0', port=port)
        
    # Iniciar em uma thread separada
    dashboard_thread = threading.Thread(target=start_dashboard, daemon=True)
    dashboard_thread.start()
    
    logger.info(f"Monitoring dashboard started on http://localhost:{port}")
    return app


# Funções para utilizar Prometheus para monitoramento
try:
    from prometheus_client import Counter, Histogram, Gauge, start_http_server
    
    # Definir métricas do Prometheus
    PROM_REQUEST_COUNT = Counter(
        'lstm_api_requests_total', 
        'Total number of requests', 
        ['endpoint', 'status']
    )
    
    PROM_REQUEST_LATENCY = Histogram(
        'lstm_api_request_duration_seconds',
        'Request latency in seconds',
        ['endpoint']
    )
    
    PROM_PREDICTION_LATENCY = Histogram(
        'lstm_prediction_duration_seconds',
        'Prediction latency in seconds'
    )
    
    PROM_CPU_USAGE = Gauge(
        'lstm_system_cpu_usage_percent',
        'CPU usage percentage'
    )
    
    PROM_MEMORY_USAGE = Gauge(
        'lstm_system_memory_usage_percent',
        'Memory usage percentage'
    )
    
    PROMETHEUS_AVAILABLE = True
    
    def start_prometheus_server(port=8000):
        """Inicia um servidor HTTP para exportar métricas do Prometheus"""
        start_http_server(port)
        logger.info(f"Prometheus metrics available at http://localhost:{port}/metrics")
        
        # Iniciar coleta de métricas do sistema
        def collect_system_metrics():
            """Coleta e exporta métricas do sistema para o Prometheus"""
            while True:
                try:
                    cpu = psutil.cpu_percent()
                    memory = psutil.virtual_memory().percent
                    
                    PROM_CPU_USAGE.set(cpu)
                    PROM_MEMORY_USAGE.set(memory)
                    
                except Exception as e:
                    logger.error(f"Error collecting system metrics: {str(e)}")
                    
                time.sleep(15)
                
        # Iniciar thread para coleta de métricas
        metrics_thread = threading.Thread(
            target=collect_system_metrics,
            daemon=True
        )
        metrics_thread.start()
        
        return metrics_thread
        
except ImportError:
    PROMETHEUS_AVAILABLE = False
    
    def start_prometheus_server(port=8000):
        """Versão simulada quando Prometheus não está disponível"""
        logger.warning("Prometheus not available. Install prometheus-client package.")
        return None


# Para medir o tempo de execução de funções
def time_execution(func):
    """
    Decorador para medir o tempo de execução de uma função
    
    Uso:
        @time_execution
        def my_function():
            ...
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        duration_ms = (end_time - start_time) * 1000
        logger.info(f"Function {func.__name__} executed in {duration_ms:.2f}ms")
        
        return result
    return wrapper


# Função para salvar métricas em JSON
def save_metrics_to_json(filename="metrics_export.json"):
    """
    Salva as métricas de monitoramento em um arquivo JSON
    
    Args:
        filename: Nome do arquivo para salvar as métricas
    """
    data = {
        "predictions": [asdict(p) for p in performance_monitor.get_recent_predictions(1000)],
        "system_metrics": [asdict(s) for s in performance_monitor.get_recent_system_metrics(1000)],
        "performance": performance_monitor.get_performance_metrics(),
        "uptime": str(performance_monitor.get_uptime())
    }
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2, default=str)
        
    logger.info(f"Metrics saved to {filename}")
    return filename


# Decorador para monitoramento de predição com tratamento de erros
def monitor_prediction(label=""):
    """
    Decorador para monitorar a execução de funções de predição
    
    Args:
        label: Rótulo opcional para identificar a função
        
    Uso:
        @monitor_prediction("stock_prediction")
        def predict_stock_price(model, data):
            return model.predict(data)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                # Determinar tamanho da entrada
                input_data = args[1] if len(args) > 1 else None
                
                # Executar a função
                result = func(*args, **kwargs)
                
                # Calcular duração
                duration_ms = (time.time() - start_time) * 1000
                
                # Registrar métricas
                performance_monitor.record_prediction(
                    input_data=input_data,
                    prediction=result,
                    duration_ms=duration_ms
                )
                
                # Registrar no Prometheus se disponível
                if PROMETHEUS_AVAILABLE:
                    PROM_PREDICTION_LATENCY.observe(duration_ms / 1000)
                    
                return result
                
            except Exception as e:
                # Registrar erro
                logger.error(f"Error in prediction {label}: {str(e)}")
                performance_monitor.record_prediction(
                    input_data=None,
                    prediction=None,
                    duration_ms=(time.time() - start_time) * 1000,
                    memory_usage_mb=0,
                    cpu_percent=0,
                    input_size=0,
                    output_size=0
                )
                raise e
        
        return wrapper
    return decorator


# Instância global do monitor
performance_monitor = ModelPerformanceMonitor()
