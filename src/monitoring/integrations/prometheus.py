"""
Configuração e Servidor Prometheus para Métricas
=======================================

Funções para configurar, exportar e servir métricas no formato Prometheus.
"""

import logging
import threading
import time
from typing import Dict, Optional, Any

# Configurar logger
logger = logging.getLogger(__name__)

# Verificar disponibilidade do Prometheus
try:
    from prometheus_client import Counter, Histogram, Gauge, Info, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("Prometheus client não está instalado. Algumas funcionalidades não estarão disponíveis.")


def setup_prometheus_metrics() -> Dict[str, Any]:
    """
    Configura e retorna as métricas do Prometheus
    
    Returns:
        Dict: Dicionário com todas as métricas configuradas
    """
    if not PROMETHEUS_AVAILABLE:
        logger.warning("Prometheus não disponível para configuração de métricas")
        return {}
        
    metrics = {}
    
    # Contadores
    metrics['request_count'] = Counter(
        'lstm_api_requests_total',
        'Total number of prediction requests',
        ['endpoint', 'status']
    )
    
    metrics['prediction_count'] = Counter(
        'lstm_predictions_total',
        'Total number of predictions made'
    )
    
    # Histogramas para latência
    metrics['request_duration'] = Histogram(
        'lstm_api_request_duration_seconds',
        'Request duration in seconds',
        ['endpoint']
    )
    
    metrics['prediction_duration'] = Histogram(
        'lstm_prediction_duration_seconds',
        'Prediction duration in seconds'
    )
    
    # Gauges para métricas do sistema
    metrics['cpu_usage'] = Gauge(
        'lstm_api_cpu_usage_percent',
        'CPU usage percentage'
    )
    
    metrics['memory_usage'] = Gauge(
        'lstm_api_memory_usage_percent',
        'Memory usage percentage'
    )
    
    metrics['memory_available'] = Gauge(
        'lstm_api_memory_available_mb',
        'Available memory in MB'
    )
    
    metrics['gpu_memory_usage'] = Gauge(
        'lstm_api_gpu_memory_usage_mb',
        'GPU memory usage in MB'
    )
    
    metrics['gpu_utilization'] = Gauge(
        'lstm_api_gpu_utilization_percent',
        'GPU utilization percentage'
    )
    
    # Gauges para métricas do modelo
    metrics['model_confidence'] = Gauge(
        'lstm_model_confidence',
        'Model prediction confidence (avg)'
    )
    
    metrics['prediction_variance'] = Gauge(
        'lstm_prediction_variance',
        'Variance in model predictions'
    )
    
    # Info sobre o modelo
    metrics['model_info'] = Info(
        'lstm_model_info',
        'Information about the LSTM model'
    )
    
    return metrics


def start_prometheus_server(port: int = 8000) -> Optional[threading.Thread]:
    """
    Inicia um servidor HTTP para exportar métricas do Prometheus
    
    Args:
        port: Porta para o servidor de métricas
        
    Returns:
        Optional[threading.Thread]: Thread do coletor de métricas ou None se não disponível
    """
    if not PROMETHEUS_AVAILABLE:
        logger.warning("Prometheus não está disponível. Instale o pacote prometheus-client.")
        return None
    
    # Importar depois de verificar disponibilidade
    import psutil
    
    # Configurar métricas
    metrics = setup_prometheus_metrics()
    
    # Iniciar servidor HTTP
    start_http_server(port)
    logger.info(f"Métricas do Prometheus disponíveis em http://localhost:{port}/metrics")
    
    # Função para coletar métricas do sistema periodicamente
    def collect_system_metrics():
        """Coleta e exporta métricas do sistema para o Prometheus"""
        while True:
            try:
                # Coletar métricas básicas
                cpu = psutil.cpu_percent()
                memory = psutil.virtual_memory()
                
                # Atualizar gauges
                metrics['cpu_usage'].set(cpu)
                metrics['memory_usage'].set(memory.percent)
                metrics['memory_available'].set(memory.available / (1024 * 1024))
                
                # Tentar coletar métricas de GPU se disponíveis
                try:
                    import GPUtil
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu = gpus[0]  # Usar a primeira GPU
                        metrics['gpu_memory_usage'].set(gpu.memoryUsed)
                        metrics['gpu_utilization'].set(gpu.load * 100)
                except (ImportError, Exception):
                    pass  # Ignorar erros relacionados à GPU
                    
            except Exception as e:
                logger.error(f"Erro ao coletar métricas do sistema: {str(e)}")
                
            time.sleep(15)  # Coletar a cada 15 segundos
    
    # Iniciar thread para coleta de métricas
    metrics_thread = threading.Thread(
        target=collect_system_metrics,
        daemon=True
    )
    metrics_thread.start()
    
    return metrics_thread


def update_model_info(
    model_version: str,
    training_date: str,
    input_features: int,
    architecture: str
) -> None:
    """
    Atualiza informações sobre o modelo nas métricas do Prometheus
    
    Args:
        model_version: Versão do modelo
        training_date: Data de treinamento
        input_features: Número de features de entrada
        architecture: Descrição da arquitetura
    """
    if not PROMETHEUS_AVAILABLE:
        logger.warning("Prometheus não disponível para atualizar informações do modelo")
        return
        
    metrics = setup_prometheus_metrics()
    
    metrics['model_info'].info({
        'version': model_version,
        'training_date': training_date,
        'input_features': str(input_features),
        'architecture': architecture
    })
