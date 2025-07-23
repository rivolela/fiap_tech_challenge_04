"""
Sistema de Monitoramento e Métricas para LSTM API
==================================================

Este módulo fornece monitoramento abrangente para o modelo LSTM em produção,
incluindo métricas de performance, recursos do sistema e qualidade do modelo.
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

import torch
import numpy as np
from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest
try:
    from memory_profiler import memory_usage
except ImportError:
    memory_usage = None

logger = logging.getLogger(__name__)


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
    """Monitor de performance do modelo LSTM"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.prediction_history = deque(maxlen=max_history)
        self.system_metrics = deque(maxlen=max_history)
        
        # Métricas Prometheus
        self._setup_prometheus_metrics()
        
        # Thread para coleta de métricas do sistema
        self._monitoring_thread = None
        self._monitoring_active = False
        
        # Cache de estatísticas
        self._stats_cache = {}
        self._cache_timestamp = None
        self._cache_duration = 30  # segundos
    
    def _setup_prometheus_metrics(self):
        """Configura métricas do Prometheus"""
        # Contadores
        self.request_count = Counter(
            'lstm_api_requests_total',
            'Total number of prediction requests',
            ['endpoint', 'status']
        )
        
        self.prediction_count = Counter(
            'lstm_predictions_total',
            'Total number of predictions made'
        )
        
        # Histogramas para latência
        self.request_duration = Histogram(
            'lstm_api_request_duration_seconds',
            'Request duration in seconds',
            ['endpoint']
        )
        
        self.prediction_duration = Histogram(
            'lstm_prediction_duration_seconds',
            'Prediction duration in seconds'
        )
        
        # Gauges para métricas do sistema
        self.cpu_usage = Gauge(
            'lstm_api_cpu_usage_percent',
            'CPU usage percentage'
        )
        
        self.memory_usage = Gauge(
            'lstm_api_memory_usage_percent',
            'Memory usage percentage'
        )
        
        self.memory_available = Gauge(
            'lstm_api_memory_available_mb',
            'Available memory in MB'
        )
        
        self.gpu_memory_usage = Gauge(
            'lstm_api_gpu_memory_usage_mb',
            'GPU memory usage in MB'
        )
        
        self.gpu_utilization = Gauge(
            'lstm_api_gpu_utilization_percent',
            'GPU utilization percentage'
        )
        
        # Gauges para métricas do modelo
        self.model_confidence = Gauge(
            'lstm_model_confidence',
            'Model prediction confidence (avg)'
        )
        
        self.prediction_variance = Gauge(
            'lstm_prediction_variance',
            'Variance in model predictions'
        )
        
        # Info sobre o modelo
        self.model_info = Info(
            'lstm_model_info',
            'Information about the LSTM model'
        )
    
    def start_monitoring(self):
        """Inicia o monitoramento do sistema"""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        self._monitoring_thread = threading.Thread(
            target=self._collect_system_metrics,
            daemon=True
        )
        self._monitoring_thread.start()
        logger.info("Sistema de monitoramento iniciado")
    
    def stop_monitoring(self):
        """Para o monitoramento do sistema"""
        self._monitoring_active = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5)
        logger.info("Sistema de monitoramento parado")
    
    def _collect_system_metrics(self):
        """Coleta métricas do sistema em background"""
        while self._monitoring_active:
            try:
                # Métricas do sistema
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                # Métricas de GPU (se disponível)
                gpu_util, gpu_memory, gpu_total = self._get_gpu_metrics()
                
                # Criar objeto de métricas
                metrics = SystemMetrics(
                    timestamp=datetime.now().isoformat(),
                    cpu_percent=cpu_percent,
                    memory_percent=memory.percent,
                    memory_available_mb=memory.available / (1024 * 1024),
                    disk_usage_percent=disk.percent,
                    gpu_utilization=gpu_util,
                    gpu_memory_used_mb=gpu_memory,
                    gpu_memory_total_mb=gpu_total
                )
                
                # Adicionar ao histórico
                self.system_metrics.append(metrics)
                
                # Atualizar métricas Prometheus
                self.cpu_usage.set(cpu_percent)
                self.memory_usage.set(memory.percent)
                self.memory_available.set(memory.available / (1024 * 1024))
                
                if gpu_util is not None:
                    self.gpu_utilization.set(gpu_util)
                if gpu_memory is not None:
                    self.gpu_memory_usage.set(gpu_memory)
                
            except Exception as e:
                logger.error(f"Erro ao coletar métricas do sistema: {e}")
            
            time.sleep(5)  # Coleta a cada 5 segundos
    
    def _get_gpu_metrics(self) -> tuple:
        """Obtém métricas de GPU se disponível"""
        try:
            if torch.cuda.is_available():
                # Utilização da GPU
                gpu_util = torch.cuda.utilization()
                
                # Memória da GPU
                gpu_memory_used = torch.cuda.memory_allocated() / (1024 * 1024)
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
                
                return gpu_util, gpu_memory_used, gpu_memory_total
        except Exception as e:
            logger.debug(f"Não foi possível obter métricas de GPU: {e}")
        
        return None, None, None
    
    def record_prediction(self, duration_ms: float, input_size: int, 
                         output_size: int, predictions: Optional[List[float]] = None):
        """Registra métricas de uma predição"""
        try:
            # Métricas de memória
            memory_mb = psutil.Process().memory_info().rss / (1024 * 1024)
            cpu_percent = psutil.cpu_percent()
            
            # Métricas de GPU
            _, gpu_memory, _ = self._get_gpu_metrics()
            
            # Métricas do modelo
            model_confidence = None
            prediction_variance = None
            
            if predictions:
                prediction_variance = float(np.var(predictions))
                # Simples heurística para confiança baseada na variância
                confidence_calc = 1.0 - (prediction_variance / float(np.mean(np.abs(predictions))))
                model_confidence = max(0.0, confidence_calc)
            
            # Criar objeto de métricas
            metrics = PredictionMetrics(
                timestamp=datetime.now().isoformat(),
                duration_ms=duration_ms,
                input_size=input_size,
                output_size=output_size,
                memory_usage_mb=memory_mb,
                cpu_percent=cpu_percent,
                gpu_memory_mb=gpu_memory,
                model_confidence=model_confidence,
                prediction_variance=prediction_variance
            )
            
            # Adicionar ao histórico
            self.prediction_history.append(metrics)
            
            # Atualizar métricas Prometheus
            self.prediction_count.inc()
            self.prediction_duration.observe(duration_ms / 1000)
            
            if model_confidence is not None:
                self.model_confidence.set(model_confidence)
            if prediction_variance is not None:
                self.prediction_variance.set(prediction_variance)
            
            logger.debug(f"Métricas registradas: {duration_ms:.2f}ms, "
                        f"mem: {memory_mb:.1f}MB, conf: {model_confidence}")
            
        except Exception as e:
            logger.error(f"Erro ao registrar métricas de predição: {e}")
    
    def get_performance_stats(self, refresh_cache: bool = False) -> Dict[str, Any]:
        """Obtém estatísticas de performance"""
        now = time.time()
        
        # Verificar cache
        if (not refresh_cache and self._cache_timestamp and 
            now - self._cache_timestamp < self._cache_duration):
            return self._stats_cache
        
        try:
            stats = {
                'timestamp': datetime.now().isoformat(),
                'total_predictions': len(self.prediction_history),
                'system_metrics_count': len(self.system_metrics)
            }
            
            if self.prediction_history:
                durations = [p.duration_ms for p in self.prediction_history]
                memory_usage = [p.memory_usage_mb for p in self.prediction_history]
                
                stats.update({
                    'prediction_stats': {
                        'avg_duration_ms': np.mean(durations),
                        'p50_duration_ms': np.percentile(durations, 50),
                        'p95_duration_ms': np.percentile(durations, 95),
                        'p99_duration_ms': np.percentile(durations, 99),
                        'max_duration_ms': np.max(durations),
                        'min_duration_ms': np.min(durations),
                        'avg_memory_mb': np.mean(memory_usage),
                        'max_memory_mb': np.max(memory_usage)
                    }
                })
                
                # Estatísticas do modelo
                confidences = [p.model_confidence for p in self.prediction_history 
                              if p.model_confidence is not None]
                if confidences:
                    stats['model_stats'] = {
                        'avg_confidence': np.mean(confidences),
                        'min_confidence': np.min(confidences),
                        'confidence_std': np.std(confidences)
                    }
            
            if self.system_metrics:
                cpu_usage = [m.cpu_percent for m in self.system_metrics]
                memory_usage = [m.memory_percent for m in self.system_metrics]
                
                stats.update({
                    'system_stats': {
                        'avg_cpu_percent': np.mean(cpu_usage),
                        'max_cpu_percent': np.max(cpu_usage),
                        'avg_memory_percent': np.mean(memory_usage),
                        'max_memory_percent': np.max(memory_usage),
                        'current_memory_available_mb': self.system_metrics[-1].memory_available_mb
                    }
                })
                
                # GPU stats se disponível
                gpu_utils = [m.gpu_utilization for m in self.system_metrics 
                            if m.gpu_utilization is not None]
                if gpu_utils:
                    stats['gpu_stats'] = {
                        'avg_utilization': np.mean(gpu_utils),
                        'max_utilization': np.max(gpu_utils),
                        'current_memory_mb': self.system_metrics[-1].gpu_memory_used_mb
                    }
            
            # Atualizar cache
            self._stats_cache = stats
            self._cache_timestamp = now
            
            return stats
            
        except Exception as e:
            logger.error(f"Erro ao calcular estatísticas: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    def get_recent_metrics(self, minutes: int = 5) -> Dict[str, Any]:
        """Obtém métricas recentes"""
        cutoff = datetime.now() - timedelta(minutes=minutes)
        
        recent_predictions = [
            asdict(p) for p in self.prediction_history
            if datetime.fromisoformat(p.timestamp) > cutoff
        ]
        
        recent_system = [
            asdict(m) for m in self.system_metrics
            if datetime.fromisoformat(m.timestamp) > cutoff
        ]
        
        return {
            'predictions': recent_predictions,
            'system': recent_system,
            'time_range_minutes': minutes
        }
    
    def export_metrics(self) -> str:
        """Exporta métricas no formato Prometheus"""
        return generate_latest().decode('utf-8')
    
    def save_metrics_to_file(self, filepath: str):
        """Salva métricas em arquivo JSON"""
        try:
            data = {
                'export_timestamp': datetime.now().isoformat(),
                'prediction_history': [asdict(p) for p in self.prediction_history],
                'system_metrics': [asdict(m) for m in self.system_metrics],
                'performance_stats': self.get_performance_stats()
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Métricas salvas em {filepath}")
            
        except Exception as e:
            logger.error(f"Erro ao salvar métricas: {e}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Verifica o status de saúde do sistema"""
        health = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'checks': {}
        }
        
        try:
            # Verificar recursos do sistema
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Check: Memória disponível
            if memory.percent > 90:
                health['status'] = 'unhealthy'
                health['checks']['memory'] = {
                    'status': 'critical',
                    'usage_percent': memory.percent,
                    'message': 'Uso de memória muito alto'
                }
            elif memory.percent > 80:
                health['status'] = 'degraded'
                health['checks']['memory'] = {
                    'status': 'warning',
                    'usage_percent': memory.percent,
                    'message': 'Uso de memória alto'
                }
            else:
                health['checks']['memory'] = {
                    'status': 'healthy',
                    'usage_percent': memory.percent
                }
            
            # Check: CPU
            if cpu_percent > 90:
                health['status'] = 'unhealthy'
                health['checks']['cpu'] = {
                    'status': 'critical',
                    'usage_percent': cpu_percent,
                    'message': 'Uso de CPU muito alto'
                }
            elif cpu_percent > 80:
                if health['status'] == 'healthy':
                    health['status'] = 'degraded'
                health['checks']['cpu'] = {
                    'status': 'warning',
                    'usage_percent': cpu_percent,
                    'message': 'Uso de CPU alto'
                }
            else:
                health['checks']['cpu'] = {
                    'status': 'healthy',
                    'usage_percent': cpu_percent
                }
            
            # Check: Performance recente
            if len(self.prediction_history) > 0:
                recent_predictions = [
                    p for p in self.prediction_history
                    if datetime.fromisoformat(p.timestamp) > datetime.now() - timedelta(minutes=5)
                ]
                
                if recent_predictions:
                    avg_duration = np.mean([p.duration_ms for p in recent_predictions])
                    if avg_duration > 5000:  # 5 segundos
                        health['status'] = 'degraded'
                        health['checks']['performance'] = {
                            'status': 'warning',
                            'avg_duration_ms': avg_duration,
                            'message': 'Tempo de resposta alto'
                        }
                    else:
                        health['checks']['performance'] = {
                            'status': 'healthy',
                            'avg_duration_ms': avg_duration
                        }
            
            # Check: GPU (se disponível)
            gpu_util, gpu_memory, gpu_total = self._get_gpu_metrics()
            if gpu_memory is not None and gpu_total is not None:
                gpu_percent = (gpu_memory / gpu_total) * 100
                if gpu_percent > 90:
                    health['status'] = 'degraded'
                    health['checks']['gpu'] = {
                        'status': 'warning',
                        'memory_usage_percent': gpu_percent,
                        'message': 'Uso de GPU alto'
                    }
                else:
                    health['checks']['gpu'] = {
                        'status': 'healthy',
                        'memory_usage_percent': gpu_percent
                    }
            
        except Exception as e:
            health['status'] = 'unhealthy'
            health['error'] = str(e)
            logger.error(f"Erro no health check: {e}")
        
        return health


def monitor_prediction_time(monitor: ModelPerformanceMonitor):
    """Decorator para monitorar tempo de predição"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                
                # Calcular métricas
                duration_ms = (time.time() - start_time) * 1000
                
                # Tentar extrair informações do resultado
                input_size = 0
                output_size = 0
                predictions = None
                
                if isinstance(result, (list, tuple)):
                    output_size = len(result)
                    if isinstance(result, list) and result and isinstance(result[0], (int, float)):
                        predictions = [float(x) for x in result]
                elif isinstance(result, dict) and 'predictions' in result:
                    pred_data = result['predictions']
                    if isinstance(pred_data, list):
                        predictions = [float(x) for x in pred_data if isinstance(x, (int, float))]
                        output_size = len(predictions) if predictions else 0
                
                # Se args contém dados de entrada, calcular tamanho
                for arg in args:
                    if hasattr(arg, '__len__') and not isinstance(arg, str):
                        input_size = len(arg)
                        break
                
                # Registrar métricas
                monitor.record_prediction(
                    duration_ms=duration_ms,
                    input_size=input_size,
                    output_size=output_size,
                    predictions=predictions
                )
                
                return result
                
            except Exception as e:
                # Registrar erro também
                duration_ms = (time.time() - start_time) * 1000
                monitor.record_prediction(
                    duration_ms=duration_ms,
                    input_size=0,
                    output_size=0
                )
                raise e
        
        return wrapper
    return decorator


# Instância global do monitor
performance_monitor = ModelPerformanceMonitor()
