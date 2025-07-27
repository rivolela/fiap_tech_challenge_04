"""
Utilitários para monitoramento
============================

Decoradores e funções utilitárias para o sistema de monitoramento.
"""

import time
import logging
import functools
from functools import wraps
from typing import Any, Callable, Dict, Optional, TypeVar, cast

# Configuração de logging
logger = logging.getLogger(__name__)

# Tentar importar memory_profiler
try:
    from memory_profiler import memory_usage
    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False

# Verificar disponibilidade do Prometheus
try:
    import prometheus_client
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

# Type para função decorada
F = TypeVar('F', bound=Callable[..., Any])


def time_execution(func: F) -> F:
    """
    Decorador para medir o tempo de execução de uma função
    
    Args:
        func: Função a ser decorada
    
    Returns:
        Função decorada
        
    Uso:
        @time_execution
        def my_function():
            ...
    """
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        duration_ms = (end_time - start_time) * 1000
        logger.info(f"Function {func.__name__} executed in {duration_ms:.2f}ms")
        
        return result
        
    return cast(F, wrapper)


def monitor_prediction(label: str = ""):
    """
    Decorador para monitorar a execução de funções de predição
    
    Args:
        label: Rótulo opcional para identificar a função
        
    Returns:
        Função decorada
        
    Uso:
        @monitor_prediction("stock_prediction")
        def predict_stock_price(model, data):
            return model.predict(data)
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Importar aqui para evitar importação circular
            from src.monitoring.core.monitor import ModelPerformanceMonitor
            from src.monitoring import performance_monitor
            
            start_time = time.time()
            
            try:
                # Determinar tamanho da entrada
                input_data = args[1] if len(args) > 1 else None
                
                # Executar a função
                result = func(*args, **kwargs)
                
                # Calcular duração
                duration_ms = (time.time() - start_time) * 1000
                
                # Identificar confiança se disponível
                confidence = None
                if isinstance(result, dict) and 'confidence' in result:
                    confidence = result['confidence']
                
                # Registrar métricas
                if hasattr(performance_monitor, 'record_prediction'):
                    performance_monitor.record_prediction(
                        input_data,
                        result,
                        duration_ms=duration_ms
                    )
                else:
                    logger.warning("Performance monitor não disponível para registrar predição")
                
                # Registrar no Prometheus se disponível
                if PROMETHEUS_AVAILABLE:
                    try:
                        from src.monitoring.integrations.prometheus import setup_prometheus_metrics
                        metrics = setup_prometheus_metrics()
                        metrics['prediction_duration'].observe(duration_ms / 1000)
                    except Exception as e:
                        logger.error(f"Erro ao registrar métricas no Prometheus: {e}")
                    
                return result
                
            except Exception as e:
                # Registrar falha de predição
                duration_ms = (time.time() - start_time) * 1000
                logger.error(f"Error in prediction {label}: {str(e)}")
                
                # Registrar falha com valores vazios
                try:
                    import numpy as np
                    empty_array = np.array([])
                    
                    # Importar o monitor
                    from src.monitoring import performance_monitor
                    
                    # Registrar falha
                    if hasattr(performance_monitor, 'record_prediction'):
                        performance_monitor.record_prediction(
                            empty_array,  # input vazio
                            0.0,          # prediction vazia
                            duration_ms=duration_ms   # duração até o erro
                        )
                except Exception as log_error:
                    logger.error(f"Erro ao registrar falha de predição: {log_error}")
                
                raise e
        
        return cast(F, wrapper)
    
    return decorator


def track_prediction(func: F) -> F:
    """
    Decorador para rastrear automaticamente o desempenho de uma função de predição
    
    Args:
        func: Função a ser decorada
    
    Returns:
        Função decorada
        
    Uso:
        @track_prediction
        def predict(model, input_data):
            return model.predict(input_data)
    """
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Importar aqui para evitar importação circular
        from src.monitoring import performance_monitor
        
        start_time = time.time()
        
        try:
            # Capturar uso de memória antes (não usado atualmente, apenas para compatibilidade)
            if MEMORY_PROFILER_AVAILABLE:
                mem_before = memory_usage(-1, interval=0.001, timeout=1, max_usage=True)
            
            # Executar a função
            result = func(*args, **kwargs)
            
            # Calcular duração
            duration_ms = (time.time() - start_time) * 1000
            
            # Identificar entrada
            input_data = args[1] if len(args) > 1 else None
            
            # Determinar confiança se disponível
            confidence = None
            if isinstance(result, dict) and 'confidence' in result:
                confidence = result['confidence']
                
            # Registrar predição se o monitor estiver disponível
            if hasattr(performance_monitor, 'record_prediction'):
                performance_monitor.record_prediction(
                    input_data=input_data,
                    prediction=result,
                    duration_ms=duration_ms
                )
            
            return result
            
        except Exception as e:
            # Calcular duração
            duration_ms = (time.time() - start_time) * 1000
            
            # Tentar registrar falha com valores vazios
            try:
                import numpy as np
                empty_array = np.array([])
                
                # Registrar falha se o monitor estiver disponível
                if hasattr(performance_monitor, 'record_prediction'):
                    performance_monitor.record_prediction(
                        input_data=empty_array,  # input vazio
                        prediction=0.0,         # prediction vazia
                        duration_ms=duration_ms  # duração até o erro
                    )
            except:
                logger.error("Erro ao registrar falha de predição")
                
            raise e
            
    return cast(F, wrapper)
