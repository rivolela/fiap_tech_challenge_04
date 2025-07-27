"""
Monitor de Desempenho do Modelo
===============================

Classes para monitorar o desempenho do modelo LSTM em produção.
"""

import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Union, Any

import numpy as np
import pandas as pd

from src.monitoring.core.metrics import PredictionMetrics, SystemMetrics


class ModelPerformanceMonitor:
    """
    Monitor para rastrear o desempenho das predições do modelo em produção
    """
    
    def __init__(self, model_version: str, log_file: str = "monitoring.log"):
        """
        Inicializa o monitor de desempenho do modelo
        
        Args:
            model_version: Versão do modelo sendo monitorado
            log_file: Arquivo para salvar logs de monitoramento
        """
        self.model_version = model_version
        self.predictions: List[PredictionMetrics] = []
        self.logger = logging.getLogger("model_monitoring")
        self.start_time = datetime.now()
        
        # Configurar o logger específico para monitoramento
        if not self.logger.handlers:
            handler = logging.FileHandler(log_file)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
            
        self.logger.info(f"Model Performance Monitor initialized for version: {model_version}")
    
    def record_prediction(
        self,
        input_data: Union[pd.DataFrame, np.ndarray],
        prediction: Union[float, np.ndarray],
        duration_ms: Optional[float] = None,
        confidence: Optional[float] = None
    ) -> PredictionMetrics:
        """
        Registra uma predição para monitoramento
        
        Args:
            input_data: Dados de entrada usados para a predição
            prediction: Valor da predição gerada pelo modelo
            duration_ms: Tempo gasto para gerar a predição (em milissegundos)
            confidence: Nível de confiança da predição (se disponível)
        
        Returns:
            PredictionMetrics: Métricas da predição registrada
        """
        # Usar valores padrão se não fornecidos
        if duration_ms is None:
            duration_ms = 0.0
            
        if confidence is None:
            confidence = 0.0
            
        # Converter entrada e saída para formato consistente
        if isinstance(prediction, np.ndarray):
            prediction_value = prediction.tolist()
        else:
            prediction_value = prediction
            
        # Determinar tamanho da entrada e saída
        if isinstance(input_data, pd.DataFrame):
            input_size = input_data.size
        elif isinstance(input_data, np.ndarray):
            input_size = input_data.size
        else:
            input_size = 1
            
        if isinstance(prediction_value, (list, tuple)):
            output_size = len(prediction_value)
        else:
            output_size = 1
            
        # Obter métricas do sistema
        import psutil
        memory_usage_mb = psutil.Process().memory_info().rss / (1024 * 1024)
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Registrar timestamp atual
        timestamp = datetime.now()
        
        # Criar o objeto de métricas
        # Converter timestamp para string
        timestamp_str = timestamp.isoformat()
        
        metrics = PredictionMetrics(
            timestamp=timestamp_str,
            duration_ms=duration_ms,
            input_size=input_size,
            output_size=output_size,
            memory_usage_mb=memory_usage_mb,
            cpu_percent=cpu_percent,
            model_confidence=confidence,
            prediction_variance=None  # Não calculamos variância neste ponto
        )
        
        # Armazenar métricas
        self.predictions.append(metrics)
        
        # Logar a predição
        self.logger.info(
            f"Prediction recorded: value={prediction_value}, "
            f"duration={duration_ms:.2f}ms, memory={memory_usage_mb:.2f}MB, "
            f"CPU={cpu_percent:.2f}%, confidence={confidence:.4f}"
        )
        
        return metrics
    
    def get_recent_metrics(self, n: int = 100) -> List[PredictionMetrics]:
        """
        Retorna as métricas das n predições mais recentes
        
        Args:
            n: Número de predições recentes para retornar
            
        Returns:
            List[PredictionMetrics]: Lista das métricas de predição mais recentes
        """
        return self.predictions[-n:] if self.predictions else []
    
    def calculate_accuracy_metrics(self) -> Dict[str, Any]:
        """
        Calcula métricas de precisão com base nas predições registradas
        
        Returns:
            Dict[str, Any]: Dicionário com métricas de precisão
        """
        if not self.predictions:
            return {
                "mean_duration_ms": 0.0,
                "mean_memory_usage_mb": 0.0,
                "mean_cpu_percent": 0.0,
                "prediction_count": 0,
                "mean_confidence": 0.0
            }
            
        # Calcular média do tempo de duração
        mean_duration_ms = np.mean([p.duration_ms for p in self.predictions])
        
        # Calcular média do uso de memória
        mean_memory_usage_mb = np.mean([p.memory_usage_mb for p in self.predictions])
        
        # Calcular média do uso de CPU
        mean_cpu_percent = np.mean([p.cpu_percent for p in self.predictions])
        
        # Calcular média de confiança (ignorando None)
        confidences = [p.model_confidence for p in self.predictions if p.model_confidence is not None]
        mean_confidence = np.mean(confidences) if confidences else 0.0
        
        return {
            "mean_duration_ms": mean_duration_ms,
            "mean_memory_usage_mb": mean_memory_usage_mb,
            "mean_cpu_percent": mean_cpu_percent,
            "prediction_count": len(self.predictions),
            "mean_confidence": mean_confidence
        }
    
    def log_performance_summary(self) -> None:
        """
        Registra um resumo do desempenho do modelo
        """
        metrics = self.calculate_accuracy_metrics()
        uptime = datetime.now() - self.start_time
        
        self.logger.info(
            f"Performance summary after {len(self.predictions)} predictions:\n"
            f"- Mean duration: {metrics['mean_duration_ms']:.2f}ms\n"
            f"- Mean memory usage: {metrics['mean_memory_usage_mb']:.2f}MB\n"
            f"- Mean CPU usage: {metrics['mean_cpu_percent']:.2f}%\n"
            f"- Mean confidence: {metrics['mean_confidence']:.4f}\n"
            f"- Monitor uptime: {uptime}"
        )
        
    def clear_old_data(self, max_records: int = 10000) -> None:
        """
        Remove dados antigos para conservar memória
        
        Args:
            max_records: Número máximo de registros para manter
        """
        if len(self.predictions) > max_records:
            self.predictions = self.predictions[-max_records:]
            self.logger.info(f"Cleared old prediction data, keeping last {max_records} records")
