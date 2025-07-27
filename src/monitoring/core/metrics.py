"""
Definições de métricas para monitoramento do LSTM
==================================================

Classes de dados para métricas de predição e métricas do sistema.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any


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
