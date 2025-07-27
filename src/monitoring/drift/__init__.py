"""
Módulos para detecção e monitoramento de drift de dados.

Este pacote contém componentes para detectar mudanças entre dados de referência e atuais,
analisar tendências e gerar relatórios de drift.
"""

from src.monitoring.drift.detector import DriftDetector
from src.monitoring.drift.monitor import DriftMonitor
from src.monitoring.drift.collector import DataCollector
from src.monitoring.drift.report import ReportGenerator

__all__ = [
    'DriftDetector',
    'DriftMonitor',
    'DataCollector',
    'ReportGenerator',
]
