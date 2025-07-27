"""
Monitoramento de Sistema
=======================

Funções para monitorar métricas do sistema operacional.
"""

import logging
import platform
import time
from datetime import datetime
from typing import Dict, Optional, List

import psutil
try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False

from src.monitoring.core.metrics import SystemMetrics


class SystemMonitor:
    """
    Monitor para rastrear métricas do sistema operacional
    """
    
    def __init__(self, log_file: str = "system_monitoring.log"):
        """
        Inicializa o monitor de sistema
        
        Args:
            log_file: Arquivo para salvar logs de monitoramento do sistema
        """
        self.system_metrics: List[SystemMetrics] = []
        self.logger = logging.getLogger("system_monitoring")
        self.start_time = datetime.now()
        self.system_info = self._get_system_info()
        
        # Configurar o logger específico para monitoramento de sistema
        if not self.logger.handlers:
            handler = logging.FileHandler(log_file)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
            
        self.logger.info(f"System Monitor initialized on {platform.node()}")
        self.logger.info(f"System info: {self.system_info}")
    
    def _get_system_info(self) -> Dict[str, str]:
        """
        Obtém informações básicas sobre o sistema
        
        Returns:
            Dict[str, str]: Informações do sistema
        """
        return {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "cpu_count": str(psutil.cpu_count(logical=True)),
            "physical_cpu_count": str(psutil.cpu_count(logical=False)),
            "total_memory": f"{psutil.virtual_memory().total / (1024**3):.2f} GB",
            "gpu_available": str(GPUTIL_AVAILABLE)
        }
    
    def collect_metrics(self) -> SystemMetrics:
        """
        Coleta métricas atuais do sistema
        
        Returns:
            SystemMetrics: Métricas do sistema coletadas
        """
        # Obter métricas básicas do sistema
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_available_mb = memory.available / (1024 * 1024)
        disk = psutil.disk_usage('/')
        disk_usage_percent = disk.percent
        
        # Timestamp atual
        timestamp = datetime.now().isoformat()
        
        # Métricas de GPU (se disponível)
        gpu_utilization: Optional[float] = None
        gpu_memory_used_mb: Optional[float] = None
        gpu_memory_total_mb: Optional[float] = None
        
        if GPUTIL_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Usar a primeira GPU disponível
                    gpu_utilization = gpu.load * 100
                    gpu_memory_used_mb = gpu.memoryUsed
                    gpu_memory_total_mb = gpu.memoryTotal
            except Exception as e:
                self.logger.warning(f"Error getting GPU metrics: {str(e)}")
        
        # Criar objeto de métricas
        metrics = SystemMetrics(
            timestamp=timestamp,
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_available_mb=memory_available_mb,
            disk_usage_percent=disk_usage_percent,
            gpu_utilization=gpu_utilization,
            gpu_memory_used_mb=gpu_memory_used_mb,
            gpu_memory_total_mb=gpu_memory_total_mb
        )
        
        # Armazenar métricas
        self.system_metrics.append(metrics)
        
        # Log de métricas
        self.logger.debug(
            f"System metrics: CPU={cpu_percent:.2f}%, "
            f"Memory={memory_percent:.2f}% ({memory_available_mb:.2f}MB free), "
            f"Disk={disk_usage_percent:.2f}%"
        )
        
        return metrics
    
    def collect_metrics_periodically(
        self, 
        interval: float = 60.0,
        callback=None
    ):
        """
        Coleta métricas periodicamente em um loop
        
        Args:
            interval: Intervalo em segundos entre as coletas
            callback: Função opcional para chamar com as métricas coletadas
        """
        self.logger.info(f"Starting periodic metrics collection every {interval}s")
        
        try:
            while True:
                metrics = self.collect_metrics()
                
                if callback:
                    callback(metrics)
                
                time.sleep(interval)
        except KeyboardInterrupt:
            self.logger.info("Periodic metrics collection stopped")
    
    def get_last_metrics(self) -> Optional[SystemMetrics]:
        """
        Retorna as métricas mais recentes do sistema
        
        Returns:
            Optional[SystemMetrics]: Métricas mais recentes ou None se não houver
        """
        if not self.system_metrics:
            return None
            
        return self.system_metrics[-1]
    
    def get_recent_metrics(self, n: int = 60) -> List[SystemMetrics]:
        """
        Retorna as n métricas mais recentes
        
        Args:
            n: Número de métricas recentes para retornar
            
        Returns:
            List[SystemMetrics]: Lista de métricas recentes
        """
        return self.system_metrics[-n:] if self.system_metrics else []
    
    def clear_old_data(self, max_records: int = 1440) -> None:
        """
        Remove dados antigos para conservar memória
        
        Args:
            max_records: Número máximo de registros para manter
        """
        if len(self.system_metrics) > max_records:
            self.system_metrics = self.system_metrics[-max_records:]
            self.logger.info(f"Cleared old system metrics, keeping last {max_records} records")
