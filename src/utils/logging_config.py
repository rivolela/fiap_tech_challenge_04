"""
Configuração de Logging
======================

Configurações padronizadas de logging para o projeto LSTM Stock Prediction.
Centraliza a configuração de logs para manter consistência entre módulos.
"""

import os
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler

# Diretórios de log
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs')


def setup_logger(name, log_type='api', level=logging.INFO):
    """
    Configura um logger com configurações padronizadas.
    
    Args:
        name: Nome do logger
        log_type: Tipo de log (api, training, monitoring, prediction)
        level: Nível de logging
        
    Returns:
        Logger configurado
    """
    # Criar diretório de log se não existir
    log_dir = os.path.join(LOGS_DIR, log_type)
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"{log_type}.log")
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Evitar duplicação de handlers se o logger já foi configurado
    if logger.hasHandlers():
        return logger
    
    # Handler para arquivo com rotação
    file_handler = RotatingFileHandler(
        log_file, maxBytes=10*1024*1024, backupCount=3
    )
    
    # Handler para console
    console_handler = logging.StreamHandler()
    
    # Formatação
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


# Loggers pré-configurados para diferentes componentes
def get_api_logger(name='api'):
    """Logger para a API"""
    return setup_logger(name, log_type='api')


def get_training_logger(name='training'):
    """Logger para treinamento de modelos"""
    return setup_logger(name, log_type='training')


def get_monitoring_logger(name='monitoring'):
    """Logger para monitoramento de modelos"""
    return setup_logger(name, log_type='monitoring')


def get_prediction_logger(name='prediction'):
    """Logger para predições"""
    return setup_logger(name, log_type='prediction')
