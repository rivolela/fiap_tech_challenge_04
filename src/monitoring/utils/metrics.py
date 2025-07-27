"""
Utilitários para métricas e registro de dados
============================================

Funções utilitárias para cálculo e formatação de métricas.
"""

import json
import logging
import numpy as np
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def format_timestamp(dt: Optional[datetime] = None) -> str:
    """
    Formata um datetime para timestamp ISO 8601 com timezone UTC
    
    Args:
        dt: Objeto datetime ou None para usar datetime.now()
        
    Returns:
        String formatada em ISO 8601
    """
    if dt is None:
        dt = datetime.now(timezone.utc)
    elif dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
        
    return dt.isoformat()


def safe_json_serialize(obj: Any) -> Any:
    """
    Serializa objetos para JSON de forma segura, tratando tipos especiais
    
    Args:
        obj: Objeto a ser serializado
        
    Returns:
        Versão serializada do objeto
    """
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, datetime):
        return format_timestamp(obj)
    elif hasattr(obj, 'to_dict'):
        return obj.to_dict()
    elif hasattr(obj, '__dict__'):
        return {k: safe_json_serialize(v) for k, v in obj.__dict__.items() 
                if not k.startswith('_')}
    else:
        return str(obj)


def safe_json_dumps(data: Any) -> str:
    """
    Converte dados para JSON de forma segura
    
    Args:
        data: Dados a serem convertidos
        
    Returns:
        String JSON
    """
    try:
        return json.dumps(data, default=safe_json_serialize)
    except Exception as e:
        logger.error(f"Error serializing to JSON: {e}")
        return json.dumps({"error": "Could not serialize data"})


def calculate_metrics(predictions: List[float], actuals: List[float]) -> Dict[str, float]:
    """
    Calcula métricas de erro para predições
    
    Args:
        predictions: Lista de valores preditos
        actuals: Lista de valores reais
        
    Returns:
        Dicionário com métricas de erro
    """
    if not predictions or not actuals:
        return {}
    
    if len(predictions) != len(actuals):
        logger.warning(f"Prediction and actuals length mismatch: {len(predictions)} vs {len(actuals)}")
        # Usar o tamanho menor
        length = min(len(predictions), len(actuals))
        predictions = predictions[:length]
        actuals = actuals[:length]
    
    # Converter para numpy arrays
    pred_array = np.array(predictions, dtype=float)
    actual_array = np.array(actuals, dtype=float)
    
    # Calcular erro
    errors = pred_array - actual_array
    abs_errors = np.abs(errors)
    
    # Calcular métricas
    metrics = {
        "mae": float(np.mean(abs_errors)),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "mape": float(np.mean(np.abs(errors / (actual_array + 1e-10))) * 100),
        "me": float(np.mean(errors)),  # Mean Error (bias)
        "max_error": float(np.max(abs_errors)),
        "min_error": float(np.min(abs_errors)),
        "std_error": float(np.std(errors)),
        "count": len(predictions)
    }
    
    return metrics
