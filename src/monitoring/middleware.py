"""
Middleware de Monitoramento para Flask API
==========================================

Este módulo integra o sistema de monitoramento com a API Flask,
fornecendo decorators e middleware para capturar métricas automaticamente.
"""

import time
import logging
from functools import wraps
from typing import Any, Dict, Optional
from flask import request, Response, g

from . import performance_monitor
from .drift_monitor import DriftMonitor
from .data_collector import DataCollector
from .drift_detector import DriftDetector  
from .report_generator import ReportGenerator

logger = logging.getLogger(__name__)


class MonitoringMiddleware:
    """Middleware para adicionar monitoramento automático à API Flask"""
    
    def __init__(self, app=None):
        self.app = app
        self.drift_monitor = DriftMonitor()
        if app is not None:
            self.init_app(app)
    
    def init_app(self, app):
        """Inicializa o middleware com a aplicação Flask"""
        app.before_request(self.before_request)
        app.after_request(self.after_request)
        app.teardown_appcontext(self.teardown)
        
        # Iniciar monitoramento do sistema
        performance_monitor.start_monitoring()
        
        logger.info("Middleware de monitoramento inicializado")
    
    def before_request(self):
        """Executado antes de cada request"""
        g.start_time = time.time()
        g.request_id = request.headers.get('X-Request-ID', 'unknown')
        
        # Registrar início do request
        endpoint = request.endpoint or 'unknown'
        performance_monitor.request_count.labels(
            endpoint=endpoint, 
            status='started'
        ).inc()
    
    def after_request(self, response: Response) -> Response:
        """Executado após cada request"""
        try:
            if hasattr(g, 'start_time'):
                duration = time.time() - g.start_time
                endpoint = request.endpoint or 'unknown'
                
                # Registrar métricas do request
                performance_monitor.request_duration.labels(
                    endpoint=endpoint
                ).observe(duration)
                
                performance_monitor.request_count.labels(
                    endpoint=endpoint,
                    status=str(response.status_code)
                ).inc()
                
                # Log detalhado para requests lentos
                if duration > 1.0:  # Mais de 1 segundo
                    logger.warning(
                        f"Request lento detectado: {request.method} {request.path} "
                        f"- {duration:.2f}s - Status: {response.status_code}"
                    )
                
        except Exception as e:
            logger.error(f"Erro no middleware de monitoramento: {e}")
        
        return response
    
    def teardown(self, exception=None):
        """Executado ao final de cada request"""
        if exception:
            endpoint = request.endpoint or 'unknown'
            performance_monitor.request_count.labels(
                endpoint=endpoint,
                status='error'
            ).inc()
            logger.error(f"Exceção no request {endpoint}: {exception}")
    
    def process_prediction(self, features, prediction, actual=None):
        """Processa e monitora predições do modelo"""
        try:
            # Adicionar ao monitor de drift
            self.drift_monitor.add_prediction(features, prediction, actual)
            
            # A cada N predições, gerar um relatório
            if len(self.drift_monitor.current_predictions) >= 100:
                self.drift_monitor.generate_report()
                self.drift_monitor.reset_current_data()
        
        except Exception as e:
            logger.error(f"Erro ao processar predição: {e}")
            raise e


def monitor_endpoint(endpoint_name: Optional[str] = None):
    """
    Decorator para monitorar endpoints específicos
    
    Args:
        endpoint_name: Nome personalizado para o endpoint
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            name = endpoint_name or func.__name__
            
            try:
                result = func(*args, **kwargs)
                
                # Métricas de sucesso
                duration = time.time() - start_time
                performance_monitor.request_duration.labels(
                    endpoint=name
                ).observe(duration)
                
                performance_monitor.request_count.labels(
                    endpoint=name,
                    status='success'
                ).inc()
                
                return result
                
            except Exception as e:
                # Métricas de erro
                duration = time.time() - start_time
                performance_monitor.request_duration.labels(
                    endpoint=name
                ).observe(duration)
                
                performance_monitor.request_count.labels(
                    endpoint=name,
                    status='error'
                ).inc()
                
                logger.error(f"Erro no endpoint {name}: {e}")
                raise e
        
        return wrapper
    return decorator


def monitor_model_prediction(func):
    """
    Decorator específico para monitorar predições do modelo
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            
            # Extrair métricas específicas da predição
            duration_ms = (time.time() - start_time) * 1000
            
            input_size = 0
            output_size = 0
            predictions = None
            
            # Analisar argumentos para encontrar dados de entrada
            for arg in args:
                if hasattr(arg, '__len__') and not isinstance(arg, str):
                    input_size = len(arg)
                    break
            
            # Analisar resultado
            if isinstance(result, list):
                output_size = len(result)
                if result and isinstance(result[0], (int, float)):
                    predictions = [float(x) for x in result]
            elif isinstance(result, dict):
                if 'predictions' in result:
                    pred_data = result['predictions']
                    if isinstance(pred_data, list):
                        predictions = [float(x) for x in pred_data if isinstance(x, (int, float))]
                        output_size = len(predictions)
            
            # Registrar métricas
            performance_monitor.record_prediction(
                duration_ms=duration_ms,
                input_size=input_size,
                output_size=output_size,
                predictions=predictions
            )
            
            return result
            
        except Exception as e:
            # Registrar erro
            duration_ms = (time.time() - start_time) * 1000
            performance_monitor.record_prediction(
                duration_ms=duration_ms,
                input_size=0,
                output_size=0
            )
            raise e
    
    return wrapper


def get_monitoring_blueprint():
    """
    Cria um Blueprint Flask com endpoints de monitoramento
    """
    from flask import Blueprint, jsonify, request
    from datetime import datetime
    
    monitoring_bp = Blueprint('monitoring', __name__, url_prefix='/monitoring')
    
    @monitoring_bp.route('/metrics', methods=['GET'])
    def prometheus_metrics():
        """Endpoint para métricas do Prometheus"""
        try:
            metrics = performance_monitor.export_metrics()
            return Response(metrics, mimetype='text/plain')
        except Exception as e:
            logger.error(f"Erro ao exportar métricas: {e}")
            return jsonify({'error': str(e)}), 500
    
    @monitoring_bp.route('/stats', methods=['GET'])
    def performance_stats():
        """Endpoint para estatísticas de performance"""
        try:
            stats = performance_monitor.get_performance_stats()
            return jsonify(stats)
        except Exception as e:
            logger.error(f"Erro ao obter estatísticas: {e}")
            return jsonify({'error': str(e)}), 500
    
    @monitoring_bp.route('/recent', methods=['GET'])
    def recent_metrics():
        """Endpoint para métricas recentes"""
        try:
            minutes = request.args.get('minutes', 5, type=int)
            metrics = performance_monitor.get_recent_metrics(minutes)
            return jsonify(metrics)
        except Exception as e:
            logger.error(f"Erro ao obter métricas recentes: {e}")
            return jsonify({'error': str(e)}), 500
    
    @monitoring_bp.route('/health', methods=['GET'])
    def detailed_health():
        """Endpoint para verificação detalhada de saúde"""
        try:
            health = performance_monitor.get_health_status()
            status_code = 200
            
            if health['status'] == 'unhealthy':
                status_code = 503
            elif health['status'] == 'degraded':
                status_code = 200  # Ainda funcional, mas com warnings
            
            return jsonify(health), status_code
        except Exception as e:
            logger.error(f"Erro no health check: {e}")
            return jsonify({
                'status': 'error',
                'error': str(e)
            }), 500
    
    @monitoring_bp.route('/dashboard', methods=['GET'])
    def monitoring_dashboard():
        """Endpoint para dashboard simples de monitoramento"""
        try:
            stats = performance_monitor.get_performance_stats()
            health = performance_monitor.get_health_status()
            recent = performance_monitor.get_recent_metrics(10)
            
            dashboard_data = {
                'overview': {
                    'status': health['status'],
                    'total_predictions': stats.get('total_predictions', 0),
                    'system_health': health.get('checks', {})
                },
                'performance': stats.get('prediction_stats', {}),
                'system': stats.get('system_stats', {}),
                'gpu': stats.get('gpu_stats', {}),
                'recent_activity': {
                    'last_10_minutes': len(recent.get('predictions', [])),
                    'avg_response_time': None
                }
            }
            
            # Calcular tempo médio das predições recentes
            recent_predictions = recent.get('predictions', [])
            if recent_predictions:
                durations = [p.get('duration_ms', 0) for p in recent_predictions]
                dashboard_data['recent_activity']['avg_response_time'] = sum(durations) / len(durations)
            
            return jsonify(dashboard_data)
        except Exception as e:
            logger.error(f"Erro no dashboard: {e}")
            return jsonify({'error': str(e)}), 500
    
    @monitoring_bp.route('/drift/report', methods=['POST'])
    def generate_drift_report():
        """Generate drift report endpoint"""
        try:
            # Get middleware components
            middleware = get_app_middleware()
            data_collector = middleware['data_collector']
            drift_detector = middleware['drift_detector']
            report_generator = middleware['report_generator']
            
            # Get request data
            request_data = request.get_json()
            
            if not request_data:
                return jsonify({
                    'message': 'No data provided for drift analysis',
                    'status': 'error'
                }), 400
            
            reference_data = request_data.get('reference_data')
            current_data = request_data.get('current_data')
            config = request_data.get('config', {})
            
            if not reference_data or not current_data:
                return jsonify({
                    'message': 'Both reference_data and current_data are required',
                    'status': 'error'
                }), 400
            
            # Collect data
            data_collector.collect_reference_data(reference_data)
            data_collector.collect_current_data(current_data)
            
            # Detect drift
            threshold = config.get('threshold', 0.05)
            methods = config.get('methods', ['statistical', 'distribution'])
            
            drift_results = drift_detector.detect_drift(
                reference_data, 
                current_data,
                threshold=threshold,
                methods=methods
            )
            
            # Generate report
            metadata = {
                'reference_samples': len(reference_data) if hasattr(reference_data, '__len__') else 1,
                'current_samples': len(current_data) if hasattr(current_data, '__len__') else 1,
                'analysis_timestamp': datetime.now().isoformat(),
                'config': config
            }
            
            report = report_generator.generate_report(drift_results, metadata)
            
            return jsonify(report)
            
        except Exception as e:
            logger.error(f"Error in drift report generation: {e}")
            return jsonify({
                'message': f'Drift report generation failed: {str(e)}',
                'status': 'error'
            }), 500
    
    return monitoring_bp


def get_logger(name):
    """Get logger instance"""
    return logging.getLogger(name)

def get_app_middleware():
    """
    Get application middleware for drift detection and reporting.
    This function should return middleware components needed for drift monitoring.
    """
    middleware_components = {
        'data_collector': DataCollector(),
        'drift_detector': DriftDetector(),
        'report_generator': ReportGenerator(),
        'logger': get_logger('drift_monitoring')
    }
    return middleware_components
