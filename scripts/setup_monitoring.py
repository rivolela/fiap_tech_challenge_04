#!/usr/bin/env python3
"""
Script de Configuração e Teste do Sistema de Monitoramento
===========================================================

Este script configura e testa o sistema de monitoramento do modelo LSTM.
"""

import os
import sys
import time
import json
import requests
import subprocess
import threading
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.monitoring import performance_monitor

performance_monitor.start_monitoring()

def install_monitoring_dependencies():
    """Instala as dependências de monitoramento"""
    print("📦 Instalando dependências de monitoramento...")
    
    dependencies = [
        "prometheus-client==0.19.0",
        "psutil==5.9.0", 
        "structlog==23.1.0",
        "colorlog==6.7.0"
    ]
    
    for dep in dependencies:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", dep], 
                         check=True, capture_output=True)
            print(f"✅ {dep} instalado com sucesso")
        except subprocess.CalledProcessError as e:
            print(f"❌ Erro ao instalar {dep}: {e}")
            return False
    
    return True

def test_monitoring_system():
    """Testa o sistema de monitoramento"""
    print("\n🔧 Testando sistema de monitoramento...")
    
    try:
        # Test imports
        from src.monitoring import performance_monitor
        from src.monitoring.integrations.flask_middleware import MonitoringMiddleware
        print("✅ Imports de monitoramento funcionando")
        
        # Test monitor initialization
        import numpy as np
        import pandas as pd
        
        # Create sample data
        sample_data = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0],
            'feature2': [4.0, 5.0, 6.0]
        })
        
        # Record a prediction
        import numpy as np
        metrics = performance_monitor.record_prediction(
            input_data=sample_data,
            prediction=np.array([10.5, 11.2, 12.1]),
            duration_ms=100.5,
            confidence=0.85
        )
        print("✅ Registro de métricas funcionando")
        
        # Test metrics retrieval
        recent_metrics = performance_monitor.get_recent_metrics(10)
        print(f"✅ Métricas recentes obtidas: {len(recent_metrics)} registros")
        
        # Test system metrics
        from src.monitoring.core.system import SystemMonitor
        system_monitor = SystemMonitor()
        system_metrics = system_monitor.collect_metrics()
        print(f"✅ Métricas do sistema obtidas: CPU={system_metrics.cpu_percent:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro no teste de monitoramento: {e}")
        return False

def create_monitoring_config():
    """Cria arquivo de configuração para monitoramento"""
    config = {
        "monitoring": {
            "enabled": True,
            "metrics_collection_interval": 5,
            "max_history_size": 1000,
            "prometheus_port": 8000,
            "health_check_thresholds": {
                "cpu_warning": 80,
                "cpu_critical": 90,
                "memory_warning": 80,
                "memory_critical": 90,
                "response_time_warning": 2000,
                "response_time_critical": 5000
            },
            "retention_days": 7
        },
        "logging": {
            "level": "INFO",
            "structured": True,
            "include_request_id": True
        }
    }
    
    config_path = os.path.join(project_root, "monitoring_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"📝 Configuração salva em {config_path}")
    return config_path

def create_dashboard_html():
    """Cria um dashboard HTML simples para monitoramento"""
    html_content = """
<!DOCTYPE html>
<html lang="pt-br">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LSTM API - Dashboard de Monitoramento</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { background: #2c3e50; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .card { background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .metric { display: inline-block; margin: 10px; padding: 15px; background: #ecf0f1; border-radius: 5px; min-width: 150px; }
        .metric-value { font-size: 24px; font-weight: bold; color: #2c3e50; }
        .metric-label { font-size: 12px; color: #7f8c8d; }
        .status-healthy { color: #27ae60; }
        .status-warning { color: #f39c12; }
        .status-critical { color: #e74c3c; }
        .chart-container { width: 100%; height: 300px; margin: 20px 0; }
        button { padding: 10px 20px; margin: 5px; border: none; border-radius: 4px; cursor: pointer; }
        .btn-primary { background: #3498db; color: white; }
        .btn-success { background: #27ae60; color: white; }
        .refresh-info { font-size: 12px; color: #7f8c8d; margin-left: 10px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 LSTM API - Dashboard de Monitoramento</h1>
            <p>Monitoramento em tempo real do modelo de predição de preços de commodities</p>
        </div>
        
        <div class="card">
            <h2>🏥 Status do Sistema</h2>
            <div id="system-status">Carregando...</div>
            <button class="btn-primary" onclick="refreshData()">🔄 Atualizar</button>
            <span class="refresh-info">Última atualização: <span id="last-update">-</span></span>
        </div>
        
        <div class="card">
            <h2>📊 Métricas de Performance</h2>
            <div id="performance-metrics">Carregando...</div>
        </div>
        
        <div class="card">
            <h2>🖥️ Recursos do Sistema</h2>
            <div id="system-metrics">Carregando...</div>
        </div>
        
        <div class="card">
            <h2>📈 Gráfico de Tempo de Resposta</h2>
            <div class="chart-container">
                <canvas id="responseTimeChart"></canvas>
            </div>
        </div>
        
        <div class="card">
            <h2>🔧 Ações Rápidas</h2>
            <button class="btn-success" onclick="exportMetrics()">📁 Exportar Métricas</button>
            <button class="btn-primary" onclick="viewLogs()">📋 Ver Logs</button>
            <button class="btn-primary" onclick="testPrediction()">🧪 Teste de Predição</button>
        </div>
        
        <div class="card">
            <h2>📝 Log de Eventos</h2>
            <div id="event-log" style="height: 200px; overflow-y: scroll; background: #f8f9fa; padding: 10px; font-family: monospace; font-size: 12px;">
                Aguardando eventos...
            </div>
        </div>
    </div>

    <script>
        let responseTimeChart;
        
        // Configurar gráfico
        function initChart() {
            const ctx = document.getElementById('responseTimeChart').getContext('2d');
            responseTimeChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Tempo de Resposta (ms)',
                        data: [],
                        borderColor: '#3498db',
                        backgroundColor: 'rgba(52, 152, 219, 0.1)',
                        tension: 0.1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: 'Tempo (ms)'
                            }
                        },
                        x: {
                            title: {
                                display: true,
                                text: 'Tempo'
                            }
                        }
                    }
                }
            });
        }
        
        function addLogEvent(message, type = 'info') {
            const log = document.getElementById('event-log');
            const timestamp = new Date().toLocaleTimeString();
            const icon = type === 'error' ? '❌' : type === 'warning' ? '⚠️' : '✅';
            log.innerHTML += `[${timestamp}] ${icon} ${message}\\n`;
            log.scrollTop = log.scrollHeight;
        }
        
        async function refreshData() {
            try {
                addLogEvent('Atualizando dados...');
                
                // Buscar health status
                const healthResponse = await fetch('/monitoring/health');
                const healthData = await healthResponse.json();
                updateSystemStatus(healthData);
                
                // Buscar stats
                const statsResponse = await fetch('/monitoring/stats');
                const statsData = await statsResponse.json();
                updatePerformanceMetrics(statsData);
                
                // Buscar métricas recentes
                const recentResponse = await fetch('/monitoring/recent?minutes=5');
                const recentData = await recentResponse.json();
                updateChart(recentData);
                
                document.getElementById('last-update').textContent = new Date().toLocaleTimeString();
                addLogEvent('Dados atualizados com sucesso');
                
            } catch (error) {
                addLogEvent(`Erro ao atualizar dados: ${error.message}`, 'error');
            }
        }
        
        function updateSystemStatus(health) {
            const container = document.getElementById('system-status');
            const statusClass = `status-${health.status === 'healthy' ? 'healthy' : 
                                health.status === 'degraded' ? 'warning' : 'critical'}`;
            
            let html = `<div class="metric">
                <div class="metric-value ${statusClass}">${health.status.toUpperCase()}</div>
                <div class="metric-label">Status Geral</div>
            </div>`;
            
            if (health.checks) {
                Object.entries(health.checks).forEach(([key, check]) => {
                    const checkClass = `status-${check.status === 'healthy' ? 'healthy' : 
                                      check.status === 'warning' ? 'warning' : 'critical'}`;
                    html += `<div class="metric">
                        <div class="metric-value ${checkClass}">${check.status}</div>
                        <div class="metric-label">${key.toUpperCase()}</div>
                    </div>`;
                });
            }
            
            container.innerHTML = html;
        }
        
        function updatePerformanceMetrics(stats) {
            const container = document.getElementById('performance-metrics');
            let html = '';
            
            if (stats.prediction_stats) {
                const ps = stats.prediction_stats;
                html += `
                    <div class="metric">
                        <div class="metric-value">${ps.avg_duration_ms?.toFixed(1) || 'N/A'}</div>
                        <div class="metric-label">Tempo Médio (ms)</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">${ps.p95_duration_ms?.toFixed(1) || 'N/A'}</div>
                        <div class="metric-label">P95 Tempo (ms)</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">${stats.total_predictions || 0}</div>
                        <div class="metric-label">Total Predições</div>
                    </div>
                `;
            }
            
            if (stats.system_stats) {
                const ss = stats.system_stats;
                html += `
                    <div class="metric">
                        <div class="metric-value">${ss.avg_cpu_percent?.toFixed(1) || 'N/A'}%</div>
                        <div class="metric-label">CPU Média</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">${ss.avg_memory_percent?.toFixed(1) || 'N/A'}%</div>
                        <div class="metric-label">Memória Média</div>
                    </div>
                `;
            }
            
            container.innerHTML = html || '<p>Sem dados disponíveis</p>';
        }
        
        function updateChart(recentData) {
            if (!recentData.predictions || recentData.predictions.length === 0) return;
            
            const predictions = recentData.predictions.slice(-20); // Últimas 20
            const labels = predictions.map(p => new Date(p.timestamp).toLocaleTimeString());
            const data = predictions.map(p => p.duration_ms);
            
            responseTimeChart.data.labels = labels;
            responseTimeChart.data.datasets[0].data = data;
            responseTimeChart.update();
        }
        
        async function exportMetrics() {
            try {
                const response = await fetch('/monitoring/metrics');
                const data = await response.text();
                
                const blob = new Blob([data], { type: 'text/plain' });
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `metrics_${new Date().toISOString().slice(0,10)}.txt`;
                a.click();
                
                addLogEvent('Métricas exportadas com sucesso');
            } catch (error) {
                addLogEvent(`Erro ao exportar métricas: ${error.message}`, 'error');
            }
        }
        
        function viewLogs() {
            addLogEvent('Função de visualização de logs não implementada');
        }
        
        async function testPrediction() {
            try {
                addLogEvent('Executando teste de predição...');
                
                const testData = {
                    "historical_data": [
                        {
                            "preco_medio_close": 100.0,
                            "lag_1_mes_preco_medio_close": 99.0,
                            "lag_2_mes_preco_medio_close": 98.0,
                            "lag_3_mes_preco_medio_close": 97.0,
                            "lag_4_mes_preco_medio_close": 96.0,
                            "lag_5_mes_preco_medio_close": 95.0,
                            "lag_6_mes_preco_medio_close": 94.0,
                            "media_movel_6_meses_preco_medio_close": 97.0,
                            "desvio_padrao_movel_6_meses_preco_medio_close": 2.0,
                            "valor_minimo_6_meses_preco_medio_close": 94.0,
                            "valor_maximo_6_meses_preco_medio_close": 100.0
                        }
                    ],
                    "forecast_horizon": 6
                };
                
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(testData)
                });
                
                if (response.ok) {
                    addLogEvent('Teste de predição executado com sucesso');
                } else {
                    addLogEvent(`Teste de predição falhou: ${response.status}`, 'error');
                }
                
            } catch (error) {
                addLogEvent(`Erro no teste de predição: ${error.message}`, 'error');
            }
        }
        
        // Inicializar
        window.onload = function() {
            initChart();
            refreshData();
            
            // Auto-refresh a cada 30 segundos
            setInterval(refreshData, 30000);
        };
    </script>
</body>
</html>
    """
    
    dashboard_path = os.path.join(project_root, "monitoring_dashboard.html")
    with open(dashboard_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"📊 Dashboard criado em {dashboard_path}")
    return dashboard_path

def create_deployment_guide():
    """Cria guia de deployment com monitoramento"""
    guide_content = """
# Guia de Deployment com Monitoramento - LSTM API

## 📊 Sistema de Monitoramento

### Recursos Implementados

#### 1. Métricas de Performance
- **Tempo de resposta**: P50, P95, P99
- **Latência de predição**: Medição detalhada
- **Throughput**: Requests por segundo
- **Taxa de erro**: Monitoramento de falhas

#### 2. Recursos do Sistema
- **CPU**: Utilização média e picos
- **Memória**: Uso e disponibilidade
- **GPU**: Utilização e memória (se disponível)
- **Disco**: Espaço disponível

#### 3. Métricas do Modelo
- **Qualidade das predições**: Confiança e variância
- **Drift detection**: Monitoramento de degradação
- **Volume de predições**: Contadores detalhados

### Endpoints de Monitoramento

```
GET /monitoring/metrics     # Métricas Prometheus
GET /monitoring/stats       # Estatísticas de performance
GET /monitoring/recent      # Métricas recentes
GET /monitoring/health      # Health check detalhado
GET /monitoring/dashboard   # Dashboard web
```

### Configuração do Prometheus

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'lstm-api'
    static_configs:
      - targets: ['localhost:5000']
    metrics_path: '/monitoring/metrics'
    scrape_interval: 5s
```

### Configuração do Grafana

#### Datasource
- URL: http://localhost:9090 (Prometheus)
- Access: Server

#### Queries Úteis
```promql
# Tempo médio de resposta
rate(lstm_api_request_duration_seconds_sum[5m]) / rate(lstm_api_request_duration_seconds_count[5m])

# Taxa de erro
rate(lstm_api_requests_total{status!~"2.."}[5m]) / rate(lstm_api_requests_total[5m])

# Uso de CPU
lstm_api_cpu_usage_percent

# Uso de memória
lstm_api_memory_usage_percent

# Predições por minuto
rate(lstm_predictions_total[1m]) * 60
```

### Alertas Recomendados

#### 1. Performance
- **Latência alta**: > 2 segundos
- **Taxa de erro alta**: > 5%
- **Predições falhando**: > 10% em 5 min

#### 2. Recursos
- **CPU alto**: > 80% por 5 min
- **Memória alta**: > 85% por 5 min
- **Disco cheio**: > 90%

#### 3. Modelo
- **Confiança baixa**: < 0.7 consistentemente
- **Sem predições**: 0 predições em 10 min

### Deployment em Produção

#### Docker Compose com Monitoramento
```yaml
version: '3.8'
services:
  lstm-api:
    build: .
    ports:
      - "5000:5000"
    environment:
      - MONITORING_ENABLED=true
      - LOG_LEVEL=INFO
    
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    
  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
```

#### Render.com
```yaml
# render.yaml
services:
  - type: web
    name: lstm-api
    env: python
    buildCommand: "pip install -r requirements.txt"
    startCommand: "python render_start.py"
    envVars:
      - key: MONITORING_ENABLED
        value: "true"
      - key: LOG_LEVEL
        value: "INFO"
```

### Logs Estruturados

O sistema gera logs estruturados em JSON:

```json
{
  "timestamp": "2024-01-20T10:30:00Z",
  "level": "INFO",
  "service": "lstm-api",
  "endpoint": "/predict",
  "duration_ms": 150.5,
  "status_code": 200,
  "input_size": 24,
  "output_size": 6,
  "memory_mb": 256.7,
  "request_id": "req_123456"
}
```

### Dashboard Web

Acesse o dashboard em: `http://your-api-url/monitoring_dashboard.html`

Recursos:
- ✅ Status do sistema em tempo real
- 📊 Gráficos de performance
- 🔄 Auto-refresh a cada 30s
- 📁 Export de métricas
- 🧪 Teste de predição

### Troubleshooting

#### Problema: Métricas não aparecem
- Verificar se `prometheus-client` está instalado
- Confirmar endpoint `/monitoring/metrics` acessível
- Checar logs da aplicação

#### Problema: Dashboard não carrega
- Verificar se todos os endpoints de monitoramento funcionam
- Confirmar JavaScript não está bloqueado
- Checar console do browser para erros

#### Problema: Alertas não funcionam
- Verificar configuração do Prometheus
- Confirmar rules de alerting
- Testar queries manualmente

### Scripts Úteis

```bash
# Testar métricas
curl http://localhost:5000/monitoring/metrics

# Verificar health
curl http://localhost:5000/monitoring/health

# Exportar métricas
curl http://localhost:5000/monitoring/stats > stats.json

# Monitoramento contínuo
watch -n 5 "curl -s http://localhost:5000/monitoring/health | jq '.status'"
```

### Performance Tuning

#### Otimizações
1. **Cache de métricas**: 30s por padrão
2. **Histórico limitado**: 1000 entradas máximo
3. **Coleta assíncrona**: Thread separada para sistema
4. **Sampling de GPU**: Apenas se disponível

#### Configurações Avançadas
```python
# Personalizar monitor
from src.monitoring import performance_monitor

performance_monitor.max_history = 2000  # Mais histórico
performance_monitor._cache_duration = 60  # Cache mais longo
```

### Monitoramento de Custos

Para deployment em cloud:
- Monitore uso de CPU/Memória para sizing adequado
- Acompanhe número de requests para billing
- Configure alertas de custo baseados em volume

### Próximos Passos

1. **APM Integration**: OpenTelemetry, DataDog, New Relic
2. **Distributed Tracing**: Para microserviços
3. **Custom Metrics**: Métricas específicas do domínio
4. **Anomaly Detection**: Detecção automática de problemas
"""
    
    guide_path = os.path.join(project_root, "docs", "MONITORING_GUIDE.md")
    os.makedirs(os.path.dirname(guide_path), exist_ok=True)
    
    with open(guide_path, 'w', encoding='utf-8') as f:
        f.write(guide_content)
    
    print(f"📚 Guia de monitoramento criado em {guide_path}")
    return guide_path

def main():
    """Função principal"""
    print("🚀 Configurando Sistema de Monitoramento para LSTM API")
    print("=" * 60)
    
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description="Configurar e testar sistema de monitoramento")
    parser.add_argument("--test-monitoring", action="store_true", help="Testar sistema de monitoramento")
    args = parser.parse_args()
    
    # Run test monitoring if requested
    if args.test_monitoring:
        print("🧪 Executando testes do sistema de monitoramento...")
        success = test_monitoring_system()
        print(f"{'✅ Testes concluídos com sucesso!' if success else '❌ Falha nos testes do sistema'}")
        return
    
    # 1. Instalar dependências
    if not install_monitoring_dependencies():
        print("❌ Falha na instalação de dependências")
        return
    
    # 2. Testar sistema
    if not test_monitoring_system():
        print("❌ Falha nos testes do sistema")
        return
    
    # 3. Criar configurações
    config_path = create_monitoring_config()
    dashboard_path = create_dashboard_html()
    guide_path = create_deployment_guide()
    
    print("\n✅ Sistema de Monitoramento Configurado com Sucesso!")
    print("=" * 60)
    print(f"📁 Configuração: {config_path}")
    print(f"📊 Dashboard: {dashboard_path}")
    print(f"📚 Guia: {guide_path}")
    print("\n🎯 Próximos Passos:")
    print("1. Execute a API: python render_start.py")
    print("2. Acesse o dashboard: http://localhost:5000/monitoring/dashboard")
    print("3. Teste predição: http://localhost:5000/predict")
    print("4. Monitore métricas: http://localhost:5000/monitoring/metrics")
    print("\n📊 Para monitoramento avançado, configure Prometheus + Grafana")
    print("📖 Consulte o guia em docs/MONITORING_GUIDE.md")

if __name__ == "__main__":
    main()

def test_prediction():
    # Exemplo: simule uma entrada e chame o modelo
    input_data = ...  # Monte um input válido (ex: pd.DataFrame({...}))
    prediction = seu_modelo.predict(input_data)
    performance_monitor.record_prediction(
        input_data=input_data,
        prediction=prediction,
        duration_ms=...,  # calcule o tempo
        confidence=...,   # se aplicável
        metadata={"auto_test": True}
    )

test_prediction()
