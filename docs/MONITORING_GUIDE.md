
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
