# 📡 Documentação Completa da API - Endpoints de Monitoramento Adicionados

## 🎯 API Endpoints Disponíveis

### 📊 **Endpoints Principais**
| Endpoint | Método | Descrição |
|----------|---------|-----------|
| `/` | GET | API information |
| `/health` | GET | Health check |
| `/predict` | POST | Make stock price predictions |
| `/model/info` | GET | Model configuration details |
| `/predictions` | GET | Get saved predictions |

### 🔍 **Endpoints de Monitoramento** ✅
| Endpoint | Método | Descrição |
|----------|---------|-----------|
| `/monitoring` | GET | Monitoring dashboard |
| `/monitoring/dashboard` | GET | Visual monitoring interface |
| `/monitoring/health` | GET | System health status |
| `/monitoring/stats` | GET | Performance statistics |
| `/monitoring/metrics` | GET | Prometheus metrics |
| `/monitoring/drift-report` | GET | Data drift analysis |

## 🚀 Como Usar os Endpoints de Monitoramento

### 1. **Dashboard Visual**
```bash
# Acessar interface web
curl http://localhost:8000/monitoring
# ou
curl http://localhost:8000/monitoring/dashboard
```

### 2. **Métricas de Performance**
```bash
# Estatísticas detalhadas
curl http://localhost:8000/monitoring/stats

# Exemplo de resposta:
{
  "total_predictions": 4,
  "prediction_stats": {
    "avg_duration_ms": 33.024,
    "max_duration_ms": 105.242,
    "min_duration_ms": 2.867,
    "p95_duration_ms": 105.242,
    "total_count": 4
  },
  "system_stats": {
    "avg_cpu_percent": 16.675,
    "avg_memory_percent": 85.9
  }
}
```

### 3. **Status de Saúde do Sistema**
```bash
# Verificar saúde do sistema
curl http://localhost:8000/monitoring/health

# Exemplo de resposta:
{
  "status": "degraded", 
  "checks": {
    "cpu": {
      "status": "healthy",
      "value": 55.7,
      "unit": "%"
    },
    "memory": {
      "status": "warning", 
      "value": 85.8,
      "unit": "%"
    }
  }
}
```

### 4. **Métricas Prometheus**
```bash
# Formato Prometheus para integração
curl http://localhost:8000/monitoring/metrics
```

### 5. **Análise de Data Drift**
```bash
# Relatório de drift detection
curl http://localhost:8000/monitoring/drift-report
```

## 📈 Interpretação dos Status de Saúde

### 🟢 **healthy**: Sistema funcionando normalmente
- CPU < 70%
- Memory < 80%

### 🟡 **degraded**: Sistema com alertas (seu status atual)
- CPU >= 70% ou Memory >= 80%
- **Sua situação**: Memory = 85.8% (acima do limite de 80%)

### 🔴 **critical**: Sistema em estado crítico
- CPU >= 90% ou Memory >= 90%

## 🎯 Agora Seu Sistema Tem:

✅ **Documentação Completa**: Todos os endpoints listados na API  
✅ **Monitoramento Visual**: Dashboard em tempo real  
✅ **Métricas Detalhadas**: Performance e recursos do sistema  
✅ **Alertas de Saúde**: Status baseado em thresholds  
✅ **Data Drift Detection**: Monitoramento de qualidade do modelo  
✅ **Integração Prometheus**: Para ferramentas de observabilidade  

## 🔗 Links Rápidos:

- **API Root**: http://localhost:8000/
- **Dashboard**: http://localhost:8000/monitoring
- **Health Check**: http://localhost:8000/monitoring/health
- **Stats**: http://localhost:8000/monitoring/stats

Seu modelo LSTM agora tem **monitoramento completo e documentado**! 🎉
