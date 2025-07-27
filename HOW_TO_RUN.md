# 🚀 Como Executar o Projeto FIAP Tech Challenge 04

## 📋 Pré-requisitos

- Python 3.11 ou superior
- pip instalado
- Git (opcional)
- Java 17+ (para Apache Spark - ETL)

## ⚡ Execução Rápida

### Método 1: Script Automático (Recomendado)
```bash
# 1. Execute o script de setup
./scripts/setup.sh

# 2. Execute a API
python scripts/run_api_server.py
```

### Método 2: Execução Direta da API
```bash
# 1. Criar ambiente virtual
python3 -m venv .venv

# 2. Ativar ambiente virtual
source .venv/bin/activate  # Linux/Mac
# ou
.venv\Scripts\activate     # Windows

# 3. Instalar dependências
pip install -r requirements.txt

# 4. Executar API (porta 8081 por padrão)
python src/api/main.py

# Ou com script runner
python scripts/run_api_server.py
```

### Método 3: Com Configurações Personalizadas
```bash
# Porta personalizada
PORT=9000 python scripts/run_api_server.py

# Modo debug
DEBUG=true PORT=8081 python scripts/run_api_server.py

# Configurar path do modelo
MODEL_PATH=./custom/model/path python scripts/run_api_server.py
```

## 🌐 Deploy em Produção (Render)

Para deployment na plataforma Render, consulte o guia completo:

```bash
# Guia detalhado de deployment no Render
cat RENDER_DEPLOYMENT_GUIDE.md
```

**Arquivos necessários para Render:**
- `render.yaml` - Configuração do serviço
- `render_start.py` - Entrypoint otimizado para Render  
- `Dockerfile.render` - Container Docker (opcional)

**Deploy rápido:**
1. Conecte seu repositório GitHub ao Render
2. Configure as variáveis de ambiente
3. Use `render.yaml` como blueprint

## 🔄 Pipeline Completo (ETL + Treinamento + API)

### 1. Executar ETL (Extração e Transformação de Dados)
```bash
# ETL com PySpark
cd src/etl
python run_etl.py

# Ou usar o módulo ETL
python -m src.etl.run_etl
```

### 2. Treinar o Modelo LSTM
```bash
# Executar treinamento
python src/ml/main.py

# Com configurações específicas
python src/ml/train_model.py --epochs 200 --batch_size 32
```

### 3. Iniciar API
```bash
# API estará disponível em http://localhost:8081
python scripts/run_api_server.py
```

## 🌐 Testando a API

### 1. Verificar se está funcionando
```bash
# Testar endpoint principal (porta 8081)
curl http://localhost:8081/

# Health check
curl http://localhost:8081/health

# Informações do modelo
curl http://localhost:8081/model/info
```

### 2. Teste de Predição Completo
```bash
curl -X POST http://localhost:8081/predict \
  -H "Content-Type: application/json" \
  -d '{
    "forecast_horizon": 6,
    "data": [
      {
        "preco_medio_close": 29.86,
        "lag_1_mes_preco_medio_close": 31.98,
        "lag_2_mes_preco_medio_close": 36.55,
        "lag_3_mes_preco_medio_close": 32.10,
        "lag_4_mes_preco_medio_close": 27.00,
        "lag_5_mes_preco_medio_close": 24.37,
        "lag_6_mes_preco_medio_close": 20.79,
        "media_movel_6_meses_preco_medio_close": 30.31,
        "desvio_padrao_movel_6_meses_preco_medio_close": 3.90,
        "valor_minimo_6_meses_preco_medio_close": 24.37,
        "valor_maximo_6_meses_preco_medio_close": 36.55
      }
    ]
  }'
```

### 3. Executar Client de Teste
```bash
# Script de teste da API
python scripts/test_api.py
```

### 4. Ver Predições Salvas
```bash
curl http://localhost:8081/predictions
```

## 📊 Executar Notebooks

```bash
# Abrir Jupyter Lab
jupyter lab notebooks/

# Ou Jupyter Notebook
jupyter notebook notebooks/

# Notebook específico do ETL
jupyter lab notebooks/etl_analysis.ipynb
```

## 🛠️ Comandos por Componente

### ETL Pipeline
```bash
# Executar ETL completo
cd src/etl && python run_etl.py

# Verificar dados transformados
ls -la data/transformed/

# Ver dados finais
ls -la data/final/
```

### Machine Learning
```bash
# Treinar modelo LSTM
python src/ml/main.py

# Verificar modelo exportado
ls -la outputs/model_export/

# Ver métricas de treinamento
cat outputs/training_metrics.json
```

### API
```bash
# Iniciar API
python src/api/main.py

# Ou usar script runner
python scripts/run_api_server.py

# Com configurações específicas
MODEL_PATH=./outputs/model_export/ PORT=8081 python scripts/run_api_server.py
```

## 🔧 Configuração Avançada

### Variáveis de Ambiente
```bash
# Configuração da API
export PORT=8081
export DEBUG=false
export MODEL_PATH=./outputs/model_export/
export PREDICTIONS_PATH=./outputs/predictions.csv

# Configuração do Spark (ETL)
export JAVA_HOME=/opt/homebrew/opt/openjdk@17
export SPARK_HOME=/usr/local/spark

# MLflow (se usar)
export MLFLOW_TRACKING_URI=file:./mlruns
```

### Configuração Docker
```bash
# Build da imagem
docker build -t fiap-lstm-api .

# Executar container
docker run -p 8081:8081 -v $(pwd)/outputs:/app/outputs fiap-lstm-api

# Com Docker Compose
docker-compose up --build
```

## 📡 Endpoints da API

| Endpoint | Método | Descrição | Porta |
|----------|--------|-----------|-------|
| `/` | GET | Informações da API | 8081 |
| `/health` | GET | Health check | 8081 |
| `/predict` | POST | Predições LSTM (6 meses) | 8081 |
| `/model/info` | GET | Detalhes do modelo | 8081 |
| `/predictions` | GET | Predições salvas (CSV) | 8081 |

### Formato de Requisição de Predição
```json
{
  "forecast_horizon": 6,
  "data": [
    {
      "preco_medio_close": 29.86,
      "lag_1_mes_preco_medio_close": 31.98,
      "lag_2_mes_preco_medio_close": 36.55,
      "lag_3_mes_preco_medio_close": 32.10,
      "lag_4_mes_preco_medio_close": 27.00,
      "lag_5_mes_preco_medio_close": 24.37,
      "lag_6_mes_preco_medio_close": 20.79,
      "media_movel_6_meses_preco_medio_close": 30.31,
      "desvio_padrao_movel_6_meses_preco_medio_close": 3.90,
      "valor_minimo_6_meses_preco_medio_close": 24.37,
      "valor_maximo_6_meses_preco_medio_close": 36.55
    }
  ]
}
```

## 🐛 Resolução de Problemas

### Erro: Port already in use (comum no macOS)
```bash
# Porta 5000 usado pelo AirPlay Receiver
PORT=8081 python scripts/run_api_server.py

# Ou desabilite AirPlay Receiver em:
# System Preferences > General > AirDrop & Handoff > AirPlay Receiver: Off
```

### Erro: Module not found 'ml'
```bash
# Adicione o projeto ao PYTHONPATH
PYTHONPATH=. python scripts/run_api_server.py

# Ou execute do diretório correto
cd src/api && python main.py
```

### Erro: Model not found
```bash
# 1. Execute o ETL primeiro
cd src/etl && python run_etl.py

# 2. Execute o treinamento
python src/ml/main.py

# 3. Verifique se os arquivos foram criados
ls -la outputs/model_export/
```

### Erro: Java/Spark não encontrado (ETL)
```bash
# macOS com Homebrew
brew install openjdk@17
export JAVA_HOME=/opt/homebrew/opt/openjdk@17

# Ubuntu/Debian
sudo apt-get install openjdk-17-jdk
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
```

### Erro: yfinance/pyspark não instalado
```bash
# Instalar dependências específicas
pip install yfinance pyspark torch pandas scikit-learn flask

# Ou reinstalar requirements
pip install -r requirements.txt --force-reinstall
```

## 📈 Monitoramento e Logs

### Logs da Aplicação
```bash
# Logs da API
tail -f logs/api/api.log

# Logs do ETL
tail -f logs/training/etl.log

# Logs do treinamento
tail -f logs/training/training.log

# Logs de monitoramento
tail -f logs/monitoring/monitoring.log

# Logs de predição
tail -f logs/prediction/prediction.log
```

### Verificar Status do Sistema
```bash
# Verificar processos Python rodando
ps aux | grep python

# Verificar portas em uso
lsof -i :8081
lsof -i :5000
```

### Monitoramento do Modelo

#### 1. Detecção de Drift de Dados
```bash
# Executar detecção de drift nos dados
python scripts/setup_monitoring.py --check-drift

# Ver último relatório de drift
cat outputs/drift_reports/latest_drift_report.json

# Visualizar relatório completo (abre no navegador)
python scripts/setup_monitoring.py --show-report
```

#### 2. Monitoramento de Performance
```bash
# Verificar métricas atuais do modelo
curl http://localhost:8081/metrics

# Comparar performance atual com baseline
python scripts/setup_monitoring.py --performance-check

# Gerar dashboard de métricas
python scripts/setup_monitoring.py --generate-dashboard
```

#### 3. Alertas e Notificações
```bash
# Configurar limites de alerta
python scripts/setup_monitoring.py --configure-alerts \
  --mse-threshold 0.5 \
  --drift-threshold 0.3

# Testar sistema de alertas
python scripts/setup_monitoring.py --test-alerts

# Ver histórico de alertas
cat outputs/monitoring/alert_history.json
```

#### 4. Interface de Monitoramento
```bash
# Iniciar interface web de monitoramento
python scripts/setup_monitoring.py --start-dashboard

# Ou acessar via endpoint da API
curl http://localhost:8081/monitoring/dashboard
```

#### 5. Retraining Automático
```bash
# Configurar critérios para retreinamento
python scripts/setup_monitoring.py --setup-retraining \
  --performance-threshold 0.8 \
  --drift-threshold-critical 0.7

# Verificar status do retreinamento automático
python scripts/setup_monitoring.py --retraining-status

# Forçar retreinamento
python scripts/setup_monitoring.py --force-retraining
```

## 🚀 Deploy em Produção

### 1. Preparar Ambiente de Produção
```bash
# Build Docker
docker build -t fiap-lstm-api:prod .

# Com Gunicorn (recomendado)
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8081 --chdir src/api main:app
```

### 2. Configurar Servidor Web
```bash
# Nginx + Gunicorn
sudo nano /etc/nginx/sites-available/fiap-lstm-api

# Configuração nginx:
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://127.0.0.1:8081;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## ✅ Validação Completa do Sistema

Execute esta sequência para validar todo o pipeline:

```bash
# 1. ETL - Processamento de dados
echo "🔄 Executando ETL..."
cd src/etl && python run_etl.py

# 2. ML - Treinamento do modelo
echo "🧠 Treinando modelo..."
python src/ml/main.py

# 3. API - Inicialização (em background)
echo "🚀 Iniciando API..."
PORT=8081 python scripts/run_api_server.py &
API_PID=$!

# 4. Aguardar inicialização
sleep 15

# 5. Teste completo
echo "🧪 Testando API..."
curl -s http://localhost:8081/health
curl -s http://localhost:8081/model/info

# 6. Parar API
kill $API_PID

echo "✅ Validação completa!"
```

## 🏗️ Estrutura do Projeto Atualizada

```
tech_challenge_04/
├── src/
│   ├── api/                 ← Flask API (Nova estrutura)
│   │   ├── __init__.py
│   │   ├── main.py          ← Aplicação Flask principal
│   │   ├── config.py        ← Configurações da API
│   │   ├── models.py        ← Service do modelo LSTM
│   │   └── utils.py         ← Utilitários da API
│   ├── etl/                 ← Pipeline ETL com PySpark
│   │   ├── __init__.py
│   │   ├── etl_pipeline.py  ← Classe principal ETL
│   │   └── run_etl.py       ← Script executor
│   └── ml/                  ← Machine Learning
│       ├── config.py        ← Configurações do modelo
│       ├── models/          ← Definições dos modelos
│       ├── training/        ← Scripts de treinamento
│       └── main.py          ← Executor principal ML
├── scripts/
│   ├── run_api_server.py    ← Runner da API
│   └── test_api.py          ← Cliente de teste
├── outputs/
│   ├── model_export/        ← Modelo treinado exportado
│   ├── predictions.csv      ← Predições salvas
│   └── *.log               ← Logs do sistema
├── data/                    ← Dados processados (ETL)
│   ├── raw/
│   ├── transformed/
│   └── final/
├── requirements.txt         ← Dependências Python
└── HOW_TO_RUN.md           ← Este arquivo
```

## 🆘 Suporte

**Ordem recomendada para resolução de problemas:**

1. **Verificar logs** em `outputs/` e arquivos `.log`
2. **Confirmar estrutura** - API deve estar em `src/api/`
3. **Testar componentes isoladamente** - ETL → ML → API
4. **Verificar dependências** - `pip list | grep -E "(torch|flask|pyspark)"`
5. **Validar arquivos do modelo** - `ls -la outputs/model_export/`

**Contatos:**
- 📧 Logs detalhados estão em `lstm_api.log`
- 🐛 Issues conhecidos em comentários do código
- 📚 Documentação técnica nos notebooks em `notebooks/`
