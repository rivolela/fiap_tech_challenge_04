# 🚀 Como Executar o Projeto FIAP Tech Challenge 04

## 📋 Pré-requisitos

- Python 3.11 ou superior
- pip ou Poetry instalado
- Git (opcional)

## ⚡ Execução Rápida

### Método 1: Script Automático (Recomendado)
```bash
# 1. Execute o script de setup
./setup.sh

# 2. Execute o projeto
python main.py
```

### Método 2: Makefile (Mais rápido)
```bash
# Ver todos os comandos disponíveis
make help

# Setup + Executar
make dev

# Ou separadamente:
make setup    # Configurar ambiente
make run      # Executar aplicação
```

### Método 3: Manual
```bash
# 1. Criar ambiente virtual
python3 -m venv .venv

# 2. Ativar ambiente virtual
source .venv/bin/activate  # Linux/Mac
# ou
.venv\Scripts\activate     # Windows

# 3. Instalar dependências
pip install -r requirements.txt

# 4. Executar aplicação
python main.py
```

## 🌐 Testando a API

### 1. Verificar se está funcionando
```bash
# Testar endpoint principal
curl http://localhost:5000/

# Health check
curl http://localhost:5000/health
```

### 2. Executar testes automatizados
```bash
# Método 1: Make
make test

# Método 2: Direto
python tests/test_api.py

# Método 3: Com pytest (se instalado)
pytest tests/
```

### 3. Testar predição
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      {"feature1": 1.0, "feature2": 2.0},
      {"feature1": 1.5, "feature2": 2.5}
    ]
  }'
```

## 📊 Executar Notebooks

```bash
# Método 1: Make
make notebook

# Método 2: Direto
jupyter lab notebooks/

# Método 3: Jupyter Notebook
jupyter notebook notebooks/
```

## 🐳 Executar com Docker

### Docker Compose (Recomendado)
```bash
# Método 1: Make
make docker-compose-up

# Método 2: Direto
cd configs/
docker-compose up
```

### Docker Manual
```bash
# Build da imagem
make docker-build

# Executar container
make docker-run
```

## 🛠️ Comandos Úteis

### Makefile Commands
```bash
make help           # Ver todos os comandos
make setup          # Configurar ambiente
make install        # Instalar dependências
make run            # Executar aplicação
make test           # Executar testes
make notebook       # Abrir Jupyter
make etl            # Executar ETL
make train          # Treinar modelo
make sample-data    # Criar dados exemplo
make clean          # Limpar temporários
make lint           # Verificar código
make format         # Formatar código
make info           # Informações do projeto
```

### Scripts Disponíveis
```bash
# Criar dados de exemplo
python scripts/create_sample_data.py

# Demo de export do modelo
python scripts/demo_model_export.py

# ETL pipeline
python -m src.etl.main

# Treinar modelo
python src/ml/main.py
```

## 🔧 Configuração Avançada

### Variáveis de Ambiente
```bash
# Porta personalizada
PORT=8080 python main.py

# Modo debug
DEBUG=true python main.py

# Configurar MLflow
export MLFLOW_TRACKING_URI=file:./mlruns
```

### Configuração Poetry
```bash
# Se preferir usar Poetry
poetry install
poetry shell
poetry run python main.py
```

## 📡 Endpoints da API

| Endpoint | Método | Descrição |
|----------|--------|-----------|
| `/` | GET | Home page |
| `/health` | GET | Health check |
| `/predict` | POST | Fazer predições |
| `/model/info` | GET | Informações do modelo |
| `/metrics` | GET | Métricas da aplicação |

## 🐛 Resolução de Problemas

### Erro: Port already in use
```bash
# Use uma porta diferente
PORT=8081 python main.py
```

### Erro: Module not found
```bash
# Certifique-se que o ambiente virtual está ativo
source .venv/bin/activate

# Reinstale as dependências
pip install -r requirements.txt
```

### Erro: Model not found
```bash
# Execute o treinamento primeiro
python src/ml/main.py

# Ou crie dados de exemplo
python scripts/create_sample_data.py
```

### Erro: Permission denied setup.sh
```bash
chmod +x setup.sh
```

## 📈 Monitoramento

### Logs
```bash
# Logs são salvos em outputs/
tail -f outputs/*.log

# Ver logs da aplicação
tail -f lstm_api.log
```

### MLflow
```bash
# Abrir interface MLflow
mlflow ui --backend-store-uri ./mlruns
```

## 🚀 Deploy em Produção

### 1. Preparar ambiente
```bash
make prod  # Build e run com Docker
```

### 2. Configurar servidor
```bash
# Com Gunicorn (recomendado)
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 main:app

# Com uWSGI
pip install uwsgi
uwsgi --http 0.0.0.0:5000 --module main:app
```

## ✅ Validação Completa

Execute esta sequência para verificar se tudo está funcionando:

```bash
# 1. Setup
make setup

# 2. Executar aplicação (em terminal separado)
make run &

# 3. Aguardar inicialização (30 segundos)
sleep 30

# 4. Executar testes
make test

# 5. Verificar endpoints
curl http://localhost:5000/health
```

## 🆘 Suporte

Se encontrar problemas:

1. Verifique os logs em `outputs/`
2. Confirme que todas as dependências estão instaladas
3. Certifique-se que o Python 3.11+ está sendo usado
4. Verifique se a estrutura de pastas está correta

**Estrutura esperada:**
```
fiap_tech_challenge_04/
├── main.py              ← Ponto de entrada
├── app/api/             ← API Flask
├── src/                 ← Código ML
├── tests/               ← Testes
├── Makefile            ← Comandos automatizados
└── setup.sh            ← Script de configuração
```
