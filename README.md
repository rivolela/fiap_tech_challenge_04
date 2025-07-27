# FIAP Tech Challenge 04

## 📊 Descrição do Desafio

Este projeto implementa um modelo preditivo de redes neurais Long Short Term Memory (LSTM) para predizer o valor de fechamento da bolsa de valores da ação BBAS3 (Banco do Brasil). A solução abrange todo o ciclo de vida de um projeto de Machine Learning, desde a coleta e processamento dos dados até a implementação de uma API que disponibiliza as previsões em produção, com monitoramento contínuo.

## 🎯 Objetivos do Projeto

Desenvolver uma solução end-to-end para previsão de preços de ações utilizando técnicas avançadas de Deep Learning, com os seguintes componentes:

1. Coleta e pré-processamento de dados históricos de ações
2. Desenvolvimento e treinamento de um modelo LSTM para previsões de séries temporais
3. Implementação de uma API para disponibilizar previsões em tempo real
4. Sistema de monitoramento para detecção de drift nos dados e qualidade das previsões
5. Infraestrutura escalável para lidar com diferentes volumes de solicitações

## 📁 Estrutura do Projeto

```
fiap_tech_challenge_04/
├── 📁 src/                     # Código fonte do projeto
│   ├── api/                    # Endpoints da API
│   ├── etl/                    # Processamento de dados
│   ├── ml/                     # Lógica de Machine Learning
│   ├── monitoring/             # Monitoramento de drift e performance
│   └── utils/                  # Utilitários compartilhados
├── 📁 scripts/                 # Scripts utilitários
│   ├── production_inference.py # Script de inferência em produção
│   ├── setup_monitoring.py    # Configuração do sistema de monitoramento
│   └── outros scripts         # Outros scripts utilitários
├── 📁 tests/                   # Testes automatizados
├── 📁 docs/                    # Documentação do projeto
├── 📁 data/                    # Armazenamento de dados
│   ├── raw/                    # Dados brutos
│   ├── transformed/            # Dados processados
│   └── final/                  # Dados prontos para modelagem
├── 📁 outputs/                 # Resultados e exportações
│   ├── model_export/           # Modelo exportado
│   └── drift_reports/          # Relatórios de monitoramento
├── 📁 notebooks/               # Jupyter notebooks para experimentação
├── render_start.py             # Ponto de entrada para o deploy
├── 📁 logs/                    # Arquivos de log organizados por categoria
│   ├── api/                    # Logs da API
│   ├── training/               # Logs de treinamento
│   ├── monitoring/             # Logs de monitoramento
│   └── prediction/             # Logs de predição
├── Dockerfile.render           # Docker para deploy
└── requirements.txt            # Dependências do projeto
```

## 🚀 Quick Start

### Configuração do Ambiente
```bash
# Instalar dependências
pip install -r requirements.txt

# Configurar ambiente de execução
python setup.py
```

### Execução
```bash
# Iniciar API localmente
python scripts/run_api_server.py

# Executar inferência com o modelo treinado
python scripts/production_inference.py

# Executar testes
python -m pytest tests/

# Iniciar API em modo de produção (utilizado no Render.com)
python render_start.py
```

## 📊 Pipeline de Desenvolvimento

### 1. Coleta e Pré-processamento dos Dados

O pipeline de dados segue três etapas principais:

#### 1.1 Extract
Os dados históricos da ação BBAS3 são obtidos através da biblioteca Yfinance e carregados em um DataFrame do PySpark. Os dados são armazenados em formato Parquet na camada 'raw', preservando todos os dados originais sem modificações.

#### 1.2 Transform
Nesta etapa realizamos:
- Remoção de registros com valores ausentes
- Agregação por mês para calcular o preço médio mensal
- Criação de features temporais (lags de até 6 meses)
- Cálculo de estatísticas móveis (média, desvio padrão, min, max)
- Extração de componentes sazonais (trimestre, mês)
- Normalização dos dados para treino do modelo LSTM

#### 1.3 Load
Os dados processados são salvos em formato Parquet na camada 'final', prontos para consumo pelo modelo de Machine Learning.

#### 1.4 ETL Pipeline
```bash
# Executar ETL completo
cd src/etl && python run_etl.py

# Verificar dados transformados
ls -la data/transformed/

# Ver dados finais
ls -la data/final/
```

### 2. Desenvolvimento do Modelo LSTM

#### 2.1 Arquitetura do Modelo
O modelo LSTM foi desenvolvido utilizando PyTorch, com a seguinte arquitetura:
- Camada de entrada com dimensões adequadas para as séries temporais
- Múltiplas camadas LSTM para capturar dependências temporais de curto e longo prazo
- Camadas de dropout para evitar overfitting
- Camada de saída para previsão dos próximos 6 meses de preços

#### 2.2 Treinamento
- Split dos dados em conjuntos de treino, validação e teste
- Otimização utilizando Adam com learning rate adaptativo
- Early stopping para evitar overfitting
- Monitoramento de métricas como MSE, MAE e MAPE durante o treinamento

#### 2.3 Avaliação
- Avaliação em dados não vistos pelo modelo
- Métricas de performance: RMSE, MAE, MAPE
- Análise visual de previsões vs. valores reais
- Teste de robustez em diferentes cenários de mercado

#### 2.4 Machine Learning
```bash
# Treinar modelo LSTM
python src/ml/main.py

# Verificar modelo exportado
ls -la outputs/model_export/

# Ver métricas de treinamento
cat outputs/training_metrics.json
```

### 3. Salvamento e Exportação do Modelo

O modelo treinado é salvo em formato otimizado para produção:
- Arquivos PyTorch (.pth) contendo os pesos e a arquitetura
- Serialização do scaler para normalização consistente dos dados
- Exportação de metadados do modelo (features, hiperparâmetros)
- Configurações necessárias para inferência em ambiente de produção



### 4. Deploy do Modelo

#### 4.1 API RESTful
- Implementação de uma API Flask que expõe endpoints para previsão
- Documentação com Swagger/OpenAPI
- Sistema de autenticação e rate limiting
- Logging e tratamento de erros

#### 4.2 Infraestrutura
- Containerização com Docker
- Deploy no Render.com para alta disponibilidade
- Cache para otimizar requisições repetidas
- Balanceamento de carga para escalabilidade

### 5. Escalabilidade e Monitoramento

#### 5.1 Sistema de Monitoramento
- Detecção de data drift usando métodos estatísticos
- Monitoramento da qualidade das previsões ao longo do tempo
- Alertas automáticos quando a performance degrada
- Dashboards com métricas operacionais e de negócio

#### 5.2 Estratégia de Atualização
- Retraining agendado com dados mais recentes
- Testes A/B para validação de novos modelos
- Rollback automático em caso de degradação de performance
- Versionamento de modelos para rastreabilidade

## 📊 Resultados e Métricas

O modelo LSTM desenvolvido apresentou os seguintes resultados:
- RMSE: 1.24 para previsões de 6 meses
- MAE: 0.98 para previsões de 6 meses
- Precisão de direção (alta/baixa): 68%

## 🌐 API Endpoints e Documentação

A API está disponível em produção através do seguinte endpoint:
```
https://fiap-tech-challenge-04.onrender.com
```

### Endpoints Disponíveis

| Endpoint | Método | Descrição | Parâmetros | Resposta |
|----------|--------|-----------|------------|----------|
| `/predict` | GET | Obtém previsão para os próximos 6 meses | Nenhum | JSON com valores previstos para cada mês |
| `/predict/custom` | POST | Previsão personalizada baseada em parâmetros | `{"ticker": "string", "months": integer, "features": {}}` | JSON com previsões personalizadas |
| `/health` | GET | Verificação de saúde da API | Nenhum | Status da API e do modelo |
| `/metrics` | GET | Métricas de performance do modelo | Nenhum | JSON com métricas atuais |
| `/drift/report` | GET | Último relatório de drift de dados | `?days=30` (opcional) | JSON com análise de drift |

### Exemplos de Uso

```bash
# Obter previsão padrão
curl https://fiap-tech-challenge-04.onrender.com/predict

# Obter previsão personalizada
curl -X POST https://fiap-tech-challenge-04.onrender.com/predict/custom \
  -H "Content-Type: application/json" \
  -d '{"ticker": "BBAS3", "months": 3}'

# Verificar saúde da API
curl https://fiap-tech-challenge-04.onrender.com/health
```

### Autenticação

A API utiliza autenticação por token. Para acessar endpoints protegidos, inclua no cabeçalho HTTP:

```
Authorization: Bearer {seu_token_api}
```

Tokens de acesso podem ser solicitados através do contato com a equipe de desenvolvimento.

## 🔗 Referências

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Render.com Deployment](https://render.com/docs)
- [Time Series Forecasting with LSTM](https://www.tensorflow.org/tutorials/structured_data/time_series)
- [MLflow for Model Tracking](https://mlflow.org/docs/latest/index.html)

## 👥 Equipe

- Romeu Ivolela Neto - Desenvolvedor Full Stack, Data Engineer, ML Engineer and DevOps & Infrastructure

## 📄 Licença

Este projeto está licenciado sob a Licença MIT - consulte o arquivo LICENSE para detalhes.
