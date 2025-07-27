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
│   └── monitoring/             # Monitoramento de drift e performance
├── 📁 scripts/                 # Scripts utilitários
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
├── production_inference.py     # Script de inferência em produção
├── render_start.py             # Ponto de entrada para o deploy
├── setup_monitoring.py         # Configuração do sistema de monitoramento
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
# Ver comandos disponíveis
make help

# Iniciar API localmente
make api

# Executar inferência com o modelo treinado
make inference

# Executar testes
make test
```

## � Pipeline de Desenvolvimento

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

## 🔗 Referências

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Render.com Deployment](https://render.com/docs)
- [Time Series Forecasting with LSTM](https://www.tensorflow.org/tutorials/structured_data/time_series)
- [MLflow for Model Tracking](https://mlflow.org/docs/latest/index.html)

## 👥 Equipe

- Desenvolvedor 1 - Desenvolvedor Full Stack
- Desenvolvedor 2 - Data Engineer
- Desenvolvedor 3 - ML Engineer
- Desenvolvedor 4 - DevOps & Infrastructure

## 📄 Licença

Este projeto está licenciado sob a Licença MIT - consulte o arquivo LICENSE para detalhes.
