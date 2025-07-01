# FIAP Tech Challenge 04

## Description

Modelo preditivo de redes neural Long Short Term Memory (LSTM) para predizer o valor de fechamento da bolsa de valores da ação BBAS3.

## ETL Architecture

O pipeline de dados segue três etapas principais:

### 1 - Extract

Os dados brutos do preço do ativo BBAS3 são obtidos através da biblioteca Yfinance e carregados em um DataFrame do PySpark. Os dados obtidos são persistidos em formato Parquet na camada 'raw'. Durante essa etapa, os dados são transferidos da origem para o ambiente de processamento, sem modificações, assegurando que todos os registros relevantes estejam disponíveis para o processamento subsequente.

### 2 - Transform

Registros com valores ausentes em "Close" são removidos para garantir a integridade dos cálculos. Em seguida, os dados são agrupados por ano e mês para calcular a média mensal do preço do ativo BBAS3. Os dados do mês corrente são excluídos para evitar distorções em análises ou previsões, já que o mês corrente pode ter dados incompletos. Além disso, lags (defasagens) de até 6 meses são criadas para capturar a dependência temporal, e estatísticas móveis como média, desvio padrão, valor mínimo e máximo dos últimos 6 meses são calculadas. Componentes sazonais, como trimestre, também são extraídos para melhor representar as variações temporais no modelo.

### 3 - Load

Os dados transformados e enriquecidos são preparados para uso em análises ou modelagem de machine learning. O DataFrame final, contendo todas as features necessárias, é salvo em formato Parquet na camada 'final'. Esta etapa garante que os dados estejam prontos para ser consumidos por outras aplicações ou processos, proporcionando um formato eficiente e otimizado para processamento adicional.

## Installation

1. Clone the repository:
   ```sh
   git clone <repository-url>
   cd fiap_tech_challenge_04
   ```

2. Install dependencies (if any):
   ```sh
   # e.g., pip install -r requirements.txt
   ```

## Usage

Describe how to use this project.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

Specify license information here.
