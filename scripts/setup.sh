#!/bin/bash
# Script de setup para o projeto LSTM Stock Prediction
# Configura o ambiente e dependências necessárias

echo "🚀 Configurando ambiente para LSTM Stock Prediction..."

# Instalar dependências do Python
echo "📦 Instalando dependências..."
pip install -r ../requirements.txt

# Criar diretórios necessários
echo "📂 Criando diretórios necessários..."
mkdir -p ../logs/{api,training,monitoring,prediction}
mkdir -p ../outputs/model_export
mkdir -p ../outputs/drift_reports

echo "✅ Setup completo! O ambiente está pronto para uso."
echo ""
echo "Para executar a API: python scripts/run_api_server.py"
echo "Para fazer previsões: python scripts/production_inference.py"
echo "Para gerenciar logs: python scripts/manage_logs.py --help"
