#!/bin/bash
# Container entrypoint script para iniciar a API e gerenciar logs

# Configurar estrutura de logs
echo "📂 Configurando estrutura de logs..."
python scripts/manage_logs.py --setup

# Configurar tarefa cron para limpeza de logs (se houver suporte a cron)
if command -v cron &> /dev/null; then
    echo "🧹 Configurando limpeza automática de logs..."
    # Adicionar job para limpar logs antigos diariamente
    echo "0 1 * * * cd /app && python scripts/manage_logs.py --clean --days ${LOG_RETENTION_DAYS:-30} > /dev/null 2>&1" > /tmp/crontab.txt
    crontab /tmp/crontab.txt
    rm /tmp/crontab.txt
    
    # Iniciar cron
    cron
    echo "✅ Limpeza automática de logs configurada!"
else
    echo "⚠️ Cron não disponível. A limpeza de logs deve ser feita manualmente."
fi

# Limpar logs antigos ao iniciar
echo "🧹 Limpando logs antigos..."
python scripts/manage_logs.py --clean --days ${LOG_RETENTION_DAYS:-30}

# Iniciar a API
echo "🚀 Iniciando servidor API..."
exec python src/api/main.py
