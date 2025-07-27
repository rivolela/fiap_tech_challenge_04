# 📊 Sistema de Logs

Este documento descreve a organização e o gerenciamento de logs no projeto LSTM Stock Prediction.

## 📁 Estrutura de Diretórios

Os logs são organizados nas seguintes categorias:

```
logs/
├── api/         # Logs da API e serviços web
├── training/    # Logs de treinamento de modelos
├── monitoring/  # Logs de monitoramento e detecção de drift
└── prediction/  # Logs de inferência e predições
```

## 🛠️ Gerenciamento de Logs

O projeto inclui um utilitário para gerenciar logs localizado em `scripts/manage_logs.py`.

### Funcionalidades:

- **Rotação de logs**: Arquivos de log são automaticamente rotacionados quando atingem um tamanho máximo.
- **Compressão**: Logs rotacionados são comprimidos para economizar espaço.
- **Limpeza automática**: Logs antigos são removidos após um período configurável.
- **Listagem de logs**: Visualização organizada de todos os arquivos de log.

### Uso:

```bash
# Criar estrutura de diretórios para logs
python scripts/manage_logs.py --setup

# Listar todos os logs
python scripts/manage_logs.py --list

# Rotacionar logs maiores que 10MB
python scripts/manage_logs.py --rotate

# Limpar logs mais antigos que 30 dias
python scripts/manage_logs.py --clean --days 30

# Ver todas as opções
python scripts/manage_logs.py --help
```

## 📝 Configuração de Logging

Para garantir consistência no logging entre diferentes módulos do projeto, 
utilize o utilitário de configuração disponível em `src/utils/logging_config.py`.

### Exemplo de uso:

```python
from src.utils.logging_config import get_api_logger

# Criar logger para API
logger = get_api_logger(__name__)

# Usar logger
logger.info("API iniciada")
logger.error("Erro ao processar requisição")
```

### Loggers disponíveis:

- `get_api_logger()` - Para componentes da API
- `get_training_logger()` - Para scripts de treinamento
- `get_monitoring_logger()` - Para monitoramento
- `get_prediction_logger()` - Para inferência e predições

## 🔄 Rotação Automática

Os logs são configurados com rotação automática:
- Tamanho máximo de 10MB por arquivo de log
- Máximo de 3 backups mantidos por arquivo
- Compressão de backups para economizar espaço
