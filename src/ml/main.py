import logging
import sys
import os

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ml.training_pipeline import TrainingPipeline

# CONFIGURAÇÃO DO LOGGING - ESSENCIAL PARA VER OS LOGS
logging.basicConfig(
    level=logging.INFO,  # Define o nível mínimo de log
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),  # Mostra no console
        logging.FileHandler('training.log')  # Salva em arquivo
    ]
)

# Configurar logger raiz para garantir que todos os módulos sejam exibidos
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

def main():
    """Main entry point for LSTM training."""
    logger = logging.getLogger(__name__)
    logger.info("=== INICIANDO PIPELINE DE TREINAMENTO LSTM ===")
    
    config = {
        'sequence_length': 24,
        'horizon': 6,
        'train_split': 0.8,
        'hidden_size': 32,
        'num_layers': 1,  # Fixed: removed the 'e' after the comma
        'dropout': 0.2,
        'learning_rate': 0.001,
        'weight_decay': 0.01,
        'scheduler_patience': 5,
        'epochs': 200
    }
    
    pipeline = TrainingPipeline(config)
    pipeline.run_training()
    
    logger.info("=== PIPELINE CONCLUÍDO COM SUCESSO ===")

if __name__ == '__main__':
    main()