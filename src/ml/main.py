import logging
import sys
import os

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def setup_logging():
    """Configure logging to ensure all messages are visible."""
    # Remove any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # File handler
    file_handler = logging.FileHandler('training.log')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # Test logging
    test_logger = logging.getLogger(__name__)
    test_logger.info("✓ Sistema de logging configurado com sucesso!")

# Setup logging immediately
setup_logging()
logger = logging.getLogger(__name__)

def main():
    """Main entry point for LSTM training."""
    logger.info("=== INICIANDO PIPELINE DE TREINAMENTO LSTM ===")
    
    # Check what files are available
    ml_dir = os.path.dirname(__file__)
    available_files = [f for f in os.listdir(ml_dir) if f.endswith('.py')]
    logger.info(f"Arquivos Python disponíveis: {available_files}")
    
    # Check for core directory
    core_dir = os.path.join(ml_dir, 'core')
    if os.path.exists(core_dir):
        core_files = [f for f in os.listdir(core_dir) if f.endswith('.py')]
        logger.info(f"Arquivos em core/: {core_files}")
    
    try:
        # Try to import the training pipeline from core
        logger.info("Tentando importar training_pipeline.py...")
        from ml.core.training_pipeline import TrainingPipeline
        
        config = {
            'sequence_length': 24,
            'horizon': 6,
            'train_split': 0.8,
            'hidden_size': 32,
            'num_layers': 1,
            'dropout': 0.2,
            'learning_rate': 0.001,
            'weight_decay': 0.01,
            'scheduler_patience': 5,
            'epochs': 200
        }
        
        pipeline = TrainingPipeline(config)
        pipeline.run_training()
        
        logger.info("=== PIPELINE CONCLUÍDO COM SUCESSO ===")
        
    except ImportError as e:
        logger.error(f"Erro ao importar training_pipeline: {e}")
        
            
    except Exception as e:
        logger.error(f"Erro durante o treinamento: {e}")
        raise

if __name__ == '__main__':
    main()