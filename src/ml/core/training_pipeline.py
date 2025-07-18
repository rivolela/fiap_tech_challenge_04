import torch
import mlflow
import mlflow.pytorch
import mlflow.models
import pandas as pd
import os
from typing import Dict, Any
import logging

from .data_handler import DataHandler
from ml.models.lstm_model import EnhancedLSTM
from .model_trainer import ModelTrainer
from ml.utils.metrics_calculator import MetricsCalculator
from ml.utils.visualizer import ModelVisualizer

logger = logging.getLogger(__name__)

class TrainingPipeline:
    """Main training pipeline orchestrating all components."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_handler = DataHandler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def run_training(self) -> None:
        """Execute the complete training pipeline."""
        logger.info("Starting LSTM training pipeline...")
        
        # Setup MLflow
        mlflow.set_experiment("Enhanced_LSTM_Predictions")
        
        with mlflow.start_run():
            # Load and prepare data
            df = self.data_handler.load_latest_parquet()
            X, y, scaler = self.data_handler.create_sequences(
                self.config['sequence_length'],
                self.config['horizon']
            )
            
            # Split data
            X_train, X_test, y_train, y_test = self._split_data(X, y)
            
            # Create and train model
            model = self._create_model(df.shape[1])
            trainer = ModelTrainer(model, str(self.device))
            trainer.setup_training(
                self.config['learning_rate'],
                self.config['weight_decay'],
                self.config['scheduler_patience']
            )
            
            trained_model, training_metrics = trainer.train_model(
                X_train, y_train, X_test, y_test, self.config['epochs']
            )
            
            # Calculate metrics and create visualizations
            metrics_calc = MetricsCalculator(scaler)
            predictions_df = metrics_calc.calculate_metrics(
                trained_model, X_test, y_test, training_metrics
            )
            
            visualizer = ModelVisualizer()
            predictions_plot = visualizer.plot_predictions(trained_model, X_test, y_test, scaler)
            losses_plot = visualizer.plot_training_losses(
                training_metrics['train_losses'],
                training_metrics['test_losses']
            )
            
            # Save results
            self._save_results(predictions_df, predictions_plot, losses_plot, trained_model, X_test)
            
        logger.info("Training pipeline completed successfully!")
    
    def _split_data(self, X, y):
        """Split data into training and testing sets."""
        train_size = int(len(X) * self.config['train_split'])
        
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        return (torch.FloatTensor(X_train), torch.FloatTensor(X_test),
                torch.FloatTensor(y_train), torch.FloatTensor(y_test))
    
    def _create_model(self, input_size: int) -> EnhancedLSTM:
        """Create the LSTM model with configured parameters."""
        return EnhancedLSTM(
            input_size=input_size,
            hidden_size=self.config['hidden_size'],
            num_layers=self.config['num_layers'],
            output_size=self.config['horizon'],
            dropout=self.config['dropout']
        )
    
    def _save_results(self, predictions_df: pd.DataFrame, predictions_plot: str, 
                     losses_plot: str, model: torch.nn.Module, X_test) -> None:
        """Save all results and artifacts."""
        # Save predictions CSV
        predictions_path = "./outputs/predictions.csv"
        os.makedirs(os.path.dirname(predictions_path), exist_ok=True)
        predictions_df.to_csv(predictions_path, index=False)
        
        # Log artifacts to MLflow
        mlflow.log_artifact(predictions_path)
        mlflow.log_artifact(predictions_plot)
        mlflow.log_artifact(losses_plot)
        
        # Create input example and signature
        input_example = X_test[:1]  # Use first test sample as example
        signature = mlflow.models.infer_signature(
            input_example.numpy(), 
            model(input_example).detach().numpy()
        )

        # Log model with signature and input example
        mlflow.pytorch.log_model(
            model, 
            "model",
            signature=signature,
            input_example=input_example.numpy()
        )
        
        # Export model for deployment
        self._export_model_for_deployment(model)
        
        logger.info(f"Results saved to: {os.path.abspath('./outputs/')}")
    
    def _export_model_for_deployment(self, model: torch.nn.Module) -> None:
        """Export model package for deployment."""
        try:
            # Import the exporter - using relative import
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from utils.model_exporter import ModelExporter
            
            # Export complete model package
            export_path = ModelExporter.export_complete_model(
                model=model,
                scaler=self.data_handler.scaler,
                feature_columns=self.data_handler.feature_columns
            )
            
            # Log export path to MLflow
            mlflow.log_param("export_path", export_path)
            
            logger.info(f"✅ Model exported for deployment to: {export_path}")
            
        except ImportError as e:
            logger.warning(f"Could not import ModelExporter: {e}")
            logger.info("Skipping model export for deployment")
        except Exception as e:
            logger.error(f"Failed to export model: {e}")
            logger.info("Model training completed, but export failed.")
        