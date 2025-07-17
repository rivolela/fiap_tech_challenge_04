import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, Tuple
import logging

# Configurar logger específico para esta classe
logger = logging.getLogger(__name__)

class MetricsCalculator:
    """Calculates and manages model performance metrics."""
    
    def __init__(self, scaler: StandardScaler):
        self.scaler = scaler
        logger.info("MetricsCalculator inicializado")
    
    def calculate_metrics(self, model: torch.nn.Module, X_test: torch.Tensor, 
                         y_test: torch.Tensor, training_metrics: Dict[str, Any]) -> pd.DataFrame:
        """Calculate comprehensive metrics for model performance."""
        logger.info("Iniciando cálculo de métricas...")
        
        model.eval()
        with torch.no_grad():
            predictions = model(X_test).cpu().numpy()
            actuals = y_test.cpu().numpy()
        
        logger.info(f"Predictions shape: {predictions.shape}, Actuals shape: {actuals.shape}")
        
        # Calculate metrics for first month prediction
        pred_unscaled = self._inverse_transform_single_feature(predictions[:, 0])
        actual_unscaled = self._inverse_transform_single_feature(actuals[:, 0])
        
        mse = np.mean((actual_unscaled - pred_unscaled) ** 2)
        mae = np.mean(np.abs(actual_unscaled - pred_unscaled))
        r2 = self._calculate_r2(actual_unscaled, pred_unscaled)
        
        self._log_metrics(mse, mae, r2)
        
        # Create comprehensive predictions DataFrame
        df_predictions = self._create_predictions_dataframe(
            predictions, actuals, training_metrics, mse, mae, r2
        )
        
        logger.info("Métricas calculadas com sucesso!")
        return df_predictions
    
    def _inverse_transform_single_feature(self, values: np.ndarray) -> np.ndarray:
        """Inverse transform a single feature (price) back to original scale."""
        dummy_array = np.zeros((len(values), self.scaler.n_features_in_))
        dummy_array[:, 0] = values
        return self.scaler.inverse_transform(dummy_array)[:, 0]
    
    def _calculate_r2(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """Calculate R-squared score."""
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - actual.mean()) ** 2)
        return 1 - (ss_res / ss_tot)
    
    def _log_metrics(self, mse: float, mae: float, r2: float) -> None:
        """Log metrics to console and MLflow."""
        print("\n" + "="*50)
        print("MÉTRICAS DO MODELO - FORECAST 6 MESES")
        print("="*50)
        print(f"MSE: {mse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R²: {r2:.4f}")
        print("="*50 + "\n")
        
        # Também usar logger
        logger.info(f"Métricas para previsão de 6 meses:")
        logger.info(f"MSE: {mse:.4f}")
        logger.info(f"MAE: {mae:.4f}")
        logger.info(f"R²: {r2:.4f}")
        
        import mlflow
        mlflow.log_metrics({
            "mse_6_months": mse,
            "mae_6_months": mae,
            "r2_6_months": r2
        })
    
    def _create_predictions_dataframe(self, predictions: np.ndarray, actuals: np.ndarray, 
                                    training_metrics: Dict[str, Any], mse: float, 
                                    mae: float, r2: float) -> pd.DataFrame:
        """Create a comprehensive predictions DataFrame."""
        logger.info("Criando DataFrame de previsões...")
        
        df_predictions = pd.DataFrame()
        
        # Add predictions for each of the 6 months
        for i in range(min(6, predictions.shape[1])):
            pred_month = self._inverse_transform_single_feature(predictions[:, i])
            actual_month = self._inverse_transform_single_feature(actuals[:, i])
            
            df_predictions[f'prediction_month_{i+1}'] = pred_month
            df_predictions[f'actual_month_{i+1}'] = actual_month
            
            logger.info(f"Mês {i+1} - Média real: {actual_month.mean():.2f}, Média prevista: {pred_month.mean():.2f}")
        
        # Add metrics
        df_predictions['mse'] = mse
        df_predictions['mae'] = mae
        df_predictions['r2'] = r2
        df_predictions['train_loss'] = training_metrics['final_train_loss']
        df_predictions['test_loss'] = training_metrics['final_test_loss']
        
        logger.info(f"DataFrame criado com {len(df_predictions)} amostras")
        return df_predictions