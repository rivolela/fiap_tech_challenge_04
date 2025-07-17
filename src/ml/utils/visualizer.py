import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)

class ModelVisualizer:
    """Handles visualization of model predictions and training metrics."""
    
    def __init__(self, output_dir: str = 'outputs'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        plt.style.use('ggplot')
    
    def plot_predictions(self, model: torch.nn.Module, X_test: torch.Tensor, 
                        y_test: torch.Tensor, scaler: StandardScaler) -> str:
        """Plot predictions vs actual values for 6 months."""
        model.eval()
        with torch.no_grad():
            predictions = model(X_test).cpu().numpy()
            actuals = y_test.cpu().numpy()
        
        # Denormalize predictions
        predictions_unscaled, actuals_unscaled = self._denormalize_predictions(
            predictions, actuals, scaler
        )
        
        # Calculate averages
        pred_avg = np.mean(predictions_unscaled, axis=0)
        actual_avg = np.mean(actuals_unscaled, axis=0)
        
        self._log_prediction_stats(pred_avg, actual_avg)
        
        return self._create_prediction_plot(pred_avg, actual_avg, predictions_unscaled, actuals_unscaled)
    
    def plot_training_losses(self, train_losses: List[float], test_losses: List[float]) -> str:
        """Plot training and validation loss curves."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(train_losses, label='Treino', linewidth=2, color='#2E86C1')
        ax.plot(test_losses, label='Teste', linewidth=2, color='#E74C3C')
        
        self._customize_loss_plot(ax, train_losses, test_losses)
        
        output_path = os.path.join(self.output_dir, 'loss_curves.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Loss curves saved to: {os.path.abspath(output_path)}")
        return output_path
    
    def _denormalize_predictions(self, predictions: np.ndarray, actuals: np.ndarray, 
                               scaler: StandardScaler) -> Tuple[np.ndarray, np.ndarray]:
        """Denormalize predictions and actuals."""
        predictions_unscaled = []
        actuals_unscaled = []
        
        for i in range(predictions.shape[0]):
            for j in range(6):  # 6 months
                dummy_pred = np.zeros(scaler.n_features_in_)
                dummy_actual = np.zeros(scaler.n_features_in_)
                
                dummy_pred[0] = predictions[i, j]
                dummy_actual[0] = actuals[i, j]
                
                pred_unscaled = scaler.inverse_transform(dummy_pred.reshape(1, -1))[0, 0]
                actual_unscaled = scaler.inverse_transform(dummy_actual.reshape(1, -1))[0, 0]
                
                predictions_unscaled.append(pred_unscaled)
                actuals_unscaled.append(actual_unscaled)
        
        predictions_unscaled = np.array(predictions_unscaled).reshape(predictions.shape[0], 6)
        actuals_unscaled = np.array(actuals_unscaled).reshape(actuals.shape[0], 6)
        
        return predictions_unscaled, actuals_unscaled
    
    def _log_prediction_stats(self, pred_avg: np.ndarray, actual_avg: np.ndarray) -> None:
        """Log prediction statistics."""
        logger.info(f"Denormalized predictions - Prediction: {pred_avg}")
        logger.info(f"Denormalized predictions - Actual: {actual_avg}")
        logger.info(f"Expected range: ~23 (average stock price)")
    
    def _create_prediction_plot(self, pred_avg: np.ndarray, actual_avg: np.ndarray, 
                              predictions_unscaled: np.ndarray, actuals_unscaled: np.ndarray) -> str:
        """Create the prediction vs actual plot."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        months = np.arange(1, 7)
        month_labels = [f'Mês {i}' for i in months]
        
        # Plot lines
        ax.plot(months, actual_avg, label='Valores Reais', color='#2E86C1',
                marker='o', markersize=8, linewidth=2, alpha=0.8)
        ax.plot(months, pred_avg, label='Previsões', color='#E74C3C',
                marker='s', markersize=8, linewidth=2, linestyle='--', alpha=0.8)
        
        # Add confidence interval
        std_dev = np.std(predictions_unscaled - actuals_unscaled, axis=0)
        ax.fill_between(months, pred_avg - std_dev, pred_avg + std_dev,
                       color='#E74C3C', alpha=0.2, label='Intervalo de Confiança')
        
        self._customize_prediction_plot(ax, months, month_labels, actual_avg, pred_avg)
        
        output_path = os.path.join(self.output_dir, 'predictions_6_months.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _customize_prediction_plot(self, ax, months: np.ndarray, month_labels: List[str], 
                                 actual_avg: np.ndarray, pred_avg: np.ndarray) -> None:
        """Customize the prediction plot."""
        ax.grid(True, linestyle='--', alpha=0.7, color='gray')
        ax.set_xlabel('Meses Futuros', fontsize=12, fontweight='bold')
        ax.set_ylabel('Preço Médio de Fechamento (R$)', fontsize=12, fontweight='bold')
        ax.set_title('Previsões vs Valores Reais - 6 Meses\nModelo LSTM', 
                    fontsize=16, fontweight='bold', pad=20)
        
        ax.set_xticks(months)
        ax.set_xticklabels(month_labels)
        
        # Add value labels
        for i, (real, pred) in enumerate(zip(actual_avg, pred_avg)):
            ax.annotate(f'{real:.2f}', (months[i], real), 
                       xytext=(0, 10), textcoords='offset points', ha='center')
            ax.annotate(f'{pred:.2f}', (months[i], pred), 
                       xytext=(0, -20), textcoords='offset points', ha='center')
        
        ax.legend(loc='best', fontsize=10, framealpha=0.9, title='Valores', title_fontsize=12)
        plt.tight_layout()
    
    def _customize_loss_plot(self, ax, train_losses: List[float], test_losses: List[float]) -> None:
        """Customize the loss plot."""
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_title('Curvas de Perda - Modelo LSTM', fontsize=14, pad=20)
        ax.set_xlabel('Época', fontsize=12)
        ax.set_ylabel('Perda', fontsize=12)
        ax.legend(loc='upper right', fontsize=10)
        
        # Add min annotations
        min_train = min(train_losses)
        min_test = min(test_losses)
        ax.annotate(f'Min Treino: {min_train:.4f}',
                   xy=(train_losses.index(min_train), min_train),
                   xytext=(10, 10), textcoords='offset points')
        ax.annotate(f'Min Teste: {min_test:.4f}',
                   xy=(test_losses.index(min_test), min_test),
                   xytext=(10, -10), textcoords='offset points')
        
        plt.tight_layout()