import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import mlflow
import mlflow.pytorch
from typing import Dict, Any, Tuple, Optional
import logging
from ml.models.lstm_model import EnhancedLSTM

logger = logging.getLogger(__name__)

# Add safe global for your custom LSTM class
torch.serialization.add_safe_globals([EnhancedLSTM])

class ModelTrainer:
    """Handles LSTM model training and evaluation."""
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler._LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None] = None
        
    def setup_training(self, learning_rate: float = 0.001, weight_decay: float = 0.01, 
                      scheduler_patience: int = 5) -> None:
        """Setup optimizer and scheduler for training."""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=scheduler_patience,
            min_lr=1e-6
        )
    
    def train_model(self, X_train: torch.Tensor, y_train: torch.Tensor, 
                   X_test: torch.Tensor, y_test: torch.Tensor, 
                   epochs: int = 200) -> Tuple[nn.Module, Dict[str, Any]]:
        """Train the LSTM model and return metrics."""
        if self.optimizer is None or self.scheduler is None:
            raise ValueError("Training setup not configured. Call setup_training() first.")
        
        # Move data to device
        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)
        X_test = X_test.to(self.device)
        y_test = y_test.to(self.device)
        
        train_losses = []
        test_losses = []
        
        logger.info(f"Starting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            train_loss = self._train_epoch(X_train, y_train)
            test_loss = self._validate_epoch(X_test, y_test)
            
            self.scheduler.step(test_loss)
            
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            
            # Log metrics
            mlflow.log_metrics({
                "train_loss": train_loss,
                "test_loss": test_loss
            }, step=epoch)
            
            if (epoch + 1) % 10 == 0:
                logger.info(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
        
        metrics = {
            'train_losses': train_losses,
            'test_losses': test_losses,
            'final_train_loss': train_losses[-1],
            'final_test_loss': test_losses[-1]
        }
        
        return self.model, metrics
    
    def _train_epoch(self, X_train: torch.Tensor, y_train: torch.Tensor) -> float:
        """Train for one epoch."""
        self.model.train()
        self.optimizer.zero_grad()
        predictions = self.model(X_train)
        loss = self.criterion(predictions, y_train)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def _validate_epoch(self, X_test: torch.Tensor, y_test: torch.Tensor) -> float:
        """Validate for one epoch."""
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_test)
            loss = self.criterion(predictions, y_test)
        return loss.item()