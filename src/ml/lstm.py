import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.pytorch
from datetime import datetime
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau

class EnhancedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(EnhancedLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,  # Reduce to single layer
            batch_first=True,
            dropout=0.2,   # Reduce dropout
            bidirectional=False  # Remove bidirectional
        )
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]  # Get last time step output
        return self.fc(out)

def load_parquet_data():
    """
    Load and preprocess the latest parquet data file.
    
    Returns:
        pd.DataFrame: Processed DataFrame with selected features
    """
    try:
        # Base directories to search
        possible_dirs = [
            "./data/transformed",
            "../data/transformed",
            "../../data/transformed"
        ]
        
        latest_file = None
        latest_time = 0
        
        # Find the latest parquet file
        for dir_path in possible_dirs:
            if not os.path.exists(dir_path):
                continue
                
            for file in os.listdir(dir_path):
                if file.endswith('.parquet'):
                    file_path = os.path.join(dir_path, file)
                    file_time = os.path.getmtime(file_path)
                    
                    if file_time > latest_time:
                        latest_time = file_time
                        latest_file = file_path
        
        if latest_file is None:
            raise FileNotFoundError("No parquet files found in any of the expected locations")
        
        print(f"Loading latest parquet file: {latest_file}")
        print(f"Last modified: {datetime.fromtimestamp(latest_time).strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Load and process the data
        df = pd.read_parquet(latest_file)
        
        # Sort chronologically
        df = df.sort_values(by=["ano", "mes"], ascending=[True, True])
        
        # Select relevant features
        feature_columns = [
            "preco_medio_close",
            "lag_1_mes_preco_medio_close",
            "lag_2_mes_preco_medio_close",
            "lag_3_mes_preco_medio_close",
            "lag_4_mes_preco_medio_close",
            "lag_5_mes_preco_medio_close",
            "lag_6_mes_preco_medio_close",
            "media_movel_6_meses_preco_medio_close",
            "desvio_padrao_movel_6_meses_preco_medio_close",
            "valor_minimo_6_meses_preco_medio_close",
            "valor_maximo_6_meses_preco_medio_close"
        ]
        
        # Filter only existing columns
        existing_columns = [col for col in feature_columns if col in df.columns]
        df = df[existing_columns]
        
        print(f"Data loaded successfully. Shape: {df.shape}")
        
        # Add data statistics
        print(f"Data statistics:")
        print(df.describe())
        print(f"\nTarget variable range: {df['preco_medio_close'].min():.2f} to {df['preco_medio_close'].max():.2f}")
        print(f"Any NaN values: {df.isnull().sum().sum()}")
        
        return df
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        print("Please ensure:")
        print("1. At least one parquet file exists in the data/transformed directory")
        print("2. You have read permissions for the files")
        raise

def create_sequences(df, sequence_length, horizon):
    """Create sequences for LSTM training"""
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df)
    
    X, y = [], []
    for i in range(len(data_scaled) - sequence_length - horizon + 1):
        X.append(data_scaled[i:(i + sequence_length)])
        y.append(data_scaled[i + sequence_length:i + sequence_length + horizon, 0])
    
    return np.array(X), np.array(y), scaler

def prepare_data(df, train_size=0.8):
    # Add feature standardization
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    
    # Create sequences with stride
    sequence_length = 12  # Increased from 6
    stride = 2           # Add stride for more samples
    
    X, y = [], []
    for i in range(0, len(scaled_data) - sequence_length, stride):
        X.append(scaled_data[i:i + sequence_length])
        y.append(scaled_data[i + sequence_length, 0])  # Target is next price
    
    return np.array(X), np.array(y), scaler

def train_model(model, X_train, y_train, X_test, y_test, optimizer, scheduler, criterion, device, epochs=200):
    """Train the LSTM model and return metrics"""
    # Move model and data to device
    model = model.to(device)
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    
    # Training history
    train_losses = []
    test_losses = []
    best_test_loss = float('inf')
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        optimizer.zero_grad()
        train_pred = model(X_train)
        train_loss = criterion(train_pred, y_train)
        train_loss.backward()
        optimizer.step()
        
        # Validation phase
        model.eval()
        with torch.no_grad():
            test_pred = model(X_test)
            test_loss = criterion(test_pred, y_test)
            
        # Update learning rate
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(test_loss)
        else:
            scheduler.step()
            
        # Save losses
        train_losses.append(train_loss.item())
        test_losses.append(test_loss.item())
        
        # Log metrics
        mlflow.log_metrics({
            "train_loss": train_loss.item(),
            "test_loss": test_loss.item()
        }, step=epoch)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss.item():.4f}, Test Loss: {test_loss.item():.4f}')
    
    metrics = {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'final_train_loss': train_losses[-1],
        'final_test_loss': test_losses[-1]
    }
    
    return model, metrics

def create_predictions_df(model, X_test, y_test, metrics, scaler):
    """Create a DataFrame with test predictions and actual values for 6 months"""
    model.eval()
    with torch.no_grad():
        predictions = model(X_test).cpu().numpy()
        actuals = y_test.cpu().numpy()
    
    # Create dummy array with correct shape for inverse transform
    dummy_array = np.zeros((predictions.shape[0], 11))
    
    # Put predictions in first column (price column)
    predictions_reshaped = dummy_array.copy()
    predictions_reshaped[:, 0] = predictions[:, 0]  # First month prediction
    
    actuals_reshaped = dummy_array.copy()
    actuals_reshaped[:, 0] = actuals[:, 0]
    
    # Inverse transform
    predictions_unscaled = scaler.inverse_transform(predictions_reshaped)[:, 0]
    actuals_unscaled = scaler.inverse_transform(actuals_reshaped)[:, 0]
    
    # Calculate metrics
    mse = np.mean((actuals_unscaled - predictions_unscaled) ** 2)
    mae = np.mean(np.abs(actuals_unscaled - predictions_unscaled))
    r2 = 1 - np.sum((actuals_unscaled - predictions_unscaled) ** 2) / np.sum((actuals_unscaled - actuals_unscaled.mean()) ** 2)
    
    print(f"\nMetrics for 6-Month Forecast:")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    
    # Log metrics to MLflow
    mlflow.log_metrics({
        "mse_6_months": mse,
        "mae_6_months": mae,
        "r2_6_months": r2
    })
    
    # Create DataFrame with 6 months of predictions
    df_predictions = pd.DataFrame()
    
    # Add predictions for each of the 6 months
    for i in range(6):
        if i < predictions.shape[1]:
            pred_month = predictions[:, i]
            actual_month = actuals[:, i]
            
            # Transform each month
            pred_reshaped = dummy_array.copy()
            pred_reshaped[:, 0] = pred_month
            actual_reshaped = dummy_array.copy()
            actual_reshaped[:, 0] = actual_month
            
            pred_unscaled = scaler.inverse_transform(pred_reshaped)[:, 0]
            actual_unscaled = scaler.inverse_transform(actual_reshaped)[:, 0]
            
            df_predictions[f'prediction_month_{i+1}'] = pred_unscaled
            df_predictions[f'actual_month_{i+1}'] = actual_unscaled
    
    # Add metrics
    df_predictions['mse'] = mse
    df_predictions['mae'] = mae
    df_predictions['r2'] = r2
    df_predictions['train_loss'] = metrics['final_train_loss']
    df_predictions['test_loss'] = metrics['final_test_loss']
    
    # VERIFICAÇÃO: Imprimir estatísticas dos dados originais
    print(f"\nEstatísticas do scaler:")
    print(f"Mean: {scaler.mean_[0]:.4f}")
    print(f"Scale: {scaler.scale_[0]:.4f}")
    
    return df_predictions

def plotar_previsoes(model, X_test, y_test, scaler, output_dir='outputs'):
    """
    Plota o gráfico de previsões vs valores reais para 6 meses - CORRIGIDO.
    """
    model.eval()
    with torch.no_grad():
        predictions = model(X_test).cpu().numpy()
        actuals = y_test.cpu().numpy()
    
    # Desnormalizar CADA amostra individualmente
    predictions_unscaled = []
    actuals_unscaled = []
    
    for i in range(predictions.shape[0]):
        for j in range(6):  # Para cada mês
            # Criar array dummy para desnormalização
            dummy_pred = np.zeros(scaler.n_features_in_)
            dummy_actual = np.zeros(scaler.n_features_in_)
            
            # Colocar valores na posição correta (assumindo que preço é a primeira feature)
            dummy_pred[0] = predictions[i, j]
            dummy_actual[0] = actuals[i, j]
            
            # Desnormalizar
            pred_unscaled = scaler.inverse_transform(dummy_pred.reshape(1, -1))[0, 0]
            actual_unscaled = scaler.inverse_transform(dummy_actual.reshape(1, -1))[0, 0]
            
            predictions_unscaled.append(pred_unscaled)
            actuals_unscaled.append(actual_unscaled)
    
    # Reorganizar em formato de matriz
    predictions_unscaled = np.array(predictions_unscaled).reshape(predictions.shape[0], 6)
    actuals_unscaled = np.array(actuals_unscaled).reshape(actuals.shape[0], 6)
    
    # Calcular médias dos valores DESNORMALIZADOS
    pred_avg = np.mean(predictions_unscaled, axis=0)
    actual_avg = np.mean(actuals_unscaled, axis=0)
    
    # Verificar se os valores estão na faixa esperada
    print(f"Valores desnormalizados - Predição: {pred_avg}")
    print(f"Valores desnormalizados - Real: {actual_avg}")
    print(f"Faixa esperada: ~23 (preço médio das ações)")
    
    plt.style.use('ggplot')
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # X-axis for 6 months
    months = np.arange(1, 7)  # 1 to 6 months
    month_labels = [f'Mês {i}' for i in months]
    
    # Plot actual vs predicted for 6 months
    ax.plot(months, actual_avg, 
           label='Valores Reais', 
           color='#2E86C1',
           marker='o', 
           markersize=8,
           linewidth=2,
           alpha=0.8)
    
    ax.plot(months, pred_avg, 
           label='Previsões',
           color='#E74C3C',
           marker='s',
           markersize=8,
           linewidth=2,
           linestyle='--',
           alpha=0.8)
    
    # Add confidence interval
    std_dev = np.std(predictions - actuals, axis=0)
    ax.fill_between(months,
                   pred_avg - std_dev,
                   pred_avg + std_dev,
                   color='#E74C3C',
                   alpha=0.2,
                   label='Intervalo de Confiança')
    
    # Customize plot
    ax.grid(True, linestyle='--', alpha=0.7, color='gray')
    ax.set_xlabel('Meses Futuros', fontsize=12, fontweight='bold')
    ax.set_ylabel('Preço Médio de Fechamento (R$)', fontsize=12, fontweight='bold')
    ax.set_title('Previsões vs Valores Reais - 6 Meses\nModelo LSTM', 
                fontsize=16, 
                fontweight='bold', 
                pad=20)
    
    # Set x-axis
    ax.set_xticks(months)
    ax.set_xticklabels(month_labels)
    
    # Add value labels
    for i, (real, pred) in enumerate(zip(actual_avg, pred_avg)):
        ax.annotate(f'{real:.2f}', 
                   (months[i], real), 
                   xytext=(0, 10), 
                   textcoords='offset points', 
                   ha='center')
        ax.annotate(f'{pred:.2f}', 
                   (months[i], pred), 
                   xytext=(0, -20), 
                   textcoords='offset points', 
                   ha='center')
    
    ax.legend(loc='best', 
             fontsize=10, 
             framealpha=0.9,
             title='Valores',
             title_fontsize=12)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'predictions_6_months.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

def plotar_perdas(train_losses, test_losses, output_dir='outputs'):
    """
    Plota o gráfico das curvas de perda do modelo aprimorado.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'loss_curves_enhanced.png')
        
        plt.clf()
        plt.cla()
        plt.close()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot loss curves
        ax.plot(train_losses, label='Treino', linewidth=2, color='#2E86C1')
        ax.plot(test_losses, label='Teste', linewidth=2, color='#E74C3C')
        
        # Enhance plot
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_title('Curvas de Perda - Modelo LSTM Aprimorado', 
                    fontsize=14, 
                    pad=20)
        ax.set_xlabel('Época', fontsize=12)
        ax.set_ylabel('Perda', fontsize=12)
        ax.legend(loc='upper right', fontsize=10)
        
        # Add min/max annotations
        min_train = min(train_losses)
        min_test = min(test_losses)
        ax.annotate(f'Min Treino: {min_train:.4f}',
                   xy=(train_losses.index(min_train), min_train),
                   xytext=(10, 10),
                   textcoords='offset points')
        ax.annotate(f'Min Teste: {min_test:.4f}',
                   xy=(test_losses.index(min_test), min_test),
                   xytext=(10, -10),
                   textcoords='offset points')
        
        plt.tight_layout()
        
        # Save figure
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        
        print(f"Gráfico salvo em: {os.path.abspath(output_path)}")
        
        return output_path  # Add this return statement
        
    except Exception as e:
        print(f"Erro ao gerar gráfico de perdas: {str(e)}")
        raise
    finally:
        plt.close(fig)

def add_noise(data, noise_level=0.01):
    noise = torch.randn_like(data) * noise_level
    return data + noise

def train():
    """Main training function"""
    print("Iniciando pipeline de treinamento...")
    
    # Load and prepare data
    df = load_parquet_data()
    print("\nPrimeiras 5 linhas do DataFrame:")
    print(df.head())
    
    # Data preparation parameters
    sequence_length = 24
    horizon = 6  # Changed from 8 to 6 months
    
    # Create sequences
    X, y, scaler = create_sequences(df, sequence_length, horizon)
    
    # Train/Test split
    train_size = int(len(X) * 0.80)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.FloatTensor(y_train)
    y_test = torch.FloatTensor(y_test)
    
    # Model configuration
    model = EnhancedLSTM(
        input_size=len(df.columns),
        hidden_size=32,
        num_layers=1,
        output_size=6  # Changed from 8 to 6
    )
    
    # Training parameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=0.01
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.3,
        patience=5,
        min_lr=1e-6,
        verbose=True
    )
    
    # MLflow logging
    mlflow.set_experiment("Enhanced_LSTM_Predictions")
    
    with mlflow.start_run():
        model, metrics = train_model(
            model, 
            X_train, 
            y_train, 
            X_test, 
            y_test,
            optimizer,
            scheduler,
            criterion,
            device,
            epochs=200
        )
        
        # Generate and save plots
        predictions_plot = plotar_previsoes(model, X_test, y_test, scaler)
        losses_plot = plotar_perdas(metrics['train_losses'], metrics['test_losses'])
        
        # Save predictions
        predictions_df = create_predictions_df(model, X_test, y_test, metrics, scaler)
        predictions_path = "./outputs/predictions.csv"
        os.makedirs(os.path.dirname(predictions_path), exist_ok=True)
        predictions_df.to_csv(predictions_path)
        
        # Log artifacts
        mlflow.log_artifact(predictions_path)
        mlflow.log_artifact(predictions_plot)
        mlflow.log_artifact(losses_plot)
        mlflow.pytorch.log_model(model, "model")

if __name__ == '__main__':
    train()