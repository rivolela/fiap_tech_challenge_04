import os
from matplotlib import pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.pytorch

class LSTMBasico(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMBasico, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Última saída da sequência
        return out


def carregar_dados():
    """
    Carrega dados dos arquivos parquet da pasta data/transformed
    
    Returns:
        pd.DataFrame: DataFrame com os dados carregados e processados
    """
    # Define o diretório dos arquivos parquet
    parquet_dir = "./data/transformed"
    
    # Lista arquivos parquet no diretório
    parquet_files = [f for f in os.listdir(parquet_dir) if f.endswith(".parquet")]
    
    if not parquet_files:
        raise FileNotFoundError(f"Nenhum arquivo parquet encontrado em {parquet_dir}")
    
    # Carrega o primeiro arquivo parquet encontrado
    parquet_path = os.path.join(parquet_dir, parquet_files[0])
    df = pd.read_parquet(parquet_path)
    
    # Seleciona colunas relevantes
    colunas_features = [
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
    
    # Filtra apenas colunas existentes
    colunas_features = [col for col in colunas_features if col in df.columns]
    
    return df[colunas_features]

def preparar_dados(df, sequence_length=6, test_size=0.2):
    """
    Prepara os dados para treinamento do modelo LSTM.
    
    Args:
        df (pd.DataFrame): DataFrame com os dados carregados
        sequence_length (int): Tamanho da sequência para input do LSTM
        test_size (float): Proporção dos dados para teste (0.0 a 1.0)
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test, scaler)
    """
    # Normalizar os dados
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df)
    
    # Criar sequências
    X, y = [], []
    for i in range(len(data_scaled) - sequence_length):
        X.append(data_scaled[i:(i + sequence_length)])
        y.append(data_scaled[i + sequence_length, 0])  # índice 0 = preco_medio_close
    
    X = np.array(X)
    y = np.array(y)
    
    # Dividir em conjuntos de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )
    
    # Converter para tensores PyTorch
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.FloatTensor(y_train).reshape(-1, 1)
    y_test = torch.FloatTensor(y_test).reshape(-1, 1)
    
    return X_train, X_test, y_train, y_test, scaler

def treinar_modelo(modelo, X_treino, y_treino, X_teste, y_teste, 
                  epochs=100, learning_rate=0.01, device='cpu'):
    """
    Treina o modelo LSTM com os dados fornecidos e registra métricas no MLflow.
    
    Args:
        modelo (LSTMBasico): Instância do modelo LSTM
        X_treino (torch.Tensor): Dados de treino
        y_treino (torch.Tensor): Labels de treino
        X_teste (torch.Tensor): Dados de teste
        y_teste (torch.Tensor): Labels de teste
        epochs (int): Número de épocas de treinamento
        learning_rate (float): Taxa de aprendizagem
        device (str): Dispositivo para treinamento ('cpu' ou 'cuda')
    
    Returns:
        tuple: (perdas_treino, perdas_teste) Listas com histórico de perdas
    """
    # Configurar experimento MLflow
    mlflow.set_experiment("LSTM Previsao Precos")
    
    with mlflow.start_run(run_name="LSTM_training"):
        # Log dos parâmetros do modelo
        mlflow.log_params({
            "hidden_size": modelo.lstm.hidden_size,
            "num_layers": modelo.lstm.num_layers,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "batch_size": X_treino.shape[0],
            "sequence_length": X_treino.shape[1],
            "input_features": X_treino.shape[2]
        })
        
        # Mover modelo e dados para o dispositivo apropriado
        device = torch.device(device)
        modelo = modelo.to(device)
        X_treino = X_treino.to(device)
        y_treino = y_treino.to(device)
        X_teste = X_teste.to(device)
        y_teste = y_teste.to(device)
        
        # Definir critério de perda e otimizador
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(modelo.parameters(), lr=learning_rate)
        
        # Listas para armazenar histórico de perdas
        perdas_treino = []
        perdas_teste = []
        
        # Loop de treinamento
        for epoch in range(epochs):
            # Modo de treinamento
            modelo.train()
            optimizer.zero_grad()
            
            # Forward pass
            outputs = modelo(X_treino)
            loss = criterion(outputs, y_treino)
            
            # Backward pass e otimização
            loss.backward()
            optimizer.step()
            
            # Avaliar no conjunto de teste
            modelo.eval()
            with torch.no_grad():
                test_outputs = modelo(X_teste)
                test_loss = criterion(test_outputs, y_teste)
            
            # Armazenar perdas
            perdas_treino.append(loss.item())
            perdas_teste.append(test_loss.item())
            
            # Log das métricas no MLflow
            mlflow.log_metrics({
                "train_loss": loss.item(),
                "test_loss": test_loss.item(),
            }, step=epoch)
            
            # Imprimir progresso a cada 10 épocas
            if (epoch + 1) % 10 == 0:
                print(f'Época [{epoch+1}/{epochs}], '
                      f'Perda Treino: {loss.item():.4f}, '
                      f'Perda Teste: {test_loss.item():.4f}')
        
        # Salvar o modelo no MLflow
        mlflow.pytorch.log_model(modelo, "model")
        
        # Log do gráfico de perdas
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(perdas_treino, label='Treino')
        ax.plot(perdas_teste, label='Teste')
        ax.set_title('Curvas de Perda')
        ax.set_xlabel('Época')
        ax.set_ylabel('Perda')
        ax.legend()
        plt.close()
        
        mlflow.log_figure(fig, "loss_curves.png")
    
    return perdas_treino, perdas_teste

def avaliar_modelo(modelo, X_teste, y_teste, scaler=None):
    """
    Avalia o modelo LSTM no conjunto de teste.
    
    Args:
        modelo (LSTMBasico): Modelo LSTM treinado
        X_teste (torch.Tensor): Dados de teste
        y_teste (torch.Tensor): Labels verdadeiros
        scaler (StandardScaler, optional): Scaler usado para desnormalizar os dados
    
    Returns:
        dict: Métricas de avaliação (MSE, MAE, R2) e previsões
    """
    # Coloca modelo em modo de avaliação
    modelo.eval()
    
    # Faz previsões
    with torch.no_grad():
        previsoes = modelo(X_teste)
    
    # Converte tensores para numpy
    y_true = y_teste.cpu().numpy()
    y_pred = previsoes.cpu().numpy()
    
    # Se fornecido scaler, desnormaliza os dados
    if scaler is not None:
        # Reshape para formato esperado pelo scaler
        y_true = scaler.inverse_transform(np.hstack([y_true, np.zeros((y_true.shape[0], X_teste.shape[2]-1))]))[: , 0:1]
        y_pred = scaler.inverse_transform(np.hstack([y_pred, np.zeros((y_pred.shape[0], X_teste.shape[2]-1))]))[: , 0:1]
    
    # Calcula métricas
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2)
    
    # Imprime resultados
    print(f"\nResultados da Avaliação:")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    
    return {
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'y_true': y_true,
        'y_pred': y_pred
    }

def main():
    """
    Função principal que executa todo o pipeline de treinamento do modelo LSTM.
    """
    try:
        print("Iniciando pipeline de treinamento...")
        
        # 1. Carregar dados
        print("\nCarregando dados...")
        df = carregar_dados()
        print(f"Dados carregados com sucesso. Shape: {df.shape}")
        
        # 2. Preparar dados
        print("\nPreparando dados para treinamento...")
        X_treino, X_teste, y_treino, y_teste, scaler = preparar_dados(
            df, 
            sequence_length=6, 
            test_size=0.2
        )
        print("Dados preparados com sucesso.")
        print(f"Shape dos dados de treino: {X_treino.shape}")
        print(f"Shape dos dados de teste: {X_teste.shape}")
        
        # 3. Criar e treinar modelo
        print("\nIniciando treinamento do modelo...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Utilizando dispositivo: {device}")
        
        modelo = LSTMBasico(
            input_size=11,
            hidden_size=32,
            num_layers=2,
            output_size=1
        )
        
        perdas_treino, perdas_teste = treinar_modelo(
            modelo=modelo,
            X_treino=X_treino,
            y_treino=y_treino,
            X_teste=X_teste,
            y_teste=y_teste,
            epochs=100,
            learning_rate=0.01,
            device=device
        )
        
        # 4. Avaliar resultados
        print("\nAvaliando modelo...")
        resultados = avaliar_modelo(modelo, X_teste, y_teste, scaler)
        
        # 5. Visualizar resultados
        print("\nGerando visualização dos resultados...")
        plt.figure(figsize=(12, 6))
        plt.plot(resultados['y_true'], label='Real')
        plt.plot(resultados['y_pred'], label='Previsto')
        plt.title('Previsões vs Valores Reais')
        plt.xlabel('Tempo')
        plt.ylabel('Preço')
        plt.legend()
        plt.show()
        
        # 6. Visualizar curvas de perda
        plt.figure(figsize=(12, 6))
        plt.plot(perdas_treino, label='Treino')
        plt.plot(perdas_teste, label='Teste')
        plt.title('Curvas de Perda')
        plt.xlabel('Época')
        plt.ylabel('Perda')
        plt.legend()
        plt.show()
        
        return modelo, resultados
        
    except Exception as e:
        print(f"\nErro durante a execução: {str(e)}")
        raise

if __name__ == "__main__":
    main()
