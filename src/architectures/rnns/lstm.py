import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import mlflow
import mlflow.pytorch



# Set device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
input_size = 11      # Number of features in the input data
hidden_size = 50     # Number of hidden units in the LSTM
num_layers = 2       # Number of LSTM layers
output_size = 1      # Number of output units (e.g., regression output)
num_epochs = 50
batch_size = 8
learning_rate = 0.001
sequence_length = 20  # Length of the input sequences
num_samples = 10000  # Number of artificial samples to generate


# Load and process real data from disk
def load_real_data(num_samples, sequence_length, input_size):
    """
    Loads real data from Parquet files in ./data/transformed,
    selects relevant numeric columns (excluding year, month, quarter),
    normalizes them, and generates sequences for LSTM.
    """

    # Carregar dados do Parquet
    import os
    parquet_dir = "./data/transformed"
    parquet_files = [f for f in os.listdir(parquet_dir) if f.endswith(".parquet")]
    if not parquet_files:
        raise FileNotFoundError(f"No Parquet files found in {parquet_dir}")
    parquet_path = os.path.join(parquet_dir, parquet_files[0])
    df = pd.read_parquet(parquet_path)

    # Selecionar apenas colunas numéricas relevantes para features
    feature_cols = [
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
    # Ajuste para garantir que só pegue colunas existentes
    feature_cols = [c for c in feature_cols if c in df.columns]

    # Normalizar os dados (opcional, mas recomendado)
    features = df[feature_cols].astype(float)
    features = (features - features.mean()) / (features.std() + 1e-8)

    # Gera sequências para LSTM
    X_list = []
    y_list = []
    for i in range(len(features) - sequence_length):
        X_seq = features.iloc[i:i+sequence_length].values
        if "preco_medio_close" in features.columns:
            y_val = features.iloc[i+sequence_length]["preco_medio_close"]
        else:
            raise KeyError('"preco_medio_close" column is missing from features.')
        X_list.append(X_seq)
        y_list.append(y_val)

    X = torch.tensor(np.array(X_list), dtype=torch.float32)
    y = torch.tensor(np.array(y_list), dtype=torch.float32).unsqueeze(1)
    return X, y
# Create training and testing datasets from real data
train_X, train_y = load_real_data(num_samples, sequence_length, input_size)
test_X, test_y = load_real_data(num_samples // 10, sequence_length, input_size)
# Create artificial training and testing dataset

# Create DataLoaders
train_dataset = TensorDataset(train_X, train_y)
test_dataset = TensorDataset(test_X, test_y)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


def get_inner_layrs(input_size, hidden_size, num_layers, output_size):
    # Only include layers relevant to the current LSTM model architecture
    return {
        str(nn.LSTM) + "_1": nn.LSTM(input_size, hidden_size, num_layers, batch_first=True),
        str(nn.Linear) + "_1": nn.Linear(hidden_size, output_size)
    }

# LSTM model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)


        self.model = nn.Sequential(
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        batch_size = x.size(0)

        # Initialize hidden and cell states
        h0_1 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0_1 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        # First LSTM layer
        out, _ = self.lstm(x, (h0_1, c0_1))
        out = self.model(out[:, -1, :])

        return out


# Training the model
def train_model():
    model = LSTM(input_size, hidden_size, num_layers, output_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    mlflow.set_experiment("LSTM Artificial Data Regression")
    with mlflow.start_run():
        # Log model parameters
        mlflow.log_param("intermediate_layers", [*get_inner_layrs(input_size, hidden_size, num_layers, output_size).keys()])
        mlflow.set_experiment("LSTM Real Data Regression")
        mlflow.log_param("hidden_size", hidden_size)
        mlflow.log_param("num_layers", num_layers)
        mlflow.log_param("output_size", output_size)
        mlflow.log_param("num_epochs", num_epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", learning_rate)

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0

            for i, (sequences, labels) in enumerate(train_loader):
                sequences, labels = sequences.to(device), labels.to(device)

                # Forward pass
                outputs = model(sequences)
                loss = criterion(outputs, labels)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                
                # Log metrics every 100 batches
                if i % 100 == 0:
                    print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
                    mlflow.log_metric("train_loss", running_loss / (i+1), step=epoch * len(train_loader) + i)

        # Save the model
        # Before saving the model, define an input example
        example_input = torch.randn(1, sequence_length, input_size).numpy()
        mlflow.pytorch.log_model(model, "lstm_artificial_data_model", input_example=example_input)


def evaluate_model(model, criterion):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

    average_test_loss = test_loss / len(test_loader)
    print(f"Test Loss: {average_test_loss:.4f}")
    mlflow.log_metric("test_loss", average_test_loss)

# Run the training and evaluation
train_model()
print("Tamanho do dataset de treino:", len(train_X))
print("Num. sequências possíveis:", len(train_X))
print("Batch size:", batch_size)
