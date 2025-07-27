"""
Monitor de Data Drift e Métricas de Erro para LSTM API
Utiliza Evidently para detectar mudanças na distribuição dos dados e avaliar métricas de erro.
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime
from evidently.report import Report
from evidently.metrics import (
    DataDriftTable,
    # DataQualityPreset,        # REMOVE or comment out
    # RegressionPreset,         # REMOVE or comment out
    ColumnQuantileMetric,
    ColumnSummaryMetric,
    DatasetSummaryMetric,
    DatasetDriftMetric,
    DatasetMissingValuesMetric,
    ColumnDriftMetric,
)
import logging
from typing import Dict, List, Optional, Union, Any

logger = logging.getLogger(__name__)

# Diretório para salvar relatórios
REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                           "outputs", "drift_reports")

# Criar diretório se não existir
if not os.path.exists(REPORTS_DIR):
    os.makedirs(REPORTS_DIR)

class DriftMonitor:
    """
    Monitor de Data Drift para modelos LSTM
    
    Monitora mudanças na distribuição dos dados de entrada e métricas de erro
    como MAE, RMSE, distribuição de erro, normalidade e viés.
    """
    def __init__(self, reference_data_path: Optional[str] = None):
        """
        Inicializa o monitor de drift
        
        Args:
            reference_data_path: Caminho para os dados de referência (treino)
                                Se None, será criado na primeira chamada
        """
        self.reference_data = None
        if reference_data_path and os.path.exists(reference_data_path):
            try:
                self.reference_data = pd.read_csv(reference_data_path)
                logger.info(f"Dados de referência carregados de {reference_data_path}")
            except Exception as e:
                logger.error(f"Erro ao carregar dados de referência: {e}")
        
        self.current_predictions = []
        logger.info("Monitor de data drift inicializado")
    
    def add_prediction(self, features: Dict[str, Any], prediction: Union[float, List[float]], actual: Optional[float] = None) -> None:
        """
        Adiciona uma predição ao monitor
        
        Args:
            features: Dicionário com features usadas na predição
            prediction: Valor predito pelo modelo (pode ser único ou lista)
            actual: Valor real (se disponível)
        """
        # Armazenar predição
        pred_data = features.copy()
        
        # Handle diferentes formatos de predição
        if isinstance(prediction, list):
            pred_data["prediction"] = prediction[0] if prediction else None
        elif isinstance(prediction, (np.ndarray, np.generic)):
            pred_data["prediction"] = float(prediction[0]) if len(prediction) > 0 else None
        else:
            pred_data["prediction"] = float(prediction) if prediction is not None else None
        
        # Adicionar valor real se disponível
        if actual is not None:
            pred_data["target"] = float(actual)
        
        # Adicionar timestamp para análises temporais
        pred_data["timestamp"] = datetime.now().isoformat()
        
        self.current_predictions.append(pred_data)
        
        # Log a cada N predições para não sobrecarregar os logs
        if len(self.current_predictions) % 100 == 0:
            logger.info(f"Monitor de drift: {len(self.current_predictions)} predições coletadas")
        
    def generate_report(self, save: bool = True) -> Optional[str]:
        """
        Gera relatório de drift e métricas de erro
        
        Args:
            save: Se True, salva o relatório em HTML
            
        Returns:
            Caminho para o relatório gerado ou None se falhar
        """
        if not self.current_predictions:
            logger.warning("Sem dados para gerar relatório de drift")
            return None
        
        # Converter lista de dicts para DataFrame
        current_data = pd.DataFrame(self.current_predictions)
        
        # Remover colunas não numéricas exceto target e prediction
        non_numeric_cols = [col for col in current_data.columns 
                           if not np.issubdtype(current_data[col].dtype, np.number)
                           and col not in ['target', 'prediction', 'timestamp']]
        
        if non_numeric_cols:
            logger.warning(f"Removendo colunas não numéricas para análise: {non_numeric_cols}")
            current_data = current_data.drop(columns=non_numeric_cols)
        
        # Se não temos dados de referência, usar os primeiros 30% como referência
        if self.reference_data is None:
            split_idx = max(1, int(len(current_data) * 0.3))
            self.reference_data = current_data.iloc[:split_idx].copy()
            current_data = current_data.iloc[split_idx:].copy()
            logger.info(f"Criado dataset de referência com {len(self.reference_data)} registros")
        
        # Verificar se temos target (valores reais)
        has_target = "target" in current_data.columns and "target" in self.reference_data.columns
        
        # Selecionar métricas apropriadas
        metrics = [DataDriftTable()]
        
        # Adicionar métricas de regressão se tiver target
        if has_target:
            # Only add ColumnDriftMetric for columns that exist in both datasets
            common_numeric_cols = [col for col in current_data.columns 
                                  if col in self.reference_data.columns 
                                  and col not in ['prediction', 'target', 'timestamp']
                                  and np.issubdtype(current_data[col].dtype, np.number)]
            
            metrics.extend([
                ColumnQuantileMetric(column_name="preco_medio_close", quantile=0.5),
                ColumnQuantileMetric(column_name="preco_medio_close", quantile=0.95),
                ColumnSummaryMetric(column_name="preco_medio_close"),
                DatasetSummaryMetric(),
                DatasetDriftMetric(),
                DatasetMissingValuesMetric(),
            ])
            
            # Add ColumnDriftMetric only for existing columns
            for col in common_numeric_cols:
                metrics.append(ColumnDriftMetric(column_name=col))
            
            logger.info("Incluindo métricas de erro de regressão no relatório")
        
        # Gerar relatório
        report = Report(metrics=metrics)
        filepath = None
        
        try:
            report.run(reference_data=self.reference_data, current_data=current_data)
            logger.info(f"Relatório de drift gerado com sucesso: {len(current_data)} registros analisados")
            
            # Salvar relatório
            if save:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"drift_report_{timestamp}.html"
                filepath = os.path.join(REPORTS_DIR, filename)
                
                report.save_html(filepath)
                logger.info(f"Relatório salvo em: {filepath}")
            
            return filepath
        
        except Exception as e:
            logger.error(f"Erro ao gerar relatório de drift: {e}")
            return None
    
    def save_reference_data(self, filepath: Optional[str] = None) -> Optional[str]:
        """
        Salva os dados de referência para uso futuro
        
        Args:
            filepath: Caminho para salvar os dados. Se None, usa o padrão.
        
        Returns:
            Caminho onde os dados foram salvos ou None se falhou
        """
        if self.reference_data is not None:
            if filepath is None:
                filepath = os.path.join(REPORTS_DIR, "reference_data.csv")
            
            try:
                self.reference_data.to_csv(filepath, index=False)
                logger.info(f"Dados de referência salvos em {filepath}")
                return filepath
            except Exception as e:
                logger.error(f"Erro ao salvar dados de referência: {e}")
        
        return None
    
    def reset_current_data(self) -> None:
        """Limpa os dados atuais, mantendo a referência"""
        count = len(self.current_predictions)  # Fixed syntax
        self.current_predictions = []
        logger.info(f"Dados de monitoramento resetados: {count} registros removidos")
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Retorna métricas básicas sobre os dados coletados
        
        Returns:
            Dicionário com métricas básicas de drift e erro
        """
        if not self.current_predictions:
            return {
                "count": 0,
                "status": "no_data",
                "has_reference_data": self.reference_data is not None,
                "reference_count": len(self.reference_data) if self.reference_data is not None else 0
            }
        
        count = len(self.current_predictions)
        df = pd.DataFrame(self.current_predictions)
        
        metrics = {
            "count": count,
            "status": "ok",
            "has_reference_data": self.reference_data is not None,
            "reference_count": len(self.reference_data) if self.reference_data is not None else 0,
        }
        
        # Estatísticas básicas das predições
        if "prediction" in df.columns:
            metrics["prediction_stats"] = {
                "mean": float(df["prediction"].mean()),
                "std": float(df["prediction"].std()),
                "min": float(df["prediction"].min()),
                "max": float(df["prediction"].max()),
                "median": float(df["prediction"].median())
            }
        
        # Métricas de erro se tiver valores reais
        if "target" in df.columns and "prediction" in df.columns:
            # Remover NaN para cálculo das métricas
            valid_mask = ~(df["target"].isna() | df["prediction"].isna())
            if valid_mask.sum() > 0:
                valid_df = df.loc[valid_mask]
                y_true = valid_df["target"]
                y_pred = valid_df["prediction"]
                
                metrics["error_metrics"] = {
                    "mae": float(np.abs(y_pred - y_true).mean()),
                    "rmse": float(np.sqrt(((y_pred - y_true) ** 2).mean())),
                    "me": float((y_pred - y_true).mean()),  # Mean Error (bias)
                    "mape": float(np.abs((y_pred - y_true) / (y_true + 1e-8)).mean() * 100)  # Mean Absolute Percentage Error
                }
        
        # Detectar drift básico (comparação com referência)
        if self.reference_data is not None and len(df) > 5:
            try:
                # Comparar distribuição das features numéricas
                common_cols = [col for col in self.reference_data.columns 
                              if col in df.columns
                              and col not in ["prediction", "target", "timestamp"]
                              and np.issubdtype(df[col].dtype, np.number)]
                
                if common_cols:
                    # Calcular divergência para cada feature
                    drift_scores = {}
                    for col in common_cols:
                        ref_mean = float(self.reference_data[col].mean())
                        ref_std = float(self.reference_data[col].std())
                        curr_mean = float(df[col].mean())
                        curr_std = float(df[col].std())
                        
                        # Distância normalizada entre médias
                        if ref_std > 0:
                            z_diff = abs(curr_mean - ref_mean) / ref_std
                            drift_scores[col] = float(z_diff)
                    
                    # Resumo de drift
                    high_drift_features = [col for col, score in drift_scores.items() if score > 0.5]
                    metrics["drift_detection"] = {
                        "drift_scores": drift_scores,
                        "high_drift_features": high_drift_features,
                        "drift_status": "high" if any(score > 1.0 for score in drift_scores.values()) else 
                                       ("medium" if any(score > 0.5 for score in drift_scores.values()) else "low")
                    }
            except Exception as e:
                logger.warning(f"Erro ao calcular métricas de drift: {e}")
        
        return metrics


# Exemplo simples de uso
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Criar monitor
    monitor = DriftMonitor()
    
    # Adicionar algumas predições simuladas
    for i in range(100):
        features = {
            "preco_medio_close": 100.0 + i * 0.5,
            "lag_1_mes_preco_medio_close": 99.0 + i * 0.5,
            "lag_2_mes_preco_medio_close": 98.0 + i * 0.5,
        }
        prediction = 105.0 + i * 0.6  # Predição simulada
        actual = 105.0 + i * 0.6 + np.random.normal(0, 2)  # Valor real simulado
        
        monitor.add_prediction(features, prediction, actual)
    
    # Gerar relatório
    report_path = monitor.generate_report()
    print(f"Relatório gerado: {report_path}")
    
    # Mostrar métricas básicas
    metrics = monitor.get_metrics()
    print(f"Métricas de drift: {metrics}")