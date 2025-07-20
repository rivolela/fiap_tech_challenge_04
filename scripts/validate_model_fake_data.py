#!/usr/bin/env python3
"""
Script para validar o modelo LSTM com dados fake
Gera dados sintéticos realistas para testar a API de predição
"""

import requests
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from typing import List, Dict, Any

class FakeDataGenerator:
    """Gerador de dados fake para validação do modelo LSTM"""
    
    def __init__(self):
        self.required_features = [
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
        
        # Parâmetros para geração de dados realistas
        self.base_price = 25.0
        self.volatility = 0.15
        self.trend = 0.02
    
    def generate_price_series(self, n_periods: int = 50) -> List[float]:
        """Gera série temporal realista de preços"""
        prices = [self.base_price]
        
        for i in range(1, n_periods):
            # Random walk with drift (tendência) e volatilidade
            change = np.random.normal(self.trend, self.volatility)
            new_price = prices[-1] * (1 + change)
            
            # Evita preços negativos
            new_price = max(new_price, 1.0)
            prices.append(new_price)
        
        return prices
    
    def calculate_technical_indicators(self, prices: List[float], window: int = 6) -> Dict[str, float]:
        """Calcula indicadores técnicos baseados nos preços"""
        recent_prices = prices[-window:] if len(prices) >= window else prices
        
        return {
            'media_movel': np.mean(recent_prices),
            'desvio_padrao': np.std(recent_prices),
            'valor_minimo': min(recent_prices),
            'valor_maximo': max(recent_prices)
        }
    
    def generate_fake_data(self, n_samples: int = 30) -> List[Dict[str, float]]:
        """Gera dados fake completos para validação"""
        print(f"🎲 Gerando {n_samples} registros de dados fake...")
        
        # Gerar série de preços mais longa para calcular lags
        extended_prices = self.generate_price_series(n_samples + 10)
        
        fake_data = []
        
        for i in range(n_samples):
            # Índice na série de preços (começando após os lags iniciais)
            price_idx = i + 10
            current_price = extended_prices[price_idx]
            
            # Calcular indicadores técnicos
            indicators = self.calculate_technical_indicators(
                extended_prices[:price_idx + 1], window=6
            )
            
            # Criar registro com todas as features necessárias
            record = {
                "preco_medio_close": round(current_price, 6),
                "lag_1_mes_preco_medio_close": round(extended_prices[price_idx - 1], 6),
                "lag_2_mes_preco_medio_close": round(extended_prices[price_idx - 2], 6),
                "lag_3_mes_preco_medio_close": round(extended_prices[price_idx - 3], 6),
                "lag_4_mes_preco_medio_close": round(extended_prices[price_idx - 4], 6),
                "lag_5_mes_preco_medio_close": round(extended_prices[price_idx - 5], 6),
                "lag_6_mes_preco_medio_close": round(extended_prices[price_idx - 6], 6),
                "media_movel_6_meses_preco_medio_close": round(indicators['media_movel'], 6),
                "desvio_padrao_movel_6_meses_preco_medio_close": round(indicators['desvio_padrao'], 6),
                "valor_minimo_6_meses_preco_medio_close": round(indicators['valor_minimo'], 6),
                "valor_maximo_6_meses_preco_medio_close": round(indicators['valor_maximo'], 6)
            }
            
            fake_data.append(record)
        
        return fake_data
    
    def validate_data_structure(self, data: List[Dict[str, float]]) -> bool:
        """Valida se os dados têm a estrutura correta"""
        if not data:
            print("❌ Dados vazios")
            return False
        
        sample_record = data[0]
        missing_features = []
        
        for feature in self.required_features:
            if feature not in sample_record:
                missing_features.append(feature)
        
        if missing_features:
            print(f"❌ Features faltando: {missing_features}")
            return False
        
        print("✅ Estrutura de dados válida")
        return True

class ModelValidator:
    """Validador do modelo usando a API"""
    
    def __init__(self, api_url: str = "http://localhost:8081"):
        self.api_url = api_url
        self.data_generator = FakeDataGenerator()
    
    def test_api_connection(self) -> bool:
        """Testa conexão com a API"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            if response.status_code == 200:
                print("✅ API conectada com sucesso")
                return True
            else:
                print(f"❌ API retornou status {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"❌ Erro de conexão: {e}")
            print("💡 Certifique-se que a API está rodando: python api_server.py")
            return False
    
    def validate_single_prediction(self, data: List[Dict[str, float]], 
                                 forecast_horizon: int = 6) -> Dict[str, Any]:
        """Valida uma predição única"""
        payload = {
            "forecast_horizon": forecast_horizon,
            "data": data
        }
        
        try:
            response = requests.post(
                f"{self.api_url}/predict",
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            
            result = response.json()
            return {
                'status_code': response.status_code,
                'success': response.status_code == 200,
                'result': result
            }
        except requests.exceptions.RequestException as e:
            return {
                'status_code': 0,
                'success': False,
                'error': str(e)
            }
    
    def run_multiple_validations(self, n_tests: int = 5) -> Dict[str, Any]:
        """Executa múltiplas validações com diferentes cenários"""
        print(f"\n🧪 Executando {n_tests} validações com dados fake...")
        print("=" * 60)
        
        results = {
            'total_tests': n_tests,
            'successful_tests': 0,
            'failed_tests': 0,
            'predictions': [],
            'errors': []
        }
        
        for i in range(1, n_tests + 1):
            print(f"\n🔬 Teste {i}/{n_tests}")
            print("-" * 30)
            
            # Gerar dados fake variando o tamanho
            data_size = random.randint(24, 50)  # Entre 24 e 50 registros
            fake_data = self.data_generator.generate_fake_data(data_size)
            
            # Validar estrutura
            if not self.data_generator.validate_data_structure(fake_data):
                results['failed_tests'] += 1
                results['errors'].append(f"Teste {i}: Estrutura de dados inválida")
                continue
            
            # Fazer predição
            print(f"📊 Enviando {len(fake_data)} registros para predição...")
            validation_result = self.validate_single_prediction(fake_data)
            
            if validation_result['success']:
                results['successful_tests'] += 1
                pred_result = validation_result['result']
                
                print(f"✅ Predição bem-sucedida!")
                print(f"   📈 Previsões: {pred_result.get('predictions', [])}")
                print(f"   📊 Horizon: {pred_result.get('forecast_horizon', 'N/A')}")
                
                results['predictions'].append({
                    'test_number': i,
                    'input_size': len(fake_data),
                    'predictions': pred_result.get('predictions', []),
                    'forecast_horizon': pred_result.get('forecast_horizon', 6)
                })
            else:
                results['failed_tests'] += 1
                error_msg = validation_result.get('error', 
                    validation_result.get('result', {}).get('error', 'Unknown error'))
                print(f"❌ Predição falhou: {error_msg}")
                results['errors'].append(f"Teste {i}: {error_msg}")
        
        return results
    
    def generate_validation_report(self, results: Dict[str, Any]) -> None:
        """Gera relatório de validação"""
        print("\n" + "=" * 60)
        print("📋 RELATÓRIO DE VALIDAÇÃO DO MODELO")
        print("=" * 60)
        
        print(f"📊 Testes Executados: {results['total_tests']}")
        print(f"✅ Testes Bem-sucedidos: {results['successful_tests']}")
        print(f"❌ Testes Falharam: {results['failed_tests']}")
        
        success_rate = (results['successful_tests'] / results['total_tests']) * 100
        print(f"📈 Taxa de Sucesso: {success_rate:.1f}%")
        
        if results['predictions']:
            print(f"\n🔮 PREDIÇÕES GERADAS:")
            for pred in results['predictions']:
                print(f"   Teste {pred['test_number']}: {pred['predictions']} "
                      f"(input: {pred['input_size']} registros)")
        
        if results['errors']:
            print(f"\n❌ ERROS ENCONTRADOS:")
            for error in results['errors']:
                print(f"   {error}")
        
        # Análise das predições
        if results['predictions']:
            all_predictions = []
            for pred in results['predictions']:
                all_predictions.extend(pred['predictions'])
            
            if all_predictions:
                print(f"\n📊 ANÁLISE ESTATÍSTICA DAS PREDIÇÕES:")
                print(f"   Média: {np.mean(all_predictions):.4f}")
                print(f"   Mediana: {np.median(all_predictions):.4f}")
                print(f"   Desvio Padrão: {np.std(all_predictions):.4f}")
                print(f"   Mín/Máx: {min(all_predictions):.4f} / {max(all_predictions):.4f}")

def main():
    """Função principal"""
    print("🚀 VALIDADOR DE MODELO LSTM COM DADOS FAKE")
    print("=" * 60)
    
    # Inicializar validador
    validator = ModelValidator()
    
    # Testar conexão
    if not validator.test_api_connection():
        return
    
    # Executar validações
    try:
        n_tests = int(input("\nQuantos testes executar? (padrão: 5): ") or "5")
        results = validator.run_multiple_validations(n_tests)
        
        # Gerar relatório
        validator.generate_validation_report(results)
        
        # Salvar dados fake em arquivo (opcional)
        save_data = input("\nSalvar dados fake gerados? (y/N): ").lower().strip()
        if save_data == 'y':
            fake_data = validator.data_generator.generate_fake_data(50)
            df = pd.DataFrame(fake_data)
            filename = f"fake_validation_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(filename, index=False)
            print(f"💾 Dados salvos em: {filename}")
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Validação interrompida pelo usuário")
    except Exception as e:
        print(f"\n❌ Erro durante validação: {e}")

if __name__ == "__main__":
    main()