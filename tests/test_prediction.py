#!/usr/bin/env python3
"""
Teste simples para validar predições com monitoramento
"""

import requests
import json
import numpy as np

# Gerar dados fake com as features corretas
def generate_test_data(n_records=30):
    """Gera dados de teste com as features esperadas pelo modelo"""
    
    features = [
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
    
    data = []
    base_price = 25.0
    
    for i in range(n_records):
        # Simular variação de preço
        price = base_price + np.random.normal(0, 2)
        
        record = {}
        for feature in features:
            if "lag" in feature:
                # Lags são preços anteriores
                record[feature] = price + np.random.normal(0, 1)
            elif "media_movel" in feature:
                # Média móvel
                record[feature] = price + np.random.normal(0, 0.5)
            elif "desvio_padrao" in feature:
                # Desvio padrão
                record[feature] = abs(np.random.normal(1, 0.3))
            elif "minimo" in feature:
                # Valor mínimo
                record[feature] = price - abs(np.random.normal(2, 0.5))
            elif "maximo" in feature:
                # Valor máximo
                record[feature] = price + abs(np.random.normal(2, 0.5))
            else:
                # Preço atual
                record[feature] = price
        
        data.append(record)
    
    return data

def test_prediction_with_monitoring():
    """Testa predição e verifica monitoramento"""
    
    # Gerar dados de teste
    test_data = generate_test_data(30)
    
    # Fazer predição
    payload = {
        "data": test_data,
        "forecast_horizon": 6
    }
    
    print("🚀 Testando predição com monitoramento...")
    print(f"📊 Enviando {len(test_data)} registros")
    
    try:
        # Fazer predição
        response = requests.post(
            "http://localhost:8000/predict",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Predição realizada com sucesso!")
            print(f"🔮 {len(result['data']['predictions'])} predições geradas")
            print(f"⏱️ Tempo de processamento: {result['data']['processing_time_ms']:.2f}ms")
            
            # Verificar estatísticas de monitoramento
            stats_response = requests.get("http://localhost:8000/monitoring/stats")
            if stats_response.status_code == 200:
                stats = stats_response.json()
                print("\n📊 ESTATÍSTICAS DE MONITORAMENTO:")
                print(f"🔢 Total de predições: {stats['total_predictions']}")
                print(f"⏱️ Duração média: {stats['prediction_stats']['avg_duration_ms']:.2f}ms")
                print(f"🖥️ CPU médio: {stats['system_stats']['avg_cpu_percent']:.1f}%")
                print(f"💻 Memória média: {stats['system_stats']['avg_memory_percent']:.1f}%")
                
                if stats['total_predictions'] > 0:
                    print("✅ Monitoramento está funcionando - predições sendo contadas!")
                else:
                    print("❌ Monitoramento não está contando predições")
            else:
                print("❌ Erro ao obter estatísticas de monitoramento")
                
        else:
            print(f"❌ Erro na predição: {response.status_code}")
            print(f"Resposta: {response.text}")
            
    except Exception as e:
        print(f"❌ Erro: {e}")

if __name__ == "__main__":
    test_prediction_with_monitoring()
