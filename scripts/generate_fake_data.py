#!/usr/bin/env python3
"""
Gerador de dados fake para desenvolvimento e testes
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
import argparse

def generate_fake_stock_data(n_samples: int = 100, save_format: str = 'csv') -> None:
    """Gera dados fake de ações com todas as features necessárias"""
    
    print(f"🎲 Gerando {n_samples} registros de dados fake de ações...")
    
    # Parâmetros base
    base_price = 25.0
    volatility = 0.15
    trend = 0.02
    
    # Gerar série de preços
    prices = [base_price]
    for i in range(1, n_samples + 10):
        change = np.random.normal(trend, volatility)
        new_price = prices[-1] * (1 + change)
        new_price = max(new_price, 1.0)  # Evitar preços negativos
        prices.append(new_price)
    
    # Criar DataFrame com todas as features
    data = []
    for i in range(n_samples):
        price_idx = i + 10
        current_price = prices[price_idx]
        
        # Calcular indicadores técnicos dos últimos 6 períodos
        recent_6 = prices[price_idx-5:price_idx+1]
        
        record = {
            "preco_medio_close": round(current_price, 6),
            "lag_1_mes_preco_medio_close": round(prices[price_idx - 1], 6),
            "lag_2_mes_preco_medio_close": round(prices[price_idx - 2], 6),
            "lag_3_mes_preco_medio_close": round(prices[price_idx - 3], 6),
            "lag_4_mes_preco_medio_close": round(prices[price_idx - 4], 6),
            "lag_5_mes_preco_medio_close": round(prices[price_idx - 5], 6),
            "lag_6_mes_preco_medio_close": round(prices[price_idx - 6], 6),
            "media_movel_6_meses_preco_medio_close": round(np.mean(recent_6), 6),
            "desvio_padrao_movel_6_meses_preco_medio_close": round(np.std(recent_6), 6),
            "valor_minimo_6_meses_preco_medio_close": round(min(recent_6), 6),
            "valor_maximo_6_meses_preco_medio_close": round(max(recent_6), 6)
        }
        data.append(record)
    
    df = pd.DataFrame(data)
    
    # Salvar dados
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if save_format.lower() == 'csv':
        filename = f"fake_stock_data_{timestamp}.csv"
        df.to_csv(filename, index=False)
        print(f"💾 Dados salvos em: {filename}")
    elif save_format.lower() == 'json':
        filename = f"fake_stock_data_{timestamp}.json"
        df.to_json(filename, orient='records', indent=2)
        print(f"💾 Dados salvos em: {filename}")
    elif save_format.lower() == 'parquet':
        filename = f"fake_stock_data_{timestamp}.parquet"
        df.to_parquet(filename, index=False)
        print(f"💾 Dados salvos em: {filename}")
    
    # Mostrar amostra dos dados
    print("\n📊 Amostra dos dados gerados:")
    print(df.head(3).to_string())
    print(f"\n📈 Estatísticas básicas:")
    print(f"   Preço médio: {df['preco_medio_close'].mean():.2f}")
    print(f"   Variação: {df['preco_medio_close'].min():.2f} - {df['preco_medio_close'].max():.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gerador de dados fake de ações")
    parser.add_argument("-n", "--samples", type=int, default=100, 
                       help="Número de amostras a gerar (padrão: 100)")
    parser.add_argument("-f", "--format", choices=['csv', 'json', 'parquet'], 
                       default='csv', help="Formato de saída (padrão: csv)")
    
    args = parser.parse_args()
    generate_fake_stock_data(args.samples, args.format)