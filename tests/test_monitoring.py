#!/usr/bin/env python3
"""
Script de Teste da API LSTM com Monitoramento
==============================================

Este script testa a API com dados fictícios e mostra as métricas de monitoramento.
"""

import requests
import json
import time
import sys

def test_api_endpoints(base_url="http://127.0.0.1:8000"):
    """Testa todos os endpoints da API"""
    
    print("🧪 Testando Endpoints da API LSTM")
    print("=" * 50)
    
    # 1. Health Check
    print("1. 🏥 Testando Health Check...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Health Check OK")
            print(f"   Status: {response.json()['data']['status']}")
        else:
            print(f"❌ Health Check Failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Erro no Health Check: {e}")
    
    # 2. API Info
    print("\n2. ℹ️ Testando API Info...")
    try:
        response = requests.get(f"{base_url}/", timeout=5)
        if response.status_code == 200:
            print("✅ API Info OK")
            data = response.json()['data']
            print(f"   API: {data.get('name', 'LSTM API')} v{data.get('version', '1.0')}")
        else:
            print(f"❌ API Info Failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Erro no API Info: {e}")
    
    # 3. Monitoramento - Health
    print("\n3. 📊 Testando Monitoramento - Health...")
    try:
        response = requests.get(f"{base_url}/monitoring/health", timeout=5)
        if response.status_code == 200:
            print("✅ Monitoring Health OK")
            health = response.json()
            print(f"   Status: {health['status']}")
            if 'checks' in health:
                for check, result in health['checks'].items():
                    print(f"   {check}: {result['status']}")
        else:
            print(f"❌ Monitoring Health Failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Erro no Monitoring Health: {e}")
    
    # 4. Monitoramento - Stats
    print("\n4. 📈 Testando Monitoramento - Stats...")
    try:
        response = requests.get(f"{base_url}/monitoring/stats", timeout=5)
        if response.status_code == 200:
            print("✅ Monitoring Stats OK")
            stats = response.json()
            print(f"   Total Predições: {stats.get('total_predictions', 0)}")
            if 'system_stats' in stats:
                sys_stats = stats['system_stats']
                print(f"   CPU Média: {sys_stats.get('avg_cpu_percent', 0):.1f}%")
                print(f"   Memória Média: {sys_stats.get('avg_memory_percent', 0):.1f}%")
        else:
            print(f"❌ Monitoring Stats Failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Erro no Monitoring Stats: {e}")
    
    # 5. Teste de Predição
    print("\n5. 🤖 Testando Predição do Modelo...")
    
    # Update test data to match expected format
    test_data = {
        "historical_data": [
            {
                "preco_medio_close": 100.0 + i,
                "lag_1_mes_preco_medio_close": 99.0 + i,
                "lag_2_mes_preco_medio_close": 98.0 + i,
                "lag_3_mes_preco_medio_close": 97.0 + i,
                "lag_4_mes_preco_medio_close": 96.0 + i,
                "lag_5_mes_preco_medio_close": 95.0 + i,
                "lag_6_mes_preco_medio_close": 94.0 + i,
                "media_movel_6_meses_preco_medio_close": 97.0 + i,
                "desvio_padrao_movel_6_meses_preco_medio_close": 2.0,
                "valor_minimo_6_meses_preco_medio_close": 94.0 + i,
                "valor_maximo_6_meses_preco_medio_close": 100.0 + i
            } for i in range(6)
        ],
        "forecast_horizon": 6
    }
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{base_url}/monitoring/predict",
            json=test_data,
            headers={'Content-Type': 'application/json'},
            timeout=30
        )
        duration = time.time() - start_time
        
        if response.status_code == 200:
            print("✅ Predição OK")
            result = response.json()
            predictions = result['predictions']
            print(f"   Tempo de resposta: {duration:.2f}s")
            print(f"   Predições geradas: {len(predictions)}")
            print(f"   Primeira predição: {predictions[0]:.2f}")
            print(f"   Última predição: {predictions[-1]:.2f}")
        else:
            print(f"❌ Predição Failed: {response.status_code}")
            print(f"   Resposta: {response.text}")
    except Exception as e:
        print(f"❌ Erro na Predição: {e}")
    
    # 6. Verificar métricas após predição
    print("\n6. 📊 Verificando métricas após predição...")
    time.sleep(1)  # Aguardar processamento das métricas
    
    try:
        response = requests.get(f"{base_url}/monitoring/stats", timeout=5)
        if response.status_code == 200:
            stats = response.json()
            print("✅ Métricas atualizadas")
            print(f"   Total Predições: {stats.get('total_predictions', 0)}")
            
            if 'prediction_stats' in stats:
                pred_stats = stats['prediction_stats']
                print(f"   Tempo médio: {pred_stats.get('avg_duration_ms', 0):.1f}ms")
                print(f"   Tempo P95: {pred_stats.get('p95_duration_ms', 0):.1f}ms")
    except Exception as e:
        print(f"❌ Erro ao verificar métricas: {e}")
    
    # 7. Teste de múltiplas predições
    print("\n7. 🔄 Testando múltiplas predições para estatísticas...")
    
    for i in range(3):
        try:
            # Generate a fresh set of 6 records for each prediction
            varied_data = {
                "historical_data": [
                    {
                        "preco_medio_close": 100.0 + i + j,
                        "lag_1_mes_preco_medio_close": 99.0 + i + j,
                        "lag_2_mes_preco_medio_close": 98.0 + i + j,
                        "lag_3_mes_preco_medio_close": 97.0 + i + j,
                        "lag_4_mes_preco_medio_close": 96.0 + i + j,
                        "lag_5_mes_preco_medio_close": 95.0 + i + j,
                        "lag_6_mes_preco_medio_close": 94.0 + i + j,
                        "media_movel_6_meses_preco_medio_close": 97.0 + i + j,
                        "desvio_padrao_movel_6_meses_preco_medio_close": 2.0,
                        "valor_minimo_6_meses_preco_medio_close": 94.0 + i + j,
                        "valor_maximo_6_meses_preco_medio_close": 100.0 + i + j
                    } for j in range(6)
                ],
                "forecast_horizon": 6
            }

            response = requests.post(
                f"{base_url}/monitoring/predict",
                json=varied_data,
                headers={'Content-Type': 'application/json'},
                timeout=30
            )

            if response.status_code == 200:
                print(f"   ✅ Predição {i+1}/3 OK")
            else:
                print(f"   ❌ Predição {i+1}/3 Failed: {response.status_code}")

        except Exception as e:
            print(f"   ❌ Erro na predição {i+1}: {e}")

        time.sleep(0.5)
    
    # 8. Estatísticas finais
    print("\n8. 📊 Estatísticas finais...")
    time.sleep(1)
    
    try:
        response = requests.get(f"{base_url}/monitoring/stats", timeout=5)
        if response.status_code == 200:
            stats = response.json()
            print("✅ Estatísticas coletadas")
            print(f"   Total Predições: {stats.get('total_predictions', 0)}")
            
            if 'prediction_stats' in stats:
                pred_stats = stats['prediction_stats']
                print(f"   Tempo médio: {pred_stats.get('avg_duration_ms', 0):.1f}ms")
                print(f"   Tempo mínimo: {pred_stats.get('min_duration_ms', 0):.1f}ms")
                print(f"   Tempo máximo: {pred_stats.get('max_duration_ms', 0):.1f}ms")
                print(f"   Memória média: {pred_stats.get('avg_memory_mb', 0):.1f}MB")
    except Exception as e:
        print(f"❌ Erro ao obter estatísticas finais: {e}")

def show_dashboard_info():
    """Mostra informações sobre como acessar o dashboard"""
    print("\n" + "=" * 50)
    print("📊 Dashboard de Monitoramento")
    print("=" * 50)
    print("🌐 Acesse o dashboard em: http://localhost:8000/monitoring/dashboard")
    print("📋 Endpoints disponíveis:")
    print("   • GET /monitoring/health     - Health check detalhado")
    print("   • GET /monitoring/stats      - Estatísticas de performance")
    print("   • GET /monitoring/recent     - Métricas recentes")
    print("   • GET /monitoring/metrics    - Métricas Prometheus")
    print("   • GET /monitoring/dashboard  - Dashboard web interativo")
    print("\n📁 Arquivos gerados:")
    print("   • monitoring_config.json     - Configuração do monitoramento")
    print("   • monitoring_dashboard.html  - Dashboard standalone")
    print("   • docs/MONITORING_GUIDE.md   - Guia completo")

if __name__ == "__main__":
    print("🚀 Teste Completo da API LSTM com Monitoramento")
    print("=" * 60)
    
    # Verificar se a API está rodando
    base_url = "http://localhost:8000"
    
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        print(f"✅ API detectada em {base_url}")
    except Exception as e:
        print(f"❌ API não está rodando em {base_url}")
        print("💡 Execute primeiro: python3 render_start.py")
        sys.exit(1)
    
    # Executar testes
    test_api_endpoints(base_url)
    
    # Mostrar informações do dashboard
    show_dashboard_info()
    
    print("\n" + "=" * 60)
    print("✅ Teste completo finalizado!")
    print("📊 Acesse o dashboard para monitoramento em tempo real")
    print("📖 Consulte docs/MONITORING_GUIDE.md para mais detalhes")
