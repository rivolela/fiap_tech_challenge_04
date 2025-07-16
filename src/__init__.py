# In the main() function, update the visualization section:
        # 5. Visualizar resultados
        print("\nGerando visualização dos resultados...")
        plotar_previsoes(resultados, output_dir='outputs')
        
        # 6. Visualizar curvas de perda
        print("\nGerando visualização das curvas de perda...")
        plotar_perdas(perdas_treino, perdas_teste, output_dir='outputs')