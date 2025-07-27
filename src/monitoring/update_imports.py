#!/usr/bin/env python3
"""
Script para atualizar referências do sistema de monitoramento antigo para o novo.

Este script deve ser executado após a refatoração do módulo de monitoramento
para ajudar na transição dos imports e uso do módulo nos arquivos existentes.
"""

import os
import re
import sys
from typing import List, Dict, Tuple


def find_python_files(base_dir: str) -> List[str]:
    """Encontra todos os arquivos Python no diretório base e subdiretórios"""
    python_files = []
    
    for root, dirs, files in os.walk(base_dir):
        # Ignorar diretórios específicos
        if ('__pycache__' in root or 
            'venv' in root or 
            '.git' in root or
            'mlruns' in root):
            continue
            
        # Adicionar arquivos Python
        for file in files:
            if file.endswith('.py') and not file == 'original_monitoring.py':
                file_path = os.path.join(root, file)
                python_files.append(file_path)
                
    return python_files


def update_imports_in_file(file_path: str) -> Tuple[bool, int]:
    """
    Atualiza os imports no arquivo especificado
    
    Returns:
        Tuple[bool, int]: (Modificado?, Número de mudanças)
    """
    with open(file_path, 'r') as f:
        content = f.read()
        
    # Padrões para substituição
    replacements = {
        # Imports
        r'from src\.monitoring import ModelPerformanceMonitor': 'from src.monitoring import ModelPerformanceMonitor',
        r'from src\.monitoring import PredictionMetrics': 'from src.monitoring import PredictionMetrics',
        r'from src\.monitoring import SystemMetrics': 'from src.monitoring import SystemMetrics',
        
        # Decoradores
        r'@monitor_prediction\(': '@src.monitoring.utils.monitor_prediction(',
        r'@track_prediction': '@src.monitoring.utils.track_prediction',
        r'@time_execution': '@src.monitoring.time_execution',
        
        # Funções
        r'setup_monitoring_dashboard\(': 'src.monitoring.setup_monitoring_dashboard(',
        r'start_prometheus_server\(': 'src.monitoring.start_prometheus_server(',
    }
    
    changes = 0
    modified_content = content
    
    for pattern, replacement in replacements.items():
        new_content, count = re.subn(pattern, replacement, modified_content)
        if count > 0:
            changes += count
            modified_content = new_content
            
    # Se houve mudanças, escrever de volta no arquivo
    if changes > 0:
        with open(file_path, 'w') as f:
            f.write(modified_content)
        return True, changes
    
    return False, 0


def main():
    """Função principal do script"""
    # Obter diretório base do projeto (assumindo que este script está na raiz)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isdir(os.path.join(base_dir, 'src')):
        base_dir = os.path.dirname(base_dir)
        
    print(f"Atualizando imports do sistema de monitoramento no diretório: {base_dir}")
    
    # Encontrar arquivos Python
    python_files = find_python_files(base_dir)
    print(f"Encontrados {len(python_files)} arquivos Python para verificar.")
    
    # Processar cada arquivo
    modified_files = []
    total_changes = 0
    
    for file_path in python_files:
        modified, changes = update_imports_in_file(file_path)
        if modified:
            modified_files.append(file_path)
            total_changes += changes
            rel_path = os.path.relpath(file_path, base_dir)
            print(f"Arquivo atualizado: {rel_path} ({changes} mudanças)")
            
    # Resumo
    print(f"\nResumo:")
    print(f"Arquivos verificados: {len(python_files)}")
    print(f"Arquivos modificados: {len(modified_files)}")
    print(f"Total de mudanças: {total_changes}")
    
    if modified_files:
        print("\nLista de arquivos modificados:")
        for file_path in modified_files:
            rel_path = os.path.relpath(file_path, base_dir)
            print(f"- {rel_path}")
    

if __name__ == "__main__":
    main()
