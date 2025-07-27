#!/usr/bin/env python3
"""
Log Management Utility
=====================

Utilitário para gerenciar arquivos de log do projeto LSTM Stock Prediction.
Realiza rotação, compressão e limpeza de logs antigos.
"""

import os
import sys
import glob
import shutil
import argparse
import logging
from datetime import datetime, timedelta
import gzip

# Configuração dos diretórios
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs')
LOG_CATEGORIES = ['api', 'training', 'monitoring', 'prediction']

# Configuração do logging para este script
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("log_manager")


def compress_log(log_file):
    """Comprime um arquivo de log usando gzip."""
    if not os.path.exists(log_file):
        logger.error(f"Arquivo não encontrado: {log_file}")
        return False
        
    try:
        with open(log_file, 'rb') as f_in:
            compressed_file = f"{log_file}.gz"
            with gzip.open(compressed_file, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        logger.info(f"Log comprimido: {compressed_file}")
        return True
    except Exception as e:
        logger.error(f"Erro ao comprimir {log_file}: {e}")
        return False


def rotate_logs(category, max_size_mb=10):
    """
    Realiza a rotação de logs quando atingem um tamanho máximo.
    
    Args:
        category: Categoria de logs (api, training, etc)
        max_size_mb: Tamanho máximo em MB
    """
    category_dir = os.path.join(LOGS_DIR, category)
    if not os.path.exists(category_dir):
        logger.warning(f"Diretório não encontrado: {category_dir}")
        return
    
    log_files = glob.glob(os.path.join(category_dir, "*.log"))
    
    for log_file in log_files:
        try:
            # Verificar tamanho do arquivo
            size_mb = os.path.getsize(log_file) / (1024 * 1024)
            if size_mb >= max_size_mb:
                # Criar novo nome com timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                base_name = os.path.basename(log_file)
                rotated_name = f"{os.path.splitext(base_name)[0]}_{timestamp}.log"
                rotated_path = os.path.join(category_dir, rotated_name)
                
                # Renomear arquivo atual
                shutil.move(log_file, rotated_path)
                logger.info(f"Log rotacionado: {base_name} -> {rotated_name}")
                
                # Comprimir arquivo rotacionado
                compress_log(rotated_path)
                
                # Remover arquivo original após compressão
                if os.path.exists(f"{rotated_path}.gz"):
                    os.remove(rotated_path)
        
        except Exception as e:
            logger.error(f"Erro ao rotacionar {log_file}: {e}")


def clean_old_logs(days=30):
    """
    Remove logs comprimidos mais antigos que o número de dias especificado.
    
    Args:
        days: Número de dias para manter logs
    """
    cutoff_date = datetime.now() - timedelta(days=days)
    
    for category in LOG_CATEGORIES:
        category_dir = os.path.join(LOGS_DIR, category)
        if not os.path.exists(category_dir):
            continue
            
        compressed_logs = glob.glob(os.path.join(category_dir, "*.gz"))
        
        for log_file in compressed_logs:
            try:
                file_time = datetime.fromtimestamp(os.path.getmtime(log_file))
                if file_time < cutoff_date:
                    os.remove(log_file)
                    logger.info(f"Log antigo removido: {log_file}")
            except Exception as e:
                logger.error(f"Erro ao limpar {log_file}: {e}")


def create_log_structure():
    """Cria a estrutura de diretórios para logs."""
    try:
        # Criar diretório principal de logs
        if not os.path.exists(LOGS_DIR):
            os.makedirs(LOGS_DIR)
        
        # Criar subdiretórios para cada categoria
        for category in LOG_CATEGORIES:
            category_dir = os.path.join(LOGS_DIR, category)
            if not os.path.exists(category_dir):
                os.makedirs(category_dir)
                
        logger.info("Estrutura de logs criada com sucesso")
        return True
    except Exception as e:
        logger.error(f"Erro ao criar estrutura de logs: {e}")
        return False


def list_logs():
    """Lista todos os arquivos de log organizados por categoria."""
    print("\n📊 LOGS POR CATEGORIA\n" + "=" * 50)
    
    total_size = 0
    total_files = 0
    
    for category in LOG_CATEGORIES:
        category_dir = os.path.join(LOGS_DIR, category)
        if not os.path.exists(category_dir):
            continue
            
        log_files = glob.glob(os.path.join(category_dir, "*.*"))
        
        if log_files:
            category_size = sum(os.path.getsize(f) for f in log_files)
            total_size += category_size
            total_files += len(log_files)
            
            print(f"\n📁 {category.upper()} ({len(log_files)} arquivos, {category_size / 1024:.2f} KB)")
            print("-" * 50)
            
            for log_file in sorted(log_files, key=os.path.getmtime, reverse=True):
                size_kb = os.path.getsize(log_file) / 1024
                mod_time = datetime.fromtimestamp(os.path.getmtime(log_file))
                print(f"  - {os.path.basename(log_file):<30} {size_kb:.2f} KB, {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n" + "=" * 50)
    print(f"Total: {total_files} arquivos, {total_size / 1024:.2f} KB")


def main():
    parser = argparse.ArgumentParser(description="Gerenciador de Logs para LSTM Stock Prediction")
    parser.add_argument("--rotate", action="store_true", help="Rotacionar logs grandes")
    parser.add_argument("--clean", action="store_true", help="Limpar logs antigos")
    parser.add_argument("--days", type=int, default=30, help="Dias para manter logs (padrão: 30)")
    parser.add_argument("--max-size", type=int, default=10, help="Tamanho máximo em MB para rotação (padrão: 10MB)")
    parser.add_argument("--list", action="store_true", help="Listar todos os logs")
    parser.add_argument("--setup", action="store_true", help="Criar estrutura de diretórios")
    
    args = parser.parse_args()
    
    if args.setup:
        create_log_structure()
        
    if args.rotate:
        for category in LOG_CATEGORIES:
            rotate_logs(category, args.max_size)
    
    if args.clean:
        clean_old_logs(args.days)
        
    if args.list:
        list_logs()
        
    # Se nenhuma opção for fornecida, mostrar ajuda
    if not (args.rotate or args.clean or args.list or args.setup):
        parser.print_help()


if __name__ == "__main__":
    main()
