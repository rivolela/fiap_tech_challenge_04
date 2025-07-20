#!/usr/bin/env python3
"""
Simple API Runner Script
"""
import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from src.api.app import run_app

if __name__ == '__main__':
    run_app()