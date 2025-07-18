#!/usr/bin/env python3
"""
Demo script showing how to use the exported LSTM model.
"""

import pandas as pd
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from ml.utils.model_exporter import ModelLoader

def demo_model_inference():
    """Demonstrate how to load and use the exported model."""
    
    print("🚀 LSTM Model Export Demo")
    print("=" * 50)
    
    # Path to exported model
    model_path = "./outputs/model_export/"
    
    if not os.path.exists(model_path):
        print("❌ No exported model found!")
        print("Run the training pipeline first to export a model.")
        return
    
    # Load the model
    print("📥 Loading model...")
    loader = ModelLoader(model_path)
    loader.load_complete_package()
    
    print("✅ Model loaded successfully!")
    print(f"Model configuration:")
    print(f"  - Input size: {loader.config['input_size']}")
    print(f"  - Hidden size: {loader.config['hidden_size']}")
    print(f"  - Num layers: {loader.config['num_layers']}")
    print(f"  - Output size: {loader.config['output_size']}")
    print(f"  - Feature columns: {len(loader.feature_columns)} features")
    
    # Load some sample data for prediction
    print("\n📊 Loading sample data...")
    try:
        # Try to load the transformed data
        data_path = "./data/transformed/"
        if os.path.exists(data_path):
            import glob
            parquet_files = glob.glob(os.path.join(data_path, "*.parquet"))
            if parquet_files:
                df = pd.read_parquet(parquet_files[0])
                print(f"✅ Loaded data with shape: {df.shape}")
                
                # Use last 50 rows for prediction (model needs last 24)
                sample_data = df.tail(50)
                
                # Make prediction
                print("\n🔮 Making prediction...")
                prediction = loader.predict(sample_data)
                
                print(f"✅ Prediction completed!")
                print(f"Raw prediction shape: {prediction.shape}")
                print(f"Predicted values (scaled):")
                if prediction.ndim == 2:
                    # Multi-output case (like 6-month forecast)
                    for i, pred in enumerate(prediction[0], 1):
                        print(f"  Month {i}: {pred:.4f}")
                else:
                    print(f"  Single prediction: {prediction}")
                
                # Note about scaling
                print("\n📝 Note: These are scaled predictions.")
                print("   In production, you would inverse transform them")
                print("   to get actual values in original scale.")
                
            else:
                print("❌ No parquet files found in transformed data folder")
        else:
            print("❌ Transformed data folder not found")
            
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
    
    print("\n" + "=" * 50)
    print("✅ Demo completed!")

def show_export_files():
    """Show what files are in the export directory."""
    model_path = "./outputs/model_export/"
    
    if not os.path.exists(model_path):
        print("❌ Export directory doesn't exist")
        return
    
    print("\n📁 Exported Model Files:")
    print("-" * 30)
    
    for file in os.listdir(model_path):
        file_path = os.path.join(model_path, file)
        size = os.path.getsize(file_path) / 1024  # Size in KB
        print(f"  📄 {file} ({size:.1f} KB)")

if __name__ == "__main__":
    print("Choose an option:")
    print("1. Run inference demo")
    print("2. Show export files")
    print("3. Both")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice in ["1", "3"]:
        demo_model_inference()
    
    if choice in ["2", "3"]:
        show_export_files()
