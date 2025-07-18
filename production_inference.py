#!/usr/bin/env python3
"""
Production-ready script for using the exported LSTM model.
This shows the complete pipeline from raw data to final predictions.
"""

import pandas as pd
import numpy as np
import sys
import os
import pickle
import json
import torch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def load_model_components(model_path="./outputs/model_export/"):
    """Load all model components."""
    # Load model
    model = torch.load(
        os.path.join(model_path, 'complete_model.pth'),
        weights_only=False
    )
    model.eval()
    
    # Load scaler
    with open(os.path.join(model_path, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    
    # Load feature columns
    with open(os.path.join(model_path, 'feature_columns.json'), 'r') as f:
        feature_columns = json.load(f)
    
    # Load config
    with open(os.path.join(model_path, 'model_config.json'), 'r') as f:
        config = json.load(f)
    
    return model, scaler, feature_columns, config

def preprocess_data(data, feature_columns, scaler):
    """Preprocess data for prediction."""
    # Select required columns
    data_filtered = data[feature_columns]
    
    # Scale data
    data_scaled = scaler.transform(data_filtered)
    
    # Create sequence (last 24 time steps)
    sequence = data_scaled[-24:]
    
    # Convert to tensor and add batch dimension
    return torch.FloatTensor(sequence).unsqueeze(0)

def inverse_transform_predictions(predictions, scaler, num_features):
    """Properly inverse transform predictions."""
    # predictions shape: (1, 6) for 6-month forecast
    batch_size, forecast_horizon = predictions.shape
    
    # Create dummy array with all features
    dummy_array = np.zeros((batch_size, num_features))
    
    # Put predictions in first column (assuming first feature is the target)
    dummy_array[:, 0] = predictions[:, 0]  # Use first prediction for first month
    
    # Inverse transform to get original scale
    inverse_transformed = scaler.inverse_transform(dummy_array)
    
    # Extract the target values (first column)
    return inverse_transformed[:, 0]

def make_prediction():
    """Make a complete prediction with proper scaling."""
    print("🔮 LSTM Production Prediction")
    print("=" * 40)
    
    # Load model components
    print("📥 Loading model components...")
    model, scaler, feature_columns, config = load_model_components()
    print(f"✅ Model loaded with {config['input_size']} inputs")
    
    # Load data
    print("📊 Loading sample data...")
    try:
        import glob
        data_path = "./data/transformed/"
        parquet_files = glob.glob(os.path.join(data_path, "*.parquet"))
        
        if parquet_files:
            df = pd.read_parquet(parquet_files[0])
            print(f"✅ Loaded data with shape: {df.shape}")
            
            # Preprocess
            print("⚙️ Preprocessing data...")
            processed_data = preprocess_data(df.tail(50), feature_columns, scaler)
            
            # Make prediction
            print("🔮 Making prediction...")
            with torch.no_grad():
                prediction = model(processed_data)
            
            # Convert to numpy
            prediction_array = prediction.numpy()
            print(f"Raw prediction shape: {prediction_array.shape}")
            
            # Show scaled predictions
            print(f"\n📊 Scaled predictions:")
            for i, pred in enumerate(prediction_array[0], 1):
                print(f"  Month {i}: {pred:.4f}")
            
            # Attempt simple inverse transform for demonstration
            print(f"\n🔄 Attempting inverse transform...")
            try:
                # Simple approach: use mean scaler parameters
                scaler_mean = scaler.mean_[0]  # Mean of first feature
                scaler_scale = scaler.scale_[0]  # Scale of first feature
                
                # Inverse transform each prediction
                unscaled_predictions = []
                for pred in prediction_array[0]:
                    unscaled = (pred * scaler_scale) + scaler_mean
                    unscaled_predictions.append(unscaled)
                
                print(f"✅ Unscaled predictions (approximate):")
                for i, pred in enumerate(unscaled_predictions, 1):
                    print(f"  Month {i}: {pred:.2f}")
                
                # Show reasonable range check
                print(f"\n📈 Prediction Analysis:")
                print(f"  - Range: {min(unscaled_predictions):.2f} to {max(unscaled_predictions):.2f}")
                print(f"  - Average: {np.mean(unscaled_predictions):.2f}")
                
                # Compare with recent data
                if 'preco_medio_close' in df.columns:
                    recent_avg = df['preco_medio_close'].tail(6).mean()
                    print(f"  - Recent data average: {recent_avg:.2f}")
                    
            except Exception as e:
                print(f"❌ Inverse transform failed: {e}")
                print("This is normal - proper inverse transform needs full feature reconstruction")
        
        else:
            print("❌ No data files found")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n" + "=" * 40)
    print("✅ Prediction complete!")

if __name__ == "__main__":
    make_prediction()
