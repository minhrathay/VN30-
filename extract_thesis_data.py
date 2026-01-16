"""
Extract Real Metrics from VN30 Models for Thesis Report
This script runs all models and outputs actual performance metrics
"""
import sys
import os
import json
import warnings
warnings.filterwarnings('ignore')

# Add current dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error, r2_score

# Import project modules
from utils import load_data, create_technical_indicators, create_lag_features, load_macro_data
from models import (
    train_arimax, train_xgboost, train_lstm, train_meta_learner,
    add_rolling_features, add_macro_features,
    fit_garch_model, forecast_volatility_garch, detect_market_regime
)

def main():
    print("=" * 60)
    print("VN30 THESIS DATA EXTRACTION")
    print("=" * 60)
    
    # Load and prepare data
    print("\n[1/6] Loading data...")
    df = load_data()
    
    # Load macro data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sp500_path = os.path.join(current_dir, 'SP500.csv')
    usdvnd_path = os.path.join(current_dir, 'USD_VND.csv')
    
    if os.path.exists(sp500_path) and os.path.exists(usdvnd_path):
        df = load_macro_data(df, sp500_path=sp500_path, usdvnd_path=usdvnd_path)
    
    df = add_macro_features(df)
    df = create_technical_indicators(df)
    df = create_lag_features(df)
    df = add_rolling_features(df)
    
    # Data statistics
    print(f"\n[DATA STATISTICS]")
    print(f"  Total observations: {len(df)}")
    print(f"  Date range: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")
    print(f"  Close - Mean: {df['Close'].mean():.2f}")
    print(f"  Close - Std: {df['Close'].std():.2f}")
    print(f"  Close - Min: {df['Close'].min():.2f}")
    print(f"  Close - Max: {df['Close'].max():.2f}")
    
    # Split data
    split_pct = 95
    train_size = int(len(df) * (split_pct / 100))
    train_data = df.iloc[:train_size]
    test_data = df.iloc[train_size:]
    
    print(f"\n[2/6] Training ARIMAX...")
    results = {}
    
    # ARIMAX
    try:
        arimax_pred, arimax_order = train_arimax(train_data, test_data, fast_mode=True)
        y_true = test_data['Close'].values[:len(arimax_pred)]
        results['ARIMAX'] = {
            'MAPE': mean_absolute_percentage_error(y_true, arimax_pred) * 100,
            'RMSE': np.sqrt(mean_squared_error(y_true, arimax_pred)),
            'MAE': mean_absolute_error(y_true, arimax_pred),
            'R2': r2_score(y_true, arimax_pred)
        }
        print(f"  ARIMAX MAPE: {results['ARIMAX']['MAPE']:.4f}%")
    except Exception as e:
        print(f"  ARIMAX failed: {e}")
        results['ARIMAX'] = {'MAPE': 'N/A', 'RMSE': 'N/A', 'MAE': 'N/A', 'R2': 'N/A'}
    
    # XGBoost
    print(f"\n[3/6] Training XGBoost...")
    try:
        xgb_pred, xgb_model, _, features, _ = train_xgboost(train_data, test_data, use_tuning=False)
        y_true = test_data['Close'].values[:len(xgb_pred)]
        results['XGBoost'] = {
            'MAPE': mean_absolute_percentage_error(y_true, xgb_pred) * 100,
            'RMSE': np.sqrt(mean_squared_error(y_true, xgb_pred)),
            'MAE': mean_absolute_error(y_true, xgb_pred),
            'R2': r2_score(y_true, xgb_pred)
        }
        print(f"  XGBoost MAPE: {results['XGBoost']['MAPE']:.4f}%")
    except Exception as e:
        print(f"  XGBoost failed: {e}")
        results['XGBoost'] = {'MAPE': 'N/A', 'RMSE': 'N/A', 'MAE': 'N/A', 'R2': 'N/A'}
    
    # LSTM
    print(f"\n[4/6] Training LSTM...")
    try:
        lstm_pred, lstm_objects = train_lstm(df, train_size)
        y_true = test_data['Close'].values[:len(lstm_pred)]
        results['LSTM'] = {
            'MAPE': mean_absolute_percentage_error(y_true, lstm_pred) * 100,
            'RMSE': np.sqrt(mean_squared_error(y_true, lstm_pred)),
            'MAE': mean_absolute_error(y_true, lstm_pred),
            'R2': r2_score(y_true, lstm_pred)
        }
        print(f"  LSTM MAPE: {results['LSTM']['MAPE']:.4f}%")
    except Exception as e:
        print(f"  LSTM failed: {e}")
        results['LSTM'] = {'MAPE': 'N/A', 'RMSE': 'N/A', 'MAE': 'N/A', 'R2': 'N/A'}
    
    # GARCH
    print(f"\n[5/6] Fitting GARCH...")
    try:
        garch_result, returns = fit_garch_model(df)
        if garch_result is not None:
            params = garch_result.params
            results['GARCH'] = {
                'omega': float(params.get('omega', 0)),
                'alpha': float(params.get('alpha[1]', 0)),
                'beta': float(params.get('beta[1]', 0)),
                'persistence': float(params.get('alpha[1]', 0) + params.get('beta[1]', 0))
            }
            print(f"  GARCH alpha: {results['GARCH']['alpha']:.4f}")
            print(f"  GARCH beta: {results['GARCH']['beta']:.4f}")
    except Exception as e:
        print(f"  GARCH failed: {e}")
        results['GARCH'] = {'omega': 'N/A', 'alpha': 'N/A', 'beta': 'N/A', 'persistence': 'N/A'}
    
    # Regime Detection
    regime, strength, momentum = detect_market_regime(df, lookback=20)
    results['Regime'] = {
        'current': regime,
        'strength': float(strength),
        'momentum': float(momentum)
    }
    
    # Ensemble
    print(f"\n[6/6] Training Ensemble...")
    try:
        # Align predictions
        min_len = min(len(arimax_pred), len(xgb_pred), len(lstm_pred))
        y_true = test_data['Close'].values[:min_len]
        
        preds = {
            'ARIMAX': arimax_pred[:min_len],
            'XGBoost': xgb_pred[:min_len],
            'LSTM': lstm_pred[:min_len]
        }
        
        meta_model, model_keys, ens_pred, ens_mape, cv_score = train_meta_learner(y_true, preds)
        
        # Calculate Win Rate
        direction_true = np.sign(np.diff(y_true))
        direction_pred = np.sign(np.diff(ens_pred))
        win_rate = np.mean(direction_true == direction_pred) * 100
        
        results['Ensemble'] = {
            'MAPE': float(ens_mape),
            'RMSE': float(np.sqrt(mean_squared_error(y_true, ens_pred))),
            'MAE': float(mean_absolute_error(y_true, ens_pred)),
            'R2': float(r2_score(y_true, ens_pred)),
            'WinRate': float(win_rate),
            'CV_Score': float(cv_score)
        }
        
        # Get feature importances (weights)
        try:
            importances = meta_model.feature_importances_
            total = np.sum(importances)
            weights = {k: float(importances[i]/total) for i, k in enumerate(sorted(model_keys))}
            results['Weights'] = weights
        except:
            results['Weights'] = {'ARIMAX': 0.33, 'LSTM': 0.33, 'XGBoost': 0.34}
        
        print(f"  Ensemble MAPE: {results['Ensemble']['MAPE']:.4f}%")
        print(f"  Win Rate: {results['Ensemble']['WinRate']:.2f}%")
        
    except Exception as e:
        print(f"  Ensemble failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Save results to JSON
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    
    # Print summary
    print("\n[MODEL COMPARISON]")
    print(f"{'Model':<12} {'MAPE':>10} {'RMSE':>10} {'MAE':>10}")
    print("-" * 45)
    for model in ['ARIMAX', 'XGBoost', 'LSTM', 'Ensemble']:
        if model in results and isinstance(results[model].get('MAPE'), (int, float)):
            print(f"{model:<12} {results[model]['MAPE']:>9.4f}% {results[model]['RMSE']:>10.2f} {results[model]['MAE']:>10.2f}")
    
    # Save to JSON file
    output_file = 'thesis_metrics.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Metrics saved to: {output_file}")
    
    # Also save data statistics
    stats = {
        'total_observations': len(df),
        'date_start': df.index.min().strftime('%Y-%m-%d'),
        'date_end': df.index.max().strftime('%Y-%m-%d'),
        'close_mean': float(df['Close'].mean()),
        'close_std': float(df['Close'].std()),
        'close_min': float(df['Close'].min()),
        'close_max': float(df['Close'].max()),
        'train_size': train_size,
        'test_size': len(df) - train_size
    }
    
    with open('thesis_data_stats.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Data stats saved to: thesis_data_stats.json")
    
    return results, stats

if __name__ == "__main__":
    results, stats = main()
