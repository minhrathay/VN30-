"""
VN30 Forecasting Model Benchmark
Compare OLD (Weighted Average) vs NEW (Stacking) vs IMPROVED approaches
"""
import pandas as pd
import numpy as np
import warnings
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.multioutput import MultiOutputRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional
import itertools
import os

warnings.filterwarnings('ignore')
np.random.seed(42)
tf.random.set_seed(42)

# ==============================================================================
# DATA LOADING & PREPROCESSING
# ==============================================================================

def load_and_preprocess():
    """Load and preprocess VN30 data"""
    filename = 'Dữ liệu Lịch sử VN 30.csv'
    if not os.path.exists(filename):
        print(f"❌ File not found: {filename}")
        return None
    
    df = pd.read_csv(filename)
    col_map = {'Ngày': 'Date', 'Lần cuối': 'Close', 'Mở': 'Open',
               'Cao': 'High', 'Thấp': 'Low', 'KL': 'Vol', '% Thay đổi': 'Change_Pct'}
    df.rename(columns=col_map, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    
    # Clean numeric columns
    def clean_volume(x):
        if isinstance(x, str):
            x = x.replace(',', '')
            if 'K' in x: return float(x.replace('K', '')) * 1e3
            if 'M' in x: return float(x.replace('M', '')) * 1e6
            if 'B' in x: return float(x.replace('B', '')) * 1e9
        return float(x)
    
    for col in ['Close', 'Open', 'High', 'Low']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '').astype(float)
    if 'Vol' in df.columns:
        df['Vol'] = df['Vol'].apply(clean_volume)
    
    # Technical indicators
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + gain/loss))
    
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    df['ATR'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1).rolling(14).mean()
    
    # Lag features
    for i in range(1, 4):
        df[f'Lag_Close_{i}'] = df['Close'].shift(i)
        df[f'Lag_Vol_{i}'] = df['Vol'].shift(i)
        df[f'Lag_RSI_{i}'] = df['RSI'].shift(i)
        df[f'Lag_MACD_{i}'] = df['MACD'].shift(i)
        df[f'Lag_ATR_{i}'] = df['ATR'].shift(i)
    
    df.dropna(inplace=True)
    return df

# ==============================================================================
# APPROACH 1: OLD LOGIC (Simple Weighted Average)
# ==============================================================================

def run_old_logic(df, train_size):
    """Original approach: Direct prediction + Scipy optimized weights"""
    train = df.iloc[:train_size]
    test = df.iloc[train_size:]
    
    # ARIMAX
    exog_cols = ['Lag_Vol_1', 'RSI', 'ATR']
    train_exog = train[exog_cols].bfill()
    test_exog = test[exog_cols].bfill()
    
    history_endog = list(train['Close'])
    history_exog = list(train_exog.values)
    arima_preds = []
    
    for t in range(len(test)):
        try:
            model = ARIMA(history_endog, exog=history_exog, order=(1,1,1)).fit()
            pred = model.forecast(steps=1, exog=test_exog.values[t].reshape(1,-1))[0]
        except:
            pred = history_endog[-1]
        arima_preds.append(pred)
        history_endog.append(test['Close'].iloc[t])
        history_exog.append(test_exog.values[t])
    
    # XGBoost (Direct)
    features = [c for c in df.columns if c.startswith('Lag_')]
    xgb = XGBRegressor(n_estimators=500, learning_rate=0.01, max_depth=6, random_state=42)
    xgb.fit(train[features], train['Close'])
    xgb_preds = xgb.predict(test[features])
    
    # LSTM (Log Returns)
    df_lstm = df.copy()
    df_lstm['Log_Ret'] = np.log(df_lstm['Close'] / df_lstm['Close'].shift(1))
    df_lstm.fillna(0, inplace=True)
    for i in range(1, 4):
        df_lstm[f'Lag_Ret_{i}'] = df_lstm['Log_Ret'].shift(i)
    df_lstm.dropna(inplace=True)
    
    feat_lstm = [c for c in df_lstm.columns if c.startswith('Lag_')]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_scaled = scaler.fit_transform(df_lstm[feat_lstm].values)
    
    split_idx = len(df_lstm) - len(test)
    X_train = X_scaled[:split_idx].reshape(-1, 1, len(feat_lstm))
    y_train = df_lstm['Log_Ret'].values[:split_idx]
    X_test = X_scaled[split_idx:].reshape(-1, 1, len(feat_lstm))
    
    model = Sequential([
        Bidirectional(LSTM(32, return_sequences=False), input_shape=(1, len(feat_lstm))),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=30, batch_size=16, verbose=0)
    
    pred_log_ret = model.predict(X_test, verbose=0).flatten()
    prev_close = df.iloc[train_size-1:-1]['Close'].values
    lstm_preds = prev_close * np.exp(pred_log_ret)
    
    # Align lengths
    min_len = min(len(arima_preds), len(xgb_preds), len(lstm_preds))
    y_true = test['Close'].values[:min_len]
    p1, p2, p3 = np.array(arima_preds[:min_len]), xgb_preds[:min_len], lstm_preds[:min_len]
    
    # Optimize weights (simple inverse MAPE)
    mapes = [
        mean_absolute_percentage_error(y_true, p1),
        mean_absolute_percentage_error(y_true, p2),
        mean_absolute_percentage_error(y_true, p3)
    ]
    inv_mapes = [1/m for m in mapes]
    weights = np.array(inv_mapes) / sum(inv_mapes)
    
    ensemble = p1 * weights[0] + p2 * weights[1] + p3 * weights[2]
    
    return {
        'ARIMAX': mean_absolute_percentage_error(y_true, p1) * 100,
        'XGBoost': mean_absolute_percentage_error(y_true, p2) * 100,
        'LSTM': mean_absolute_percentage_error(y_true, p3) * 100,
        'Ensemble': mean_absolute_percentage_error(y_true, ensemble) * 100,
        'Weights': weights
    }

# ==============================================================================
# APPROACH 2: NEW LOGIC (Polynomial Trend + Stacking)
# ==============================================================================

def run_new_logic(df, train_size):
    """Current approach: Polynomial Trend + Residual Models + RidgeCV Stacking"""
    train = df.iloc[:train_size]
    test = df.iloc[train_size:]
    
    # Fit Polynomial Trend
    time_idx = np.arange(len(train)).reshape(-1, 1)
    log_close = np.log(train['Close'].values)
    
    poly = PolynomialFeatures(degree=3)
    time_poly = poly.fit_transform(time_idx)
    weights = np.exp(np.linspace(-5, 0, len(train)))
    
    trend_model = Ridge(alpha=0.5)
    trend_model.fit(time_poly, log_close, sample_weight=weights)
    
    train_trend = trend_model.predict(time_poly)
    train_resid = log_close - train_trend
    
    # Prepare residual features
    df_train = train.copy()
    df_train['Resid'] = train_resid
    for i in range(1, 4):
        df_train[f'Lag_Resid_{i}'] = df_train['Resid'].shift(i)
    df_train.dropna(inplace=True)
    
    features = [c for c in df_train.columns if c.startswith('Lag_') or c in ['RSI', 'MACD', 'ATR']]
    
    # XGBoost on Residuals
    xgb = XGBRegressor(n_estimators=500, learning_rate=0.01, max_depth=6, random_state=42)
    xgb.fit(df_train[features], df_train['Resid'])
    
    # Predict on Test
    full_df = pd.concat([train, test])
    full_time = np.arange(len(full_df)).reshape(-1, 1)
    full_time_poly = poly.transform(full_time)
    full_trend = trend_model.predict(full_time_poly)
    full_df['Resid'] = np.log(full_df['Close']) - full_trend
    for i in range(1, 4):
        full_df[f'Lag_Resid_{i}'] = full_df['Resid'].shift(i)
    
    test_df = full_df.iloc[train_size:]
    pred_resid = xgb.predict(test_df[features].fillna(0))
    pred_log = full_trend[train_size:] + pred_resid
    xgb_preds = np.exp(pred_log)
    
    # ARIMAX (simple)
    exog_cols = ['Lag_Vol_1', 'RSI', 'ATR']
    history_endog = list(train['Close'])
    history_exog = list(train[exog_cols].bfill().values)
    arima_preds = []
    
    for t in range(len(test)):
        try:
            model = ARIMA(history_endog, exog=history_exog, order=(1,1,1)).fit()
            pred = model.forecast(steps=1, exog=test[exog_cols].bfill().values[t].reshape(1,-1))[0]
        except:
            pred = history_endog[-1]
        arima_preds.append(pred)
        history_endog.append(test['Close'].iloc[t])
        history_exog.append(test[exog_cols].bfill().values[t])
    
    # Align
    min_len = min(len(arima_preds), len(xgb_preds))
    y_true = test['Close'].values[:min_len]
    p1, p2 = np.array(arima_preds[:min_len]), xgb_preds[:min_len]
    
    # Stacking with RidgeCV
    X_stack = np.column_stack([p1, p2])
    meta = RidgeCV(alphas=[0.1, 1.0, 10.0])
    meta.fit(X_stack, y_true)
    ensemble = meta.predict(X_stack)
    
    return {
        'ARIMAX': mean_absolute_percentage_error(y_true, p1) * 100,
        'XGBoost': mean_absolute_percentage_error(y_true, p2) * 100,
        'Ensemble': mean_absolute_percentage_error(y_true, ensemble) * 100,
        'Meta_Coeffs': meta.coef_
    }

# ==============================================================================
# APPROACH 3: IMPROVED (Holt-Winters Trend + Boosted Stacking)
# ==============================================================================

def run_improved_logic(df, train_size):
    """Improved: Holt-Winters Trend + XGBoost Meta-Learner"""
    train = df.iloc[:train_size]
    test = df.iloc[train_size:]
    
    # Holt-Winters for Trend
    try:
        hw_model = ExponentialSmoothing(
            train['Close'], 
            trend='add', 
            seasonal=None,
            damped_trend=True
        ).fit(optimized=True)
        
        hw_forecast = hw_model.forecast(len(test))
    except:
        hw_forecast = np.array([train['Close'].iloc[-1]] * len(test))
    
    # XGBoost (Direct on Price)
    features = [c for c in df.columns if c.startswith('Lag_')]
    xgb = XGBRegressor(n_estimators=500, learning_rate=0.01, max_depth=6, random_state=42)
    xgb.fit(train[features], train['Close'])
    xgb_preds = xgb.predict(test[features])
    
    # ARIMAX
    exog_cols = ['Lag_Vol_1', 'RSI', 'ATR']
    history_endog = list(train['Close'])
    history_exog = list(train[exog_cols].bfill().values)
    arima_preds = []
    
    for t in range(len(test)):
        try:
            model = ARIMA(history_endog, exog=history_exog, order=(1,1,1)).fit()
            pred = model.forecast(steps=1, exog=test[exog_cols].bfill().values[t].reshape(1,-1))[0]
        except:
            pred = history_endog[-1]
        arima_preds.append(pred)
        history_endog.append(test['Close'].iloc[t])
        history_exog.append(test[exog_cols].bfill().values[t])
    
    # Align
    min_len = min(len(arima_preds), len(xgb_preds), len(hw_forecast))
    y_true = test['Close'].values[:min_len]
    p1 = np.array(arima_preds[:min_len])
    p2 = xgb_preds[:min_len]
    p3 = np.array(hw_forecast[:min_len])
    
    # XGBoost Meta-Learner (Boosted Stacking)
    X_stack = np.column_stack([p1, p2, p3])
    meta = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    meta.fit(X_stack, y_true)
    ensemble = meta.predict(X_stack)
    
    return {
        'ARIMAX': mean_absolute_percentage_error(y_true, p1) * 100,
        'XGBoost': mean_absolute_percentage_error(y_true, p2) * 100,
        'Holt-Winters': mean_absolute_percentage_error(y_true, p3) * 100,
        'Ensemble': mean_absolute_percentage_error(y_true, ensemble) * 100,
        'Meta_Importance': meta.feature_importances_
    }

# ==============================================================================
# MAIN BENCHMARK
# ==============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print(" VN30 FORECASTING MODEL BENCHMARK")
    print("=" * 60)
    
    df = load_and_preprocess()
    if df is None:
        exit()
    
    train_size = int(len(df) * 0.95)
    print(f"\n📊 Data: {len(df)} samples | Train: {train_size} | Test: {len(df)-train_size}")
    
    print("\n⏳ Running OLD logic (Weighted Average)...")
    old_results = run_old_logic(df, train_size)
    
    print("⏳ Running NEW logic (Polynomial + Stacking)...")
    new_results = run_new_logic(df, train_size)
    
    print("⏳ Running IMPROVED logic (Holt-Winters + XGB Meta)...")
    improved_results = run_improved_logic(df, train_size)
    
    print("\n" + "=" * 60)
    print(" BENCHMARK RESULTS (MAPE %)")
    print("=" * 60)
    
    print(f"\n{'Model':<20} {'OLD':<12} {'NEW':<12} {'IMPROVED':<12}")
    print("-" * 56)
    print(f"{'ARIMAX':<20} {old_results['ARIMAX']:<12.4f} {new_results['ARIMAX']:<12.4f} {improved_results['ARIMAX']:<12.4f}")
    print(f"{'XGBoost':<20} {old_results['XGBoost']:<12.4f} {new_results['XGBoost']:<12.4f} {improved_results['XGBoost']:<12.4f}")
    print(f"{'LSTM/HW':<20} {old_results['LSTM']:<12.4f} {'N/A':<12} {improved_results['Holt-Winters']:<12.4f}")
    print(f"{'ENSEMBLE':<20} {old_results['Ensemble']:<12.4f} {new_results['Ensemble']:<12.4f} {improved_results['Ensemble']:<12.4f}")
    
    print("\n" + "=" * 60)
    print(" WINNER DETERMINATION")
    print("=" * 60)
    
    ensemble_mapes = {
        'OLD (Weighted Avg)': old_results['Ensemble'],
        'NEW (Poly + Ridge)': new_results['Ensemble'],
        'IMPROVED (HW + XGB)': improved_results['Ensemble']
    }
    
    winner = min(ensemble_mapes, key=ensemble_mapes.get)
    print(f"\n🏆 BEST APPROACH: {winner}")
    print(f"   MAPE: {ensemble_mapes[winner]:.4f}%")
    
    print("\n✅ Benchmark Complete!")
