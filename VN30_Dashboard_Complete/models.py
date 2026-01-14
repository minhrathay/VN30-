"""
Model training and prediction functions for VN30 Forecasting
HYBRID STRATEGY: Trend (Robust Linear) + Residuals (Direct Multi-step ML)
"""
import pandas as pd
import numpy as np
import warnings
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import HuberRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_percentage_error
from scipy.optimize import minimize
import itertools

warnings.filterwarnings('ignore')
# Enforce determinism
np.random.seed(42)
tf.random.set_seed(42)

# --- HYBRID UTILS ---

def fit_trend_model(df):
    """
    Fit Holt-Winters Exponential Smoothing for Trend.
    More responsive to recent data than Polynomial Trend.
    Returns: trend_model, residuals, None (no poly object needed)
    """
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    
    close_prices = df['Close'].values
    
    try:
        # Holt-Winters with additive trend, no seasonality, damped trend
        trend_model = ExponentialSmoothing(
            close_prices,
            trend='add',
            seasonal=None,
            damped_trend=True,
            initialization_method='estimated'
        ).fit(optimized=True)
        
        # Get fitted values (trend component)
        fitted_trend = trend_model.fittedvalues
        residuals = close_prices - fitted_trend
        
    except Exception as e:
        # Fallback: Simple moving average as trend
        print(f"Holt-Winters failed, using SMA fallback: {e}")
        fitted_trend = pd.Series(close_prices).rolling(window=20, min_periods=1).mean().values
        residuals = close_prices - fitted_trend
        trend_model = None
        
    return trend_model, residuals, None  # None replaces poly object

def create_direct_targets(residuals, n_days=14):
    """
    Create Multi-step targets for Direct Forecasting.
    X[t] -> y[t+1, t+2, ..., t+14]
    """
    Y = []
    # We need to ensure we don't go out of bounds
    for i in range(len(residuals) - n_days):
        Y.append(residuals[i+1 : i+1+n_days])
    
    return np.array(Y)

# --- MODELS ---

def train_arimax(train_data, test_data, exog_cols=['Lag_Vol_1', 'RSI', 'ATR']):
    """Train ARIMAX model"""
    # ARIMAX is naturally recursive, but we'll keep it as a baseline
    train_exog = train_data[exog_cols].bfill()
    test_exog = test_data[exog_cols].bfill()
    train_endog = train_data['Close']
    
    # Grid search for best order
    best_aic = float("inf")
    best_order = (1, 1, 1)
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        for p, d, q in itertools.product(range(0, 3), [1], range(0, 3)):
            try:
                model = ARIMA(endog=train_endog, exog=train_exog, order=(p,d,q)).fit()
                if model.aic < best_aic:
                    best_aic = model.aic
                    best_order = (p,d,q)
            except:
                continue
    
    # Walk-forward forecast (standard ARIMAX)
    history_endog = list(train_endog)
    history_exog = list(train_exog.values)
    predictions = []
    test_exog_vals = test_exog.values
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        for t in range(len(test_data)):
            model = ARIMA(history_endog, exog=history_exog, order=best_order).fit()
            pred = model.forecast(steps=1, exog=test_exog_vals[t].reshape(1,-1))[0]
            predictions.append(pred)
            history_endog.append(test_data['Close'].iloc[t])
            history_exog.append(test_exog_vals[t])
    
    return np.array(predictions), best_order

def train_xgboost(train_data, test_data, n_days=14):
    """
    Train XGBoost using Direct Price Prediction.
    Simpler approach that won in benchmark testing.
    Returns: predictions, model, None (no trend), features, None (no poly)
    """
    # Features: All lag columns + technical indicators
    features = [c for c in train_data.columns if c.startswith('Lag_') or c in ['RSI', 'MACD', 'ATR']]
    
    X_train = train_data[features].fillna(0)
    y_train = train_data['Close']
    X_test = test_data[features].fillna(0)
    
    # Train XGBoost
    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.01,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Predict on test
    predictions = model.predict(X_test)
    
    # For compatibility, return same tuple structure
    # (predictions, model, trend_model=None, features, poly=None)
    return predictions, model, None, features, None

def train_lstm(df, train_size, n_days=14):
    """
    Train LSTM using Log Returns prediction.
    Simpler approach matching benchmark winner.
    """
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Prepare Log Returns
    df_lstm = df.copy()
    df_lstm['Log_Ret'] = np.log(df_lstm['Close'] / df_lstm['Close'].shift(1))
    df_lstm.fillna(0, inplace=True)
    
    # Create lag features
    for i in range(1, 4):
        df_lstm[f'Lag_Ret_{i}'] = df_lstm['Log_Ret'].shift(i)
    df_lstm.dropna(inplace=True)
    
    features = [c for c in df_lstm.columns if c.startswith('Lag_')]
    X_vals = df_lstm[features].values
    y_vals = df_lstm['Log_Ret'].values
    
    # Scale
    scaler_X = MinMaxScaler(feature_range=(-1, 1))
    X_scaled = scaler_X.fit_transform(X_vals)
    
    # Split
    split_idx = len(df_lstm) - (len(df) - train_size)
    X_train = X_scaled[:split_idx].reshape(-1, 1, len(features))
    y_train = y_vals[:split_idx]
    X_test = X_scaled[split_idx:].reshape(-1, 1, len(features))
    
    # Build Model
    model = Sequential([
        Bidirectional(LSTM(32, return_sequences=False), input_shape=(1, len(features))),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=30, batch_size=16, verbose=0)
    
    # Predict Log Returns
    pred_log_ret = model.predict(X_test, verbose=0).flatten()
    
    # Reconstruct Prices
    prev_close = df.iloc[train_size-1:-1]['Close'].values
    min_len = min(len(prev_close), len(pred_log_ret))
    pred_prices = prev_close[:min_len] * np.exp(pred_log_ret[:min_len])
    
    # Pack objects for future forecasting
    lstm_objects = {
        'model': model,
        'scaler_X': scaler_X,
        'features': features
    }
    
    return pred_prices, lstm_objects


def train_meta_learner(y_true, model_predictions):
    """
    Train a Stacking Meta-Learner (XGBoost - Benchmark Winner).
    Input: True values and dict of predictions from base models.
    Output: Trained Meta-Model.
    """
    # Align lengths
    min_len = min(len(y_true), *[len(p) for p in model_predictions.values()])
    y_target = y_true[-min_len:]
    
    # Create Feature Matrix for Meta-Learner [n_samples, n_models]
    keys = sorted(model_predictions.keys())
    X_meta = np.column_stack([model_predictions[k][-min_len:] for k in keys])
    
    # Train XGBoost Meta-Learner (Benchmark Winner)
    meta_model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    meta_model.fit(X_meta, y_target)
    
    # Calculate score
    final_pred = meta_model.predict(X_meta)
    mape = mean_absolute_percentage_error(y_target, final_pred) * 100
    
    return meta_model, keys, final_pred, mape

def forecast_ensemble_stacking(meta_model, model_keys, future_predictions):
    """
    Generate future forecast using the trained Meta-Learner.
    """
    # Stack future predictions in the same order as training
    # Check if we have all models
    X_future = np.column_stack([future_predictions[k] for k in model_keys])
    
    # Predict
    final_forecast = meta_model.predict(X_future)
    return final_forecast

# --- FORECASTING WRAPPERS ---

def forecast_future_arimax(model_order, df, exog_cols, n_days=14):
    """Forecast future days using ARIMAX"""
    train_exog = df[exog_cols].bfill()
    train_endog = df['Close']
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        try:
            model = ARIMA(endog=train_endog, exog=train_exog, order=model_order).fit()
            
            # DYNAMIC EXOG GENERATION (Mean Reversion)
            # Instead of static tile, we evolve inputs towards their long-term means
            # exog_cols are ['Lag_Vol_1', 'RSI', 'ATR']
            
            last_vals = train_exog.iloc[-1].values
            future_exog_list = []
            
            # Calculate Means from history
            means = train_exog.mean().values
            # RSI mean is effectively 50 (Neutral)
            means[1] = 50.0 
            
            # Generate 14 steps
            for i in range(1, n_days + 1):
                # Interpolate: Alpha moves from 0 to 1 (revert to mean)
                # alpha = i / n_days # Linear decay
                alpha = 1 - np.exp(-0.2 * i) # Exponential decay to mean
                
                next_val = last_vals * (1 - alpha) + means * alpha
                future_exog_list.append(next_val)
                
            future_exog = np.array(future_exog_list)
            
            forecast = model.forecast(steps=n_days, exog=future_exog)
        except:
            # Fallback if ARIMAX fails
            forecast = [df['Close'].iloc[-1]] * n_days
    
    return forecast

def forecast_future_xgboost(model_objs, df, n_days=14):
    """
    Future Forecast using Holt-Winters Exponential Smoothing.
    XGBoost is used for backtest, HW is used for future (cleaner, more stable).
    """
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    
    try:
        # Holt-Winters with damped trend for realistic forecasts
        hw = ExponentialSmoothing(
            df['Close'].values,
            trend='add',
            seasonal=None,
            damped_trend=True,
            initialization_method='estimated'
        ).fit(optimized=True)
        
        # Forecast future
        forecast = hw.forecast(n_days)
        
    except Exception as e:
        # Fallback: Linear extrapolation from last few points
        print(f"HW failed: {e}, using linear fallback")
        last_price = df['Close'].iloc[-1]
        # Calculate recent trend (last 5 days)
        recent_change = (df['Close'].iloc[-1] - df['Close'].iloc[-5]) / 5
        forecast = np.array([last_price + recent_change * (i+1) for i in range(n_days)])
    
    return forecast

def forecast_future_lstm(lstm_objects, df, n_days=14):
    """
    Direct Forecast using LSTM (Log Returns).
    """
    model = lstm_objects['model']
    scaler_X = lstm_objects['scaler_X']
    features = lstm_objects['features']
    
    # Prepare Log Returns
    df_lstm = df.copy()
    df_lstm['Log_Ret'] = np.log(df_lstm['Close'] / df_lstm['Close'].shift(1))
    df_lstm.fillna(0, inplace=True)
    
    for i in range(1, 4):
        df_lstm[f'Lag_Ret_{i}'] = df_lstm['Log_Ret'].shift(i)
    
    # Iterative forecast
    predictions = []
    last_close = df['Close'].iloc[-1]
    last_lags = [df_lstm['Log_Ret'].iloc[-i] for i in range(1, 4)]
    
    for day in range(n_days):
        # Build feature vector
        X_vals = np.array(last_lags + [0] * (len(features) - 3)).reshape(1, -1)
        X_scaled = scaler_X.transform(X_vals).reshape(1, 1, len(features))
        
        # Predict log return
        pred_log_ret = model.predict(X_scaled, verbose=0)[0, 0]
        
        # Convert to price
        new_close = last_close * np.exp(pred_log_ret)
        predictions.append(new_close)
        
        # Update for next iteration
        last_lags = [pred_log_ret] + last_lags[:2]
        last_close = new_close
    
    return np.array(predictions)

def create_future_dates(last_date, n_days=14):
    """Create future business dates"""
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=n_days)
    return future_dates
