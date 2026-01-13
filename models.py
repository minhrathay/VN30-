"""
Model training and prediction functions for VN30 Forecasting
"""
import pandas as pd
import numpy as np
import warnings
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error
import itertools

warnings.filterwarnings('ignore')
np.random.seed(42)
tf.random.set_seed(42)

def train_arimax(train_data, test_data, exog_cols=['Lag_Vol_1', 'RSI', 'ATR']):
    """Train ARIMAX model"""
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
    
    # Walk-forward forecast
    history_endog = list(train_endog)
    history_exog = list(train_exog.values)
    predictions = []
    test_exog_vals = test_exog.values
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        for t in range(len(test_data)):
            model = ARIMA(history_endog, exog=history_exog, order=best_order).fit()
            forecast_result = model.forecast(steps=1, exog=test_exog_vals[t].reshape(1,-1))
            pred = forecast_result[0] if isinstance(forecast_result, np.ndarray) else forecast_result.iloc[0]
            predictions.append(pred)
            history_endog.append(test_data['Close'].iloc[t])
            history_exog.append(test_exog_vals[t])
    
    return np.array(predictions), best_order

def train_xgboost(train_data, test_data):
    """Train XGBoost model on Log Returns"""
    # Create Log Return features if not exist
    train_df = train_data.copy()
    test_df = test_data.copy()
    
    # Calculate Log Returns
    train_df['Log_Ret'] = np.log(train_df['Close'] / train_df['Close'].shift(1))
    test_df['Log_Ret'] = np.log(test_df['Close'] / test_df['Close'].shift(1))
    
    # Fill NaN from shift
    train_df.fillna(0, inplace=True)
    test_df.fillna(0, inplace=True)
    
    # Feature Engineering for Returns
    # Important: Add Lag_Log_Ret features for better momentum prediction
    for i in range(1, 4):
        train_df[f'Lag_Log_Ret_{i}'] = train_df['Log_Ret'].shift(i)
        test_df[f'Lag_Log_Ret_{i}'] = test_df['Log_Ret'].shift(i)
        
    # Drop NaNs created by new lags
    train_df.dropna(inplace=True)
    test_df.dropna(inplace=True)
    
    # Select features: Use Lag_Log_Ret and other technical lags (Vol, RSI, etc)
    # Exclude Lag_Close features if we want purely relative model, but keeping them might help with volatility regime
    features = [col for col in train_df.columns if col.startswith('Lag_')]
    
    X_train = train_df[features]
    y_train = train_df['Log_Ret']
    X_test = test_df[features]
    y_test = test_df['Log_Ret']
    
    model = XGBRegressor(
        n_estimators=3000,
        learning_rate=0.005,
        max_depth=8,
        subsample=0.7,
        colsample_bytree=0.7,
        n_jobs=-1,
        random_state=42,
        early_stopping_rounds=50,
        eval_metric='mae'
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False
    )
    
    # Predict Log Returns
    pred_log_ret = model.predict(X_test)
    
    # Convert Log Returns back to Prices
    # Price_t = Price_{t-1} * exp(Log_Ret_t)
    initial_price = train_data['Close'].iloc[-1]
    predictions = []
    
    # Use actual previous prices for one-step ahead test (Teacher Forcing style for evaluation)
    # OR reconstruct sequentially. For "test set evaluation", we usually want sequential reconstruction 
    # if we treat it as multi-step, but here standard 'predict' usually implies using known features.
    # But since features are fixed, let's reconstruct simply:
    
    # Wait, 'metrics' calculation expects aligned comparisons.
    # For simplicity in 'train_xgboost' (backtest), we usually predict one-step ahead given actual history features.
    # So: Prev_Close (Actual) * Exp(Pred_Log_Ret)
    
    prev_closes = pd.concat([train_data['Close'].tail(1), test_data['Close'].shift(1).iloc[1:]])
    # The above shift logic interacts complexly with concatenation.
    # Simpler: Test data Close lagged by 1.
    
    # Correct Shift for comparison:
    # Test[0] prediction needs Train[-1] close.
    # Test[1] prediction needs Test[0] close.
    
    ref_closes = pd.Series([train_data['Close'].iloc[-1]]).append(test_data['Close'][:-1], ignore_index=True)
    ref_closes = ref_closes.values
    
    predictions = ref_closes * np.exp(pred_log_ret)
    
    return predictions, model

def train_lstm(df, train_size):
    """Train Bi-LSTM model"""
    # Enforce Determinism for every run
    np.random.seed(42)
    tf.random.set_seed(42)
    
    df_lstm = df.copy()
    df_lstm['Log_Ret'] = np.log(df_lstm['Close'] / df_lstm['Close'].shift(1))
    df_lstm.fillna(0, inplace=True)
    
    # Create lag features
    for i in range(1, 4):
        df_lstm[f'Lag_Ret_{i}'] = df_lstm['Log_Ret'].shift(i)
        df_lstm[f'Lag_Vol_{i}'] = df[f'Lag_Vol_{i}']
        df_lstm[f'Lag_ATR_{i}'] = df[f'Lag_ATR_{i}']
    
    df_lstm.dropna(inplace=True)
    features = [col for col in df_lstm.columns if col.startswith('Lag_')]
    
    X_vals = df_lstm[features].values
    y_vals = df_lstm['Log_Ret'].values
    
    # Scale
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_scaled = scaler.fit_transform(X_vals)
    
    # Important: Find where train_size falls in df_lstm (after dropna)
    # Map from original df index to df_lstm index
    lstm_train_size = len(df_lstm) - (len(df) - train_size)
    if lstm_train_size < 0 or lstm_train_size > len(df_lstm):
        lstm_train_size = int(len(df_lstm) * 0.95)  # Fallback to 95%
    
    X_train = X_scaled[:lstm_train_size].reshape(-1, 1, len(features))
    X_test = X_scaled[lstm_train_size:].reshape(-1, 1, len(features))
    y_train = y_vals[:lstm_train_size]
    
    # Build model
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True), input_shape=(1, len(features))),
        Dropout(0.2),
        Bidirectional(LSTM(32)),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    
    # Train
    model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=16,
        verbose=0,
        validation_split=0.1,
        callbacks=[
            EarlyStopping(patience=15, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5)
        ]
    )
    
    # Predict log returns
    pred_log_ret = model.predict(X_test, verbose=0).flatten()
    
    # Reconstruct actual prices from log returns
    # Use Close prices from df_lstm (aligned with predictions)
    test_close_prices = df_lstm.iloc[lstm_train_size:]['Close'].values
    prev_close_prices = df_lstm.iloc[lstm_train_size-1:-1]['Close'].values
    
    # Ensure lengths match
    min_len = min(len(pred_log_ret), len(prev_close_prices))
    pred_log_ret = pred_log_ret[:min_len]
    prev_close_prices = prev_close_prices[:min_len]
    
    # Reconstruct predictions
    predictions = prev_close_prices * np.exp(pred_log_ret)
    
    return predictions, model, scaler

from scipy.optimize import minimize

def optimize_ensemble(y_true, *predictions):
    """
    Optimize ensemble weights using SLSQP (Convex Optimization).
    Finds exact weights that minimize MAPE.
    """
    n_models = len(predictions)
    # Stack predictions: shape (n_samples, n_models)
    preds_matrix = np.column_stack(predictions)
    
    # Objective Function: Minimize MAPE
    def objective(weights):
        # weights shape: (n_models,)
        weighted_pred = np.dot(preds_matrix, weights)
        return mean_absolute_percentage_error(y_true, weighted_pred) * 100

    # Constraint: sum(weights) = 1
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    
    # Bounds: 0 <= weight <= 1
    bounds = tuple((0, 1) for _ in range(n_models))
    
    # Initial guess: Equal weights
    initial_weights = [1/n_models] * n_models
    
    # Run Optimization
    # method='SLSQP' is standard for constrained optimization
    result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=cons)
    
    if result.success:
        best_weights = result.x
    else:
        # Fallback to inverse error weighting if optimization fails
        errors = [mean_absolute_percentage_error(y_true, p) for p in predictions]
        # Avoid div by zero
        inv_errors = [1/(e + 1e-6) for e in errors]
        sum_inv = sum(inv_errors)
        best_weights = [w/sum_inv for w in inv_errors]
        
    # Calculate final ensemble
    best_pred = np.dot(preds_matrix, best_weights)
    best_mape = mean_absolute_percentage_error(y_true, best_pred) * 100
    
    # Force clean zeros for very small weights (e.g. 1e-10 becomes 0)
    best_weights = [w if w > 0.001 else 0.0 for w in best_weights]
    # Re-normalize
    total = sum(best_weights)
    if total > 0:
        best_weights = [w/total for w in best_weights]
    
    return best_pred, best_weights, best_mape

def forecast_future_arimax(model_order, df, exog_cols, n_days=14):
    """
    Autoregressive multi-step forecast using ARIMAX
    Model uses its own predictions to generate future forecasts
    """
    from utils import create_technical_indicators, create_lag_features
    
    # Prepare training data
    train_exog = df[exog_cols].bfill()
    train_endog = df['Close']
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        
        # Fit model on full historical data
        base_model = ARIMA(endog=train_endog, exog=train_exog, order=model_order).fit()
        
        # Initialize history with actual data
        history_close = list(train_endog.values)
        
        # Handle both 'Vol' and 'Volume' column names
        if 'Vol' in df.columns:
            history_volume = list(df['Vol'].values)
        elif 'Volume' in df.columns:
            history_volume = list(df['Volume'].values)
        else:
            history_volume = [1000000] * len(df)  # Default volume if not present
            
        history_high = list(df['High'].values) if 'High' in df.columns else list(train_endog.values)
        history_low = list(df['Low'].values) if 'Low' in df.columns else list(train_endog.values)
        
        forecasts = []
        
        # Use fixed RNG for consistent results across different forecast horizons
        rng = np.random.RandomState(42)
        
        # Iteratively forecast each day
        for step in range(n_days):
            # Build temporary dataframe with history
            temp_df = pd.DataFrame({
                'Close': history_close,
                'Volume': history_volume,
                'High': history_high,
                'Low': history_low
            })
            
            # Recalculate technical indicators AND lag features with current history
            temp_df = create_technical_indicators(temp_df)
            temp_df = create_lag_features(temp_df)
            
            # Drop NaN rows created by lag features
            temp_df = temp_df.dropna()
            
            # Check if we have the required exog columns
            available_cols = list(temp_df.columns)
            if not all(col in available_cols for col in exog_cols):
                current_exog = train_exog.iloc[-1:].values
            else:
                current_exog = temp_df[exog_cols].iloc[-1:].values
            
            # Forecast next step
            forecast_result = base_model.forecast(steps=1, exog=current_exog)
            pred = forecast_result[0] if isinstance(forecast_result, np.ndarray) else forecast_result.iloc[0]
            forecasts.append(pred)
            
            # Update history with prediction for next iteration
            history_close.append(pred)
            
            # Estimate future volume (Mean reversion + Random Shock)
            recent_vol = np.array(history_volume[-10:])
            vol_mean = np.mean(recent_vol)
            vol_std = np.std(recent_vol) if len(recent_vol) > 1 else vol_mean * 0.05
            
            # Simulated volume with fixed random seed
            shock = rng.normal(0, vol_std * 0.5) 
            future_volume = max(1000, vol_mean + shock) # Ensure non-negative
            history_volume.append(future_volume)
            
            # Estimate High/Low based on predicted Close + Fixed Random Volatility
            recent_close_vol = np.std(history_close[-20:]) if len(history_close) >= 20 else np.std(history_close)
            if recent_close_vol == 0: recent_close_vol = pred * 0.01
            
            # Add some randomness to High/Low spread
            spread_shock = rng.uniform(0.8, 1.2)
            history_high.append(pred + (recent_close_vol * 0.5 * spread_shock))
            history_low.append(pred - (recent_close_vol * 0.5 * spread_shock))
        
        return np.array(forecasts)

def forecast_future_xgboost(model, df, n_days=14):
    """
    Autoregressive multi-step forecast using XGBoost
    Predicts Log Returns and reconstructs price path
    """
    from utils import create_lag_features, create_technical_indicators
    
    # Get lag features
    features = [col for col in df.columns if col.startswith('Lag_')]
    
    # Initialize with last known values
    history_df = df.copy()
    current_close = history_df['Close'].iloc[-1]
    
    # Prepare lists for faster appending
    hist_close = list(history_df['Close'].values)
    hist_high = list(history_df['High'].values) if 'High' in history_df.columns else list(history_df['Close'].values)
    hist_low = list(history_df['Low'].values) if 'Low' in history_df.columns else list(history_df['Close'].values)
    
    if 'Vol' in history_df.columns:
        hist_vol = list(history_df['Vol'].values)
        vol_col = 'Vol'
    elif 'Volume' in history_df.columns:
        hist_vol = list(history_df['Volume'].values)
        vol_col = 'Volume'
    else:
        hist_vol = [1000000] * len(history_df)
        vol_col = 'Vol'

    forecasts = []
    
    # Use fixed RNG for consistent future auxiliary data
    rng = np.random.RandomState(42)
    
    for step in range(n_days):
        # Re-build dataframe from lists to calculate indicators
        temp_df = pd.DataFrame({
            'Close': hist_close,
            'High': hist_high,
            'Low': hist_low,
            vol_col: hist_vol
        })
        
        # Recalculate indicators and lags
        temp_df = create_technical_indicators(temp_df)
        
        # Calculate Log Returns & Lag Log Returns locally required for XGBoost
        temp_df['Log_Ret'] = np.log(temp_df['Close'] / temp_df['Close'].shift(1))
        temp_df.fillna(0, inplace=True)
        
        for i in range(1, 4):
            temp_df[f'Lag_Log_Ret_{i}'] = temp_df['Log_Ret'].shift(i)
        
        temp_df = create_lag_features(temp_df)
        temp_df = temp_df.dropna()
        
        # Get most recent features
        # Ensure we capture the new Lag_Log_Ret features
        features = [col for col in temp_df.columns if col.startswith('Lag_')]
        current_features = temp_df[features].iloc[-1:].values
        
        # Predict next LOG RETURN
        pred_log_ret = model.predict(current_features)[0]
        
        # Convert Log Return -> Price
        pred_price = current_close * np.exp(pred_log_ret)
        forecasts.append(pred_price)
        
        # Update current close
        current_close = pred_price
        
        # --- Estimate Auxiliary for next step (High/Low/Vol) ---
        # 1. Volume: Mean reversion + noise
        recent_vol = np.array(hist_vol[-10:])
        vol_mean = np.mean(recent_vol)
        vol_std = np.std(recent_vol) if len(recent_vol) > 1 else vol_mean * 0.05
        next_vol = max(1000, vol_mean + rng.normal(0, vol_std * 0.5))
        
        # 2. High/Low: Based on Volatility
        recent_std = np.std(hist_close[-20:]) if len(hist_close) >= 20 else np.std(hist_close)
        if recent_std == 0: recent_std = pred_price * 0.01
        
        spread_shock = rng.uniform(0.8, 1.2)
        next_high = pred_price + (recent_std * 0.5 * spread_shock)
        next_low = pred_price - (recent_std * 0.5 * spread_shock)
        
        # Append to lists
        hist_close.append(pred_price)
        hist_high.append(next_high)
        hist_low.append(next_low)
        hist_vol.append(next_vol)
    
    return np.array(forecasts)

def forecast_future_lstm(model, scaler, df, n_days=14):
    """
    Autoregressive multi-step forecast using LSTM
    Properly recalculates all features at each step
    """
    from utils import create_technical_indicators
    
    # Initialize history with last known full state
    history_df = df.copy()
    current_close = history_df['Close'].iloc[-1]
    
    # Prepare independent lists for easier appending
    hist_close = list(history_df['Close'].values)
    hist_high = list(history_df['High'].values) if 'High' in history_df.columns else list(history_df['Close'].values)
    hist_low = list(history_df['Low'].values) if 'Low' in history_df.columns else list(history_df['Close'].values)
    
    if 'Vol' in history_df.columns:
        hist_vol = list(history_df['Vol'].values)
        vol_col = 'Vol'
    elif 'Volume' in history_df.columns:
        hist_vol = list(history_df['Volume'].values)
        vol_col = 'Volume'
    else:
        hist_vol = [1000000] * len(history_df)
        vol_col = 'Vol'
        
    forecasts = []
    
    # Use fixed RNG
    rng = np.random.RandomState(42)
    
    # Identify feature columns used by the model
    # Note: We need to reconstruct the EXACT dataframe structure expected by model
    # logic must match train_lstm feature creation
    
    for step in range(n_days):
        # 1. Rebuild DataFrame
        temp_df = pd.DataFrame({
            'Close': hist_close,
            'High': hist_high,
            'Low': hist_low,
            vol_col: hist_vol
        })
        
        # 2. Recalculate Base Technical Indicators (ATR is needed)
        temp_df = create_technical_indicators(temp_df)
        
        # 3. Create LSTM specific features (Log Returns & Lags)
        # We need to replicate train_lstm logic here locally as it's specific
        temp_df['Log_Ret'] = np.log(temp_df['Close'] / temp_df['Close'].shift(1))
        temp_df.fillna(0, inplace=True)
        
        for i in range(1, 4):
            temp_df[f'Lag_Ret_{i}'] = temp_df['Log_Ret'].shift(i)
            # Legacy Note: train_lstm code assumed Lag_Vol_i existed in df. 
            # We must create them here if we want them to update!
            # Using simple shift of Vol/ATR column
            temp_df[f'Lag_Vol_{i}'] = temp_df[vol_col].shift(i)
            if 'ATR' in temp_df.columns:
                temp_df[f'Lag_ATR_{i}'] = temp_df['ATR'].shift(i)
                
        # Drop NaN
        valid_df = temp_df.dropna()
        
        # Get Features
        # Ensure column order matches training (startswith Lag_)
        features = [col for col in valid_df.columns if col.startswith('Lag_')]
        
        # Get most recent inputs
        last_features = valid_df[features].iloc[-1:].values
        
        # Scale
        scaled_features = scaler.transform(last_features)
        X_input = scaled_features.reshape(1, 1, len(features))
        
        # Predict Log Return
        pred_log_ret = model.predict(X_input, verbose=0)[0, 0]
        
        # Convert to Price
        pred_close = current_close * np.exp(pred_log_ret)
        forecasts.append(pred_close)
        
        # Update current close
        current_close = pred_close
        
        # --- Estimate Auxiliary for next step ---
        # Volume
        recent_vol = np.array(hist_vol[-10:])
        vol_mean = np.mean(recent_vol)
        vol_std = np.std(recent_vol) if len(recent_vol) > 1 else vol_mean * 0.05
        next_vol = max(1000, vol_mean + rng.normal(0, vol_std * 0.5))
        
        # High/Low
        recent_std = np.std(hist_close[-20:]) if len(hist_close) >= 20 else np.std(hist_close)
        if recent_std == 0: recent_std = pred_close * 0.01
        spread_shock = rng.uniform(0.8, 1.2)
        next_high = pred_close + (recent_std * 0.5 * spread_shock)
        next_low = pred_close - (recent_std * 0.5 * spread_shock)
        
        # Append
        hist_close.append(pred_close)
        hist_high.append(next_high)
        hist_low.append(next_low)
        hist_vol.append(next_vol)
    
    return np.array(forecasts)

def create_future_dates(last_date, n_days=14):
    """Create future business dates"""
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=n_days)
    return future_dates
