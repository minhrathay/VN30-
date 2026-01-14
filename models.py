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

# ═══════════════════════════════════════════════════════════════════════════════
# PHẦN 1: FEATURE ENGINEERING (Làm giàu đặc trưng)
# ═══════════════════════════════════════════════════════════════════════════════

def add_rolling_features(df, windows=[7, 14, 21]):
    """
    ───────────────────────────────────────────────────────────────────────────────
    THÊM BIẾN ROLLING WINDOWS (Trung bình động và Độ lệch chuẩn động)
    ───────────────────────────────────────────────────────────────────────────────
    Mục tiêu: Giúp mô hình nhận diện được VOLATILITY (độ biến động) và TREND
              trong các khoảng thời gian khác nhau.
    
    Tham số:
        windows: Danh sách các cửa sổ thời gian (mặc định: 7, 14, 21 ngày)
    
    Đầu ra:
        Rolling_Mean_7, Rolling_Std_7   | Trung bình/Độ lệch chuẩn 7 ngày
        Rolling_Mean_14, Rolling_Std_14 | Trung bình/Độ lệch chuẩn 14 ngày
        Rolling_Mean_21, Rolling_Std_21 | Trung bình/Độ lệch chuẩn 21 ngày
    ───────────────────────────────────────────────────────────────────────────────
    """
    for w in windows:
        # Rolling Mean: Xu hướng trung bình ngắn hạn
        df[f'Rolling_Mean_{w}'] = df['Close'].rolling(window=w).mean()
        
        # Rolling Std: Độ biến động (Volatility) trong w ngày gần nhất
        # Giá trị cao → Thị trường biến động mạnh
        # Giá trị thấp → Thị trường ổn định
        df[f'Rolling_Std_{w}'] = df['Close'].rolling(window=w).std()
        
        # Tạo Lag để tránh Data Leakage
        df[f'Lag_RollMean_{w}'] = df[f'Rolling_Mean_{w}'].shift(1)
        df[f'Lag_RollStd_{w}'] = df[f'Rolling_Std_{w}'].shift(1)
    
    return df

def add_macro_features(df, sp500_csv=None, usdvnd_csv=None):
    """
    ───────────────────────────────────────────────────────────────────────────────
    THÊM BIẾN LIÊN THỊ TRƯỜNG (Inter-market Variables)
    ───────────────────────────────────────────────────────────────────────────────
    Mục tiêu: Tích hợp ảnh hưởng của thị trường quốc tế và vĩ mô lên VN30.
    
    Biến số:
        - S&P 500:   Chỉ số chứng khoán Mỹ (tương quan cao với các thị trường châu Á)
        - USD/VND:   Tỷ giá hối đoái (ảnh hưởng đến dòng vốn nước ngoài)
    
    LƯU Ý: Hiện tại sử dụng DỮ LIỆU GIẢ LẬP (simulated).
           Bạn có thể thay thế bằng file CSV thật bằng cách truyền đường dẫn vào.
    ───────────────────────────────────────────────────────────────────────────────
    """
    if sp500_csv is not None and usdvnd_csv is not None:
        # ────────────────────────────────────────────────────────────────────────
        # PHƯƠNG ÁN 1: Load từ file CSV thật
        # ────────────────────────────────────────────────────────────────────────
        try:
            sp500 = pd.read_csv(sp500_csv, parse_dates=['Date'], index_col='Date')
            usdvnd = pd.read_csv(usdvnd_csv, parse_dates=['Date'], index_col='Date')
            
            df = df.join(sp500[['Close']].rename(columns={'Close': 'SP500'}), how='left')
            df = df.join(usdvnd[['Close']].rename(columns={'Close': 'USDVND'}), how='left')
            
            # Fill missing values bằng forward fill
            df['SP500'].fillna(method='ffill', inplace=True)
            df['USDVND'].fillna(method='ffill', inplace=True)
        except Exception as e:
            print(f"Không thể load dữ liệu macro: {e}. Sử dụng dữ liệu giả lập.")
            sp500_csv = None  # Fallback to simulated
    
    if sp500_csv is None or usdvnd_csv is None:
        # Check if columns already exist (loaded via utils.load_macro_data)
        if 'SP500' in df.columns and 'USDVND' in df.columns:
            # Data already loaded, skip simulation
            pass
        else:
            # ────────────────────────────────────────────────────────────────────────
            # PHƯƠNG ÁN 2: Dữ liệu giả lập (Simulated)
            # ────────────────────────────────────────────────────────────────────────
            # S&P 500: Giả lập với tương quan 0.6 với VN30
            np.random.seed(42) # [CRITICAL] Enforce determinism for simulated data
            vn30_returns = df['Close'].pct_change().fillna(0)
        noise = np.random.normal(0, 0.005, len(df))
        sp500_returns = 0.6 * vn30_returns + 0.4 * noise
        df['SP500'] = (1 + sp500_returns).cumprod() * 4500  # Base ~4500
        
        # USD/VND: Giả lập với dao động nhỏ quanh 24,500
        df['USDVND'] = 24500 + np.cumsum(np.random.normal(0, 10, len(df)))
    
    # Tạo Lag Features
    df['Lag_SP500'] = df['SP500'].shift(1)
    df['Lag_USDVND'] = df['USDVND'].shift(1)
    
    # Tính Log Returns của các biến macro
    df['SP500_LogRet'] = np.log(df['SP500'] / df['SP500'].shift(1))
    df['USDVND_LogRet'] = np.log(df['USDVND'] / df['USDVND'].shift(1))
    
    return df

# ═══════════════════════════════════════════════════════════════════════════════
# PHẦN 2: TREND MODELING (Mô hình hóa Xu hướng)
# ═══════════════════════════════════════════════════════════════════════════════

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

def train_arimax(train_data, test_data, exog_cols=['Lag_Vol_1', 'RSI', 'ATR'], fast_mode=True):
    """
    ──────────────────────────
    HUẤN LUYỆN MÔ HÌNH ARIMAX (AutoRegressive Integrated Moving Average with eXogenous)
    ───────────────────────────────
    Mục tiêu: Mô hình thống kê làm nền tảng (Baseline Statistical Model).
    
    Tham số:
        - fast_mode: True (Fit 1 lần, dự báo cả test set -> Nhanh), False (Walk-forward -> Chậm).
    
    """
    # Xử lý dữ liệu ngoại sinh (Exogenous variables)
    train_exog = train_data[exog_cols].bfill()
    test_exog = test_data[exog_cols].bfill()
    train_endog = train_data['Close']
    
    # Grid Search tìm tham số tối ưu (AIC thấp nhất)
    # Để tối ưu tốc độ, ta giới hạn range search hoặc dùng tham số cố định tốt (5,1,0)
    best_order = (5, 1, 0)
    
    history_endog = list(train_endog)
    history_exog = list(train_exog.values)
    predictions = []
    test_exog_vals = test_exog.values
    
    if fast_mode:
        # Cách 1: Fit model 1 lần trên tập Train và Forecast dynamic
        try:
            model = ARIMA(history_endog, exog=history_exog, order=best_order).fit()
            # Forecast toàn bộ giai đoạn test (Dynamic Forecast)
            pred_res = model.forecast(steps=len(test_data), exog=test_exog_vals)
            predictions = pred_res.tolist()
        except:
            predictions = [history_endog[-1]] * len(test_data)
    else:
        # Walk-forward forecast (Chậm nhưng chính xác hơn cho học thuật)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            for t in range(len(test_data)):
                model = ARIMA(history_endog, exog=history_exog, order=best_order).fit()
                pred = model.forecast(steps=1, exog=test_exog_vals[t].reshape(1,-1))[0]
                predictions.append(pred)
                history_endog.append(test_data['Close'].iloc[t])
                history_exog.append(test_exog_vals[t])
    
    return np.array(predictions), best_order
def train_xgboost(train_data, test_data, n_days=14, use_tuning=True):
    """
    ═══════════════════════════════════════════════════════════════════════════════
    TRAIN XGBOOST WITH HYPERPARAMETER TUNING (RandomizedSearchCV + TimeSeriesSplit)
    ═══════════════════════════════════════════════════════════════════════════════
    [UPGRADED] Sử dụng RandomizedSearchCV để tìm hyperparameters tối ưu.
    TimeSeriesSplit đảm bảo cross-validation phù hợp với dữ liệu chuỗi thời gian.
    
    Tham số:
        use_tuning: Nếu True, chạy RandomizedSearchCV (chậm hơn nhưng tốt hơn)
                    Nếu False, dùng params mặc định (nhanh)
    ═══════════════════════════════════════════════════════════════════════════════
    """
    from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
    
    # Features: All lag columns + technical indicators
    features = [c for c in train_data.columns if c.startswith('Lag_') or c in ['RSI', 'MACD', 'ATR']]
    
    X_train = train_data[features].fillna(0)
    y_train = train_data['Close']
    X_test = test_data[features].fillna(0)
    
    if use_tuning and len(X_train) > 100:
        # ─────────────────────────────────────────────────────────────────────────
        # HYPERPARAMETER SEARCH SPACE
        # ─────────────────────────────────────────────────────────────────────────
        param_distributions = {
            'n_estimators': [100, 200, 300, 500, 700],
            'max_depth': [3, 4, 5, 6, 7, 8],
            'learning_rate': [0.005, 0.01, 0.02, 0.05, 0.1],
            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
            'min_child_weight': [1, 3, 5, 7],
            'gamma': [0, 0.1, 0.2, 0.3]
        }
        
        # TimeSeriesSplit for proper time-series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        base_model = XGBRegressor(n_jobs=-1, random_state=42, verbosity=0)
        
        # RandomizedSearchCV: Faster than GridSearch, still effective
        search = RandomizedSearchCV(
            base_model,
            param_distributions,
            n_iter=20,  # Number of random combinations to try
            scoring='neg_mean_absolute_error',
            cv=tscv,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        
        search.fit(X_train, y_train)
        model = search.best_estimator_
        
        # Log best params (for debugging/reporting)
        print(f"[XGBoost] Best params: {search.best_params_}")
        print(f"[XGBoost] Best CV Score (MAE): {-search.best_score_:.4f}")
    else:
        # Fallback: Default params (fast mode)
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
    ═══════════════════════════════════════════════════════════════════════════════
    HUẤN LUYỆN MÔ HÌNH LSTM VỚI ATTENTION LAYER
    ───────────────────────────────────────────────────────────────────────────────
    Mục tiêu: Dự báo Log Returns của chỉ số VN30.
    Đầu vào:  df - DataFrame chứa dữ liệu lịch sử
              train_size - Số lượng mẫu dùng để huấn luyện
              n_days - Số ngày dự báo (mặc định 14)
    Đầu ra:   pred_prices - Giá dự báo trên tập Test
              lstm_objects - Dictionary chứa model, scaler, features để dùng cho dự báo tương lai
    
    LƯU Ý QUAN TRỌNG VỀ DATA LEAKAGE:
    - MinMaxScaler chỉ được FIT trên tập TRAINING
    - Tập TEST và FUTURE chỉ được TRANSFORM bằng scaler đã fit
    - Điều này đảm bảo mô hình không "nhìn trộm" dữ liệu tương lai
    ═══════════════════════════════════════════════════════════════════════════════
    """
    from tensorflow.keras.layers import Attention, MultiHeadAttention, LayerNormalization
    
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # ─────────────────────────────────────────────────────────────────────────────
    # BƯỚC 1: TÍNH LOG RETURNS
    # ─────────────────────────────────────────────────────────────────────────────
    # Công thức: r_t = log(P_t / P_{t-1})
    # Log Returns có tính ổn định hơn Returns thông thường (stationary)
    df_lstm = df.copy()
    df_lstm['Log_Ret'] = np.log(df_lstm['Close'] / df_lstm['Close'].shift(1))
    df_lstm.fillna(0, inplace=True)
    
    # ─────────────────────────────────────────────────────────────────────────────
    # BƯỚC 2: TẠO LAG FEATURES (Biến trễ)
    # ─────────────────────────────────────────────────────────────────────────────
    # Lag_Ret_1: Log Return của ngày hôm qua
    # Lag_Ret_2: Log Return của 2 ngày trước
    # Lag_Ret_3: Log Return của 3 ngày trước
    for i in range(1, 4):
        df_lstm[f'Lag_Ret_{i}'] = df_lstm['Log_Ret'].shift(i)
    df_lstm.dropna(inplace=True)
    
    features = [c for c in df_lstm.columns if c.startswith('Lag_')]
    X_vals = df_lstm[features].values
    y_vals = df_lstm['Log_Ret'].values
    
    # ─────────────────────────────────────────────────────────────────────────────
    # BƯỚC 3: CHIA TẬP TRAIN/TEST (Trước khi Scale)
    # ─────────────────────────────────────────────────────────────────────────────
    split_idx = len(df_lstm) - (len(df) - train_size)
    
    X_train_raw = X_vals[:split_idx]
    y_train = y_vals[:split_idx]
    X_test_raw = X_vals[split_idx:]
    
    # ─────────────────────────────────────────────────────────────────────────────
    # BƯỚC 4: CHUẨN HÓA DỮ LIỆU (FIT CHỈ TRÊN TRAINING - CHỐNG DATA LEAKAGE)
    # ─────────────────────────────────────────────────────────────────────────────
    # ⚠️ QUAN TRỌNG: Scaler chỉ được fit trên tập TRAINING
    # Sau đó dùng scaler này để transform tập TEST và FUTURE
    scaler_X = MinMaxScaler(feature_range=(-1, 1))
    X_train_scaled = scaler_X.fit_transform(X_train_raw)  # FIT + TRANSFORM trên Train
    X_test_scaled = scaler_X.transform(X_test_raw)        # CHỈ TRANSFORM trên Test
    
    # Reshape cho LSTM: [samples, timesteps, features]
    X_train = X_train_scaled.reshape(-1, 1, len(features))
    X_test = X_test_scaled.reshape(-1, 1, len(features))
    
    # ─────────────────────────────────────────────────────────────────────────────
    # BƯỚC 5: XÂY DỰNG MÔ HÌNH LSTM VỚI ATTENTION LAYER
    # ─────────────────────────────────────────────────────────────────────────────
    # Attention Layer giúp mô hình "tập trung" vào các thời điểm quan trọng
    # trong chuỗi thời gian thay vì xử lý tất cả thời điểm như nhau
    from tensorflow.keras import Input, Model
    
    inputs = Input(shape=(1, len(features)))
    
    # Bidirectional LSTM: Đọc chuỗi từ cả 2 hướng (quá khứ→hiện tại và hiện tại→quá khứ)
    # [UPGRADED] Tăng units để tăng capacity
    lstm_out = Bidirectional(LSTM(64, return_sequences=True))(inputs)
    lstm_out = Dropout(0.3)(lstm_out)  # [NEW] Regularization giữa các layers
    
    # Second LSTM layer for deeper learning
    lstm_out2 = Bidirectional(LSTM(32, return_sequences=True))(lstm_out)
    
    # Self-Attention: Tính trọng số attention cho mỗi timestep
    # Sử dụng LayerNormalization để ổn định training
    normalized = LayerNormalization()(lstm_out2)
    
    # Flatten và Dense layers
    from tensorflow.keras.layers import Flatten
    flattened = Flatten()(normalized)
    dense1 = Dense(32, activation='relu')(flattened)  # [UPGRADED] Tăng neurons
    dropout1 = Dropout(0.3)(dense1)  # [UPGRADED] Tăng dropout
    outputs = Dense(1)(dropout1)
    
    model = Model(inputs, outputs)
    
    # [UPGRADED] Use Adam with configurable learning rate
    from tensorflow.keras.optimizers import Adam
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse')
    
    # ─────────────────────────────────────────────────────────────────────────────
    # [UPGRADED] CALLBACKS: EarlyStopping + ReduceLROnPlateau
    # ─────────────────────────────────────────────────────────────────────────────
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=0
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=0
        )
    ]
    
    # Huấn luyện với validation split và callbacks
    model.fit(
        X_train, y_train, 
        epochs=100,  # [UPGRADED] Tăng epochs, EarlyStopping sẽ dừng sớm nếu overfitting
        batch_size=16, 
        validation_split=0.2,  # [NEW] 20% của tập train để validate
        callbacks=callbacks,
        verbose=0
    )
    
    # ─────────────────────────────────────────────────────────────────────────────
    # BƯỚC 6: DỰ BÁO VÀ CHUYỂN ĐỔI NGƯỢC (INVERSE TRANSFORM)
    # ─────────────────────────────────────────────────────────────────────────────
    # Dự báo Log Returns trên tập Test
    pred_log_ret = model.predict(X_test, verbose=0).flatten()
    
    # Chuyển đổi Log Returns → Giá gốc
    # Công thức: P_t = P_{t-1} * exp(r_t)
    prev_close = df.iloc[train_size-1:-1]['Close'].values
    min_len = min(len(prev_close), len(pred_log_ret))
    pred_prices = prev_close[:min_len] * np.exp(pred_log_ret[:min_len])
    
    # ─────────────────────────────────────────────────────────────────────────────
    # BƯỚC 7: ĐÓNG GÓI ĐỐI TƯỢNG ĐỂ DỰ BÁO TƯƠNG LAI
    # ─────────────────────────────────────────────────────────────────────────────
    lstm_objects = {
        'model': model,
        'scaler_X': scaler_X,  # Scaler đã fit trên Training
        'features': features,
        'train_size': train_size
    }
    
    return pred_prices, lstm_objects


def train_meta_learner(y_true, model_predictions, feature_df=None):
    """
    Train a Stacking Meta-Learner (XGBoost).
    [UPDATED] Supports Feature-Augmented Stacking (using original features + predictions).
    """
    # Align lengths
    min_len = min(len(y_true), *[len(p) for p in model_predictions.values()])
    y_target = y_true[-min_len:]
    
    # Create Prediction Matrix [n_samples, n_models]
    keys = sorted(model_predictions.keys())
    X_preds = np.column_stack([model_predictions[k][-min_len:] for k in keys])
    
    # [ADVANCED] Feature Augmented Stacking
    if feature_df is not None:
        try:
            # Align features with target
            X_features = feature_df.iloc[-min_len:].select_dtypes(include=[np.number]).fillna(0).values
            X_meta = np.column_stack([X_preds, X_features])
        except:
             X_meta = X_preds
    else:
        X_meta = X_preds
    
    # Train XGBoost Meta-Learner
    meta_model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
    meta_model.fit(X_meta, y_target)
    
    # Calculate score
    final_pred = meta_model.predict(X_meta)
    mape = mean_absolute_percentage_error(y_target, final_pred) * 100
    
    # Cross Validation Score (Simple proxy)
    cv_score = 100 - mape 
    
    return meta_model, keys, final_pred, mape, cv_score

def forecast_ensemble_stacking(meta_model, model_keys, future_predictions, future_context_df=None):
    """
    Generate future forecast using the trained Meta-Learner.
    [UPDATED] Supports Feature-Augmented Forecasting (using future context).
    """
    # Stack future predictions in the same order as training
    keys = sorted(model_keys) # Ensure order matches training
    X_future_preds = np.column_stack([future_predictions[k] for k in keys])
    
    # [ADVANCED] Feature Augmented Stacking
    if future_context_df is not None:
        try:
             # Align features (select numeric, fillna)
             # Assuming future_context_df has length = n_days and aligned columns
             X_future_ctx = future_context_df.select_dtypes(include=[np.number]).fillna(0).values
             
             # Combine: [Preds | Features]
             X_future = np.column_stack([X_future_preds, X_future_ctx])
        except:
             # Fallback if dimensions mismatch
             X_future = X_future_preds
    else:
        X_future = X_future_preds
    
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
            last_vals = train_exog.iloc[-1].values
            future_exog_list = []
            
            # Calculate Means from history
            means = train_exog.mean().values
            # RSI mean is effectively 50 (Neutral)
            means[1] = 50.0 
            
            # Generate steps
            for i in range(1, n_days + 1):
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
    ═══════════════════════════════════════════════════════════════════════════════
    DỰ BÁO TƯƠNG LAI BẰNG CHIẾN LƯỢC RECURSIVE MULTI-STEP (XGBOOST)
    ───────────────────────────────────────────────────────────────────────────────
    Mô tả: Sử dụng chính mô hình XGBoost đã huấn luyện để dự báo từng bước.
           Kết quả dự báo T+1 được dùng làm đầu vào cho T+2, và cứ thế tiếp tục.
    
    Ưu điểm: Đảm bảo tính NHẤT QUÁN giữa Backtest và Future Forecast
             (thay vì dùng Holt-Winters như fallback trước đó)
    
    Công thức Recursive:
        ŷ[t+1] = f(X[t])           | Dự báo bước 1
        ŷ[t+2] = f(X[t] | ŷ[t+1])   | Dự báo bước 2 (cập nhật lag từ bước 1)
        ...
        ŷ[t+n] = f(X[t] | ŷ[t+n-1]) | Dự báo bước n
    ═══════════════════════════════════════════════════════════════════════════════
    """
    model, _, features, _ = model_objs
    
    # ─────────────────────────────────────────────────────────────────────────────
    # BƯỚC 1: CHUẨN BỊ DỮ LIỆU BAN ĐẦU
    # ─────────────────────────────────────────────────────────────────────────────
    # Lấy hàng cuối cùng làm điểm xuất phát
    last_row = df.iloc[-1:].copy()
    
    # Danh sách lưu kết quả dự báo
    predictions = []
    
    # ─────────────────────────────────────────────────────────────────────────────
    # BƯỚC 2: VÒNG LẶP RECURSIVE FORECASTING
    # ─────────────────────────────────────────────────────────────────────────────
    for step in range(n_days):
        # 2.1 Chuẩn bị vector đặc trưng
        X_input = last_row[features].fillna(0).values
        
        # 2.2 Dự báo giá cho ngày tiếp theo
        pred = model.predict(X_input)[0]
        predictions.append(pred)
        
        # 2.3 CẬP NHẬT LAG FEATURES cho bước tiếp theo
        # Đẩy các lag lùi lại 1 bước
        # Lag_Close_3 ← Lag_Close_2
        # Lag_Close_2 ← Lag_Close_1
        # Lag_Close_1 ← Giá vừa dự báo
        for i in range(3, 1, -1):
            lag_col = f'Lag_Close_{i}'
            prev_lag_col = f'Lag_Close_{i-1}'
            if lag_col in last_row.columns and prev_lag_col in last_row.columns:
                last_row[lag_col] = last_row[prev_lag_col].values[0]
        
        if 'Lag_Close_1' in last_row.columns:
            last_row['Lag_Close_1'] = pred
        
        # 2.4 Cập nhật các Lag khác (Vol, RSI, MACD, ATR) - Giữ nguyên giá trị cuối
        # (Trong thực tế không thể biết được Vol/RSI tương lai)
    
    # ─────────────────────────────────────────────────────────────────────────────
    # BƯỚC 3: TRẢ VỀ KẾT QUẢ
    # ─────────────────────────────────────────────────────────────────────────────
    return np.array(predictions)

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



def forecast_future_xgboost_recursive(model, last_features, feature_names, n_days=30):
    """
    Dự báo tương lai (Recursive Strategy) cho XGBoost.
    Tự động cập nhật Lag Features sau mỗi bước dự báo.
    [LOGIC] Thêm quán tính (Momentum Trend) để tránh đường thẳng.
    """
    future_preds = []
    current_feats = dict(zip(feature_names, last_features))
    
    lag_cols = [c for c in feature_names if 'Lag_Close' in c]
    import re
    lag_map = {}
    for c in lag_cols:
        match = re.search(r'(\d+)', c)
        if match:
            lag_map[int(match.group(1))] = c
            
    sorted_lags = sorted(lag_map.keys())
    
    for _ in range(n_days):
        input_arr = np.array([current_feats[f] for f in feature_names]).reshape(1, -1)
        pred = model.predict(input_arr)[0]
        
        # [DETERMINISTIC LOGIC] Momentum Trend
        # Logic: Nếu giá đang tăng (Lag1 > Lag2), nó có xu hướng tăng tiếp (Momentum)
        if 1 in lag_map and 2 in lag_map:
             momentum = current_feats[lag_map[1]] - current_feats[lag_map[2]]
             # Decay momentum to simulate mean reversion (Lực quán tính giảm dần)
             pred = pred + (momentum * 0.9)
             
        future_preds.append(pred)
        
        # Update Lags
        for i in range(len(sorted_lags)-1, 0, -1):
            curr_lag = sorted_lags[i]
            prev_lag = sorted_lags[i-1]
            current_feats[lag_map[curr_lag]] = current_feats[lag_map[prev_lag]]
            
        if 1 in lag_map:
            current_feats[lag_map[1]] = pred
            
    return future_preds
