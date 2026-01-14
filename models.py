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

def add_macro_features(df):
    """
    ───────────────────────────────────────────────────────────────────────────────
    FEATURE ENGINEERING LIÊN THỊ TRƯỜNG (Feature-Augmented)
    ───────────────────────────────────────────────────────────────────────────────
    Input: df đã tích hợp dữ liệu SP500 và USDVND.
    Output: df với các biến phái sinh mới.
    
    New Feature: Correlation_VN30_SP500 (Rolling 20 days)
    ───────────────────────────────────────────────────────────────────────────────
    """
    # Kiểm tra dữ liệu đầu vào
    if 'SP500' not in df.columns:
        df['SP500'] = 0
    if 'USDVND' not in df.columns:
        df['USDVND'] = 0
        
    # 1. Rolling Correlation: Đo lường mức độ đồng pha giữa VN30 và S&P500
    # Window = 20 (khoảng 1 tháng giao dịch)
    try:
        df['Correlation_VN30_SP500'] = df['Close'].rolling(window=20).corr(df['SP500'])
        df['Correlation_VN30_SP500'] = df['Correlation_VN30_SP500'].fillna(0)
    except:
        df['Correlation_VN30_SP500'] = 0

    # 2. Log Returns: Biến động tương đối (Stationary)
    # Tránh chia cho 0
    with np.errstate(divide='ignore', invalid='ignore'):
        df['SP500_LogRet'] = np.log(df['SP500'] / df['SP500'].shift(1))
        df['USDVND_LogRet'] = np.log(df['USDVND'] / df['USDVND'].shift(1))
    
    df['SP500_LogRet'] = df['SP500_LogRet'].replace([np.inf, -np.inf], 0).fillna(0)
    df['USDVND_LogRet'] = df['USDVND_LogRet'].replace([np.inf, -np.inf], 0).fillna(0)

    # 3. Lag Features (Trễ): Dùng thông tin t-1 để dự báo t
    # Đây là bước quan trọng để tránh Data Leakage
    df['Lag_SP500_Ret_1'] = df['SP500_LogRet'].shift(1)
    df['Lag_USDVND_Ret_1'] = df['USDVND_LogRet'].shift(1)
    df['Lag_Corr_VN30_SP500_1'] = df['Correlation_VN30_SP500'].shift(1)
    
    # Drop missing rows created by lags
    df.dropna(inplace=True)
    
    return df

# ═══════════════════════════════════════════════════════════════════════════════
# PHẦN 2: TREND MODELING (Mô hình hóa Xu hướng)
# ═══════════════════════════════════════════════════════════════════════════════

def fit_trend_model(df):
    """
    ───────────────────────────────────────────────────────────────────────────────
    MÔ HÌNH HÓA XU HƯỚNG (TREND MODELING)
    ───────────────────────────────────────────────────────────────────────────────
    Phương pháp: Holt-Winters Exponential Smoothing (San bằng mũ)
    Lý do chọn: 
        - Phản ứng tốt hơn với dữ liệu gần đây so với Polynomial Regression.
        - Phù hợp cho chuỗi thời gian tài chính có xu hướng thay đổi (Stochastic Trend).
    
    Đầu ra:
        - trend_model: Mô hình đã huấn luyện
        - residuals: Phần dư (Chuỗi dừng để dự báo bằng ML)
    ───────────────────────────────────────────────────────────────────────────────
    """
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    
    close_prices = df['Close'].values
    
    try:
        # Holt-Winters dạng cộng tính (Additive), có xu hướng tắt dần (Damped Trend)
        trend_model = ExponentialSmoothing(
            close_prices,
            trend='add',
            seasonal=None,
            damped_trend=True,
            initialization_method='estimated'
        ).fit(optimized=True)
        
        # Tách thành phần xu hướng
        fitted_trend = trend_model.fittedvalues
        residuals = close_prices - fitted_trend
        
    except Exception as e:
        # Fallback: Trung bình động (SMA) nếu HW thất bại
        print(f"Holt-Winters failed, using SMA fallback: {e}")
        fitted_trend = pd.Series(close_prices).rolling(window=20, min_periods=1).mean().values
        residuals = close_prices - fitted_trend
        trend_model = None
        
    return trend_model, residuals, None  # None thay thế cho poly object cũ

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
    """
    ───────────────────────────────────────────────────────────────────────────────
    HUẤN LUYỆN MÔ HÌNH ARIMAX (AutoRegressive Integrated Moving Average with eXogenous)
    ───────────────────────────────────────────────────────────────────────────────
    Mục tiêu: Mô hình thống kê làm nền tảng (Baseline Statistical Model).
    
    Tham số:
        - p (AR): Tự hồi quy
        - d (I): Sai phân (để đảm bảo tính dừng)
        - q (MA): Trung bình trượt
        - exog: Biến ngoại sinh (Volume, RSI, ATR)
    
    Quy trình:
        1. Grid Search tìm bộ tham số (p,d,q) tối ưu dựa trên AIC.
        2. Walk-forward Forecast trên tập Test.
    ───────────────────────────────────────────────────────────────────────────────
    """
    # Xử lý dữ liệu ngoại sinh (Exogenous variables)
    train_exog = train_data[exog_cols].bfill()
    test_exog = test_data[exog_cols].bfill()
    train_endog = train_data['Close']
    
    # Grid Search tìm tham số tối ưu (AIC thấp nhất)
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
    ───────────────────────────────────────────────────────────────────────────────
    HUẤN LUYỆN XGBOOST (DIRECT MULTI-STEP FORECASTING)
    ───────────────────────────────────────────────────────────────────────────────
    Chiến lược: Direct Forecasting (Dự báo trực tiếp)
        - Thay vì Recursive (dễ tích lũy sai số), mô hình học ánh xạ trực tiếp từ X -> Y.
        - Tuy nhiên, code dưới đây sử dụng Recursive cho đơn giản hóa (cần lưu ý trong luận văn).
    
    Đặc trưng đầu vào:
        - Lag Features (Trễ)
        - Technical Indicators (RSI, MACD, ATR)
    
    Hyperparameters (được chọn qua Grid Search trước đó):
        - n_estimators=500
        - learning_rate=0.01
        - max_depth=6
    ───────────────────────────────────────────────────────────────────────────────
    """
    # Features: All lag columns + technical indicators
    features = [c for c in train_data.columns if c.startswith('Lag_') or c in ['RSI', 'MACD', 'ATR']]
    
    X_train = train_data[features].fillna(0)
    y_train = train_data['Close']
    X_test = test_data[features].fillna(0)
    
    # Huấn luyện XGBoost (Gradient Boosting Decision Tree)
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
    
    # Dự báo trên tập Test
    predictions = model.predict(X_test)
    
    # Trả về tuple chuẩn: (predictions, model, trend_model, features, poly)
    return predictions, model, None, features, None

def train_lstm(df, train_size, n_days=14):
    """
    ═══════════════════════════════════════════════════════════════════════════════
    HUẤN LUYỆN MÔ HÌNH LSTM VỚI ATTENTION LAYER (Deep Learning)
    ───────────────────────────────────────────────────────────────────────────────
    Mục tiêu: Dự báo Log Returns của chỉ số VN30.
    
    Kiến trúc mạng (Network Architecture):
        1. Bidirectional LSTM: Học sự phụ thuộc hai chiều (Past & Future context trong training).
        2. Attention Mechanism: (Đã comment out để tối ưu tốc độ, có thể bật lại).
        3. LayerNormalization: Ổn định quá trình Gradient Descent.
        4. Dropout (0.2): Chống Overfitting.
    
    LƯU Ý QUAN TRỌNG VỀ DATA LEAKAGE:
    - MinMaxScaler chỉ được FIT trên tập TRAINING.
    - Tập TEST và FUTURE chỉ được TRANSFORM bằng scaler đã fit.
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
    lstm_out = Bidirectional(LSTM(32, return_sequences=True))(inputs)
    
    # Self-Attention: Tính trọng số attention cho mỗi timestep
    # attention_output = Attention()([lstm_out, lstm_out])
    # Sử dụng LayerNormalization để ổn định training
    normalized = LayerNormalization()(lstm_out)
    
    # Flatten và Dense layers
    from tensorflow.keras.layers import Flatten
    flattened = Flatten()(normalized)
    dense1 = Dense(16, activation='relu')(flattened)
    dropout1 = Dropout(0.2)(dense1)
    outputs = Dense(1)(dropout1)
    
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    
    # Huấn luyện với verbose=0 để không in log
    model.fit(X_train, y_train, epochs=30, batch_size=16, verbose=0)
    
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
    ───────────────────────────────────────────────────────────────────────────────
    HUẤN LUYỆN META-LEARNER (FEATURE-AUGMENTED STACKING)
    ───────────────────────────────────────────────────────────────────────────────
    Kiến trúc Ensemble: Stacking (Level 1 Model)
    
    Đổi mới (Novelty):
        - Không chỉ dùng dự báo của các mô hình cơ sở (Base Models) làm đầu vào.
        - Bổ sung thêm "Context Augmentation" (RSI, ATR, Correlation, Macro) để
          Meta-learner hiểu bối cảnh thị trường nào thì mô hình nào hoạt động tốt hơn.
    
    Quy trình kiểm định (Validation):
        - TimeSeriesSplit Cross-Validation (K=3) để đảm bảo tính ổn định (Robustness).
        - Tránh hiện tượng Overfitting khi Meta-learner học thuộc lòng kết quả quá khứ.
    
    Tham số XGBoost Meta-learner:
        - Tinh chỉnh nhẹ nhàng (max_depth=3) để tránh High Variance.
    ───────────────────────────────────────────────────────────────────────────────
    """
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_absolute_percentage_error
    
    # Align lengths
    min_len = min(len(y_true), *[len(p) for p in model_predictions.values()])
    if feature_df is not None:
        min_len = min(min_len, len(feature_df))
        
    y_target = y_true[-min_len:]
    
    # 1. Base Model Predictions (Level 0)
    keys = sorted(model_predictions.keys())
    X_base = np.column_stack([model_predictions[k][-min_len:] for k in keys])
    
    # 2. Context Features (Augmentation)
    if feature_df is not None:
        # Chọn các biến ngữ cảnh quan trọng
        ctx_cols = ['RSI', 'ATR', 'Correlation_VN30_SP500', 'SP500_LogRet'] 
        valid_cols = [c for c in ctx_cols if c in feature_df.columns]
        X_context = feature_df[valid_cols].iloc[-min_len:].values
        
        # Kết hợp (Stacking)
        X_meta = np.column_stack([X_base, X_context])
        meta_features = keys + valid_cols
    else:
        X_meta = X_base
        meta_features = keys
        
    # 3. TimeSeries Cross-Validation (Tính học thuật)
    tscv = TimeSeriesSplit(n_splits=3)
    cv_scores = []
    
    # Simple Hyperparam for Meta-learner (XGBoost is robust)
    meta_model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    
    # CV Loop
    for train_index, val_index in tscv.split(X_meta):
        X_train_cv, X_val_cv = X_meta[train_index], X_meta[val_index]
        y_train_cv, y_val_cv = y_target[train_index], y_target[val_index]
        
        meta_model.fit(X_train_cv, y_train_cv)
        pred_cv = meta_model.predict(X_val_cv)
        score = mean_absolute_percentage_error(y_val_cv, pred_cv)
        cv_scores.append(score)
        
    avg_cv_mape = np.mean(cv_scores) * 100 if cv_scores else 0
    
    # 4. Final Train on All Data
    meta_model.fit(X_meta, y_target)
    final_pred = meta_model.predict(X_meta)
    mape = mean_absolute_percentage_error(y_target, final_pred) * 100
    
    return meta_model, meta_features, final_pred, mape, avg_cv_mape

def forecast_ensemble_stacking(meta_model, feature_names, future_predictions, future_context_df=None):
    """
    ───────────────────────────────────────────────────────────────────────────────
    DỰ BÁO TƯƠNG LAI VỚI META-LEARNER
    ───────────────────────────────────────────────────────────────────────────────
    Quy trình:
        1. Thu thập dự báo từ các Base Models (Tương lai).
        2. Thu thập/Dự báo các biến ngữ cảnh (Future Context).
        3. Đưa qua Meta-learner để tổng hợp kết quả cuối cùng.
    
    Chú ý:
        - Thứ tự các cột (Features) phải khớp CHÍNH XÁC với lúc training.
    ───────────────────────────────────────────────────────────────────────────────
    """
    # Keys should match what was trained (first N columns are models)
    # Identify model keys vs context keys
    # Issue: We stored 'feature_names' (Models + Context).
    # We need to reconstruct X_future exactly.
    
    # Separate Model Keys from Context Keys
    # We assume 'future_predictions' keys are the Model Keys
    model_keys = sorted(list(future_predictions.keys()))
    
    # 1. Base Predictions
    X_base = np.column_stack([future_predictions[k] for k in model_keys])
    
    # 2. Context Features
    if future_context_df is not None:
        # Filter cols that are in feature_names AND in dataframe
        # Note: feature_names contains [ARIMAX, LSTM... RSI, ATR...]
        # We need to extract just the Context parts from feature_names
        context_keys = [f for f in feature_names if f not in model_keys]
        
        if context_keys:
            # Ensure dataframe has them
            X_context = future_context_df[context_keys].values
            X_future = np.column_stack([X_base, X_context])
        else:
            X_future = X_base
    else:
        X_future = X_base
        
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
