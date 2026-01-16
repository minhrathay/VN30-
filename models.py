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
# PHẦN 0: GARCH VOLATILITY MODEL (Mô hình hóa Biến động)
# ═══════════════════════════════════════════════════════════════════════════════

def fit_garch_model(df, return_col='Close'):
    """
    ───────────────────────────────────────────────────────────────────────────────
    FIT GARCH(1,1) MODEL - Mô hình hóa Volatility Clustering
    ───────────────────────────────────────────────────────────────────────────────
    
    GARCH (Generalized Autoregressive Conditional Heteroskedasticity) là mô hình
    tiêu chuẩn trong tài chính để dự báo độ biến động (volatility).
    
    Đặc điểm quan trọng:
    - Volatility Clustering: Biến động mạnh thường đi theo biến động mạnh
    - Mean Reversion: Volatility có xu hướng quay về mức trung bình dài hạn
    
    Tham số:
        df: DataFrame chứa dữ liệu giá
        return_col: Tên cột giá để tính returns
    
    Trả về:
        garch_result: Kết quả đã fit của GARCH model
        returns: Chuỗi log returns đã tính
    ───────────────────────────────────────────────────────────────────────────────
    """
    try:
        from arch import arch_model
        
        # Tính Log Returns
        prices = df[return_col].values
        returns = np.diff(np.log(prices)) * 100  # Scale to percentage
        
        # Loại bỏ NaN và Inf
        returns = returns[np.isfinite(returns)]
        
        # Fit GARCH(1,1) model
        model = arch_model(returns, vol='Garch', p=1, q=1, mean='AR', lags=1, rescale=False)
        result = model.fit(disp='off', show_warning=False)
        
        return result, returns
        
    except Exception as e:
        print(f"GARCH fitting failed: {e}. Using fallback volatility.")
        return None, None


def forecast_volatility_garch(garch_result, n_days=14):
    """
    ───────────────────────────────────────────────────────────────────────────────
    DỰ BÁO VOLATILITY TƯƠNG LAI TỪ GARCH MODEL
    ───────────────────────────────────────────────────────────────────────────────
    
    Trả về: Array of daily volatility forecasts (as decimal, not percentage)
    ───────────────────────────────────────────────────────────────────────────────
    """
    try:
        forecast = garch_result.forecast(horizon=n_days)
        # Variance forecast → Volatility (standard deviation)
        # Chia 100 để chuyển từ percentage về decimal
        vol_forecast = np.sqrt(forecast.variance.values[-1, :]) / 100
        return vol_forecast
    except:
        # Fallback: constant volatility based on historical
        return np.full(n_days, 0.015)  # 1.5% daily volatility


def detect_market_regime(df, lookback=20):
    """
    ───────────────────────────────────────────────────────────────────────────────
    NHẬN DIỆN CHẾ ĐỘ THỊ TRƯỜNG (Market Regime Detection)
    ───────────────────────────────────────────────────────────────────────────────
    
    Phân loại thị trường thành 3 chế độ:
    - TRENDING_UP:   Đang trong xu hướng tăng mạnh
    - TRENDING_DOWN: Đang trong xu hướng giảm mạnh
    - SIDEWAYS:      Đi ngang, không có xu hướng rõ

    Trả về:
        regime: str ('trending_up', 'trending_down', 'sideways')
        strength: float (0-1, độ mạnh của regime)
        momentum: float (hướng và cường độ momentum)
    ───────────────────────────────────────────────────────────────────────────────
    """
    prices = df['Close'].tail(lookback).values
    
    # Tính momentum (linear regression slope)
    x = np.arange(len(prices))
    slope, intercept = np.polyfit(x, prices, 1)
    
    # Normalize slope by price level
    momentum = slope / prices.mean()
    
    # Tính R² để đo độ mạnh của trend
    y_pred = slope * x + intercept
    ss_res = np.sum((prices - y_pred) ** 2)
    ss_tot = np.sum((prices - prices.mean()) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Classify regime
    if r_squared > 0.6:  # Strong trend
        if momentum > 0.001:
            regime = 'trending_up'
        else:
            regime = 'trending_down'
        strength = r_squared
    else:
        regime = 'sideways'
        strength = 1 - r_squared
    
    return regime, strength, momentum


def calculate_technical_score(df):
    """
    ═══════════════════════════════════════════════════════════════════════════════
    TECHNICAL SCORE SYSTEM - Tính điểm hướng dựa trên chỉ báo kỹ thuật
    ═══════════════════════════════════════════════════════════════════════════════
    
    Kết hợp 4 chỉ báo kỹ thuật để xác định hướng forecast:
    - RSI: Overbought/Oversold
    - MACD: Crossover và Momentum
    - Bollinger Bands: Position relative to bands
    - MA Crossover: Short-term vs Medium-term trend
    
    Trả về:
        score: float (-8 to +8)
        direction: str ('bullish', 'bearish', 'neutral')
        daily_drift: float (expected daily % change)
        details: dict (breakdown of each indicator's contribution)
    ═══════════════════════════════════════════════════════════════════════════════
    """
    score = 0
    details = {}
    
    # Get latest values
    close = df['Close'].iloc[-1]
    
    # ─────────────────────────────────────────────────────────────────────────────
    # 1. RSI Analysis (-2 to +2)
    # ─────────────────────────────────────────────────────────────────────────────
    rsi = df['RSI'].iloc[-1] if 'RSI' in df.columns else 50
    
    if rsi < 30:
        rsi_score = 2  # Strong oversold → expect bounce UP
        rsi_signal = "Oversold (Strong Buy)"
    elif rsi < 40:
        rsi_score = 1  # Mild oversold
        rsi_signal = "Mildly Oversold"
    elif rsi > 70:
        rsi_score = -2  # Strong overbought → expect pullback DOWN
        rsi_signal = "Overbought (Strong Sell)"
    elif rsi > 60:
        rsi_score = -1  # Mild overbought
        rsi_signal = "Mildly Overbought"
    else:
        rsi_score = 0
        rsi_signal = "Neutral"
    
    score += rsi_score
    details['RSI'] = {'value': rsi, 'score': rsi_score, 'signal': rsi_signal}
    
    # ─────────────────────────────────────────────────────────────────────────────
    # 2. MACD Analysis (-2 to +2)
    # ─────────────────────────────────────────────────────────────────────────────
    if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
        macd = df['MACD'].iloc[-1]
        macd_signal = df['MACD_Signal'].iloc[-1]
        macd_prev = df['MACD'].iloc[-2] if len(df) > 1 else macd
        
        # MACD histogram
        histogram = macd - macd_signal
        histogram_prev = macd_prev - df['MACD_Signal'].iloc[-2] if len(df) > 1 else histogram
        
        if macd > macd_signal:
            if histogram > histogram_prev:  # Increasing bullish momentum
                macd_score = 2
                macd_sig = "Bullish Crossover + Momentum"
            else:
                macd_score = 1
                macd_sig = "Bullish"
        elif macd < macd_signal:
            if histogram < histogram_prev:  # Increasing bearish momentum
                macd_score = -2
                macd_sig = "Bearish Crossover + Momentum"
            else:
                macd_score = -1
                macd_sig = "Bearish"
        else:
            macd_score = 0
            macd_sig = "Neutral"
    else:
        macd_score = 0
        macd_sig = "N/A"
    
    score += macd_score
    details['MACD'] = {'score': macd_score, 'signal': macd_sig}
    
    # ─────────────────────────────────────────────────────────────────────────────
    # 3. Bollinger Bands Analysis (-1 to +1)
    # ─────────────────────────────────────────────────────────────────────────────
    if 'BB_Upper' in df.columns and 'BB_Lower' in df.columns:
        bb_upper = df['BB_Upper'].iloc[-1]
        bb_lower = df['BB_Lower'].iloc[-1]
        bb_middle = df['BB_Middle'].iloc[-1] if 'BB_Middle' in df.columns else (bb_upper + bb_lower) / 2
        
        if close < bb_lower:
            bb_score = 1  # Below lower band → oversold
            bb_sig = "Below Lower Band (Oversold)"
        elif close > bb_upper:
            bb_score = -1  # Above upper band → overbought
            bb_sig = "Above Upper Band (Overbought)"
        elif close > bb_middle:
            bb_score = 0.5  # Upper half
            bb_sig = "Upper Half"
        else:
            bb_score = -0.5  # Lower half
            bb_sig = "Lower Half"
    else:
        bb_score = 0
        bb_sig = "N/A"
    
    score += bb_score
    details['Bollinger'] = {'score': bb_score, 'signal': bb_sig}
    
    # ─────────────────────────────────────────────────────────────────────────────
    # 4. Moving Average Crossover (-2 to +2)
    # ─────────────────────────────────────────────────────────────────────────────
    # Calculate SMAs if not present
    if len(df) >= 21:
        sma_7 = df['Close'].tail(7).mean()
        sma_21 = df['Close'].tail(21).mean()
        sma_7_prev = df['Close'].iloc[-8:-1].mean() if len(df) > 8 else sma_7
        
        if sma_7 > sma_21:
            if sma_7 > sma_7_prev:  # Rising trend
                ma_score = 2
                ma_sig = "Uptrend + Accelerating"
            else:
                ma_score = 1
                ma_sig = "Uptrend"
        elif sma_7 < sma_21:
            if sma_7 < sma_7_prev:  # Falling trend
                ma_score = -2
                ma_sig = "Downtrend + Accelerating"
            else:
                ma_score = -1
                ma_sig = "Downtrend"
        else:
            ma_score = 0
            ma_sig = "Neutral"
    else:
        ma_score = 0
        ma_sig = "Insufficient Data"
    
    score += ma_score
    details['MA_Crossover'] = {'score': ma_score, 'signal': ma_sig}
    
    # ─────────────────────────────────────────────────────────────────────────────
    # Calculate Direction and Daily Drift
    # ─────────────────────────────────────────────────────────────────────────────
    # Score range: -8 to +8
    if score >= 4:
        direction = 'strong_bullish'
        daily_drift = 0.003  # +0.3% per day
    elif score >= 2:
        direction = 'bullish'
        daily_drift = 0.0015  # +0.15% per day
    elif score <= -4:
        direction = 'strong_bearish'
        daily_drift = -0.003  # -0.3% per day
    elif score <= -2:
        direction = 'bearish'
        daily_drift = -0.0015  # -0.15% per day
    else:
        direction = 'neutral'
        daily_drift = 0.0
    
    return score, direction, daily_drift, details


def forecast_probabilistic_ensemble(
    base_forecasts,     # Dict: {'ARIMAX': array, 'XGBoost': array, ...}
    garch_vol,          # Array: volatility forecast từ GARCH
    last_price,         # Float: giá đóng cửa cuối cùng
    regime_info,        # Tuple: (regime, strength, momentum)
    technical_drift=0.0, # Float: daily drift từ technical score
    n_simulations=500,  # Số Monte Carlo paths
    n_days=14
):
    """
    ═══════════════════════════════════════════════════════════════════════════════
    TẠO DỰ BÁO XÁC SUẤT VỚI FAN CHART (Probabilistic Ensemble Forecast)
    ═══════════════════════════════════════════════════════════════════════════════
    
    Kết hợp:
    1. Base forecasts từ các models (ARIMAX, XGBoost, LSTM)
    2. GARCH volatility để scale uncertainty
    3. Regime information để điều chỉnh drift
    4. Monte Carlo simulation để tạo distribution
    
    Trả về:
        dict: {
            'median': array,      # 50th percentile (đường chính)
            'lower_10': array,    # 10th percentile (pessimistic)
            'upper_90': array,    # 90th percentile (optimistic)
            'lower_25': array,    # 25th percentile
            'upper_75': array,    # 75th percentile
            'paths': array        # Tất cả simulated paths (for debugging)
        }
    ═══════════════════════════════════════════════════════════════════════════════
    """
    regime, strength, momentum = regime_info
    
    # ─────────────────────────────────────────────────────────────────────────────
    # BƯỚC 1: Tính Weighted Average của base forecasts (làm drift/center)
    # ─────────────────────────────────────────────────────────────────────────────
    if base_forecasts:
        # Stack all forecasts and take mean
        all_preds = np.array(list(base_forecasts.values()))
        # Ensure all have same length
        min_len = min(len(p) for p in base_forecasts.values())
        all_preds = np.array([p[:min_len] for p in base_forecasts.values()])
        base_path = np.mean(all_preds, axis=0)
    else:
        # Fallback: flat forecast at last price
        base_path = np.full(n_days, last_price)
    
    # Pad if needed
    if len(base_path) < n_days:
        last_val = base_path[-1] if len(base_path) > 0 else last_price
        base_path = np.pad(base_path, (0, n_days - len(base_path)), constant_values=last_val)
    
    # ─────────────────────────────────────────────────────────────────────────────
    # BƯỚC 2: Điều chỉnh drift dựa trên Regime
    # ─────────────────────────────────────────────────────────────────────────────
    # Trong trending regime, giữ momentum
    # Trong sideways, mean-revert mạnh hơn
    
    drift_adjustment = np.zeros(n_days)
    cumulative_momentum = 0
    
    for t in range(n_days):
        if regime == 'trending_up':
            # Momentum decay but still positive
            decay_factor = np.exp(-0.05 * t)  # Slow decay
            cumulative_momentum += momentum * last_price * decay_factor
        elif regime == 'trending_down':
            decay_factor = np.exp(-0.05 * t)
            cumulative_momentum += momentum * last_price * decay_factor
        else:  # sideways
            # Mean reversion - pull towards recent mean
            decay_factor = np.exp(-0.1 * t)  # Faster decay to mean
            cumulative_momentum *= decay_factor
            
        drift_adjustment[t] = cumulative_momentum * strength
    
    # Apply drift adjustment to base path
    adjusted_path = base_path + drift_adjustment
    
    # ─────────────────────────────────────────────────────────────────────────────
    # BƯỚC 3: Monte Carlo Simulation với GARCH volatility + Technical Drift
    # ─────────────────────────────────────────────────────────────────────────────
    np.random.seed(42)  # For reproducibility in demo
    
    # Ensure garch_vol has correct length
    if garch_vol is None or len(garch_vol) < n_days:
        # Fallback volatility: historical std scaled by sqrt(t)
        base_vol = 0.012  # ~1.2% daily
        garch_vol = np.array([base_vol * np.sqrt(t+1) for t in range(n_days)])
    else:
        garch_vol = garch_vol[:n_days]
    
    # Generate paths
    paths = np.zeros((n_simulations, n_days))
    
    for sim in range(n_simulations):
        path = np.zeros(n_days)
        price = last_price
        
        for t in range(n_days):
            # Random shock scaled by GARCH volatility
            shock = np.random.normal(0, garch_vol[t])
            
            # TECHNICAL DRIFT: Apply direction from technical indicators
            # Optimized for 7-day forecast - slower decay to maintain signal
            decay_factor = np.exp(-0.01 * t)  # Very slow decay for 7-day
            daily_return = technical_drift * decay_factor
            
            # Add drift to adjusted path target
            target_return = (adjusted_path[t] - price) / price + daily_return
            
            # Blend target return with random shock
            # Optimized for 7-day: higher model weight throughout
            model_weight = np.exp(-0.05 * t)  # Slower decay for 7-day accuracy
            actual_return = model_weight * target_return + (1 - model_weight) * (shock + daily_return)
            
            # Apply return
            price = price * (1 + actual_return)
            path[t] = price
            
        paths[sim, :] = path
    
    # ─────────────────────────────────────────────────────────────────────────────
    # BƯỚC 4: Tính Percentiles từ distribution
    # ─────────────────────────────────────────────────────────────────────────────
    result = {
        'median': np.percentile(paths, 50, axis=0),
        'lower_10': np.percentile(paths, 10, axis=0),
        'upper_90': np.percentile(paths, 90, axis=0),
        'lower_25': np.percentile(paths, 25, axis=0),
        'upper_75': np.percentile(paths, 75, axis=0),
        'mean': np.mean(paths, axis=0),
        'regime': regime,
        'regime_strength': strength
    }
    
    return result

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
    HUẤN LUYỆN MÔ HÌNH LSTM VỚI PROPER SEQUENCE LEARNING
    ───────────────────────────────────────────────────────────────────────────────
    [REFACTORED] Sử dụng lookback window thực sự để LSTM học được patterns
    
    Đầu vào:  df - DataFrame chứa dữ liệu lịch sử
              train_size - Số lượng mẫu dùng để huấn luyện
              n_days - Số ngày dự báo (mặc định 14)
    
    KIẾN TRÚC MỚI:
    - Lookback: 10 ngày (chuỗi thời gian thực sự)
    - Features: Log_Ret, RSI_norm, ATR_norm, Volume_norm (4 features)
    - Shape: (samples, 10, 4) - đúng format cho sequence learning
    - Output: Dự báo Log Return của ngày tiếp theo
    ═══════════════════════════════════════════════════════════════════════════════
    """
    from tensorflow.keras.layers import LayerNormalization
    from tensorflow.keras import Input, Model
    from tensorflow.keras.layers import Flatten
    from tensorflow.keras.optimizers import Adam
    
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # ─────────────────────────────────────────────────────────────────────────────
    # BƯỚC 1: CHUẨN BỊ FEATURES ĐA CHIỀU
    # ─────────────────────────────────────────────────────────────────────────────
    LOOKBACK = 10  # Số ngày nhìn lại (sequence length)
    
    df_lstm = df.copy()
    
    # Feature 1: Log Returns (stationary)
    df_lstm['Log_Ret'] = np.log(df_lstm['Close'] / df_lstm['Close'].shift(1))
    
    # Feature 2: RSI normalized (0-1) 
    if 'RSI' in df_lstm.columns:
        df_lstm['RSI_norm'] = df_lstm['RSI'] / 100.0
    else:
        df_lstm['RSI_norm'] = 0.5  # Neutral fallback
    
    # Feature 3: ATR normalized (% of price)
    if 'ATR' in df_lstm.columns:
        df_lstm['ATR_norm'] = df_lstm['ATR'] / df_lstm['Close']
    else:
        df_lstm['ATR_norm'] = 0.01  # 1% fallback
    
    # Feature 4: Volume change (normalized)
    if 'Volume' in df_lstm.columns:
        df_lstm['Vol_Change'] = df_lstm['Volume'].pct_change()
        df_lstm['Vol_Change'] = df_lstm['Vol_Change'].clip(-1, 1)  # Clip outliers
    else:
        df_lstm['Vol_Change'] = 0.0
    
    # Fill NaN
    df_lstm.fillna(0, inplace=True)
    
    # Features list
    feature_cols = ['Log_Ret', 'RSI_norm', 'ATR_norm', 'Vol_Change']
    
    # ─────────────────────────────────────────────────────────────────────────────
    # BƯỚC 2: TẠO SEQUENCES VỚI SLIDING WINDOW
    # ─────────────────────────────────────────────────────────────────────────────
    # Mỗi sample X gồm LOOKBACK ngày, target y là Log_Ret của ngày tiếp theo
    
    feature_data = df_lstm[feature_cols].values
    target_data = df_lstm['Log_Ret'].values
    
    X_sequences = []
    y_targets = []
    
    for i in range(LOOKBACK, len(df_lstm)):
        # X: 10 ngày trước (i-10 đến i-1)
        X_sequences.append(feature_data[i-LOOKBACK:i])
        # y: Log return của ngày i
        y_targets.append(target_data[i])
    
    X_all = np.array(X_sequences)  # Shape: (samples, 10, 4)
    y_all = np.array(y_targets)    # Shape: (samples,)
    
    # ─────────────────────────────────────────────────────────────────────────────
    # BƯỚC 3: CHIA TẬP TRAIN/TEST (Điều chỉnh theo lookback)
    # ─────────────────────────────────────────────────────────────────────────────
    # Số samples sau khi tạo sequences
    adjusted_train_size = train_size - LOOKBACK
    if adjusted_train_size < 50:
        adjusted_train_size = len(X_all) - (len(df) - train_size)
    
    X_train_raw = X_all[:adjusted_train_size]
    y_train = y_all[:adjusted_train_size]
    X_test_raw = X_all[adjusted_train_size:]
    y_test = y_all[adjusted_train_size:]
    
    # ─────────────────────────────────────────────────────────────────────────────
    # BƯỚC 4: CHUẨN HÓA DỮ LIỆU (FIT CHỈ TRÊN TRAINING)
    # ─────────────────────────────────────────────────────────────────────────────
    # Reshape để fit scaler: (samples * timesteps, features)
    n_train, n_timesteps, n_features = X_train_raw.shape
    
    scaler_X = MinMaxScaler(feature_range=(-1, 1))
    
    # Flatten, fit, reshape
    X_train_flat = X_train_raw.reshape(-1, n_features)
    scaler_X.fit(X_train_flat)
    
    X_train_scaled = scaler_X.transform(X_train_flat).reshape(n_train, n_timesteps, n_features)
    
    if len(X_test_raw) > 0:
        X_test_flat = X_test_raw.reshape(-1, n_features)
        X_test_scaled = scaler_X.transform(X_test_flat).reshape(len(X_test_raw), n_timesteps, n_features)
    else:
        X_test_scaled = np.array([]).reshape(0, n_timesteps, n_features)
    
    # ─────────────────────────────────────────────────────────────────────────────
    # BƯỚC 5: XÂY DỰNG MÔ HÌNH LSTM VỚI PROPER SEQUENCE ARCHITECTURE
    # ─────────────────────────────────────────────────────────────────────────────
    inputs = Input(shape=(LOOKBACK, n_features))
    
    # Layer 1: Bidirectional LSTM - học patterns từ cả 2 hướng
    lstm_out = Bidirectional(LSTM(64, return_sequences=True))(inputs)
    lstm_out = Dropout(0.2)(lstm_out)
    
    # Layer 2: LSTM để aggregate thông tin
    lstm_out2 = LSTM(32, return_sequences=False)(lstm_out)
    lstm_out2 = Dropout(0.2)(lstm_out2)
    
    # Dense layers
    dense1 = Dense(16, activation='relu')(lstm_out2)
    outputs = Dense(1, activation='tanh')(dense1)  # tanh vì log return thường trong (-1, 1)
    
    # Scale output để phù hợp với log return thực tế (thường < 0.1)
    from tensorflow.keras.layers import Lambda
    outputs = Lambda(lambda x: x * 0.1)(outputs)
    
    model = Model(inputs, outputs)
    
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    # ─────────────────────────────────────────────────────────────────────────────
    # BƯỚC 6: TRAINING VỚI CALLBACKS
    # ─────────────────────────────────────────────────────────────────────────────
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=0
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=0
        )
    ]
    
    model.fit(
        X_train_scaled, y_train,
        epochs=150,
        batch_size=32,
        validation_split=0.15,
        callbacks=callbacks,
        verbose=0
    )
    
    # ─────────────────────────────────────────────────────────────────────────────
    # BƯỚC 7: DỰ BÁO TRÊN TẬP TEST
    # ─────────────────────────────────────────────────────────────────────────────
    if len(X_test_scaled) > 0:
        pred_log_ret = model.predict(X_test_scaled, verbose=0).flatten()
        
        # Chuyển Log Returns → Giá
        # Lấy giá đóng cửa trước mỗi ngày test
        test_start_idx = train_size
        prev_close = df.iloc[test_start_idx-1:-1]['Close'].values
        min_len = min(len(prev_close), len(pred_log_ret))
        pred_prices = prev_close[:min_len] * np.exp(pred_log_ret[:min_len])
    else:
        pred_prices = np.array([df['Close'].iloc[-1]])
    
    # ─────────────────────────────────────────────────────────────────────────────
    # BƯỚC 8: ĐÓNG GÓI ĐỐI TƯỢNG
    # ─────────────────────────────────────────────────────────────────────────────
    lstm_objects = {
        'model': model,
        'scaler_X': scaler_X,
        'feature_cols': feature_cols,
        'lookback': LOOKBACK,
        'n_features': n_features
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
    ═══════════════════════════════════════════════════════════════════════════════
    DỰ BÁO TƯƠNG LAI BẰNG LSTM VỚI PROPER SEQUENCE LEARNING
    ───────────────────────────────────────────────────────────────────────────────
    [REFACTORED] Sử dụng lookback window thực sự để dự báo iterative
    
    Logic:
    1. Lấy 10 ngày cuối cùng làm sequence đầu tiên
    2. Dự báo Log Return ngày T+1
    3. Cập nhật sequence: bỏ ngày đầu, thêm ngày mới vào cuối
    4. Lặp lại cho đến khi đủ n_days
    ═══════════════════════════════════════════════════════════════════════════════
    """
    model = lstm_objects['model']
    scaler_X = lstm_objects['scaler_X']
    feature_cols = lstm_objects['feature_cols']
    LOOKBACK = lstm_objects['lookback']
    n_features = lstm_objects['n_features']
    
    # ─────────────────────────────────────────────────────────────────────────────
    # BƯỚC 1: CHUẨN BỊ DỮ LIỆU FEATURES
    # ─────────────────────────────────────────────────────────────────────────────
    df_lstm = df.copy()
    
    # Tính các features giống như training
    df_lstm['Log_Ret'] = np.log(df_lstm['Close'] / df_lstm['Close'].shift(1))
    
    if 'RSI' in df_lstm.columns:
        df_lstm['RSI_norm'] = df_lstm['RSI'] / 100.0
    else:
        df_lstm['RSI_norm'] = 0.5
    
    if 'ATR' in df_lstm.columns:
        df_lstm['ATR_norm'] = df_lstm['ATR'] / df_lstm['Close']
    else:
        df_lstm['ATR_norm'] = 0.01
    
    if 'Volume' in df_lstm.columns:
        df_lstm['Vol_Change'] = df_lstm['Volume'].pct_change().clip(-1, 1)
    else:
        df_lstm['Vol_Change'] = 0.0
    
    df_lstm.fillna(0, inplace=True)
    
    # ─────────────────────────────────────────────────────────────────────────────
    # BƯỚC 2: LẤY SEQUENCE CUỐI CÙNG (10 ngày gần nhất)
    # ─────────────────────────────────────────────────────────────────────────────
    feature_data = df_lstm[feature_cols].values
    current_sequence = feature_data[-LOOKBACK:].copy()  # Shape: (10, 4)
    
    # ─────────────────────────────────────────────────────────────────────────────
    # BƯỚC 3: DỰ BÁO ITERATIVE
    # ─────────────────────────────────────────────────────────────────────────────
    predictions = []
    last_close = df['Close'].iloc[-1]
    
    # Lấy giá trị cuối của các features không đổi (RSI, ATR sẽ mean-revert dần)
    last_rsi = df_lstm['RSI_norm'].iloc[-1]
    last_atr = df_lstm['ATR_norm'].iloc[-1]
    mean_rsi = df_lstm['RSI_norm'].mean()
    mean_atr = df_lstm['ATR_norm'].mean()
    
    for day in range(n_days):
        # Scale sequence
        seq_flat = current_sequence.reshape(-1, n_features)
        seq_scaled = scaler_X.transform(seq_flat).reshape(1, LOOKBACK, n_features)
        
        # Predict log return
        pred_log_ret = model.predict(seq_scaled, verbose=0)[0, 0]
        
        # Clipping để tránh giá trị cực đoan (±7% là limit hợp lý cho VN30)
        pred_log_ret = np.clip(pred_log_ret, -0.07, 0.07)
        
        # Convert to price
        new_close = last_close * np.exp(pred_log_ret)
        predictions.append(new_close)
        
        # ─────────────────────────────────────────────────────────────────────────
        # BƯỚC 4: CẬP NHẬT SEQUENCE CHO NGÀY TIẾP THEO
        # ─────────────────────────────────────────────────────────────────────────
        # Mean reversion cho RSI và ATR
        decay = 0.1  # Tốc độ mean reversion
        new_rsi = last_rsi + decay * (mean_rsi - last_rsi)
        new_atr = last_atr + decay * (mean_atr - last_atr)
        
        # Tạo feature vector mới
        new_features = np.array([
            pred_log_ret,   # Log return vừa dự báo
            new_rsi,        # RSI mean-reverted
            new_atr,        # ATR mean-reverted
            0.0             # Volume change = 0 (không biết trước)
        ])
        
        # Shift sequence: bỏ ngày đầu, thêm ngày mới vào cuối
        current_sequence = np.vstack([current_sequence[1:], new_features])
        
        # Update states
        last_close = new_close
        last_rsi = new_rsi
        last_atr = new_atr
    
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
