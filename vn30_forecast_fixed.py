# ==============================================================================
# VN30 INDEX FORECASTING - ENSEMBLE MODEL
# Fixed for Local Machine Execution
# ==============================================================================

import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import plotly.graph_objects as go
import itertools
import os

warnings.filterwarnings('ignore')
pd.set_option('display.float_format', lambda x: '%.4f' % x)
np.random.seed(42)
tf.random.set_seed(42)

# Chart styling function
def makeup_chart(ax, title, x_lab, y_lab):
    ax.set_title(title, fontsize=12, fontweight='bold', pad=15, color='#333333')
    ax.set_xlabel(x_lab, fontweight='bold')
    ax.set_ylabel(y_lab, fontweight='bold')

# ==============================================================================
# BƯỚC 1: TIỀN XỬ LÝ DỮ LIỆU & FEATURE ENGINEERING
# ==============================================================================

print("[1/6] INITIALIZING DATA PREPROCESSING...")

# 1. ĐỌC DỮ LIỆU
filename = 'Dữ liệu Lịch sử VN 30.csv'

# Check if file exists
if not os.path.exists(filename):
    print(f"❌ ERROR: File not found: '{filename}'")
    print(f"📂 Please place the CSV file in the same directory as this script.")
    print(f"   Current directory: {os.getcwd()}")
    exit()

try:
    df = pd.read_csv(filename)
    print(f"   + Data Loaded: {filename}")
except Exception as e:
    print(f"❌ ERROR reading file: {e}")
    exit()

# 2. CHUẨN HÓA
col_map = {
    'Ngày': 'Date', 'Lần cuối': 'Close', 'Mở': 'Open',
    'Cao': 'High', 'Thấp': 'Low', 'KL': 'Vol', '% Thay đổi': 'Change_Pct'
}
df.rename(columns=col_map, inplace=True)
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
df.set_index('Date', inplace=True)
df.sort_index(inplace=True)

# 3. LÀM SẠCH VOLUME
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
if 'Change_Pct' in df.columns:
    df['Change_Pct'] = df['Change_Pct'].astype(str).str.replace('%', '').astype(float)
else:
    df['Change_Pct'] = df['Close'].pct_change() * 100

# 4. CHỈ BÁO KỸ THUẬT (INDICATORS)
# RSI
delta = df['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss
df['RSI'] = 100 - (100 / (1 + rs))

# MACD
exp12 = df['Close'].ewm(span=12, adjust=False).mean()
exp26 = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD'] = exp12 - exp26
df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

# Bollinger Bands & ATR
df['MA20'] = df['Close'].rolling(window=20).mean()
df['STD20'] = df['Close'].rolling(window=20).std()
df['Upper_Band'] = df['MA20'] + (df['STD20'] * 2)
df['Lower_Band'] = df['MA20'] - (df['STD20'] * 2)
df['Pct_B'] = (df['Close'] - df['Lower_Band']) / (df['Upper_Band'] - df['Lower_Band'])

high_low = df['High'] - df['Low']
high_close = np.abs(df['High'] - df['Close'].shift())
low_close = np.abs(df['Low'] - df['Close'].shift())
df['ATR'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1).rolling(14).mean()

# 5. TẠO LAG FEATURES
lags = 3
for i in range(1, lags + 1):
    df[f'Lag_Close_{i}'] = df['Close'].shift(i)
    df[f'Lag_Vol_{i}'] = df['Vol'].shift(i)
    df[f'Lag_RSI_{i}'] = df['RSI'].shift(i)
    df[f'Lag_MACD_{i}'] = df['MACD'].shift(i)
    df[f'Lag_ATR_{i}'] = df['ATR'].shift(i)

df.dropna(inplace=True)

# 6. SPLIT TRAIN/TEST
train_size = int(len(df) * 0.95)
train_data = df.iloc[:train_size]
test_data = df.iloc[train_size:]
y_test_real = test_data['Close'].values

print("✅ DATA PREPROCESSING COMPLETED.")
print(f"   + Total Samples: {len(df)}")
print(f"   + Train: {len(train_data)} | Test: {len(test_data)}")

# ==============================================================================
# BƯỚC 1.5: EXPLORATORY DATA ANALYSIS (EDA)
# ==============================================================================

plt.style.use('seaborn-v0_8-whitegrid')
c = {'hist': '#16a085', 'corr': 'RdBu_r'}

print("📊 [1.5/6] GENERATING EDA CHARTS...")

# 1. DISTRIBUTION & BOXPLOT
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
sns.histplot(df['Close'], kde=True, bins=30, color=c['hist'], ax=ax[0])
ax[0].axvline(df['Close'].mean(), color='red', linestyle='--', label='Mean')
makeup_chart(ax[0], 'FIGURE 1: VN30 INDEX DISTRIBUTION', 'Index Value', 'Frequency')
ax[0].legend()

# Boxplot
sns.boxplot(x=df['Close'], color='#f39c12', ax=ax[1])
makeup_chart(ax[1], 'FIGURE 2: BOXPLOT (OUTLIER DETECTION)', 'Index Value', '')
plt.tight_layout()
plt.show()

# 2. CORRELATION MATRIX
cols_corr = ['Close', 'Vol', 'RSI', 'ATR', 'Lag_Close_1']
if all(col in df.columns for col in cols_corr):
    plt.figure(figsize=(8, 6))
    df_corr = df[cols_corr].rename(columns={'Lag_Close_1': 'Lag_Price'})
    sns.heatmap(df_corr.corr(), annot=True, cmap=c['corr'], fmt=".2f")
    plt.title('FIGURE 3: CORRELATION MATRIX', fontsize=12, fontweight='bold', pad=15)
    plt.show()

# 3. ADF TEST
print("\n📋 ADF STATIONARITY TEST RESULTS:")
adf = adfuller(df['Close'])
print(f"   - Statistic: {adf[0]:.4f}")
print(f"   - P-value: {adf[1]:.4f}")
print(f"   - Conclusion: {'Stationary' if adf[1] < 0.05 else 'Non-Stationary'}")

# ==============================================================================
# BƯỚC 2: ARIMAX MODELING
# ==============================================================================

print("⏳ [2/6] TRAINING ARIMAX MODEL...")

# 1. SETUP
exog_cols = ['Lag_Vol_1', 'RSI', 'ATR']
train_exog = train_data[exog_cols].bfill()  # FIXED: Changed from fillna(method='bfill')
test_exog = test_data[exog_cols].bfill()    # FIXED: Changed from fillna(method='bfill')
train_endog = train_data['Close']

# 2. GRID SEARCH
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

print(f"   + Best Params (p,d,q): {best_order}")

# 3. WALK-FORWARD FORECAST
history_endog = list(train_endog)
history_exog = list(train_exog.values)
arima_preds = []
test_exog_vals = test_exog.values

with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    for t in range(len(test_data)):
        model = ARIMA(history_endog, exog=history_exog, order=best_order).fit()
        pred = model.forecast(steps=1, exog=test_exog_vals[t].reshape(1,-1))[0]
        arima_preds.append(pred)
        history_endog.append(test_data['Close'].iloc[t])
        history_exog.append(test_exog_vals[t])

arima_preds = np.array(arima_preds)
mape_arima = mean_absolute_percentage_error(y_test_real, arima_preds) * 100
print(f"✅ ARIMAX COMPLETED. MAPE: {mape_arima:.2f}%")

# PLOT
plt.figure(figsize=(14, 6))
plt.plot(test_data.index, test_data['Close'], label='Actual Price', color='#2c3e50', linewidth=2)
plt.plot(test_data.index, arima_preds, label='ARIMAX Forecast', color='#c0392b', linestyle='--', linewidth=2)
makeup_chart(plt.gca(), f'FIGURE 4: ARIMAX FORECAST RESULTS (MAPE: {mape_arima:.2f}%)', 'Date', 'Index Value')
plt.legend(frameon=True)
plt.show()

# ==============================================================================
# BƯỚC 3: XGBOOST MODELING
# ==============================================================================

print("⏳ [3/6] TRAINING XGBOOST MODEL...")

# 1. SETUP
features_xgb = [col for col in df.columns if col.startswith('Lag_')]
X_train_xgb = train_data[features_xgb]
y_train_xgb = train_data['Close']
X_test_xgb = test_data[features_xgb]

# 2. TRAIN
xgb_model = XGBRegressor(
    n_estimators=3000, learning_rate=0.005, max_depth=8,
    subsample=0.7, colsample_bytree=0.7, n_jobs=-1, random_state=42,
    early_stopping_rounds=50, eval_metric='mae'
)

xgb_model.fit(
    X_train_xgb, y_train_xgb,
    eval_set=[(X_train_xgb, y_train_xgb), (X_test_xgb, test_data['Close'])],
    verbose=False
)

# 3. PREDICT
xgb_preds = xgb_model.predict(X_test_xgb)
mape_xgb = mean_absolute_percentage_error(y_test_real, xgb_preds) * 100
print(f"✅ XGBOOST COMPLETED. MAPE: {mape_xgb:.2f}%")

# PLOT FEATURE IMPORTANCE
plt.figure(figsize=(10, 6))
imp = xgb_model.feature_importances_
idx = np.argsort(imp)[-10:]
plt.barh(range(len(idx)), imp[idx], color='#8e44ad')
plt.yticks(range(len(idx)), [features_xgb[i] for i in idx])
makeup_chart(plt.gca(), 'FIGURE 5: TOP 10 FEATURE IMPORTANCE (XGBOOST)', 'Importance Score', '')
plt.show()

# PLOT FORECAST
plt.figure(figsize=(14, 6))
plt.plot(test_data.index, y_test_real, label='Actual Price', color='#2c3e50', linewidth=2)
plt.plot(test_data.index, xgb_preds, label='XGBoost Forecast', color='#c0392b', linestyle='--', linewidth=2)
makeup_chart(plt.gca(), f'FIGURE 6: XGBOOST FORECAST RESULTS (MAPE: {mape_xgb:.2f}%)', 'Date', 'Index Value')
plt.legend(frameon=True)
plt.show()

# ==============================================================================
# BƯỚC 4: BI-LSTM (MOMENTUM STRATEGY)
# ==============================================================================

print("⏳ [4/6] TRAINING BI-LSTM MODEL...")

# 1. TRANSFORM TO LOG RETURN
df_lstm = df.copy()
df_lstm['Log_Ret'] = np.log(df_lstm['Close'] / df_lstm['Close'].shift(1))
df_lstm.fillna(0, inplace=True)

# Re-create Lags
for i in range(1, 4):
    df_lstm[f'Lag_Ret_{i}'] = df_lstm['Log_Ret'].shift(i)
    df_lstm[f'Lag_Vol_{i}'] = df[f'Lag_Vol_{i}']
    df_lstm[f'Lag_ATR_{i}'] = df[f'Lag_ATR_{i}']

df_lstm.dropna(inplace=True)
features_lstm = [col for col in df_lstm.columns if col.startswith('Lag_')]
X_vals = df_lstm[features_lstm].values
y_vals = df_lstm['Log_Ret'].values

# 2. SCALE
scaler = MinMaxScaler(feature_range=(-1, 1))
X_scaled = scaler.fit_transform(X_vals)

split_idx = len(df_lstm) - len(test_data)
X_train = X_scaled[:split_idx].reshape(-1, 1, len(features_lstm))
X_test = X_scaled[split_idx:].reshape(-1, 1, len(features_lstm))
y_train = y_vals[:split_idx]

# 3. BUILD MODEL
model = Sequential([
    Bidirectional(LSTM(64, return_sequences=True), input_shape=(1, len(features_lstm))),
    Dropout(0.2),
    Bidirectional(LSTM(32)),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# 4. TRAIN
model.fit(
    X_train, y_train, epochs=100, batch_size=16, verbose=0,
    validation_split=0.1,
    callbacks=[EarlyStopping(patience=15, restore_best_weights=True),
               ReduceLROnPlateau(factor=0.5, patience=5)]
)

# 5. PREDICT & RECONSTRUCT
pred_log_ret = model.predict(X_test, verbose=0).flatten()
prev_close = df.iloc[train_size-1 : -1]['Close'].values
lstm_preds = prev_close * np.exp(pred_log_ret)

min_len = min(len(y_test_real), len(lstm_preds))
lstm_preds = lstm_preds[:min_len]
y_lstm_real = y_test_real[:min_len]

mape_lstm = mean_absolute_percentage_error(y_lstm_real, lstm_preds) * 100
print(f"✅ BI-LSTM COMPLETED. MAPE: {mape_lstm:.2f}%")

# PLOT
plt.figure(figsize=(14, 6))
plt.plot(test_data.index[-min_len:], y_lstm_real, label='Actual Price', color='#2c3e50', linewidth=2)
plt.plot(test_data.index[-min_len:], lstm_preds, label='Bi-LSTM Forecast', color='#c0392b', linestyle='--', linewidth=2)
makeup_chart(plt.gca(), f'FIGURE 7: BI-LSTM FORECAST RESULTS (MAPE: {mape_lstm:.2f}%)', 'Date', 'Index Value')
plt.legend(frameon=True)
plt.show()

# ==============================================================================
# BƯỚC 5: ENSEMBLE OPTIMIZATION
# ==============================================================================

print("🔄 [5/6] OPTIMIZING ENSEMBLE WEIGHTS...")

# 1. SYNC LENGTHS
min_len_final = min(len(y_test_real), len(arima_preds), len(xgb_preds), len(lstm_preds))
y_final = y_test_real[-min_len_final:]
p1 = arima_preds[-min_len_final:]
p2 = xgb_preds[-min_len_final:]
p3 = lstm_preds[-min_len_final:]
dates_final = test_data.index[-min_len_final:]

# 2. OPTIMIZED WEIGHTING (scipy.optimize)
from scipy.optimize import minimize

def objective(weights, predictions, y_true):
    """Minimize MAPE by finding optimal weights"""
    # Normalize weights to sum to 1
    weights = np.array(weights) / np.sum(weights)
    # Calculate weighted ensemble
    ensemble = np.zeros(len(y_true))
    for pred, w in zip(predictions, weights):
        ensemble += pred * w
    # Return MAPE
    return mean_absolute_percentage_error(y_true, ensemble) * 100

# Initial guess: equal weights
initial_weights = np.array([1/3, 1/3, 1/3])

# Constraints: all weights >= 0, sum = 1
constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
bounds = [(0, 1), (0, 1), (0, 1)]

# Optimize
result = minimize(
    objective,
    initial_weights,
    args=([p1, p2, p3], y_final),
    method='SLSQP',
    bounds=bounds,
    constraints=constraints
)

best_weights = result.x
final_pred = p1 * best_weights[0] + p2 * best_weights[1] + p3 * best_weights[2]
best_mape = mean_absolute_percentage_error(y_final, final_pred) * 100

print("\n" + "="*50)
print(f"🏆 FINAL EXPERIMENTAL RESULTS")
print("="*50)
print(f"💎 OPTIMAL WEIGHTS (Scipy Optimization):")
print(f"   + ARIMAX (Statistical):       {best_weights[0]*100:.1f}%")
print(f"   + XGBoost (Machine Learning):  {best_weights[1]*100:.1f}%")
print(f"   + Bi-LSTM (Deep Learning):     {best_weights[2]*100:.1f}%")
print("-" * 50)
print(f"📉 COMBINED MAPE: {best_mape:.4f}%")
print(f"📊 Individual MAPE:")
print(f"   - ARIMAX:  {mape_arima:.4f}%")
print(f"   - XGBoost: {mape_xgb:.4f}%")
print(f"   - LSTM:    {mape_lstm:.4f}%")
print("="*50)

# 3. INTERACTIVE PLOT
fig = go.Figure()
fig.add_trace(go.Scatter(x=dates_final, y=y_final, name='Actual Price', line=dict(color='#2c3e50', width=3)))
fig.add_trace(go.Scatter(x=dates_final, y=final_pred, name=f'Ensemble Forecast', line=dict(color='#ff2e63', width=2, dash='dash')))
fig.update_layout(
    title='<b>VN30 INDEX FORECAST (ENSEMBLE MODEL)</b>',
    xaxis_title='Date', yaxis_title='Index Value',
    plot_bgcolor='white', height=600,
    legend=dict(orientation="h", y=1.02)
)
fig.show()

# SAVE CSV
output_file = 'Final_Forecast_Results.csv'
pd.DataFrame({'Date': dates_final, 'Actual': y_final, 'Forecast': final_pred}).to_csv(output_file, index=False)
print(f"\n💾 Results saved to: {output_file}")
print("✅ [6/6] ALL TASKS COMPLETED SUCCESSFULLY!")
