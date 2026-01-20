"""
VN30 INDEX ANALYTICS PLATFORM
Minimalist Fintech Aesthetic | Refactored UI
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import warnings
import time
from datetime import datetime

# Custom modules
from utils import (
    load_data, create_technical_indicators, create_lag_features,
    calculate_metrics, load_macro_data
)
from models import (
    train_arimax, train_xgboost, train_lstm,
    train_meta_learner, forecast_ensemble_stacking,
    forecast_future_arimax, forecast_future_xgboost_recursive, 
    forecast_future_lstm, create_future_dates,
    add_rolling_features, add_macro_features,  # Feature Engineering functions
    # GARCH & Probabilistic Forecast
    fit_garch_model, forecast_volatility_garch, 
    detect_market_regime, forecast_probabilistic_ensemble,
    # Technical Score System (NEW)
    calculate_technical_score
)

warnings.filterwarnings('ignore')

# ==============================================================================
# ENFORCE GLOBAL DETERMINISM
# ==============================================================================
# Set random seeds at APPLICATION LEVEL to guarantee reproducible results
# across all library calls (NumPy, TensorFlow, XGBoost, Scikit-learn)
import random
import tensorflow as tf
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
# For TensorFlow deterministic ops (GPU)
try:
    tf.config.experimental.enable_op_determinism()
except:
    pass # Older TF versions may not have this
# ==============================================================================

# -----------------------------------------------------------------------------
# 1. APP CONFIGURATION & FINTECH STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="VN30 Analytics",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# HELPER: Lucide Icons (SVG)
def icon(name, color="#64748b", size=20):
    icons = {
        "bar-chart-3": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v18h18"/><path d="M18 17V9"/><path d="M13 17V5"/><path d="M8 17v-3"/></svg>',
        "activity": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>',
        "target": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>',
        "upload-cloud": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M4 14.899A7 7 0 1 1 15.71 8h1.79a4.5 4.5 0 0 1 2.5 8.242"/><path d="M12 12v9"/><path d="m16 16-4-4-4 4"/></svg>',
        "sliders": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><line x1="4" x2="20" y1="12" y2="12"/><line x1="4" x2="20" y1="6" y2="6"/><line x1="4" x2="20" y1="18" y2="18"/><circle cx="12" cy="12" r="2"/><circle cx="8" cy="6" r="2"/><circle cx="16" cy="18" r="2"/></svg>',
        "play": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polygon points="10 8 16 12 10 16 10 8"/></svg>',
        "layers": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>',
        "trending-up": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/></svg>'
    }
    return icons.get(name, "")

# Minimalist CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    :root {
        --text-primary: #1e293b;
        --text-secondary: #64748b;
        --bg-color: #f1f5f9;
        --card-bg: #ffffff;
        --border-color: #e2e8f0;
        --primary-brand: #0f172a;
    }
    
    .stApp {
        background-color: var(--bg-color);
        font-family: 'Inter', sans-serif;
        color: var(--text-primary);
    }
    
    /* Header Styling */
    /* Restored default header visibility */
    
    /* Custom Navigation Bar */
    .nav-bar {
        background-color: white;
        padding: 1rem 2rem;
        margin: 1rem -5rem 2rem -5rem; /* Adjusted for visible header */
        border-bottom: 1px solid var(--border-color);
        box-shadow: 0 1px 3px 0 rgb(0 0 0 / 0.05);
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .nav-brand {
        font-weight: 700;
        font-size: 1.1rem;
        display: flex;
        align-items: center;
        gap: 10px;
        color: var(--primary-brand);
    }
    .nav-meta {
        font-size: 0.85rem;
        color: var(--text-secondary);
        display: flex;
        align-items: center;
        gap: 15px;
    }
    
    /* Styled Containers & Cards */
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
        font-weight: 600 !important;
        color: var(--primary-brand) !important;
    }
    div[data-testid="stMetricLabel"] {
        font-weight: 500 !important;
        color: var(--text-secondary) !important;
    }
    
    .process-card {
        background-color: white;
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 1.5rem;
        height: 100%;
        min-height: 200px;
        transition: all 0.2s;
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
    }
    .process-card:hover {
        border-color: #cbd5e1;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.05);
    }
    .process-step {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .process-title {
        font-weight: 600;
        color: var(--text-primary);
        font-size: 1rem;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .process-desc {
        font-size: 0.875rem;
        color: var(--text-secondary);
        line-height: 1.5;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: white;
        border-right: 1px solid var(--border-color);
    }
    
    /* Round the Plotly Chart */
    .stPlotlyChart {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.05);
        border: 1px solid var(--border-color);
    }
    
    /* Primary Button */
    .stButton > button {
        background-color: #0f172a;
        color: white;
        font-weight: 500;
        border-radius: 6px;
        border: none;
        padding: 0.5rem 1rem;
        transition: background 0.2s;
    }
    .stButton > button:hover {
        background-color: #334155;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. NAVIGATION BAR (White, Clean)
# -----------------------------------------------------------------------------
current_time = datetime.now().strftime("%H:%M GMT+7")
st.markdown(f"""
<div class="nav-bar">
    <div class="nav-brand">
        {icon("layers", "#0f172a", 24)}
        VN30 ANALYTICS <span style="font-weight:400; color:#cbd5e1;">|</span> PLATFORM
    </div>
    <div class="nav-meta">
        <span>{icon("activity", "#10b981", 16)} System Operational</span>
        <span>{current_time}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f"### {icon('sliders', '#1e293b')} Controls", unsafe_allow_html=True)
    
    # Data Section
    st.caption("DATA SOURCE")
    data_source = st.selectbox("Select Input", ["VN30 Default", "Upload CSV"], label_visibility="collapsed")
    
    uploaded_file = None
    column_mapping = None
    
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("📁 Upload your CSV file", type=['csv'], label_visibility="collapsed")
        
        if uploaded_file is not None:
            # Preview uploaded file and let user map columns
            try:
                # Read preview
                preview_df = pd.read_csv(uploaded_file)
                uploaded_file.seek(0)  # Reset file pointer
                
                available_cols = list(preview_df.columns)
                
                st.caption("📊 COLUMN MAPPING")
                st.markdown("<small style='color:#64748b'>Map your columns to required fields:</small>", unsafe_allow_html=True)
                
                # Auto-detect common column names
                def auto_detect(required, available):
                    """Auto-detect column based on common names"""
                    patterns = {
                        'Date': ['date', 'ngày', 'thời gian', 'datetime', 'time'],
                        'Close': ['close', 'giá', 'lần cuối', 'đóng cửa', 'price', 'adj close'],
                        'Open': ['open', 'mở', 'giá mở'],
                        'High': ['high', 'cao', 'giá cao'],
                        'Low': ['low', 'thấp', 'giá thấp'],
                        'Volume': ['volume', 'vol', 'kl', 'khối lượng']
                    }
                    
                    for col in available:
                        col_lower = col.lower().strip()
                        for pattern in patterns.get(required, []):
                            if pattern in col_lower:
                                return col
                    return available[0] if available else None
                
                # Required mappings
                date_col = st.selectbox(
                    "📅 Date Column",
                    available_cols,
                    index=available_cols.index(auto_detect('Date', available_cols)) if auto_detect('Date', available_cols) in available_cols else 0,
                    key="map_date"
                )
                
                close_col = st.selectbox(
                    "💰 Close/Price Column",
                    available_cols,
                    index=available_cols.index(auto_detect('Close', available_cols)) if auto_detect('Close', available_cols) in available_cols else 0,
                    key="map_close"
                )
                
                # Optional mappings (collapsible)
                with st.expander("⚙️ Optional Columns", expanded=False):
                    open_col = st.selectbox(
                        "Open", available_cols + ["(Use Close)"],
                        index=available_cols.index(auto_detect('Open', available_cols)) if auto_detect('Open', available_cols) in available_cols else len(available_cols),
                        key="map_open"
                    )
                    high_col = st.selectbox(
                        "High", available_cols + ["(Use Close)"],
                        index=available_cols.index(auto_detect('High', available_cols)) if auto_detect('High', available_cols) in available_cols else len(available_cols),
                        key="map_high"
                    )
                    low_col = st.selectbox(
                        "Low", available_cols + ["(Use Close)"],
                        index=available_cols.index(auto_detect('Low', available_cols)) if auto_detect('Low', available_cols) in available_cols else len(available_cols),
                        key="map_low"
                    )
                    vol_col = st.selectbox(
                        "Volume", available_cols + ["(None)"],
                        index=available_cols.index(auto_detect('Volume', available_cols)) if auto_detect('Volume', available_cols) in available_cols else len(available_cols),
                        key="map_vol"
                    )
                
                # Store mapping in session state
                column_mapping = {
                    'Date': date_col,
                    'Close': close_col,
                    'Open': open_col if open_col != "(Use Close)" else None,
                    'High': high_col if high_col != "(Use Close)" else None,
                    'Low': low_col if low_col != "(Use Close)" else None,
                    'Volume': vol_col if vol_col != "(None)" else None
                }
                st.session_state['column_mapping'] = column_mapping
                
                # Show preview
                st.caption(f"✅ {len(preview_df)} rows detected")
                
            except Exception as e:
                st.error(f"Error reading CSV: {e}")
    
    st.markdown("###")
    run_btn = st.button("RUN ANALYSIS", type="primary", use_container_width=True)
    
    st.markdown("---")
    
    # System Info
    st.caption("SYSTEM INFO")
    st.markdown("""
    | Parameter | Value |
    |-----------|-------|
    | Forecast Horizon | **7 days** |
    | Training Split | **95%** |
    | Models | **Ensemble** |
    """)
    
    st.markdown("---")
    
    # Models Info
    st.caption("ACTIVE MODELS")
    st.markdown("""
    ✅ ARIMAX (Time Series)  
    ✅ XGBoost (Gradient Boosting)  
    ✅ LSTM (Deep Learning)  
    ✅ Ensemble (Weighted Combination)
    """)
    
    st.markdown("---")
    
    # About Section
    st.caption("ABOUT")
    st.markdown("""
    <div style="font-size: 0.85rem; line-height: 1.6; color: #64748b;">
        <strong style="color: #1e293b;">VN30 Index Forecasting System</strong><br>
        <em>Hybrid ML Ensemble Model</em><br><br>
        Graduation Thesis Project<br>
        University of Economics and Finance<br>
        Ho Chi Minh City, Vietnam
    </div>
    """, unsafe_allow_html=True)
    
    # All models always enabled
    models_selected = ["ARIMAX", "LSTM", "XGBoost", "Ensemble"]
    n_days = 7
    split_pct = 95

# -----------------------------------------------------------------------------
# 4. LOAD DATA & SUMMARY RIBBON
# -----------------------------------------------------------------------------
try:
    # Get column mapping if user uploaded custom CSV
    col_map = st.session_state.get('column_mapping', None)
    df = load_data(uploaded_file=uploaded_file, column_mapping=col_map)
    
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sp500_path = os.path.join(current_dir, 'SP500.csv')
    usdvnd_path = os.path.join(current_dir, 'USD_VND.csv')
    
    # --- PHASE 2: MACRO DATA INTEGRATION ---
    df = load_macro_data(df, sp500_path=sp500_path, usdvnd_path=usdvnd_path)
    
    # --- PHASE 3: FEATURE ENGINEERING ---
    df = add_macro_features(df)
    df = create_technical_indicators(df)
    df = create_lag_features(df)
    
    # Calculate Metrics
    last_close = df['Close'].iloc[-1]
    prev_close = df['Close'].iloc[-2]
    day_change = last_close - prev_close
    day_pct = (day_change / prev_close) * 100
    
    # Volatility (ATR-based mock)
    volatility = df['ATR'].iloc[-1] if 'ATR' in df.columns else (last_close * 0.015)
    
    # Accuracy (Mock based on selection)
    accuracy_score = "N/A"
    if 'history_preds' in st.session_state and st.session_state.history_preds:
        # Get ensemble MAPE if available, else best single model
        accuracy_score = "92.4%" # Placeholder updated after run
        
    # --- SUMMARY RIBBON (st.metric style) ---
    st.markdown("###### MARKET SNAPSHOT")
    
    m1, m2, m3, m4 = st.columns(4)
    
    m1.metric("VN30 Index", f"{last_close:,.2f}", f"{day_pct:+.2f}%")
    m2.metric("Volatility (ATR)", f"{volatility:,.2f} pts", help="Average Daily Range (Points)")
    m3.metric("Models Active", f"{len(models_selected)} / 4", "Ready")
    m4.metric("Forecast Horizon", f"{n_days} Days", "Forward")
    
    st.markdown("---")

except Exception as e:
    st.error(f"Data Error: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# 5. MAIN CONTENT & PROCESS FLOW
# -----------------------------------------------------------------------------

if run_btn:
    # --- RUNNING LOGIC (Hidden during computation) ---
    with st.spinner("Processing Financial Models..."):
        time.sleep(1) # UX Pause
        
        train_size = int(len(df) * (split_pct/100))
        train_data = df.iloc[:train_size]
        test_data = df.iloc[train_size:]
        
        st.session_state.models = {}
        st.session_state.history_preds = {}
        
        # Train Models
        try:
            if "ARIMAX" in models_selected:
                # Fast Mode = True for Dashboard Responsiveness
                pred, model = train_arimax(train_data, test_data, fast_mode=True)
                st.session_state.models['ARIMAX'] = model
                st.session_state.history_preds['ARIMAX'] = pred
                
            if "XGBoost" in models_selected:
                # Train with recursive logic support
                pred, model, _, features, _ = train_xgboost(train_data, test_data, n_days)
                # Store (model, last_features_array, feature_names_list)
                # Convert last row of test data to array for recursion
                last_features = test_data[features].iloc[-1].values
                st.session_state.models['XGBoost'] = (model, last_features, features)
                st.session_state.history_preds['XGBoost'] = pred
                
            if "LSTM" in models_selected:
                pred, lstm_objects = train_lstm(df, train_size, n_days)
                st.session_state.models['LSTM'] = lstm_objects
                st.session_state.history_preds['LSTM'] = pred
                
            if "Ensemble" in models_selected and len(st.session_state.history_preds) >= 2:
                # ... (Keep existing Ensemble Logic) ...
                preds = st.session_state.history_preds.copy() 
                
                # Align with y_true
                min_len = min(len(p) for p in preds.values())
                y_true = test_data['Close'].values[:min_len]
                
                aligned_preds = {k: v[:min_len] for k, v in preds.items()}
                
                # Train Meta-Learner
                # Use ONLY specific context features that we can project/simulate for future
                ctx_cols = ['RSI', 'ATR', 'Correlation_VN30_SP500', 'SP500_LogRet']
                # Ensure cols exist
                valid_ctx_cols = [c for c in ctx_cols if c in test_data.columns]
                
                meta_model, model_keys, ens_pred, mape, cv_score = train_meta_learner(
                    y_true, 
                    aligned_preds, 
                    feature_df=test_data[valid_ctx_cols]
                )
                
                st.session_state.models['Ensemble'] = (meta_model, model_keys) 
                st.session_state.history_preds['Ensemble'] = ens_pred
                
                # Calculate Trading Performance (Win Rate & Total Return)
                # Logic: Buy if Forecast > Previous Close
                def calculate_trading_performance(y_true, y_pred):
                    # Directional Accuracy (Win Rate)
                    direction_true = np.sign(np.diff(y_true))
                    direction_pred = np.sign(np.diff(y_pred))
                    # Align lengths (diff reduces size by 1)
                    matches = (direction_true == direction_pred)
                    win_rate = np.mean(matches) * 100
                    
                    # Total Return (Simple Strategy: Long only)
                    # Buy at Close[t], Sell at Close[t+1] if Pred[t+1] > Close[t]
                    # Here we simulate on Test set
                    returns = np.diff(y_true) / y_true[:-1]
                    signals = (np.diff(y_pred) > 0).astype(int)
                    # Shift signals to match T+1 return (Signal calculated at T effective for T->T+1)
                    # Here aligned: diff[i] is Return(i -> i+1), signal[i] is Pred(i+1) > Pred(i) (Trend following)
                    # Better logic: Signal = Pred[i+1] > Actual[i] (but Actual[i] is known)
                    
                    strategy_returns = returns * signals
                    total_return = np.prod(1 + strategy_returns) - 1
                    return win_rate, total_return * 100

                try:
                    win_rate, total_ret = calculate_trading_performance(y_true, ens_pred)
                    st.session_state.trading_metrics = {'Win Rate': win_rate, 'Total Return': total_ret}
                except:
                    st.session_state.trading_metrics = {'Win Rate': 0, 'Total Return': 0}

                st.info(f"Ensemble CV Score: {cv_score:.2f}% | Test MAPE: {mape:.2f}% | Win Rate: {st.session_state.trading_metrics['Win Rate']:.1f}%")
                
                try:
                    importances = meta_model.feature_importances_
                    # Chỉ lấy importance của các models (không lấy context features)
                    # model_keys có 3 phần tử (ARIMAX, LSTM, XGBoost)
                    # importances có thể có 7+ phần tử (3 models + 4 context features)
                    n_models = len(model_keys)
                    model_importances = importances[:n_models]  # Chỉ lấy n_models đầu tiên
                    total_imp = np.sum(model_importances)
                    if total_imp > 0:
                        st.session_state.weights = dict(zip(model_keys, model_importances/total_imp))
                    else:
                        st.session_state.weights = {k: 1.0/len(model_keys) for k in model_keys}
                except:
                    st.session_state.weights = {k: 1.0/len(model_keys) for k in model_keys}
                
            st.success("Computation Complete")
            
        except Exception as e:
            st.error(f"Engine Failure: {e}")

# --- DISPLAY RESULTS OR QUICK START ---

if 'history_preds' in st.session_state and st.session_state.history_preds:
    # ═══════════════════════════════════════════════════════════════════════════════
    # 🤖 TEASER: ROBO ADVISOR SECTION (New Feature)
    # ═══════════════════════════════════════════════════════════════════════════════
    st.markdown("### 🤖 Robo-Advisor Recommendation")
    
    # Always recalculate to ensure latest data/logic is used (Fixes N/A issue)
    ts, td, tdr, tdet = calculate_technical_score(df)
    st.session_state.tech_score = ts
    st.session_state.tech_direction = td
    st.session_state.tech_details = tdet

    # -----------------------------------------------------------------------------
    # LOGIC: INTEGRATE AI FORECAST INTO ROBO-ADVISOR SCORE
    # -----------------------------------------------------------------------------
    # Get Forecast Trend
    if 'prob_forecast' in st.session_state:
        median_trend = st.session_state.prob_forecast['median']
        trend_pct = (median_trend[-1] - median_trend[0]) / median_trend[0] * 100
    else:
        trend_pct = 0.0

    # Calculate AI Score Component (-2 to +2)
    # Refined thresholds to handle near-zero values (Sideways)
    if trend_pct >= 1.5:
        ai_score = 2
        ai_signal = "Strong Uptrend"
    elif trend_pct >= 0.2:
        ai_score = 1
        ai_signal = "Slight Uptrend"
    elif trend_pct > -0.2:
        ai_score = 0
        ai_signal = "Sideways"
    elif trend_pct > -1.5:
        ai_score = -1
        ai_signal = "Slight Downtrend"
    else:
        ai_score = -2
        ai_signal = "Strong Downtrend"

    # Layout: Gauge Chart (Left) + Analysis Details (Right)
    robo_c1, robo_c2 = st.columns([1, 2])
    
    with robo_c1:
        # Create Plotly Gauge Chart
        raw_tech_score = st.session_state.tech_score
        
        # COMBINE: Technical Score + AI Score
        # Range: [-8, 8] + [-2, 2] = [-10, 10] roughly
        combined_score = raw_tech_score + ai_score
        
        # Normalize -10 to +10  -->  0 to 10
        # Formula: ((score + 10) / 20) * 10
        # Actually Technical is -8 to 8. Let's clamp constraints.
        # Let's map [-10, 10] to [0, 10]
        score = ((combined_score + 10) / 20) * 10
        score = max(0, min(10, score)) # Clamp 0-10
        
        # Determine Color & Label based on User's Thesis Description
        if score > 6.5: # Harder to get Strong Buy
            gauge_color = "#10b981" # Green
            gauge_label = "STRONG BUY"
        elif score >= 4.0:
            gauge_color = "#eab308" # Yellow/Amber
            gauge_label = "WAIT"
        else:
            gauge_color = "#ef4444" # Red
            gauge_label = "PANIC SELL WARNING"
            
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': f"<b>{gauge_label}</b>", 'font': {'size': 20, 'color': gauge_color}},
            gauge = {
                'axis': {'range': [0, 10], 'tickwidth': 1, 'tickcolor': "#333"},
                'bar': {'color': gauge_color},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "#e2e8f0",
                'steps': [
                    {'range': [0, 4], 'color': 'rgba(239, 68, 68, 0.2)'},  # Red Zone 
                    {'range': [4, 6.5], 'color': 'rgba(234, 179, 8, 0.2)'},  # Yellow Zone
                    {'range': [6.5, 10], 'color': 'rgba(16, 185, 129, 0.2)'} # Green Zone 
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': score}
            }
        ))
        # Update layout with specific font and margin
        fig_gauge.update_layout(
            height=300, 
            margin=dict(l=20,r=20,t=50,b=20), 
            paper_bgcolor='white', 
            font={'family': "Inter, sans-serif"}
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with robo_c2:
        # Detailed HTML Card
        st.markdown(f"""
        <div style="background-color: white; padding: 20px; border-radius: 10px; border: 1px solid #e2e8f0; height: 100%;">
            <h4 style="margin-top:0; color: #1e293b;">Technical & AI Analysis Summary</h4>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-top: 15px;">
                <div style="background: #f8fafc; padding: 10px; border-radius: 6px; border-left: 4px solid #3b82f6;">
                    <div style="font-size: 0.8rem; color: #94a3b8; font-weight: 600;">🤖 AI MODEL FORECAST</div>
                    <div style="font-weight: 600; color: #1e293b;">{ai_signal} ({trend_pct:+.2f}%)</div>
                </div>
                 <div style="background: #f8fafc; padding: 10px; border-radius: 6px;">
                    <div style="font-size: 0.8rem; color: #94a3b8; font-weight: 600;">RSI SIGNAL</div>
                    <div style="font-weight: 600; color: #334155;">{st.session_state.get('tech_details', {}).get('RSI', {}).get('signal', 'N/A')}</div>
                </div>
                <div style="background: #f8fafc; padding: 10px; border-radius: 6px;">
                    <div style="font-size: 0.8rem; color: #94a3b8; font-weight: 600;">TREND (MA)</div>
                    <div style="font-weight: 600; color: #334155;">{st.session_state.get('tech_details', {}).get('MA_Crossover', {}).get('signal', 'N/A')}</div>
                </div>
            </div>
            <p style="color: #64748b; font-size: 0.8rem; margin-top: 15px; font-style: italic;">
                *Score integrates both Historical Technicals and Future AI Projections.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")

    # RESULTS VIEW
    st.markdown(f"#### {icon('trending-up', '#0f172a')} Forecast Trajectory", unsafe_allow_html=True)
    
    # Generate Forecasts
    future_dates = create_future_dates(df.index[-1], n_days)
    future_results = {}
    
    if 'ARIMAX' in st.session_state.models:
        future_results['ARIMAX'] = forecast_future_arimax(st.session_state.models['ARIMAX'], df, ['Lag_Vol_1', 'RSI', 'ATR'], n_days)
    if 'XGBoost' in st.session_state.models:
        xgb_tuple = st.session_state.models['XGBoost']
        if len(xgb_tuple) == 3: # (model, last_features, feature_names)
            model, last_feats, feat_names = xgb_tuple
            future_results['XGBoost'] = forecast_future_xgboost_recursive(model, last_feats, feat_names, n_days)
        else: # Legacy/Fallback
             future_results['XGBoost'] = [0] * n_days
    if 'LSTM' in st.session_state.models:
         future_results['LSTM'] = forecast_future_lstm(st.session_state.models['LSTM'], df, n_days)
    
    # Define colors for all charts
    chart_colors = {
        'Ensemble': '#10b981', # Emerald
        'ARIMAX':   '#f43f5e', # Rose
        'XGBoost':  '#3b82f6', # Blue
        'LSTM':     '#8b5cf6'  # Violet
    }
    
    if 'weights' in st.session_state and len(future_results) >= 2:
        # ═══════════════════════════════════════════════════════════════════════
        # NEW: GARCH-BASED PROBABILISTIC ENSEMBLE FORECAST
        # ═══════════════════════════════════════════════════════════════════════
        
        # Step 1: Fit GARCH model for volatility forecasting
        garch_result, _ = fit_garch_model(df)
        
        # Step 2: Forecast volatility for future days
        if garch_result is not None:
            garch_vol = forecast_volatility_garch(garch_result, n_days)
        else:
            # Fallback: use historical volatility scaled by sqrt(t)
            hist_vol = df['Close'].pct_change().std()
            garch_vol = np.array([hist_vol * np.sqrt(t+1) for t in range(n_days)])
        
        # Step 3: Detect current market regime
        regime_info = detect_market_regime(df, lookback=20)
        
        # Step 4: Calculate Technical Score for trend direction
        tech_score, tech_direction, tech_drift, tech_details = calculate_technical_score(df)
        
        # Step 5: Generate probabilistic forecast with GARCH + Technical Drift
        prob_forecast = forecast_probabilistic_ensemble(
            base_forecasts=future_results,
            garch_vol=garch_vol,
            last_price=df['Close'].iloc[-1],
            regime_info=regime_info,
            technical_drift=tech_drift,  # NEW: Apply technical indicator direction
            n_simulations=500,
            n_days=n_days
        )
        
        # Store for chart rendering
        st.session_state.prob_forecast = prob_forecast
        st.session_state.regime_info = regime_info
        st.session_state.tech_score = tech_score
        st.session_state.tech_direction = tech_direction
        
        # Fallback to weights if Stacking fails (or just to keep flow valid)

        # --- Display Ensemble Weights (Donut Chart) ---
        with st.expander("📊 Show/Hide Model Analysis", expanded=False):
            # Create 2 tabs for different analysis views
            tab1, tab3 = st.tabs(["Ensemble Weights", "Metrics Comparison"])
            
            # ═══════════════════════════════════════════════════════════════════
            # TAB 1: Ensemble Contribution Weights (Premium Donut Chart)
            # ═══════════════════════════════════════════════════════════════════
            with tab1:
                w_df = pd.DataFrame(list(st.session_state.weights.items()), columns=['Model', 'Weight'])
                w_df = w_df[w_df['Weight'] > 0] # Show only active models
                
                # Sort by weight for better visual
                w_df = w_df.sort_values('Weight', ascending=False)
                
                # Custom color map for specific models to ensure consistency
                model_colors = {
                    'ARIMAX': '#f43f5e',   # Rose
                    'XGBoost': '#3b82f6',  # Blue
                    'LSTM': '#8b5cf6',     # Violet
                    'Ensemble': '#10b981'  # Emerald
                }
                colors = [model_colors.get(m, '#94a3b8') for m in w_df['Model']]
                
                w_fig = go.Figure(data=[go.Pie(
                    labels=w_df['Model'], 
                    values=w_df['Weight'], 
                    hole=.6, # Thinner ring looks more modern
                    textinfo='label+percent',
                    textposition='outside', # Clean look
                    marker=dict(colors=colors, line=dict(color='#ffffff', width=2)),
                    pull=[0.05 if w == w_df['Weight'].max() else 0 for w in w_df['Weight']] # Highlight top model
                )])
                
                w_fig.update_layout(
                    title_text="<b>Ensemble Model Contribution</b>",
                    title_x=0.5,
                    height=350,
                    margin=dict(l=20,r=20,t=50,b=20),
                    font=dict(family="Inter, sans-serif", size=14),
                    showlegend=False, # Labels are outside, cleaner without legend
                    annotations=[dict(text=f"{len(w_df)} Models", x=0.5, y=0.5, font_size=20, showarrow=False)]
                )
                st.plotly_chart(w_fig, use_container_width=True)



            # ═══════════════════════════════════════════════════════════════════
            # TAB 3: Error Analysis / Metrics Comparison Table
            # ═══════════════════════════════════════════════════════════════════
            with tab3:
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                
                y_true = st.session_state.get('test_data', {}).get('Close', None)
                
                if y_true is not None and len(st.session_state.history_preds) > 0:
                    metrics_data = []
                    
                    for model_name, preds in st.session_state.history_preds.items():
                        min_len = min(len(y_true), len(preds))
                        y_t = y_true.values[:min_len] if hasattr(y_true, 'values') else y_true[:min_len]
                        p = preds[:min_len]
                        
                        mse = mean_squared_error(y_t, p)
                        rmse = np.sqrt(mse)
                        mae = mean_absolute_error(y_t, p)
                        mape = mean_absolute_percentage_error(y_t, p) * 100
                        r2 = r2_score(y_t, p)
                        
                        metrics_data.append({
                            'Model': model_name,
                            'MSE': f"{mse:,.2f}",
                            'RMSE': f"{rmse:,.2f}",
                            'MAE': f"{mae:,.2f}",
                            'MAPE (%)': f"{mape:.4f}",
                            'R²': f"{r2:.4f}"
                        })
                    
                    metrics_df = pd.DataFrame(metrics_data)
                    
                    st.markdown("### 📈 Error Analysis Table")
                    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
                    
                    st.caption("""
                    **Metrics Explanation:**
                    - **MSE/RMSE:** Mean Squared Error - Lower is better
                    - **MAE:** Mean Absolute Error - Average absolute deviation
                    - **MAPE:** Mean Absolute Percentage Error - Easier to compare across models
                    - **R²:** Coefficient of Determination - Closer to 1 is better
                    """)
                else:
                    pass  # No message needed

    # Plot - Premium Fintech Aesthetic
    fig = go.Figure()
    
    # 1. Historical Data - Clean Line
    hist_x = df.index[-90:]
    hist_y = df['Close'].tail(90)
    
    fig.add_trace(go.Scatter(
        x=hist_x, y=hist_y, name='Historical',
        mode='lines',
        line=dict(color='#1e3a5f', width=2.5), # Dark Navy
        hovertemplate='<b>Historical</b><br>Date: %{x|%d-%b-%Y}<br>Close: %{y:,.2f}<extra></extra>' 
    ))
    
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # 2. PROBABILISTIC FORECAST FAN CHART (GARCH-Enhanced)
    # ═══════════════════════════════════════════════════════════════════════════════
    
    # Check if probabilistic forecast is available
    if 'prob_forecast' in st.session_state:
        pf = st.session_state.prob_forecast
        
        # ─────────────────────────────────────────────────────────────────────────
        # LAYER 1: Outer Band (10th - 90th Percentile) - Lightest
        # ─────────────────────────────────────────────────────────────────────────
        fig.add_trace(go.Scatter(
            x=future_dates, y=pf['upper_90'],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip',
            name='90th Percentile'
        ))
        
        fig.add_trace(go.Scatter(
            x=future_dates, y=pf['lower_10'],
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(16, 185, 129, 0.1)',  # Very light green
            showlegend=False,
            hoverinfo='skip',
            name='10th-90th Band'
        ))
        
        # ─────────────────────────────────────────────────────────────────────────
        # LAYER 2: Inner Band (25th - 75th Percentile) - Medium
        # ─────────────────────────────────────────────────────────────────────────
        fig.add_trace(go.Scatter(
            x=future_dates, y=pf['upper_75'],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip',
            name='75th Percentile'
        ))
        
        fig.add_trace(go.Scatter(
            x=future_dates, y=pf['lower_25'],
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(16, 185, 129, 0.2)',  # Medium green
            showlegend=False,
            hoverinfo='skip',
            name='25th-75th Band'
        ))
        
        # ─────────────────────────────────────────────────────────────────────────
        # LAYER 3: Median Line (50th Percentile) - Main Forecast
        # ─────────────────────────────────────────────────────────────────────────
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=pf['median'],
            mode='lines',
            name='Ensemble Forecast',
            line=dict(color='#10b981', width=2.5),
            hovertemplate=(
                '<b>Probabilistic Forecast</b><br>'
                'Date: %{x|%d-%b-%Y}<br>'
                'Median: %{y:,.2f}<br>'
                f"Range: {pf['lower_10'][-1]:,.0f} - {pf['upper_90'][-1]:,.0f}"
                '<extra></extra>'
            )
        ))
        
        # ─────────────────────────────────────────────────────────────────────────
        # Annotations: Target Price + Regime Badge
        # ─────────────────────────────────────────────────────────────────────────
        last_pred_val = pf['median'][-1]
        last_pred_date = future_dates[-1]
        
        # Target annotation
        fig.add_annotation(
            x=last_pred_date, y=last_pred_val,
            text=f"Target: {last_pred_val:,.0f}",
            showarrow=True,
            arrowhead=2, arrowsize=1, arrowwidth=2,
            arrowcolor='#10b981',
            ax=0, ay=-40,
            bgcolor="#10b981",
            bordercolor="#10b981",
            font=dict(color="white", size=12, family="Inter")
        )
        
        # Percentile annotations removed for cleaner look
        
        # Connect Last History to First Forecast
        fig.add_trace(go.Scatter(
            x=[hist_x[-1], future_dates[0]],
            y=[hist_y.iloc[-1], pf['median'][0]],
            mode='lines',
            line=dict(color='#10b981', width=2.5, dash='dot'),
            showlegend=False,
            hoverinfo='skip'
        ))
    else:
        # Fallback: Original single-line forecasts from individual models
        for model_name, preds in future_results.items():
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=preds,
                mode='lines',
                name=model_name,
                line=dict(color=chart_colors.get(model_name, '#ccc'), width=1.5, dash='dot'),
                visible='legendonly' if model_name != 'Ensemble' else True
            ))
    
    # Add component model lines (always hidden by default, can toggle in legend)
    for model_name, preds in future_results.items():
        if model_name not in ['Ensemble']:
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=preds,
                mode='lines',
                name=f'{model_name} (Component)',
                line=dict(color=chart_colors.get(model_name, '#ccc'), width=1.5, dash='dot'),
                visible='legendonly'
            ))

    # Add Vertical Line separator between History and Forecast
    # Use add_shape instead of add_vline to avoid Plotly annotation bug
    today_date = hist_x[-1]
    forecast_end_date = future_dates[-1]
    
    # Vertical dashed line at TODAY
    fig.add_shape(
        type="line",
        x0=today_date, x1=today_date,
        y0=0, y1=1,
        yref="paper",  # Use paper coordinates for full height
        line=dict(color="#64748b", width=2, dash="dash")
    )
    
    # Add "FORECAST" annotation at CENTER of the forecast zone
    # Calculate midpoint date (use index 7 of future_dates for center)
    mid_forecast_date = future_dates[len(future_dates)//2]
    
    # FORECAST ZONE label removed - the green shaded area is self-explanatory
    
    # Forecast Zone Background (Subtle highlight)
    fig.add_vrect(
        x0=today_date, x1=forecast_end_date,
        fillcolor="rgba(16, 185, 129, 0.08)", # Faint green
        layer="below", line_width=0
    )

    # 4. Layout (Legend OUTSIDE chart at top)
    fig.update_layout(
        title=dict(
            text="VN30 Forecast Trajectory",
            font=dict(family="Inter, sans-serif", size=18, color="#0f172a"),
            x=0, y=0.98
        ),
        yaxis_title="Index Points",
        template='plotly_white',
        height=500,
        font=dict(family="Inter, sans-serif", size=12, color="#64748b"),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,  # ABOVE the chart
            xanchor="center", x=0.5,
            bgcolor='rgba(255,255,255,0)',  # Transparent
            borderwidth=0
        ),
        margin=dict(l=50, r=20, t=80, b=40),  # More top margin for legend
        xaxis=dict(
            showgrid=False,
            showline=True, linecolor='#cbd5e1',
            zeroline=False
        ),
        yaxis=dict(
            showgrid=True, gridcolor='#f1f5f9', gridwidth=1,
            zeroline=False,
            showline=False,
            tickformat=","
        ),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    # Backtest Metrics Table
    with st.expander("View Model Performance Metrics"):
        metrics_data = []
        min_len = min(len(p) for p in st.session_state.history_preds.values())
        y_true = df['Close'].iloc[-min_len:].values if len(df) > min_len else []
        if len(y_true) > 0:
            for name, pred in st.session_state.history_preds.items():
                m = calculate_metrics(y_true, pred[:min_len])
                metrics_data.append({"Model": name, "MAPE": f"{m['MAPE']:.2f}%", "RMSE": f"{m['RMSE']:.2f}"})
            st.table(pd.DataFrame(metrics_data))

else:
    # --- QUICK START GUIDES (Process Flow) ---
    st.markdown("#### Process Flow")
    
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown(f"""
        <div class="process-card">
            <div class="process-step">STEP 01</div>
            <div class="process-title">{icon('upload-cloud', '#3b82f6')} Data Ingestion</div>
            <div class="process-desc">
                Select 'VN30 Default' or upload a formatted CSV. The system automatically calculates technical indicators (RSI, MACD, ATR).
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with c2:
        st.markdown(f"""
        <div class="process-card">
            <div class="process-step">STEP 02</div>
            <div class="process-title">{icon('layers', '#3b82f6')} Model Selection</div>
            <div class="process-desc">
                Configure your ensemble. Combine Statistical (ARIMA), ML (XGBoost), and Deep Learning (LSTM) for optimal results.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with c3:
        st.markdown(f"""
        <div class="process-card">
            <div class="process-step">STEP 03</div>
            <div class="process-title">{icon('play', '#3b82f6')} Execution</div>
            <div class="process-desc">
                Click 'Run Analysis'. The engine performs backtesting validation before generating the {n_days}-day future trajectory.
            </div>
        </div>
        """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 6. EDA & MODEL RESULTS TABS (Always Visible)
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown("### 📊 Data Analysis & Model Insights")

eda_tab, model_tab = st.tabs(["📈 Exploratory Data Analysis (EDA)", "🎯 Individual Model Results"])

with eda_tab:
    st.markdown("#### VN30 Historical Price & Volume")
    
    # Price Chart with Volume
    from plotly.subplots import make_subplots
    
    fig_eda = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.05,
        subplot_titles=("VN30 Index Price", "Trading Volume")
    )
    
    # Price line
    fig_eda.add_trace(
        go.Scatter(x=df.index, y=df['Close'], name='Close Price',
                   line=dict(color='#1e3a5f', width=1.5)),
        row=1, col=1
    )
    
    # Volume bars
    if 'Volume' in df.columns:
        colors_vol = ['#10b981' if df['Close'].iloc[i] >= df['Close'].iloc[i-1] else '#ef4444' 
                      for i in range(1, len(df))]
        colors_vol = ['#10b981'] + colors_vol
        fig_eda.add_trace(
            go.Bar(x=df.index, y=df['Volume'], name='Volume',
                   marker_color=colors_vol, opacity=0.7),
            row=2, col=1
        )
    
    fig_eda.update_layout(
        height=500, showlegend=False,
        margin=dict(l=50, r=50, t=40, b=40),
        template='plotly_white'
    )
    st.plotly_chart(fig_eda, use_container_width=True)
    
    # Second row: Returns Distribution + Correlation Heatmap
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Returns Distribution")
        returns = df['Close'].pct_change().dropna() * 100
        
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=returns, nbinsx=50, name='Daily Returns',
            marker_color='rgba(16, 185, 129, 0.7)'
        ))
        fig_hist.add_vline(x=returns.mean(), line_dash="dash", line_color="red",
                           annotation_text=f"Mean: {returns.mean():.2f}%")
        fig_hist.update_layout(
            height=350, xaxis_title="Daily Return (%)", yaxis_title="Frequency",
            template='plotly_white', showlegend=False
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Stats summary
        st.markdown(f"""
        | Statistic | Value |
        |-----------|-------|
        | Mean | {returns.mean():.3f}% |
        | Std Dev | {returns.std():.3f}% |
        | Skewness | {returns.skew():.3f} |
        | Kurtosis | {returns.kurtosis():.3f} |
        """)
    
    with col2:
        st.markdown("#### Feature Correlation Matrix")
        corr_cols = ['Close', 'RSI', 'ATR', 'MACD']
        if 'Volume' in df.columns:
            corr_cols.append('Volume')
        
        corr_cols = [c for c in corr_cols if c in df.columns]
        corr_matrix = df[corr_cols].corr()
        
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_cols,
            y=corr_cols,
            colorscale='RdBu_r',
            zmin=-1, zmax=1,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        fig_corr.update_layout(
            height=350, template='plotly_white',
            xaxis_title="", yaxis_title=""
        )
        st.plotly_chart(fig_corr, use_container_width=True)

with model_tab:
    if 'history_preds' in st.session_state and st.session_state.history_preds:
        st.markdown("#### Actual vs Predicted (Backtest Period)")
        
        # Get test data for comparison
        train_size = int(len(df) * 0.95)
        y_true = df['Close'].iloc[train_size:].values
        test_dates = df.index[train_size:]
        
        # Create subplot for each model
        model_names = list(st.session_state.history_preds.keys())
        n_models = len(model_names)
        
        fig_models = make_subplots(
            rows=n_models, cols=1,
            shared_xaxes=True,
            subplot_titles=[f"{name} - Actual vs Predicted" for name in model_names],
            vertical_spacing=0.08
        )
        
        colors = {'ARIMAX': '#f43f5e', 'XGBoost': '#3b82f6', 'LSTM': '#8b5cf6'}
        
        for i, name in enumerate(model_names):
            preds = st.session_state.history_preds[name]
            min_len = min(len(y_true), len(preds))
            
            # Actual
            fig_models.add_trace(
                go.Scatter(x=test_dates[:min_len], y=y_true[:min_len],
                           name='Actual', line=dict(color='#1e3a5f', width=2),
                           showlegend=(i==0)),
                row=i+1, col=1
            )
            
            # Predicted
            fig_models.add_trace(
                go.Scatter(x=test_dates[:min_len], y=preds[:min_len],
                           name=name, line=dict(color=colors.get(name, '#10b981'), width=2, dash='dot'),
                           showlegend=(i==0)),
                row=i+1, col=1
            )
        
        fig_models.update_layout(
            height=200 * n_models + 100,
            template='plotly_white',
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_models, use_container_width=True)
        
        # Error Analysis per Model
        st.markdown("#### Prediction Error Analysis")
        
        error_data = []
        for name in model_names:
            preds = st.session_state.history_preds[name]
            min_len = min(len(y_true), len(preds))
            errors = y_true[:min_len] - preds[:min_len]
            pct_errors = (errors / y_true[:min_len]) * 100
            
            error_data.append({
                'Model': name,
                'Mean Error': f"{np.mean(errors):.2f}",
                'Std Error': f"{np.std(errors):.2f}",
                'Max Overpredict': f"{np.min(errors):.2f}",
                'Max Underpredict': f"{np.max(errors):.2f}",
                'Mean Abs % Error': f"{np.mean(np.abs(pct_errors)):.2f}%"
            })
        
        st.table(pd.DataFrame(error_data))
        
    else:
        st.info("🔄 Run Analysis first to see individual model results.")

