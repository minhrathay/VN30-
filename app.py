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
    calculate_metrics
)
from models import (
    train_arimax, train_xgboost, train_lstm,
    optimize_ensemble, forecast_future_arimax,
    forecast_future_xgboost, forecast_future_lstm,
    create_future_dates
)

warnings.filterwarnings('ignore')

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
        margin: -6rem -5rem 2rem -5rem; /* Expanded top */
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
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("CSV File", type=['csv'], label_visibility="collapsed")
    else:
        uploaded_file = None
        
    st.markdown("---")
    
    # Models Section
    st.caption("MODEL CONFIGURATION")
    models_selected = []
    
    m_col1, m_col2 = st.columns(2)
    with m_col1:
        if st.checkbox("ARIMAX", value=True): models_selected.append("ARIMAX")
        if st.checkbox("LSTM", value=True): models_selected.append("LSTM")
    with m_col2:
        if st.checkbox("XGBoost", value=True): models_selected.append("XGBoost")
        if st.checkbox("Ensemble", value=True): models_selected.append("Ensemble")
        
    st.markdown("---")
    
    # Settings Section
    st.caption("FORECAST SETTINGS")
    n_days = st.slider("Horizon (Days)", 7, 30, 14)
    split_pct = st.slider("Training Split", 70, 95, 95)
    
    st.markdown("###")
    run_btn = st.button("RUN ANALYSIS", type="primary")

# -----------------------------------------------------------------------------
# 4. LOAD DATA & SUMMARY RIBBON
# -----------------------------------------------------------------------------
try:
    df = load_data(uploaded_file=uploaded_file)
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
    m2.metric("Market Volatility", f"{volatility:,.2f}", help="Average True Range (14)")
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
                pred, model = train_arimax(train_data, test_data)
                st.session_state.models['ARIMAX'] = model
                st.session_state.history_preds['ARIMAX'] = pred
                
            if "XGBoost" in models_selected:
                pred, model = train_xgboost(train_data, test_data)
                st.session_state.models['XGBoost'] = model
                st.session_state.history_preds['XGBoost'] = pred
                
            if "LSTM" in models_selected:
                pred, model, scaler = train_lstm(df, train_size)
                st.session_state.models['LSTM'] = {'model': model, 'scaler': scaler}
                st.session_state.history_preds['LSTM'] = pred
                
            if "Ensemble" in models_selected and len(st.session_state.history_preds) >= 2:
                preds = st.session_state.history_preds
                min_len = min(len(p) for p in preds.values())
                y_true = test_data['Close'].values[:min_len]
                synced_preds = [p[:min_len] for p in preds.values()]
                ens_pred, weights, mape = optimize_ensemble(y_true, *synced_preds)
                st.session_state.history_preds['Ensemble'] = ens_pred
                st.session_state.weights = dict(zip(preds.keys(), weights))
                
            st.success("Computation Complete")
            
        except Exception as e:
            st.error(f"Engine Failure: {e}")

# --- DISPLAY RESULTS OR QUICK START ---

if 'history_preds' in st.session_state and st.session_state.history_preds:
    # RESULTS VIEW
    st.markdown(f"#### {icon('trending-up', '#0f172a')} Forecast Trajectory", unsafe_allow_html=True)
    
    # Generate Forecasts
    future_dates = create_future_dates(df.index[-1], n_days)
    future_results = {}
    
    if 'ARIMAX' in st.session_state.models:
        future_results['ARIMAX'] = forecast_future_arimax(st.session_state.models['ARIMAX'], df, ['Lag_Vol_1', 'RSI', 'ATR'], n_days)
    if 'XGBoost' in st.session_state.models:
        future_results['XGBoost'] = forecast_future_xgboost(st.session_state.models['XGBoost'], df, n_days)
    if 'LSTM' in st.session_state.models:
         future_results['LSTM'] = forecast_future_lstm(st.session_state.models['LSTM']['model'], st.session_state.models['LSTM']['scaler'], df, n_days)
    
    # Define colors for all charts
    chart_colors = {
        'Ensemble': '#10b981', # Emerald
        'ARIMAX':   '#f43f5e', # Rose
        'XGBoost':  '#3b82f6', # Blue
        'LSTM':     '#8b5cf6'  # Violet
    }
    
    if 'weights' in st.session_state and len(future_results) >= 2:
        ens_future = np.zeros(n_days)
        w_sum = 0
        for name, res in future_results.items():
            if name in st.session_state.weights:
                w = st.session_state.weights[name]
                ens_future += res * w
                w_sum += w
        if w_sum > 0: future_results['Ensemble'] = ens_future / w_sum

        # --- Display Ensemble Weights (Donut Chart) ---
        with st.expander("Show/Hide Ensemble Details", expanded=False):
            w_df = pd.DataFrame(list(st.session_state.weights.items()), columns=['Model', 'Weight'])
            w_df = w_df[w_df['Weight'] > 0] # Show only active models
            
            w_fig = go.Figure(data=[go.Pie(
                labels=w_df['Model'], 
                values=w_df['Weight'], 
                hole=.4,
                marker=dict(colors=[chart_colors.get(m, '#94a3b8') for m in w_df['Model']])
            )])
            w_fig.update_layout(
                title_text="Ensemble Contribution Weights",
                height=300,
                margin=dict(l=0,r=0,t=30,b=0),
                font=dict(family="Inter, sans-serif")
            )
            st.plotly_chart(w_fig, use_container_width=True)

    # Plot - Expert Financial Standard Refactor
    fig = go.Figure()
    
    # 1. Historical Data - Truth Line
    hist_x = df.index[-90:]
    hist_y = df['Close'].tail(90)
    
    fig.add_trace(go.Scatter(
        x=hist_x, y=hist_y, name='Historical',
        mode='lines',
        line=dict(color='#1e293b', width=2.5), # Dark Slate, Thick
        hovertemplate='<b>Historical</b><br>Date: %{x|%d-%b-%Y}<br>Close: %{y:,.2f}<extra></extra>' 
    ))
    
    # 2. Add "Today" Vertical Dashed Line
    last_date = hist_x[-1]
    
    # 3. Forecast Lines
    # Define colors and styles
    style_map = {
        'Ensemble': {'color': '#10b981', 'width': 3.0, 'dash': 'solid', 'opacity': 1.0}, # Emerald
        'ARIMAX':   {'color': '#f43f5e', 'width': 1.5, 'dash': 'dot', 'opacity': 0.8},   # Rose
        'XGBoost':  {'color': '#3b82f6', 'width': 1.5, 'dash': 'dot', 'opacity': 0.8},   # Blue
        'LSTM':     {'color': '#8b5cf6', 'width': 1.5, 'dash': 'dot', 'opacity': 0.8}    # Violet
    }
    
    # We must ensure 'Ensemble' is plotted last to be on top, or control z-order
    # Sort keys so Ensemble is last
    sorted_models = sorted(future_results.keys(), key=lambda x: 1 if x == 'Ensemble' else 0)
    
    for name in sorted_models:
        res = future_results[name]
        style = style_map.get(name, {'color': 'gray', 'width': 1, 'dash': 'dot', 'opacity': 0.5})
        
        # Connect to last historical point
        plot_x = [last_date] + list(future_dates)
        plot_y = [hist_y.iloc[-1]] + list(res)
        
        fig.add_trace(go.Scatter(
            x=plot_x, y=plot_y, name=name,
            mode='lines',
            line=dict(
                color=style['color'], 
                width=style['width'], 
                dash=style['dash']
            ),
            opacity=style['opacity'],
            hovertemplate=f'<b>{name}</b>: %{{y:,.2f}}<extra></extra>' 
        ))
    
    # 4. Forecast Zone Highlighting
    # Calculate forecast range for background rect
    start_forecast = last_date
    end_forecast = future_dates[-1]
    
    fig.add_vrect(
        x0=start_forecast, x1=end_forecast,
        fillcolor="#f0f9ff", opacity=0.4,
        layer="below", line_width=0,
        annotation_text="Forecast Zone", annotation_position="top left",
        annotation_font_color="#3b82f6", annotation_font_size=10
    )
    
    fig.add_vline(
        x=start_forecast, 
        line_width=1.5, line_dash="dash", line_color="#94a3b8"
    )

    # 5. Layout Refinement
    fig.update_layout(
        title=dict(
            text="VN30 Forecast Trajectory",
            font=dict(family="Inter, sans-serif", size=18, color="#0f172a"),
            x=0, y=0.98
        ),
        yaxis_title="Index Points",
        template='plotly_white',
        height=550,
        font=dict(family="Inter, sans-serif", size=12, color="#64748b"),
        hovermode="x unified", # Unified Tooltip
        legend=dict(
            orientation="h",
            yanchor="top", y=0.99,
            xanchor="left", x=0.01,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='#e2e8f0',
            borderwidth=1
        ),
        margin=dict(l=0, r=10, t=60, b=0),
        xaxis=dict(
            showgrid=False, # Remove vertical grid
            showline=True, linecolor='#cbd5e1',
            zeroline=False
        ),
        yaxis=dict(
            showgrid=True, gridcolor='#f1f5f9', gridwidth=1, # Thin horizontal grid
            zeroline=False,
            showline=False,
            tickformat="," # Comma formatting
        ),
        plot_bgcolor='white', # Transparent/White
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
