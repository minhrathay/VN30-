import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
from tensorflow.keras.models import load_model
from statsmodels.tsa.arima.model import ARIMA
from datetime import timedelta

# --- 1. CẤU HÌNH TRANG & CSS NÂNG CAO (UI/UX) ---
st.set_page_config(page_title="VN30 Quant Terminal Pro", layout="wide")

st.markdown("""
    <style>
    /* Nền gradient sâu */
    .stApp {
        background: radial-gradient(circle at top right, #1e2631, #0e1117);
        color: #e0e6ed;
    }
    
    /* Hiệu ứng kính mờ (Glassmorphism) cho Metric */
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 20px !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        transition: transform 0.3s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-5px);
        border-color: rgba(0, 255, 0, 0.3);
    }

    /* Tối ưu Sidebar */
    section[data-testid="stSidebar"] {
        background-color: rgba(22, 27, 34, 0.95) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Nổi khối cho Card */
    .status-card {
        background: linear-gradient(135deg, rgba(0,255,0,0.1) 0%, rgba(0,0,0,0) 100%);
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid #00FF00;
        margin-bottom: 25px;
        box-shadow: 5px 5px 15px rgba(0,0,0,0.2);
    }

    /* Tab Header tinh tế */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255,255,255,0.05);
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        color: #8b949e;
    }
    .stTabs [data-baseweb="tab--active"] {
        background-color: rgba(255,255,255,0.1) !important;
        color: #ffffff !important;
        border-bottom: 2px solid #00FF00 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. HÀM XỬ LÝ (GIỮ NGUYÊN LOGIC 70/30) ---
@st.cache_resource
def load_ai_models():
    try:
        lstm = load_model('lstm_model.h5', compile=False)
        scaler = joblib.load('scaler.pkl')
        return lstm, scaler
    except: return None, None

def feature_engineering(df):
    col_map = {'Ngày': 'Date', 'Lần cuối': 'Close', 'Mở': 'Open', 'Cao': 'High', 'Thấp': 'Low', 'KL': 'Volume'}
    df.rename(columns=col_map, inplace=True)
    df.columns = [c.strip().capitalize() for c in df.columns]
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    for col in ['Close', 'Open', 'High', 'Low']:
        df[col] = df[col].astype(str).str.replace(',', '').astype(float)
    
    def clean_vol(x):
        x = str(x).upper().replace(',', '').strip()
        if 'B' in x: return float(x.replace('B', '')) * 1e9
        if 'M' in x: return float(x.replace('M', '')) * 1e6
        if 'K' in x: return float(x.replace('K', '')) * 1e3
        return float(x) if x not in ['NAN', ''] else 0.0
    
    df['Vol'] = df['Volume'].apply(clean_vol)
    df.sort_values('Date', inplace=True)
    
    delta = df['Close'].diff()
    df['RSI'] = 100 - (100 / (1 + (delta.where(delta > 0, 0).rolling(14).mean() / -delta.where(delta < 0, 0).rolling(14).mean())))
    df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
    df['ATR'] = pd.concat([df['High']-df['Low'], np.abs(df['High']-df['Close'].shift()), np.abs(df['Low']-df['Close'].shift())], axis=1).max(axis=1).rolling(14).mean()

    for col in ['Close', 'Vol', 'RSI', 'MACD', 'ATR']:
        for i in range(1, 4):
            df[f'Lag_{col}_{i}'] = df[col].shift(i)
    return df.dropna()

def multi_step_forecast(df_ready, lstm, scaler, w_arima, steps):
    model_arima = ARIMA(df_ready['Close'].values, order=(5,1,0)).fit()
    p_arima = model_arima.forecast(steps=steps)
    f_lstm = [col for col in df_ready.columns if 'Lag_' in col] + ['Close', 'Vol', 'RSI']
    current_f = df_ready[f_lstm].iloc[-1:].values.copy()
    last_c = df_ready['Close'].iloc[-1]
    p_lstm = []
    for _ in range(steps):
        ret = lstm.predict(scaler.transform(current_f).reshape(-1, 1, 18), verbose=0).flatten()[0]
        next_c = last_c * np.exp(ret)
        p_lstm.append(next_c)
        new_row = current_f[0].copy()
        new_row[10], new_row[5], new_row[0] = new_row[5], new_row[0], next_c
        current_f[0] = new_row
        last_c = next_c
    return (p_arima * w_arima) + (np.array(p_lstm) * (1 - w_arima))

# --- 3. GIAO DIỆN CHÍNH ---
st.markdown('# 🛡️ VN30 QUANTITATIVE TERMINAL')
st.markdown('<div class="status-card">📊 <b>HỆ THỐNG TRỰC TUYẾN:</b> Mô hình Hybrid ARIMA-LSTM đang phân tích chuỗi thời gian...</div>', unsafe_allow_html=True)

lstm_model, scaler_tool = load_ai_models()

with st.sidebar:
    st.markdown("### 🖥️ BÀN ĐIỀU KHIỂN")
    uploaded_file = st.file_uploader("Nạp dữ liệu thị trường (.CSV)", type=["csv"])
    st.markdown("---")
    f_days = st.slider("Cửa sổ dự báo (Ngày)", 1, 14, 7)
    w_arima = st.slider("Tỷ trọng Thống kê (ARIMA)", 0, 100, 70) / 100
    st.caption(f"Trí tuệ nhân tạo (LSTM) đóng góp: {(1-w_arima)*100:.0f}%")

if uploaded_file and lstm_model:
    df = feature_engineering(pd.read_csv(uploaded_file))
    
    # Phân tầng Dashboard
    tab1, tab2 = st.tabs(["📈 PHÂN TÍCH CHIẾN THUẬT", "🗄️ DỮ LIỆU ĐỊNH LƯỢNG"])
    
    with tab1:
        f_res = multi_step_forecast(df, lstm_model, scaler_tool, w_arima, f_days)
        
        # Grid hiển thị nổi khối
        m1, m2, m3 = st.columns(3)
        m1.metric("CHỈ SỐ HIỆN TẠI", f"{df['Close'].iloc[-1]:,.2f}")
        m2.metric(f"MỤC TIÊU {f_days} NGÀY", f"{f_res[-1]:,.2f}", f"{f_res[-1]-df['Close'].iloc[-1]:+,.2f}")
        m3.metric("XU HƯỚNG DỰ KIẾN", "TĂNG" if f_res[-1] > df['Close'].iloc[-1] else "GIẢM")

        # Biểu đồ Plotly với Style đồng bộ
        l_date = df['Date'].iloc[-1]
        f_dates = [l_date + timedelta(days=i) for i in range(f_days + 1)]
        f_prices = [df['Close'].iloc[-1]] + list(f_res)
        model_h = ARIMA(df['Close'].values, order=(5,1,0)).fit().fittedvalues * w_arima + df['Close']*(1-w_arima)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Thực tế', line=dict(color='#00FF00', width=3)))
        fig.add_trace(go.Scatter(x=df['Date'], y=model_h, name='Mô hình (Backtest)', line=dict(color='#FFFF00', width=2)))
        fig.add_trace(go.Scatter(x=f_dates, y=f_prices, name='Dự báo Tương lai', line=dict(color='#FF0000', width=4)))
        
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color="#ffffff"), height=650,
            margin=dict(l=0, r=0, t=20, b=0),
            xaxis=dict(gridcolor='rgba(255,255,255,0.05)', zeroline=False),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)', zeroline=False),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Bảng dữ liệu dự báo chi tiết")
        r_df = pd.DataFrame({"Ngày": [d.strftime('%d/%m/%Y') for d in f_dates[1:]], "Giá kỳ vọng": f_res})
        st.dataframe(r_df.style.background_gradient(cmap='Greens', subset=['Giá kỳ vọng']), use_container_width=True)
else:
    st.info("👋 Sẵn sàng phân tích. Hãy nạp file CSV VN30 để bắt đầu.")
