"""
Utility functions for VN30 Forecasting Dashboard
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
import os

def clean_volume(x):
    """Clean volume data from string format"""
    if isinstance(x, str):
        x = x.replace(',', '')
        if 'K' in x: return float(x.replace('K', '')) * 1e3
        if 'M' in x: return float(x.replace('M', '')) * 1e6
        if 'B' in x: return float(x.replace('B', '')) * 1e9
    return float(x)

def load_data(filename='Dữ liệu Lịch sử VN 30.csv', uploaded_file=None):
    """Load and preprocess stock data from file or uploaded file"""
    
    # Load from uploaded file or local file
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File not found: {filename}")
        df = pd.read_csv(filename)
    
    # Auto-detect and rename columns (support multiple formats)
    col_map = {
        # Vietnamese format
        'Ngày': 'Date', 'Lần cuối': 'Close', 'Mở': 'Open',
        'Cao': 'High', 'Thấp': 'Low', 'KL': 'Vol', '% Thay đổi': 'Change_Pct',
        # English format
        'Date': 'Date', 'Close': 'Close', 'Open': 'Open',
        'High': 'High', 'Low': 'Low', 'Volume': 'Vol', 'Vol.': 'Vol',
        'Change %': 'Change_Pct', 'Adj Close': 'Close',
        # Other formats
        'Giá': 'Close', 'Đóng cửa': 'Close', 'Price': 'Close',
        'Khối lượng': 'Vol', 'Thời gian': 'Date'
    }
    
    df.rename(columns=col_map, inplace=True)
    
    # Ensure required columns exist
    if 'Date' not in df.columns:
        # Try to find date column
        date_cols = [col for col in df.columns if 'date' in col.lower() or 'ngày' in col.lower()]
        if date_cols:
            df.rename(columns={date_cols[0]: 'Date'}, inplace=True)
        else:
            raise ValueError("Cannot find Date column in CSV file")
    
    if 'Close' not in df.columns:
        # Try to find price column
        price_cols = [col for col in df.columns if any(x in col.lower() for x in ['close', 'price', 'giá', 'cuối'])]
        if price_cols:
            df.rename(columns={price_cols[0]: 'Close'}, inplace=True)
        else:
            raise ValueError("Cannot find Close/Price column in CSV file")
    
    # Process date
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df.dropna(subset=['Date'], inplace=True)
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    
    # Clean numeric columns
    for col in ['Close', 'Open', 'High', 'Low']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '').astype(float)
    
    # Create missing OHLC columns if not exist
    if 'Open' not in df.columns:
        df['Open'] = df['Close']
    if 'High' not in df.columns:
        df['High'] = df['Close']
    if 'Low' not in df.columns:
        df['Low'] = df['Close']
    
    # Handle volume
    if 'Vol' in df.columns:
        df['Vol'] = df['Vol'].apply(clean_volume)
    else:
        df['Vol'] = 1000000  # Default volume if not present
    
    if 'Change_Pct' in df.columns:
        df['Change_Pct'] = df['Change_Pct'].astype(str).str.replace('%', '').astype(float)
    else:
        df['Change_Pct'] = df['Close'].pct_change() * 100
    
    return df

def create_technical_indicators(df):
    """Create technical indicators"""
    df = df.copy()
    
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
    
    # Bollinger Bands
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['STD20'] = df['Close'].rolling(window=20).std()
    df['Upper_Band'] = df['MA20'] + (df['STD20'] * 2)
    df['Lower_Band'] = df['MA20'] - (df['STD20'] * 2)
    df['Pct_B'] = (df['Close'] - df['Lower_Band']) / (df['Upper_Band'] - df['Lower_Band'])
    
    # ATR
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    df['ATR'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1).rolling(14).mean()
    
    return df

def create_lag_features(df, lags=3):
    """Create lag features"""
    df = df.copy()
    
    for i in range(1, lags + 1):
        df[f'Lag_Close_{i}'] = df['Close'].shift(i)
        
        # Check for Volume column (can be 'Vol' or 'Volume')
        if 'Vol' in df.columns:
            df[f'Lag_Vol_{i}'] = df['Vol'].shift(i)
        elif 'Volume' in df.columns:
            df[f'Lag_Vol_{i}'] = df['Volume'].shift(i)
        
        # Only create lags for columns that exist
        if 'RSI' in df.columns:
            df[f'Lag_RSI_{i}'] = df['RSI'].shift(i)
        if 'MACD' in df.columns:
            df[f'Lag_MACD_{i}'] = df['MACD'].shift(i)
        if 'ATR' in df.columns:
            df[f'Lag_ATR_{i}'] = df['ATR'].shift(i)
    
    df.dropna(inplace=True)
    return df

def calculate_metrics(y_true, y_pred):
    """Calculate performance metrics"""
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    return {
        'MAPE': mape,
        'MAE': mae,
        'RMSE': rmse
    }

def create_forecast_plot(dates, actual, predictions_dict, title="VN30 Forecast"):
    """Create interactive forecast plot with Plotly - Premium Vibrant Design"""
    fig = go.Figure()
    
    # Actual prices with gradient-like effect
    fig.add_trace(go.Scatter(
        x=dates, y=actual,
        name='Actual',
        line=dict(color='#1e293b', width=4),
        mode='lines',
        fill='tozeroy',
        fillcolor='rgba(30, 41, 59, 0.1)'
    ))
    
    # Model predictions with vibrant colors
    colors = {
        'ARIMAX': '#ef4444',      # Vibrant red
        'XGBoost': '#3b82f6',     # Electric blue
        'LSTM': '#8b5cf6',        # Vibrant purple
        'Ensemble': '#10b981'     # Neon green
    }
    
    for name, preds in predictions_dict.items():
        fig.add_trace(go.Scatter(
            x=dates, y=preds,
            name=name,
            line=dict(color=colors.get(name, '#64748b'), width=3, dash='dot'),
            mode='lines'
        ))
    
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(size=24, family="Poppins, sans-serif", color='#1e293b')
        ),
        xaxis_title=dict(
            text='<b>Date</b>',
            font=dict(size=14, family="Inter, sans-serif")
        ),
        yaxis_title=dict(
            text='<b>Index Value</b>',
            font=dict(size=14, family="Inter, sans-serif")
        ),
        hovermode='x unified',
        template='plotly_white',
        height=600,
        plot_bgcolor='rgba(248, 250, 252, 0.5)',
        paper_bgcolor='white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='#e2e8f0',
            borderwidth=2,
            font=dict(size=12, family="Inter, sans-serif")
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(226, 232, 240, 0.5)',
            gridwidth=1
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(226, 232, 240, 0.5)',
            gridwidth=1
        )
    )
    
    return fig

def create_comparison_chart(metrics_df):
    """Create model comparison bar chart - Premium Vibrant Design"""
    fig = go.Figure()
    
    models = metrics_df['Model'].tolist()
    
    # Vibrant gradient colors for each model
    colors = {
        'ARIMAX': '#ef4444',
        'XGBoost': '#3b82f6',
        'LSTM': '#8b5cf6',
        'Ensemble': '#10b981'
    }
    
    bar_colors = [colors.get(model, '#64748b') for model in models]
    
    fig.add_trace(go.Bar(
        name='MAPE (%)',
        x=models,
        y=metrics_df['MAPE'],
        marker=dict(
            color=bar_colors,
            line=dict(color='white', width=2),
            pattern=dict(shape="")
        ),
        text=[f"{val:.2f}%" for val in metrics_df['MAPE']],
        textposition='outside',
        textfont=dict(size=14, family="JetBrains Mono, monospace", color='#1e293b')
    ))
    
    fig.update_layout(
        title=dict(
            text='<b>Model Performance Comparison</b><br><sub>Lower MAPE is Better</sub>',
            font=dict(size=22, family="Poppins, sans-serif", color='#1e293b')
        ),
        yaxis_title=dict(
            text='<b>MAPE (%)</b>',
            font=dict(size=14, family="Inter, sans-serif")
        ),
        xaxis=dict(
            tickfont=dict(size=13, family="Inter, sans-serif", color='#475569')
        ),
        template='plotly_white',
        height=450,
        plot_bgcolor='rgba(248, 250, 252, 0.5)',
        paper_bgcolor='white',
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(226, 232, 240, 0.5)',
            gridwidth=1
        ),
        xaxis_showgrid=False
    )
    
    return fig

def format_number(num):
    """Format number for display"""
    return f"{num:,.2f}"

def standardize_column_names(df):
    """
    ───────────────────────────────────────────────────────────────────────────────
    Standardize column names to English format (auto-mapping).
    ───────────────────────────────────────────────────────────────────────────────
    Tự động nhận diện và đổi tên các cột sang chuẩn tiếng Anh:
    - Ngày -> Date
    - Lần cuối -> Close
    - Mở -> Open
    - Cao -> High
    - Thấp -> Low
    - KL -> Volume
    """
    col_map = {
        'Ngày': 'Date', 'Date': 'Date',
        'Lần cuối': 'Close', 'Close': 'Close', 'Price': 'Close', 'Last': 'Close', 'Giá': 'Close', 'Đóng cửa': 'Close',
        'Mở': 'Open', 'Open': 'Open',
        'Cao': 'High', 'High': 'High',
        'Thấp': 'Low', 'Low': 'Low',
        'KL': 'Volume', 'Vol.': 'Volume', 'Volume': 'Volume', 'Khối lượng': 'Volume'
    }
    
    # Rename columns based on map (case-insensitive search)
    new_cols = {}
    for col in df.columns:
        for k, v in col_map.items():
            if k.lower() == col.lower():
                new_cols[col] = v
                break
    
    if new_cols:
        df = df.rename(columns=new_cols)
        
    return df

def load_macro_data(df_vn30, sp500_path='SP500.csv', usdvnd_path='USD_VND.csv'):
    """
    ───────────────────────────────────────────────────────────────────────────────
    NẠP DỮ LIỆU LIÊN THỊ TRƯỜNG (Thực tế hóa) v2.0
    ───────────────────────────────────────────────────────────────────────────────
    Input:
        df_vn30: DataFrame VN30 chuẩn (có cột Date)
        sp500_path: Đường dẫn file SP500.csv
        usdvnd_path: Đường dẫn file USD_VND.csv
    
    Logic:
        1. Đọc file & Chuẩn hóa tên cột.
        2. Shift(1) macro data:
           - Lý do: Thị trường Mỹ đóng cửa vào rạng sáng ngày T (giờ VN).
           - Do đó, giá Close của SP500 phiên T-1 mới là thông tin khả dụng cho phiên giao dịch T của VN30.
           - Việc Shift(1) đảm bảo tính nhân quả (Causality) và tránh Look-ahead Bias.
        3. Merge & Fillna.
    ───────────────────────────────────────────────────────────────────────────────
    """
    # Helper for Excel support
    def read_smart(path):
        # 1. Check exact path
        if os.path.exists(path):
            if path.lower().endswith(('.xlsx', '.xls')):
                return pd.read_excel(path)
            return pd.read_csv(path)
        
        # 2. Check alternative extension
        base, ext = os.path.splitext(path)
        
        # If input was .csv, check .xlsx
        if ext == '.csv':
            alt_path = base + '.xlsx'
            if os.path.exists(alt_path):
                return pd.read_excel(alt_path)
        
        return None

    try:
        # 1. Load S&P 500
        sp500 = read_smart(sp500_path)
        if sp500 is not None:
            # Standardize
            sp500 = standardize_column_names(sp500)
            
            # Ensure Date parsing
            if 'Date' in sp500.columns:
                sp500['Date'] = pd.to_datetime(sp500['Date'], dayfirst=True) # Try dayfirst just in case
            
            if 'Close' in sp500.columns and 'Date' in sp500.columns:
                sp500 = sp500[['Date', 'Close']].set_index('Date').sort_index()
                # Clean numeric
                if sp500['Close'].dtype == object:
                     sp500['Close'] = sp500['Close'].astype(str).str.replace(',', '').astype(float)
                
                # Shift 1 day to prevent look-ahead bias
                sp500['Close'] = sp500['Close'].shift(1)
                sp500 = sp500.rename(columns={'Close': 'SP500'})
            else:
                sp500 = pd.DataFrame()
        else:
            sp500 = pd.DataFrame()

        # 2. Load USD/VND
        usdvnd = read_smart(usdvnd_path)
        if usdvnd is not None:
            usdvnd = standardize_column_names(usdvnd)
            
            if 'Date' in usdvnd.columns:
                usdvnd['Date'] = pd.to_datetime(usdvnd['Date'], dayfirst=True)
                
            if 'Close' in usdvnd.columns and 'Date' in usdvnd.columns:
                usdvnd = usdvnd[['Date', 'Close']].set_index('Date').sort_index()
                if usdvnd['Close'].dtype == object:
                     usdvnd['Close'] = usdvnd['Close'].astype(str).str.replace(',', '').astype(float)
                
                # Tỷ giá thường cập nhật sáng sớm, có thể dùng ngay hoặc shift tùy nguồn.
                # Để an toàn, cũng shift 1 nếu dữ liệu là end-of-day.
                usdvnd['Close'] = usdvnd['Close'].shift(1)
                usdvnd = usdvnd.rename(columns={'Close': 'USDVND'})
            else:
                 usdvnd = pd.DataFrame()
        else:
            usdvnd = pd.DataFrame()

        # 3. Merge vào VN30
        # Đảm bảo df_vn30 có index là Date hoặc cột Date chuẩn
        df_out = df_vn30.copy()
        date_col_name = None
        if 'Date' in df_out.columns:
            date_col_name = 'Date'
        
        # Nếu Date đang là Index, reset để merge dễ dàng (hoặc join trực tiếp index)
        # Cách tốt nhất: Set index là Date cho cả 3 rồi join
        is_index_date = isinstance(df_out.index, pd.DatetimeIndex)
        
        if not is_index_date and date_col_name:
            df_out['Date'] = pd.to_datetime(df_out['Date'])
            df_out.set_index('Date', inplace=True)
        
        # Join (Left Join để giữ cấu trúc VN30)
        df_out = df_out.join(sp500, how='left')
        df_out = df_out.join(usdvnd, how='left')

        # 4. Xử lý Missing Data (Quan trọng cho tính thực tế)
        # Forward fill: Dùng giá ngày liền trước cho ngày nghỉ
        df_out[['SP500', 'USDVND']] = df_out[['SP500', 'USDVND']].fillna(method='ffill')
        
        # Backward fill cho những ngày đầu tiên nếu thiếu
        df_out[['SP500', 'USDVND']] = df_out[['SP500', 'USDVND']].fillna(method='bfill')

        # Nếu vẫn còn NaN (do cả cột ko có dữ liệu), fill bằng 0 hoặc drop?
        # Fill 0 để tránh lỗi model, nhưng cảnh báo
        if 'SP500' in df_out.columns and df_out['SP500'].isnull().all():
            df_out['SP500'] = 0
        if 'USDVND' in df_out.columns and df_out['USDVND'].isnull().all():
            df_out['USDVND'] = 24000 # Default fallback
            
        return df_out

    except Exception as e:
        # print(f"Lỗi nạp dữ liệu Macro: {e}")
        return df_vn30
