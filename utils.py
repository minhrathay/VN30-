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
