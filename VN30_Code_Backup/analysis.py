"""
═══════════════════════════════════════════════════════════════════════════════
ANALYSIS MODULE cho VN30 Forecasting System
───────────────────────────────────────────────────────────────────────────────
Phase 4: Feature Importance Analysis & Error Analysis
═══════════════════════════════════════════════════════════════════════════════
"""
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ═══════════════════════════════════════════════════════════════════════════════
# 1. FEATURE IMPORTANCE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_feature_importance(model, feature_names, top_k=15):
    """
    ───────────────────────────────────────────────────────────────────────────────
    PHÂN TÍCH FEATURE IMPORTANCE
    ───────────────────────────────────────────────────────────────────────────────
    
    Hỗ trợ: XGBoost, LightGBM, Random Forest, và các tree-based models.
    
    Parameters:
        model: Trained model với thuộc tính feature_importances_
        feature_names: List các tên feature
        top_k: Số lượng top features để hiển thị
    
    Returns:
        dict: {
            'importance_df': DataFrame với feature rankings,
            'top_features': List top_k features quan trọng nhất,
            'importance_dict': Dict {feature: importance}
        }
    ───────────────────────────────────────────────────────────────────────────────
    """
    result = {
        'importance_df': None,
        'top_features': [],
        'importance_dict': {}
    }
    
    try:
        # Lấy feature importances
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'get_booster'):
            # XGBoost specific
            booster = model.get_booster()
            importance_dict = booster.get_score(importance_type='gain')
            # Map to feature names if needed
            importances = np.zeros(len(feature_names))
            for i, name in enumerate(feature_names):
                if f'f{i}' in importance_dict:
                    importances[i] = importance_dict[f'f{i}']
                elif name in importance_dict:
                    importances[i] = importance_dict[name]
        else:
            print("[Feature Importance] Model không hỗ trợ feature_importances_")
            return result
        
        # Tạo DataFrame
        df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        # Normalize to percentages
        df['Importance_Pct'] = (df['Importance'] / df['Importance'].sum()) * 100
        df['Rank'] = range(1, len(df) + 1)
        
        result['importance_df'] = df
        result['top_features'] = df.head(top_k)['Feature'].tolist()
        result['importance_dict'] = dict(zip(df['Feature'], df['Importance']))
        
        return result
        
    except Exception as e:
        print(f"[Feature Importance] Error: {e}")
        return result


def plot_feature_importance(importance_df, top_k=15, title="Feature Importance Analysis"):
    """
    ───────────────────────────────────────────────────────────────────────────────
    VẼ BIỂU ĐỒ FEATURE IMPORTANCE (Premium Design)
    ───────────────────────────────────────────────────────────────────────────────
    """
    if importance_df is None or len(importance_df) == 0:
        return None
    
    df = importance_df.head(top_k).copy()
    df = df.sort_values('Importance_Pct', ascending=True)  # Để vẽ nằm ngang đúng
    
    # Color gradient based on importance
    colors = [f'rgba(16, 185, 129, {0.4 + 0.6 * (i/len(df))})' for i in range(len(df))]
    
    fig = go.Figure(go.Bar(
        x=df['Importance_Pct'],
        y=df['Feature'],
        orientation='h',
        marker=dict(color=colors[::-1]),
        text=[f'{val:.1f}%' for val in df['Importance_Pct']],
        textposition='outside',
        textfont=dict(size=11, family="Inter, sans-serif")
    ))
    
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(size=18, family="Inter, sans-serif", color="#0f172a"),
            x=0
        ),
        xaxis_title="Importance (%)",
        yaxis_title="",
        height=max(400, top_k * 30),
        template='plotly_white',
        margin=dict(l=150, r=50, t=60, b=40),
        font=dict(family="Inter, sans-serif", size=12),
        yaxis=dict(tickfont=dict(size=11)),
        xaxis=dict(showgrid=True, gridcolor='rgba(226, 232, 240, 0.5)')
    )
    
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 2. ERROR ANALYSIS MODULE
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_prediction_errors(y_true, y_pred, df_context=None, date_index=None):
    """
    ───────────────────────────────────────────────────────────────────────────────
    PHÂN TÍCH CHI TIẾT KHI NÀO MODEL DỰ BÁO SAI
    ───────────────────────────────────────────────────────────────────────────────
    
    Phân tích:
    1. Error Distribution: Phân bố sai số
    2. Worst Predictions: Những ngày dự báo sai nhất
    3. Error by Volatility: Sai số theo độ biến động
    4. Error by Market Regime: Sai số theo regime (trending/sideways)
    5. Directional Accuracy: Dự đoán đúng hướng?
    
    Parameters:
        y_true: np.array giá thực
        y_pred: np.array giá dự báo
        df_context: DataFrame chứa context data (ATR, RSI, etc.)
        date_index: DatetimeIndex của test period
    
    Returns:
        dict: Error analysis results
    ───────────────────────────────────────────────────────────────────────────────
    """
    # Ensure same length
    min_len = min(len(y_true), len(y_pred))
    y_true = np.array(y_true[:min_len])
    y_pred = np.array(y_pred[:min_len])
    
    # ─────────────────────────────────────────────────────────────────────────────
    # 1. BASIC ERROR METRICS
    # ─────────────────────────────────────────────────────────────────────────────
    errors = y_true - y_pred
    abs_errors = np.abs(errors)
    pct_errors = (errors / y_true) * 100
    abs_pct_errors = np.abs(pct_errors)
    
    results = {
        'basic_metrics': {
            'mape': float(np.mean(abs_pct_errors)),
            'mae': float(np.mean(abs_errors)),
            'rmse': float(np.sqrt(np.mean(errors**2))),
            'max_error': float(np.max(abs_errors)),
            'max_pct_error': float(np.max(abs_pct_errors)),
            'bias': float(np.mean(errors)),  # Positive = under-predict, Negative = over-predict
        }
    }
    
    # ─────────────────────────────────────────────────────────────────────────────
    # 2. DIRECTIONAL ACCURACY (Win Rate)
    # ─────────────────────────────────────────────────────────────────────────────
    if len(y_true) > 1:
        direction_true = np.sign(np.diff(y_true))
        direction_pred = np.sign(np.diff(y_pred))
        
        correct_directions = (direction_true == direction_pred)
        win_rate = np.mean(correct_directions) * 100
        
        results['directional'] = {
            'win_rate': float(win_rate),
            'total_predictions': int(len(direction_true)),
            'correct_predictions': int(np.sum(correct_directions))
        }
    else:
        results['directional'] = {'win_rate': 0, 'total_predictions': 0, 'correct_predictions': 0}
    
    # ─────────────────────────────────────────────────────────────────────────────
    # 3. WORST PREDICTIONS
    # ─────────────────────────────────────────────────────────────────────────────
    worst_indices = np.argsort(abs_pct_errors)[-5:][::-1]  # Top 5 worst
    
    worst_predictions = []
    for idx in worst_indices:
        pred_info = {
            'index': int(idx),
            'y_true': float(y_true[idx]),
            'y_pred': float(y_pred[idx]),
            'error': float(errors[idx]),
            'pct_error': float(pct_errors[idx])
        }
        if date_index is not None and idx < len(date_index):
            pred_info['date'] = str(date_index[idx])
        worst_predictions.append(pred_info)
    
    results['worst_predictions'] = worst_predictions
    
    # ─────────────────────────────────────────────────────────────────────────────
    # 4. ERROR BY VOLATILITY (nếu có ATR data)
    # ─────────────────────────────────────────────────────────────────────────────
    if df_context is not None and 'ATR' in df_context.columns:
        atr_values = df_context['ATR'].values[:min_len]
        
        # Split by ATR quartiles
        atr_median = np.median(atr_values)
        
        high_vol_mask = atr_values > atr_median
        low_vol_mask = atr_values <= atr_median
        
        results['error_by_volatility'] = {
            'high_volatility': {
                'mape': float(np.mean(abs_pct_errors[high_vol_mask])) if np.any(high_vol_mask) else 0,
                'count': int(np.sum(high_vol_mask))
            },
            'low_volatility': {
                'mape': float(np.mean(abs_pct_errors[low_vol_mask])) if np.any(low_vol_mask) else 0,
                'count': int(np.sum(low_vol_mask))
            }
        }
    
    # ─────────────────────────────────────────────────────────────────────────────
    # 5. ERROR DISTRIBUTION
    # ─────────────────────────────────────────────────────────────────────────────
    results['error_distribution'] = {
        'mean_error': float(np.mean(errors)),
        'std_error': float(np.std(errors)),
        'skewness': float(pd.Series(errors).skew()),
        'kurtosis': float(pd.Series(errors).kurtosis()),
        'percentile_25': float(np.percentile(pct_errors, 25)),
        'percentile_75': float(np.percentile(pct_errors, 75)),
    }
    
    return results


def plot_error_analysis(y_true, y_pred, date_index=None):
    """
    ───────────────────────────────────────────────────────────────────────────────
    VẼ BIỂU ĐỒ ERROR ANALYSIS (2x2 Grid)
    ───────────────────────────────────────────────────────────────────────────────
    """
    min_len = min(len(y_true), len(y_pred))
    y_true = np.array(y_true[:min_len])
    y_pred = np.array(y_pred[:min_len])
    
    errors = y_true - y_pred
    pct_errors = (errors / y_true) * 100
    
    if date_index is not None:
        x_axis = date_index[:min_len]
    else:
        x_axis = list(range(min_len))
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Actual vs Predicted",
            "Prediction Errors Over Time",
            "Error Distribution",
            "Error Scatter Plot"
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.08
    )
    
    # 1. Actual vs Predicted
    fig.add_trace(
        go.Scatter(x=x_axis, y=y_true, name="Actual", 
                   line=dict(color='#1e3a5f', width=2)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=x_axis, y=y_pred, name="Predicted",
                   line=dict(color='#10b981', width=2, dash='dot')),
        row=1, col=1
    )
    
    # 2. Errors over time
    colors = ['#ef4444' if e > 0 else '#10b981' for e in pct_errors]
    fig.add_trace(
        go.Bar(x=x_axis, y=pct_errors, name="Error %",
               marker_color=colors, showlegend=False),
        row=1, col=2
    )
    
    # 3. Error Distribution (Histogram)
    fig.add_trace(
        go.Histogram(x=pct_errors, nbinsx=20, name="Error Dist",
                     marker_color='rgba(16, 185, 129, 0.7)',
                     showlegend=False),
        row=2, col=1
    )
    
    # 4. Scatter: Actual vs Predicted
    fig.add_trace(
        go.Scatter(x=y_true, y=y_pred, mode='markers',
                   name="Predictions",
                   marker=dict(color='#3b82f6', size=6, opacity=0.6),
                   showlegend=False),
        row=2, col=2
    )
    # Add perfect prediction line
    min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    fig.add_trace(
        go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                   mode='lines', name="Perfect",
                   line=dict(color='#ef4444', dash='dash'),
                   showlegend=False),
        row=2, col=2
    )
    
    fig.update_layout(
        title=dict(
            text="<b>Error Analysis Dashboard</b>",
            font=dict(size=20, family="Inter, sans-serif", color="#0f172a"),
            x=0.5
        ),
        height=700,
        template='plotly_white',
        font=dict(family="Inter, sans-serif", size=11),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=1, col=2)
    fig.update_xaxes(title_text="Error %", row=2, col=1)
    fig.update_xaxes(title_text="Actual", row=2, col=2)
    
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Error %", row=1, col=2)
    fig.update_yaxes(title_text="Frequency", row=2, col=1)
    fig.update_yaxes(title_text="Predicted", row=2, col=2)
    
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 3. MODEL COMPARISON REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def generate_model_report(models_results, y_true):
    """
    ───────────────────────────────────────────────────────────────────────────────
    TẠO BÁO CÁO SO SÁNH MODELS
    ───────────────────────────────────────────────────────────────────────────────
    
    Parameters:
        models_results: Dict {model_name: predictions array}
        y_true: Actual values
    
    Returns:
        DataFrame với metrics của các models
    ───────────────────────────────────────────────────────────────────────────────
    """
    from sklearn.metrics import mean_squared_error, r2_score
    
    report_data = []
    
    for model_name, y_pred in models_results.items():
        min_len = min(len(y_true), len(y_pred))
        y_t = y_true[:min_len] if hasattr(y_true, '__getitem__') else y_true.values[:min_len]
        y_p = y_pred[:min_len]
        
        # Calculate metrics
        mse = mean_squared_error(y_t, y_p)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_t, y_p)
        mape = mean_absolute_percentage_error(y_t, y_p) * 100
        r2 = r2_score(y_t, y_p)
        
        # Directional accuracy
        if len(y_t) > 1:
            dir_true = np.sign(np.diff(y_t))
            dir_pred = np.sign(np.diff(y_p))
            win_rate = np.mean(dir_true == dir_pred) * 100
        else:
            win_rate = 0
        
        report_data.append({
            'Model': model_name,
            'MAPE (%)': f'{mape:.4f}',
            'RMSE': f'{rmse:.2f}',
            'MAE': f'{mae:.2f}',
            'R²': f'{r2:.4f}',
            'Win Rate (%)': f'{win_rate:.1f}'
        })
    
    return pd.DataFrame(report_data)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. EXPORT UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def export_analysis_to_json(analysis_results, filepath):
    """Export analysis results to JSON file for reporting"""
    import json
    
    # Convert numpy types to Python types
    def convert_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(i) for i in obj]
        return obj
    
    converted = convert_types(analysis_results)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(converted, f, indent=2, ensure_ascii=False)
    
    print(f"[Analysis] Exported to {filepath}")
