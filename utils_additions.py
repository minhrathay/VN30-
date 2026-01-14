
def load_macro_data(df_vn30, sp500_path='SP500.csv', usdvnd_path='USD_VND.csv'):
    """
    ───────────────────────────────────────────────────────────────────────────────
    NẠP DỮ LIỆU LIÊN THỊ TRƯỜNG (Thực tế hóa)
    ───────────────────────────────────────────────────────────────────────────────
    Input:
        df_vn30: DataFrame VN30 chuẩn (có cột Date)
        sp500_path: Đường dẫn file SP500.csv
        usdvnd_path: Đường dẫn file USD_VND.csv
    
    Logic:
        - Đọc file CSV ngoài
        - Merge vào VN30 theo Date (Left Join)
        - Xử lý lệch ngày (Lễ, Tết, Cuối tuần) bằng ffill (Forward Fill)
    ───────────────────────────────────────────────────────────────────────────────
    """
    try:
        # 1. Load S&P 500
        if os.path.exists(sp500_path):
            sp500 = pd.read_csv(sp500_path)
            # Tự động nhận diện cột ngày
            if 'Date' in sp500.columns:
                sp500['Date'] = pd.to_datetime(sp500['Date'])
            elif 'Ngày' in sp500.columns:
                sp500['Date'] = pd.to_datetime(sp500['Ngày'], dayfirst=True)
            
            # Đổi tên cột giá đóng cửa thành SP500
            # Ưu tiên cột 'Close' hoặc 'Lần cuối'
            col_map = {'Close': 'SP500', 'Lần cuối': 'SP500'}
            sp500.rename(columns=col_map, inplace=True)
            sp500 = sp500[['Date', 'SP500']].set_index('Date')
        else:
            print(f"Warning: Không tìm thấy {sp500_path}")
            sp500 = pd.DataFrame()

        # 2. Load USD/VND
        if os.path.exists(usdvnd_path):
            usdvnd = pd.read_csv(usdvnd_path)
            if 'Date' in usdvnd.columns:
                usdvnd['Date'] = pd.to_datetime(usdvnd['Date'])
            elif 'Ngày' in usdvnd.columns:
                usdvnd['Date'] = pd.to_datetime(usdvnd['Ngày'], dayfirst=True)
                
            col_map = {'Close': 'USDVND', 'Lần cuối': 'USDVND'}
            usdvnd.rename(columns=col_map, inplace=True)
            usdvnd = usdvnd[['Date', 'USDVND']].set_index('Date')
        else:
            print(f"Warning: Không tìm thấy {usdvnd_path}")
            usdvnd = pd.DataFrame()

        # 3. Merge vào VN30
        # Đảm bảo df_vn30 có index là Date hoặc cột Date chuẩn
        df_out = df_vn30.copy()
        if 'Date' not in df_out.columns and isinstance(df_out.index, pd.DatetimeIndex):
             df_out = df_out.reset_index()
        
        if 'Date' in df_out.columns:
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

        return df_out

    except Exception as e:
        print(f"Lỗi nạp dữ liệu Macro: {e}")
        return df_vn30
