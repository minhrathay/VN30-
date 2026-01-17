# VN30 Stock Index Forecasting - Ensemble Model

Dự án dự đoán chỉ số VN30 sử dụng mô hình ensemble kết hợp ARIMAX, XGBoost và Bi-LSTM.

## ⚠️ YÊU CẦU QUAN TRỌNG

**Python 3.11 hoặc 3.12 là BẮT BUỘC** (TensorFlow không hỗ trợ Python 3.14)

## 🔧 Cài Đặt

### Phương Án 1: Virtual Environment (Khuyên dùng)

```bash
# Tải và cài Python 3.11: https://www.python.org/downloads/release/python-3110/

# Tạo môi trường ảo
C:\Users\Admin\AppData\Local\Programs\Python\Python311\python.exe -m venv vn30_env

# Kích hoạt
.\vn30_env\Scripts\Activate.ps1

# Cài dependencies
pip install -r requirements.txt
```

### Phương Án 2: Anaconda

```bash
# Tạo môi trường Python 3.11
conda create -n vn30 python=3.11 -y

# Kích hoạt
conda activate vn30

# Cài dependencies
pip install -r requirements.txt
```

## 📂 Chuẩn Bị Dữ Liệu

- Đặt file CSV tên `Dữ liệu Lịch sử VN 30.csv` trong cùng thư mục với script
- File CSV phải có các cột: `Ngày`, `Lần cuối`, `Mở`, `Cao`, `Thấp`, `KL`, `% Thay đổi`

## 🚀 Chạy Chương Trình

```bash
cd C:\Users\Admin\.gemini\antigravity\scratch
python vn30_forecast_fixed.py
```

## 📊 Kết Quả

Script sẽ:
1. Tiền xử lý dữ liệu và tạo các chỉ báo kỹ thuật (RSI, MACD, ATR, Bollinger Bands)
2. Hiển thị các biểu đồ phân tích (EDA)
3. Huấn luyện 3 mô hình:
   - **ARIMAX**: Mô hình thống kê chuỗi thời gian
   - **XGBoost**: Mô hình học máy gradient boosting
   - **Bi-LSTM**: Mô hình học sâu với LSTM hai chiều
4. Tối ưu trọng số ensemble
5. Lưu kết quả vào file `Final_Forecast_Results.csv`

## ⚙️ Yêu Cầu Hệ Thống

- **Python**: 3.11 hoặc 3.12 (KHÔNG dùng 3.14)
- **RAM**: Tối thiểu 4GB
- **Thời gian chạy**: 5-15 phút (tùy kích thước dữ liệu)

## 🔍 Các Thay Đổi So Với Code Gốc

1. ✅ Sửa lỗi `fillna(method='bfill')` → `bfill()`
2. ✅ Cập nhật matplotlib style → `seaborn-v0_8-whitegrid`
3. ✅ Thêm kiểm tra file tồn tại
4. ✅ Tắt verbose output của LSTM model
5. ✅ Gộp tất cả code vào 1 file duy nhất

## 📝 Lưu Ý

- **Bắt buộc dùng Python 3.11 hoặc 3.12** để có đầy đủ 3 mô hình
- Nếu máy chậm, giảm `n_estimators=3000` xuống `1000` trong XGBoost
- Các biểu đồ sẽ hiển thị trong các cửa sổ riêng biệt
