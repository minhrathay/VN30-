# VN30 Dashboard - Quick Start Guide

## 🚀 Cách Chạy Dashboard

### Khởi động
```bash
cd C:\Users\Admin\.gemini\antigravity\scratch
C:\Users\Admin\.gemini\antigravity\scratch\vn30_env\Scripts\streamlit.exe run dashboard.py
```

Dashboard sẽ mở tại: **http://localhost:8501**

---

## 📤 Upload File CSV Tùy Chỉnh

### Tính Năng Mới!
Dashboard giờ đây hỗ trợ upload **bất kỳ file CSV nào** có dữ liệu giá chứng khoán!

### Cách Sử Dụng

1. **Chọn Data Source** ở sidebar
   - Chọn "Upload Custom CSV"

2. **Click vào "Upload CSV file"**
   - Chọn file từ máy tính

3. **Kiểm tra file đã upload**
   - Hệ thống sẽ hiển thị: tên file, kích thước, loại file

4. **Click "Run Forecast"**
   - Dashboard tự động phân tích dữ liệu

---

## ✅ Format CSV Được Hỗ Trợ

### Cột Bắt Buộc

**Date Column** (chọn 1 trong các tên sau):
- `Date`, `date`, `DATE`
- `Ngày`, `Thời gian`
- Bất kỳ cột nào có chữ "date" hoặc "ngày"

**Price Column** (chọn 1 trong các tên sau):
- `Close`, `close`, `CLOSE`
- `Price`, `Adj Close`
- `Lần cuối`, `Giá`, `Đóng cửa`
- Bất kỳ cột nào có chữ "price", "close", hoặc "giá"

### Cột Tùy Chọn

- `Open` - Giá mở cửa
- `High` - Giá cao nhất
- `Low` - Giá thấp nhất
- `Volume`, `Vol`, `KL` - Khối lượng giao dịch

**Lưu ý:** Nếu không có các cột tùy chọn, hệ thống sẽ tự động tạo dựa trên giá Close

---

## 📋 Ví Dụ CSV

### Format 1: Tiếng Anh (Yahoo Finance style)
```csv
Date,Close,Open,High,Low,Volume
2024-01-01,1250.5,1245.0,1255.0,1240.0,1500000
2024-01-02,1260.2,1250.5,1265.0,1248.0,1800000
```

### Format 2: Tiếng Việt (Investing.com style)
```csv
Ngày,Lần cuối,Mở,Cao,Thấp,KL
01/01/2024,"1.250,50","1.245,00","1.255,00","1.240,00",1.5M
02/01/2024,"1.260,20","1.250,50","1.265,00","1.248,00",1.8M
```

### Format 3: Minimal (Chỉ cần Date + Price)
```csv
Date,Price
2024-01-01,1250.5
2024-01-02,1260.2
2024-01-03,1255.8
```

---

## 🔧 Xử Lý Tự Động

Dashboard tự động xử lý:

✅ **Định dạng số:**
- Dấu phẩy trong số: `1,250.50` → `1250.50`
- Ký hiệu K/M/B: `1.5M` → `1500000`

✅ **Định dạng ngày:**
- `2024-01-01` (ISO)
- `01/01/2024` (DD/MM/YYYY)
- `01-01-2024` (DD-MM-YYYY)

✅ **Tên cột:**
- Tự động nhận diện tiếng Việt/Anh
- Không phân biệt hoa thường

✅ **Dữ liệu thiếu:**
- Tự động điền giá trị cho cột thiếu
- Xóa dòng có ngày không hợp lệ

---

## 💡 Tips Sử Dụng

### Cho Kết Quả Tốt Nhất

1. **Độ dài dữ liệu:** Tối thiểu 200 ngày, khuyến nghị 500+ ngày
2. **Tần suất:** Dữ liệu theo ngày (daily)
3. **Liên tục:** Ít khoảng trống (missing dates)
4. **Chất lượng:** Dữ liệu sạch, ít outliers

### Test với Sample File

File mẫu có sẵn:
```
C:\Users\Admin\.gemini\antigravity\scratch\sample_stock_data.csv
```

Upload file này để test dashboard!

---

## 🎯 Use Cases

### 1. Phân tích cổ phiếu riêng lẻ
- Download dữ liệu từ Yahoo Finance
- Upload và dự đoán giá cổ phiếu

### 2. So sánh nhiều mã chứng khoán
- Chạy từng mã một
- So sánh kết quả MAPE

### 3. Backtest chiến lược
- Upload dữ liệu lịch sử
- Kiểm tra độ chính xác mô hình

---

## ⚠️ Xử Lý Lỗi

### Lỗi thường gặp

**"Cannot find Date column"**
- Đảm bảo có cột chứa ngày tháng
- Rename cột thành "Date"

**"Cannot find Close/Price column"**
- Đảm bảo có cột chứa giá
- Rename cột thành "Close" hoặc "Price"

**"Error loading data"**
- Check encoding: CSV phải là UTF-8
- Check delimiter: Dùng dấu phẩy (,)
- Check format: Đúng format số và ngày

---

## 📊 Sau Khi Upload

Dashboard sẽ:
1. ✅ Load và validate dữ liệu
2. ✅ Tạo technical indicators (RSI, MACD, ATR, Bollinger Bands)
3. ✅ Tạo lag features
4. ✅ Train các mô hình AI
5. ✅ Hiển thị kết quả dự đoán
6. ✅ Cho phép export CSV

Thời gian xử lý: 1-3 phút tùy kích thước file

---

## 🎉 Hoàn Tất!

Giờ bạn có thể dự đoán **bất kỳ chứng khoán nào** với dashboard này! 🚀
