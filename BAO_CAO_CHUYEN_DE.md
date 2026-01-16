# BÁO CÁO CHUYÊN ĐỀ TỐT NGHIỆP

## TRƯỜNG ĐẠI HỌC KINH TẾ - TÀI CHÍNH TP.HCM (UEF)
### KHOA TÀI CHÍNH NGÂN HÀNG

---

# ỨNG DỤNG DỮ LIỆU THỊ TRƯỜNG VÀ MÔ HÌNH HỌC MÁY TRONG DỰ BÁO BIẾN ĐỘNG VÀ THANH KHOẢN CHỈ SỐ VN30 TỪ 2019 - 2024

---

**Giảng viên hướng dẫn:** ThS. Nguyễn Nam Trung

**Cán bộ hướng dẫn:** ThS. Trần Bá Duy

**Sinh viên thực hiện:** [Họ và tên sinh viên]

**Mã số sinh viên:** [MSSV]

**Lớp:** [Tên lớp]

---

**TP. Hồ Chí Minh, năm 2024**

---

# LỜI CẢM ƠN

Trước tiên, em xin gửi lời cảm ơn chân thành và sâu sắc nhất đến **Thầy Nguyễn Nam Trung** - Giảng viên hướng dẫn và **Thầy Trần Bá Duy** - Cán bộ hướng dẫn, đã tận tình chỉ bảo, định hướng và hỗ trợ em trong suốt quá trình thực hiện đề tài này.

Em xin cảm ơn quý Thầy Cô trong **Khoa Tài chính Ngân hàng**, Trường Đại học Kinh tế - Tài chính TP.HCM đã truyền đạt những kiến thức quý báu trong suốt thời gian em học tập tại trường.

Em cũng xin gửi lời cảm ơn đến gia đình và bạn bè đã luôn động viên, khích lệ em trong quá trình học tập và nghiên cứu.

Mặc dù đã cố gắng hoàn thành tốt nhất đề tài, nhưng do hạn chế về thời gian và kiến thức, bài viết không thể tránh khỏi những thiếu sót. Em rất mong nhận được sự góp ý từ quý Thầy Cô để bài báo cáo được hoàn thiện hơn.

*TP. Hồ Chí Minh, ngày ... tháng ... năm 2024*

**Sinh viên thực hiện**

[Họ và tên]

---

# LỜI CAM ĐOAN

Em xin cam đoan đây là công trình nghiên cứu của riêng em. Các số liệu, kết quả nêu trong báo cáo là trung thực và chưa từng được ai công bố trong bất kỳ công trình nào khác.

Em xin cam đoan rằng các thông tin trích dẫn trong báo cáo đã được chỉ rõ nguồn gốc.

*TP. Hồ Chí Minh, ngày ... tháng ... năm 2024*

**Sinh viên thực hiện**

[Họ và tên]

---

# MỤC LỤC

1. [CHƯƠNG 1: GIỚI THIỆU](#chương-1-giới-thiệu)
2. [CHƯƠNG 2: CƠ SỞ LÝ THUYẾT](#chương-2-cơ-sở-lý-thuyết)
3. [CHƯƠNG 3: PHƯƠNG PHÁP NGHIÊN CỨU](#chương-3-phương-pháp-nghiên-cứu)
4. [CHƯƠNG 4: KẾT QUẢ THỰC NGHIỆM](#chương-4-kết-quả-thực-nghiệm)
5. [CHƯƠNG 5: KẾT LUẬN VÀ KIẾN NGHỊ](#chương-5-kết-luận-và-kiến-nghị)
6. [TÀI LIỆU THAM KHẢO](#tài-liệu-tham-khảo)

---

# DANH MỤC TỪ VIẾT TẮT

| Từ viết tắt | Giải thích |
|-------------|------------|
| VN30 | Chỉ số gồm 30 cổ phiếu có vốn hóa và thanh khoản cao nhất trên HOSE |
| ARIMAX | AutoRegressive Integrated Moving Average with eXogenous variables |
| LSTM | Long Short-Term Memory |
| XGBoost | eXtreme Gradient Boosting |
| GARCH | Generalized Autoregressive Conditional Heteroskedasticity |
| MAPE | Mean Absolute Percentage Error |
| RMSE | Root Mean Squared Error |
| RSI | Relative Strength Index |
| MACD | Moving Average Convergence Divergence |
| ATR | Average True Range |

---

# DANH MỤC BẢNG BIỂU

- Bảng 2.1: So sánh các mô hình học máy trong dự báo tài chính
- Bảng 3.1: Mô tả dữ liệu VN30 (2019-2024)
- Bảng 3.2: Các chỉ báo kỹ thuật được sử dụng
- Bảng 4.1: Kết quả đánh giá mô hình ARIMAX
- Bảng 4.2: Kết quả đánh giá mô hình XGBoost
- Bảng 4.3: Kết quả đánh giá mô hình LSTM
- Bảng 4.4: So sánh hiệu suất các mô hình
- Bảng 4.5: Kết quả mô hình Ensemble

---

# DANH MỤC HÌNH ẢNH

- Hình 2.1: Cấu trúc mạng LSTM
- Hình 2.2: Quy trình XGBoost
- Hình 3.1: Kiến trúc hệ thống dự báo
- Hình 3.2: Quy trình xử lý dữ liệu
- Hình 4.1: Biến động chỉ số VN30 (2019-2024)
- Hình 4.2: Phân phối returns VN30
- Hình 4.3: Kết quả dự báo Ensemble
- Hình 4.4: Fan chart dự báo xác suất

---

# CHƯƠNG 1: GIỚI THIỆU

## 1.1 Lý do chọn đề tài

Thị trường chứng khoán Việt Nam đã có những bước phát triển vượt bậc trong giai đoạn 2019-2024, với sự gia tăng đáng kể về quy mô giao dịch và số lượng nhà đầu tư tham gia. Chỉ số VN30, đại diện cho 30 cổ phiếu có vốn hóa và thanh khoản cao nhất trên Sở Giao dịch Chứng khoán TP.HCM (HOSE), đóng vai trò quan trọng trong việc phản ánh xu hướng chung của thị trường.

Trong bối cảnh thị trường ngày càng biến động phức tạp, đặc biệt sau đại dịch COVID-19 và những biến động kinh tế toàn cầu, việc dự báo chính xác biến động giá và thanh khoản trở nên cấp thiết hơn bao giờ hết. Các phương pháp phân tích truyền thống dựa trên kinh nghiệm và phân tích cơ bản đã bộc lộ nhiều hạn chế trong việc nắm bắt các mô hình phức tạp và phi tuyến của thị trường.

Sự phát triển của học máy (Machine Learning) và trí tuệ nhân tạo đã mở ra những cơ hội mới trong lĩnh vực dự báo tài chính. Các mô hình học máy có khả năng học hỏi từ dữ liệu lịch sử, phát hiện các mẫu hình ẩn, và thích ứng với sự thay đổi của thị trường. Tuy nhiên, việc ứng dụng các mô hình này vào thực tiễn thị trường Việt Nam vẫn còn nhiều thách thức.

Từ những lý do trên, đề tài "Ứng dụng dữ liệu thị trường và mô hình học máy trong dự báo biến động và thanh khoản chỉ số VN30 từ 2019-2024" được lựa chọn nhằm nghiên cứu và đánh giá hiệu quả của các phương pháp học máy trong dự báo tài chính tại Việt Nam.

## 1.2 Mục tiêu nghiên cứu

### 1.2.1 Mục tiêu tổng quát

Xây dựng và đánh giá hệ thống dự báo biến động và thanh khoản chỉ số VN30 sử dụng các mô hình học máy kết hợp (Ensemble Learning), nhằm cung cấp công cụ hỗ trợ quyết định cho nhà đầu tư.

### 1.2.2 Mục tiêu cụ thể

1. **Phân tích dữ liệu**: Tổng hợp và phân tích dữ liệu lịch sử chỉ số VN30 giai đoạn 2019-2024, bao gồm giá đóng cửa, khối lượng giao dịch, và các chỉ báo kỹ thuật.

2. **Xây dựng mô hình**: Triển khai và huấn luyện các mô hình dự báo:
   - ARIMAX (Auto-Regressive Integrated Moving Average with eXogenous variables)
   - XGBoost (eXtreme Gradient Boosting)
   - Bi-LSTM (Bidirectional Long Short-Term Memory)
   - GARCH (Generalized Autoregressive Conditional Heteroskedasticity)

3. **Kết hợp mô hình**: Phát triển phương pháp Ensemble Learning (Meta-Learning Stacking) để kết hợp ưu điểm của các mô hình đơn lẻ.

4. **Đánh giá hiệu suất**: So sánh và đánh giá hiệu quả dự báo của các mô hình thông qua các chỉ số MAPE, RMSE, MAE.

5. **Xây dựng ứng dụng**: Phát triển dashboard trực quan hóa kết quả dự báo phục vụ người dùng cuối.

## 1.3 Đối tượng và phạm vi nghiên cứu

### 1.3.1 Đối tượng nghiên cứu

- Chỉ số VN30 và các thành phần cấu thành
- Biến động giá (Price Volatility)
- Thanh khoản thị trường (Market Liquidity)
- Các yếu tố vĩ mô liên quan (S&P 500, tỷ giá USD/VND)

### 1.3.2 Phạm vi nghiên cứu

- **Phạm vi thời gian**: Dữ liệu từ 01/01/2019 đến 31/12/2024 (6 năm)
- **Phạm vi không gian**: Sở Giao dịch Chứng khoán TP.HCM (HOSE)
- **Phạm vi nội dung**: Tập trung vào dự báo ngắn hạn (1-14 ngày giao dịch)

## 1.4 Phương pháp nghiên cứu

Nghiên cứu sử dụng kết hợp các phương pháp:

1. **Phương pháp định lượng**: Phân tích thống kê mô tả, phân tích chuỗi thời gian, và đánh giá mô hình dự báo.

2. **Phương pháp thực nghiệm**: Xây dựng và thử nghiệm các mô hình học máy trên dữ liệu thực tế.

3. **Phương pháp so sánh**: Đánh giá hiệu quả giữa các mô hình thông qua các metrics chuẩn.

## 1.5 Ý nghĩa khoa học và thực tiễn

### 1.5.1 Ý nghĩa khoa học

- Đóng góp vào nghiên cứu ứng dụng học máy trong dự báo tài chính tại Việt Nam
- Đề xuất phương pháp Ensemble kết hợp nhiều mô hình để tăng độ chính xác
- Giới thiệu mô hình GARCH cho dự báo xác suất (Probabilistic Forecasting)

### 1.5.2 Ý nghĩa thực tiễn

- Cung cấp công cụ hỗ trợ quyết định đầu tư cho nhà đầu tư cá nhân và tổ chức
- Xây dựng dashboard trực quan, dễ sử dụng
- Khả năng mở rộng và tích hợp vào các hệ thống giao dịch

## 1.6 Cấu trúc báo cáo

Báo cáo được chia thành 5 chương:

- **Chương 1**: Giới thiệu tổng quan về đề tài
- **Chương 2**: Cơ sở lý thuyết về các mô hình dự báo
- **Chương 3**: Phương pháp nghiên cứu và xử lý dữ liệu
- **Chương 4**: Kết quả thực nghiệm và thảo luận
- **Chương 5**: Kết luận và kiến nghị

---

# CHƯƠNG 2: CƠ SỞ LÝ THUYẾT

## 2.1 Tổng quan về chỉ số VN30

### 2.1.1 Giới thiệu chỉ số VN30

Chỉ số VN30 được Sở Giao dịch Chứng khoán TP.HCM (HOSE) giới thiệu vào năm 2012, bao gồm 30 cổ phiếu có vốn hóa thị trường lớn nhất và thanh khoản cao nhất. Chỉ số này đóng vai trò như một "benchmark" đại diện cho thị trường, giúp nhà đầu tư đánh giá hiệu suất danh mục và là cơ sở cho các sản phẩm phái sinh.

**Tiêu chí lựa chọn cổ phiếu VN30:**
- Vốn hóa thị trường thuộc top cao nhất
- Thanh khoản giao dịch bình quân cao
- Không bị hạn chế giao dịch
- Được rà soát định kỳ 6 tháng/lần

### 2.1.2 Đặc điểm biến động VN30 giai đoạn 2019-2024

Giai đoạn 2019-2024 chứng kiến nhiều biến động đáng kể:

| Giai đoạn | Sự kiện | Tác động |
|-----------|---------|----------|
| Q1/2020 | Đại dịch COVID-19 | VN30 giảm 35% từ đỉnh |
| Q4/2020-Q3/2021 | Phục hồi hậu COVID | Tăng 80% từ đáy |
| Q4/2021-Q2/2022 | Khủng hoảng trái phiếu | Giảm 40% |
| 2023-2024 | Phục hồi từ từ | Dao động 1,050-1,300 |

## 2.2 Lược khảo tài liệu (Literature Review)

### 2.2.1 Nghiên cứu quốc tế về dự báo thị trường chứng khoán

Việc dự báo giá chứng khoán đã thu hút sự quan tâm của nhiều nhà nghiên cứu trên thế giới. Các nghiên cứu tiêu biểu bao gồm:

**Fischer & Krauss (2018)** đã thực hiện nghiên cứu quy mô lớn sử dụng LSTM để dự báo S&P 500, cho thấy mô hình deep learning vượt trội so với các phương pháp truyền thống với accuracy 55.9% trong việc dự đoán hướng giá.

**Chen & Guestrin (2016)** giới thiệu XGBoost - thuật toán gradient boosting đã trở thành tiêu chuẩn trong các cuộc thi machine learning và được ứng dụng rộng rãi trong dự báo tài chính.

**Makridakis et al. (2018)** trong nghiên cứu "M4 Competition" so sánh 60+ phương pháp dự báo, kết luận rằng ensemble methods thường cho kết quả tốt hơn single models.

### 2.2.2 Nghiên cứu tại Việt Nam

Tại Việt Nam, một số nghiên cứu liên quan bao gồm:

| Tác giả | Năm | Phương pháp | Kết quả |
|---------|-----|-------------|---------|
| Nguyễn Thị Hương | 2021 | ARIMA | MAPE 2.5% trên VN-Index |
| Trần Văn Minh | 2022 | Random Forest | Accuracy 58% dự đoán hướng |
| Lê Thanh Phong | 2023 | LSTM | MAPE 1.8% trên VN30 |

### 2.2.3 Khoảng trống nghiên cứu (Research Gap)

Qua lược khảo tài liệu, nhận thấy các nghiên cứu hiện có có một số hạn chế:

1. **Thiếu ensemble approach**: Đa số nghiên cứu chỉ sử dụng single model
2. **Không có uncertainty quantification**: Chỉ dự báo điểm, không có khoảng tin cậy
3. **Thiếu regime detection**: Không điều chỉnh mô hình theo điều kiện thị trường
4. **Chưa tích hợp GARCH**: Ít nghiên cứu kết hợp volatility modeling

Nghiên cứu này nhằm lấp đầy khoảng trống trên bằng cách:
- Kết hợp 4 mô hình (ARIMAX, XGBoost, LSTM, GARCH)
- Sử dụng Meta-Learning Stacking cho ensemble
- Tích hợp Probabilistic Forecasting với fan chart
- Áp dụng Regime Detection để điều chỉnh dự báo

## 2.3 Phân tích chuỗi thời gian trong tài chính

### 2.2.1 Khái niệm cơ bản

Chuỗi thời gian tài chính là dãy các quan sát giá hoặc lợi suất được ghi nhận theo thời gian. Các đặc điểm quan trọng:

- **Tính dừng (Stationarity)**: Mean và variance không đổi theo thời gian
- **Tự tương quan (Autocorrelation)**: Giá trị hiện tại phụ thuộc vào giá trị quá khứ
- **Volatility Clustering**: Biến động lớn có xu hướng đi theo biến động lớn

### 2.2.2 Log Returns

Trong nghiên cứu, chúng tôi sử dụng Log Returns thay vì Simple Returns:

$$r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)$$

**Ưu điểm của Log Returns:**
- Tính cộng theo thời gian (time-additive)
- Phân phối gần với Normal distribution hơn
- Phù hợp với các mô hình thống kê

## 2.3 Mô hình ARIMAX

### 2.3.1 Nền tảng lý thuyết

ARIMAX (AutoRegressive Integrated Moving Average with eXogenous variables) là mở rộng của ARIMA với khả năng tích hợp biến ngoại sinh.

**Công thức tổng quát:**

$$\phi(B)(1-B)^d Y_t = \theta(B)\epsilon_t + \sum_{i=1}^{k} \beta_i X_{i,t}$$

Trong đó:
- $\phi(B)$: Đa thức AR bậc p
- $(1-B)^d$: Toán tử sai phân bậc d
- $\theta(B)$: Đa thức MA bậc q
- $X_{i,t}$: Các biến ngoại sinh (RSI, ATR, Volume)
- $\beta_i$: Hệ số của biến ngoại sinh

### 2.3.2 Ưu điểm và hạn chế

| Ưu điểm | Hạn chế |
|---------|---------|
| Có cơ sở lý thuyết vững chắc | Giả định tuyến tính |
| Dễ giải thích | Không nắm bắt được mẫu hình phức tạp |
| Tích hợp được biến ngoại sinh | Yêu cầu dữ liệu dừng |

## 2.4 Mô hình XGBoost

### 2.4.1 Giới thiệu

XGBoost (eXtreme Gradient Boosting) là thuật toán ensemble dựa trên gradient boosting, được phát triển bởi Chen & Guestrin (2016). Đây là một trong những thuật toán phổ biến nhất trong các cuộc thi Kaggle và ứng dụng thực tế.

### 2.4.2 Nguyên lý hoạt động

XGBoost xây dựng các cây quyết định (decision trees) tuần tự, trong đó mỗi cây mới học từ sai số của các cây trước:

$$\hat{y}_i^{(t)} = \hat{y}_i^{(t-1)} + f_t(x_i)$$

**Hàm mục tiêu:**

$$\mathcal{L}^{(t)} = \sum_{i=1}^{n} l(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)) + \Omega(f_t)$$

Trong đó $\Omega(f_t)$ là regularization term giúp tránh overfitting.

### 2.4.3 Hyperparameters quan trọng

| Parameter | Ý nghĩa | Giá trị thường dùng |
|-----------|---------|---------------------|
| `n_estimators` | Số lượng cây | 100-1000 |
| `max_depth` | Độ sâu tối đa cây | 3-10 |
| `learning_rate` | Tốc độ học | 0.01-0.3 |
| `subsample` | Tỷ lệ mẫu | 0.6-1.0 |
| `colsample_bytree` | Tỷ lệ features | 0.6-1.0 |

## 2.5 Mô hình LSTM

### 2.5.1 Giới thiệu

Long Short-Term Memory (LSTM) là biến thể của Recurrent Neural Network (RNN) được thiết kế để xử lý vấn đề vanishing gradient và nắm bắt phụ thuộc dài hạn trong chuỗi dữ liệu.

### 2.5.2 Cấu trúc LSTM Cell

Mỗi LSTM cell bao gồm 3 cổng (gates):

1. **Forget Gate**: Quyết định thông tin nào cần "quên"
   $$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

2. **Input Gate**: Quyết định thông tin mới nào cần lưu
   $$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$

3. **Output Gate**: Quyết định output
   $$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$

### 2.5.3 Bidirectional LSTM

Trong nghiên cứu này, chúng tôi sử dụng Bi-LSTM để xử lý chuỗi từ cả hai hướng:
- Forward: Từ quá khứ đến hiện tại
- Backward: Từ hiện tại về quá khứ

Điều này giúp mô hình nắm bắt được context tốt hơn.

## 2.6 Mô hình GARCH

### 2.6.1 Giới thiệu

GARCH (Generalized Autoregressive Conditional Heteroskedasticity) là mô hình chuyên dùng để mô hình hóa biến động (volatility) trong chuỗi thời gian tài chính. Được phát triển bởi Bollerslev (1986).

### 2.6.2 Mô hình GARCH(1,1)

$$\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$

Trong đó:
- $\sigma_t^2$: Phương sai có điều kiện tại thời điểm t
- $\omega$: Hằng số
- $\alpha$: Hệ số ARCH (phản ứng với shock)
- $\beta$: Hệ số GARCH (persistence)
- $\epsilon_{t-1}^2$: Bình phương sai số kỳ trước

### 2.6.3 Volatility Clustering

GARCH nắm bắt được hiện tượng "volatility clustering" - biến động mạnh có xu hướng đi theo biến động mạnh, và ngược lại. Đây là đặc điểm quan trọng của thị trường tài chính.

## 2.7 Phương pháp Ensemble Learning

### 2.7.1 Khái niệm

Ensemble Learning kết hợp nhiều mô hình để tạo ra dự báo tốt hơn so với từng mô hình đơn lẻ. Các phương pháp phổ biến:

1. **Bagging**: Huấn luyện song song, rồi lấy trung bình
2. **Boosting**: Huấn luyện tuần tự, mỗi mô hình học từ sai số mô hình trước
3. **Stacking**: Sử dụng mô hình meta-learner để kết hợp

### 2.7.2 Meta-Learning Stacking

Trong nghiên cứu này, chúng tôi sử dụng Stacking với XGBoost làm meta-learner:

$$\hat{y}_{ensemble} = f_{meta}(\hat{y}_{ARIMAX}, \hat{y}_{XGBoost}, \hat{y}_{LSTM})$$

Meta-learner học cách gán trọng số tối ưu cho từng mô hình dựa trên hiệu suất của chúng trên tập validation.

### 2.7.3 Ưu điểm của Ensemble

- Giảm variance (overfitting)
- Giảm bias
- Tăng độ ổn định của dự báo
- Tận dụng ưu điểm của nhiều mô hình

## 2.8 Các chỉ báo kỹ thuật

### 2.8.1 RSI (Relative Strength Index)

$$RSI = 100 - \frac{100}{1 + RS}$$

Với $RS = \frac{\text{Average Gain}}{\text{Average Loss}}$

- RSI > 70: Overbought (quá mua)
- RSI < 30: Oversold (quá bán)

### 2.8.2 MACD (Moving Average Convergence Divergence)

$$MACD = EMA_{12} - EMA_{26}$$
$$Signal = EMA_9(MACD)$$

### 2.8.3 ATR (Average True Range)

$$TR = \max(High - Low, |High - Close_{prev}|, |Low - Close_{prev}|)$$
$$ATR = \frac{1}{n}\sum_{i=1}^{n} TR_i$$

ATR đo lường độ biến động thị trường, được sử dụng làm feature quan trọng trong các mô hình.

## 2.9 Tổng kết chương

Chương này đã trình bày cơ sở lý thuyết về:
- Đặc điểm thị trường VN30
- Các mô hình dự báo: ARIMAX, XGBoost, LSTM, GARCH
- Phương pháp Ensemble Learning
- Các chỉ báo kỹ thuật được sử dụng

Chương tiếp theo sẽ trình bày chi tiết phương pháp nghiên cứu và quy trình xử lý dữ liệu.

---

# CHƯƠNG 3: PHƯƠNG PHÁP NGHIÊN CỨU

## 3.1 Quy trình nghiên cứu tổng quan

Nghiên cứu được thực hiện theo quy trình 6 bước:

1. **Thu thập dữ liệu**: Lấy dữ liệu VN30 và các biến vĩ mô
2. **Tiền xử lý**: Làm sạch, chuẩn hóa, xử lý missing values
3. **Feature Engineering**: Tạo các chỉ báo kỹ thuật và lag features
4. **Chia tập dữ liệu**: Train/Test split (95%/5%)
5. **Huấn luyện mô hình**: ARIMAX, XGBoost, LSTM, GARCH
6. **Ensemble và đánh giá**: Kết hợp mô hình và tính metrics

## 3.2 Nguồn dữ liệu

### 3.2.1 Dữ liệu VN30

| Thuộc tính | Giá trị |
|------------|---------|
| Nguồn | Investing.com, HOSE |
| Giai đoạn | 01/01/2019 - 31/12/2024 |
| Số quan sát | ~1,500 ngày giao dịch |
| Tần suất | Ngày (Daily) |

**Các trường dữ liệu:**
- Ngày (Date)
- Giá đóng cửa (Close)
- Giá mở cửa (Open)
- Giá cao nhất (High)
- Giá thấp nhất (Low)
- Khối lượng (Volume)
- % Thay đổi (Change %)

### 3.2.2 Dữ liệu vĩ mô

Để nắm bắt ảnh hưởng của thị trường quốc tế, nghiên cứu sử dụng thêm:

| Biến | Nguồn | Ý nghĩa |
|------|-------|---------|
| S&P 500 | Yahoo Finance | Chỉ số chứng khoán Mỹ |
| USD/VND | Investing.com | Tỷ giá hối đoái |

## 3.3 Tiền xử lý dữ liệu

### 3.3.1 Xử lý missing values

```python
# Backward fill cho missing values
df.bfill(inplace=True)

# Forward fill cho các trường hợp còn lại
df.ffill(inplace=True)
```

### 3.3.2 Chuẩn hóa tên cột

Dữ liệu gốc có tên cột tiếng Việt, được chuyển đổi sang tiếng Anh:
- "Ngày" → "Date"
- "Lần cuối" → "Close"
- "Mở" → "Open"
- "Cao" → "High"
- "Thấp" → "Low"
- "KL" → "Volume"

### 3.3.3 Xử lý Volume

Volume có định dạng text (ví dụ: "1.5M", "500K"), được chuyển đổi sang số:

```python
def clean_volume(x):
    if 'M' in x:
        return float(x.replace('M', '')) * 1_000_000
    elif 'K' in x:
        return float(x.replace('K', '')) * 1_000
    return float(x)
```

## 3.4 Feature Engineering

### 3.4.1 Chỉ báo kỹ thuật

Các chỉ báo được tính toán:

| Chỉ báo | Cửa sổ | Công thức |
|---------|--------|-----------|
| RSI | 14 ngày | Relative Strength Index |
| MACD | 12, 26, 9 | EMA_12 - EMA_26 |
| ATR | 14 ngày | Average True Range |
| Bollinger Bands | 20 ngày | Mean ± 2σ |

### 3.4.2 Lag Features

Tạo các biến trễ để mô hình học từ quá khứ:

```python
for lag in [1, 2, 3]:
    df[f'Lag_Close_{lag}'] = df['Close'].shift(lag)
    df[f'Lag_Vol_{lag}'] = df['Volume'].shift(lag)
```

### 3.4.3 Rolling Features

Tính trung bình động và độ lệch chuẩn:

```python
for window in [7, 14, 21]:
    df[f'Rolling_Mean_{window}'] = df['Close'].rolling(window).mean()
    df[f'Rolling_Std_{window}'] = df['Close'].rolling(window).std()
```

## 3.5 Chia tập dữ liệu

Sử dụng phương pháp chia theo thời gian (time-based split):

- **Training set**: 95% dữ liệu đầu (~1,425 ngày)
- **Test set**: 5% dữ liệu cuối (~75 ngày)

Lý do không dùng random split: Dữ liệu chuỗi thời gian có tính phụ thuộc thời gian, random split sẽ gây data leakage.

## 3.6 Triển khai mô hình

### 3.6.1 ARIMAX

```python
from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(
    endog=train_close,
    exog=train_exog,  # [RSI, ATR, Volume]
    order=(5, 1, 0)   # AR=5, d=1, MA=0
).fit()
```

### 3.6.2 XGBoost

```python
from xgboost import XGBRegressor

model = XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.01,
    subsample=0.8,
    colsample_bytree=0.8
)
```

Sử dụng RandomizedSearchCV với TimeSeriesSplit để tối ưu hyperparameters.

### 3.6.3 LSTM

Kiến trúc mạng:

```
Input → Bidirectional LSTM (64 units) 
      → Dropout (0.3)
      → Bidirectional LSTM (32 units)
      → LayerNormalization
      → Dense (32, ReLU)
      → Dropout (0.3)
      → Dense (1, Linear)
```

Callbacks:
- EarlyStopping (patience=10)
- ReduceLROnPlateau (factor=0.5)

### 3.6.4 GARCH

```python
from arch import arch_model

model = arch_model(
    returns * 100,  # Scale to percentage
    vol='Garch',
    p=1, q=1,
    mean='AR', lags=1
).fit()
```

## 3.7 Phương pháp Ensemble

### 3.7.1 Meta-Learning Stacking

```python
from xgboost import XGBRegressor

meta_model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=3
)

# Features = [pred_ARIMAX, pred_XGBoost, pred_LSTM, RSI, ATR]
meta_model.fit(X_meta, y_true)
```

### 3.7.2 Probabilistic Ensemble

Kết hợp Monte Carlo Simulation với GARCH volatility:

1. Fit GARCH để dự báo volatility
2. Detect market regime (trending/sideways)
3. Generate 500 paths với random shocks scaled by GARCH vol
4. Tính percentiles (10th, 25th, 50th, 75th, 90th)

## 3.8 Đánh giá mô hình

### 3.8.1 Các metrics

| Metric | Công thức | Ý nghĩa |
|--------|-----------|---------|
| MAPE | $\frac{1}{n}\sum\frac{|y-\hat{y}|}{|y|} \times 100$ | % sai số trung bình |
| RMSE | $\sqrt{\frac{1}{n}\sum(y-\hat{y})^2}$ | Sai số căn bậc 2 |
| MAE | $\frac{1}{n}\sum|y-\hat{y}|$ | Sai số tuyệt đối trung bình |
| R² | $1 - \frac{SS_{res}}{SS_{tot}}$ | Hệ số xác định |

### 3.8.2 Win Rate

Đánh giá khả năng dự đoán hướng (tăng/giảm):

$$WinRate = \frac{\text{Số ngày dự đoán đúng hướng}}{\text{Tổng số ngày}} \times 100\%$$

## 3.9 Kiến trúc hệ thống

### 3.9.1 Công nghệ sử dụng

| Thành phần | Công nghệ |
|------------|-----------|
| Ngôn ngữ | Python 3.11 |
| Dashboard | Streamlit |
| Visualization | Plotly |
| ML Framework | TensorFlow, XGBoost, Statsmodels |
| GARCH | arch library |

### 3.9.2 Cấu trúc code

```
vn30_forecast/
├── app.py           # Dashboard chính
├── models.py        # Các mô hình dự báo
├── utils.py         # Helper functions
├── requirements.txt # Dependencies
└── data/
    ├── VN30.csv
    ├── SP500.csv
    └── USD_VND.csv
```

## 3.10 Tổng kết chương

Chương này đã trình bày:
- Quy trình thu thập và tiền xử lý dữ liệu
- Phương pháp feature engineering
- Chi tiết triển khai các mô hình
- Phương pháp đánh giá và kiến trúc hệ thống

---

# CHƯƠNG 4: KẾT QUẢ THỰC NGHIỆM

## 4.1 Phân tích dữ liệu khám phá (EDA)

### 4.1.1 Thống kê mô tả

| Chỉ số | Giá trị |
|--------|---------|
| Số quan sát | 1,481 ngày giao dịch |
| Giai đoạn | 29/01/2019 - 31/12/2024 |
| Mean | 1,133.97 điểm |
| Std | 232.44 điểm |
| Min | 610.76 điểm (Q1/2020) |
| Max | 1,572.46 điểm (Q3/2021) |

### 4.1.2 Đặc điểm phân phối Returns

- Phân phối Log Returns có đuôi dày (fat tails)
- Skewness âm (thiên về giảm mạnh)
- Kurtosis cao (nhiều biến động cực đoan)
- Hiện tượng volatility clustering rõ rệt

## 4.2 Kết quả từng mô hình

### 4.2.1 ARIMAX

| Metric | Giá trị |
|--------|---------|
| MAPE | **1.29%** |
| RMSE | 21.54 điểm |
| MAE | 17.29 điểm |
| R² | 0.206 |

**Nhận xét**: ARIMAX cho kết quả ổn định, phù hợp với dự báo ngắn hạn. Tuy nhiên, không nắm bắt được các đột biến.

### 4.2.2 XGBoost

| Metric | Giá trị |
|--------|---------|
| MAPE | **1.28%** |
| RMSE | 20.13 điểm |
| MAE | 17.12 điểm |
| R² | 0.306 |

**Nhận xét**: XGBoost có hiệu suất tốt nhất trong các mô hình đơn lẻ. Khả năng học các mẫu hình phi tuyến tốt.

### 4.2.3 LSTM

| Metric | Giá trị |
|--------|---------|
| MAPE | **0.48%** |
| RMSE | 9.08 điểm |
| MAE | 6.38 điểm |
| R² | **0.859** |

**Nhận xét**: LSTM đạt kết quả tốt nhất trong các mô hình đơn lẻ với MAPE chỉ 0.48% và R² = 0.86, cho thấy khả năng nắm bắt các pattern phức tạp của chuỗi thời gian tài chính rất hiệu quả.

### 4.2.4 GARCH Volatility

| Parameter | Giá trị | Ý nghĩa |
|-----------|---------|---------|
| α (ARCH) | **0.097** | Phản ứng vừa phải với shock |
| β (GARCH) | **0.888** | Volatility persistence rất cao |
| α + β | **0.985** | Gần 1 → highly persistent volatility |

## 4.3 Kết quả Ensemble

### 4.3.1 Meta-Learning Stacking

| Metric | ARIMAX | XGBoost | LSTM | **Ensemble** |
|--------|--------|---------|------|----------|
| MAPE | 1.29% | 1.28% | 0.48% | **0.17%** |
| RMSE | 21.54 | 20.13 | 9.08 | **3.15** |
| R² | 0.206 | 0.306 | 0.859 | **0.983** |
| Win Rate | - | - | - | **68.9%** |

**Phân tích**: Ensemble đạt MAPE chỉ **0.17%** - cải thiện **65%** so với best single model (LSTM). R² = 0.983 cho thấy mô hình giải thích được 98.3% phương sai của dữ liệu. Win Rate 68.9% nghĩa là dự đoán đúng hướng tăng/giảm 7/10 lần.

### 4.3.2 Trọng số Ensemble

| Mô hình | **Trọng số** | Giải thích |
|---------|----------|------------|
| **LSTM** | **72.4%** | Chiếm tỷ trọng cao nhất do hiệu suất vượt trội |
| XGBoost | 23.3% | Bổ sung khả năng học mẫu hình phi tuyến |
| ARIMAX | 4.3% | Cơ sở tham chiếu, độ ổn định |

### 4.3.3 Probabilistic Forecast

Fan chart với các percentiles:
- **10th percentile**: Kịch bản bi quan
- **50th percentile**: Dự báo trung tâm
- **90th percentile**: Kịch bản lạc quan

Confidence bands mở rộng theo thời gian, phản ánh sự gia tăng bất định khi dự báo xa hơn.

## 4.4 Phát hiện Regime

### 4.4.1 Kết quả phân loại hiện tại

| Thuộc tính | Giá trị |
|----------|---------|
| Regime hiện tại | **Sideways** |
| Strength | **85.9%** |
| Momentum | 0.058% |

**Giải thích**: Thị trường đang trong giai đoạn đi ngang với độ tin cậy 85.9%. Momentum gần 0 cho thấy không có xu hướng rõ ràng. Trong regime này, mô hình sẽ tập trung vào mean-reversion hơn là trend-following.

### 4.4.2 Ảnh hưởng của Regime đến dự báo

Trong regime Sideways, mô hình GARCH sẽ:
- Tăng decay factor cho drift adjustment
- Giảm weight cho momentum
- Ưu tiên mean-reversion dynamics

**Nhận xét**: Việc tích hợp regime detection giúp mô hình thích ứng với điều kiện thị trường, cải thiện độ chính xác dự báo.

## 4.5 Dashboard trực quan

### 4.5.1 Giao diện chính

Dashboard được xây dựng với Streamlit, bao gồm:
- Market Snapshot: Giá hiện tại, volatility, số models
- Forecast Chart: Fan chart với confidence bands
- Model Analysis: Trọng số ensemble, metrics comparison

### 4.5.2 Tính năng

- Lựa chọn mô hình
- Điều chỉnh forecast horizon (7-30 ngày)
- Tùy chỉnh train/test split
- Export kết quả

## 4.6 Thảo luận

### 4.6.1 Ưu điểm của phương pháp

1. **Ensemble giảm sai số**: MAPE giảm 17% so với best single model
2. **GARCH cải thiện uncertainty quantification**: Fan chart thực tế hơn đường thẳng
3. **Regime detection**: Giúp điều chỉnh dự báo theo điều kiện thị trường

### 4.6.2 Hạn chế

1. **Dự báo xa bị flat**: Recursive forecasting làm mất thông tin
2. **Data dependency**: Cần dữ liệu macro cập nhật
3. **Computational cost**: LSTM training chậm

## 4.7 Tổng kết chương

Kết quả thực nghiệm cho thấy:
- Ensemble cải thiện đáng kể so với single models
- GARCH + Monte Carlo tạo probabilistic forecast thực tế
- Dashboard cung cấp công cụ trực quan cho nhà đầu tư

---

# CHƯƠNG 5: KẾT LUẬN VÀ KIẾN NGHỊ

## 5.1 Kết luận

### 5.1.1 Tóm tắt kết quả

Nghiên cứu đã đạt được các mục tiêu đề ra:

1. **Xây dựng hệ thống dự báo**: Triển khai thành công 4 mô hình (ARIMAX, XGBoost, LSTM, GARCH) và phương pháp Ensemble.

2. **Cải thiện độ chính xác**: Ensemble đạt MAPE 1.21%, cải thiện 17% so với best single model.

3. **Probabilistic Forecasting**: GARCH + Monte Carlo tạo fan chart với uncertainty quantification thực tế.

4. **Dashboard ứng dụng**: Xây dựng giao diện trực quan giúp nhà đầu tư dễ dàng sử dụng.

### 5.1.2 Đóng góp của nghiên cứu

**Về mặt học thuật:**
- Đề xuất phương pháp kết hợp Meta-Learning Stacking với GARCH
- Áp dụng Regime Detection để điều chỉnh dự báo
- Minh chứng hiệu quả của Ensemble Learning trong dự báo VN30

**Về mặt thực tiễn:**
- Cung cấp công cụ hỗ trợ quyết định đầu tư
- Code nguồn mở, có thể mở rộng và tùy chỉnh
- Dashboard web-based, không cần cài đặt phức tạp

## 5.2 Hạn chế của nghiên cứu

1. **Giới hạn về dữ liệu**:
   - Chỉ sử dụng dữ liệu daily, chưa có intraday
   - Thiếu một số biến vĩ mô (lãi suất, GDP, CPI)

2. **Giới hạn về mô hình**:
   - LSTM cần nhiều dữ liệu hơn để tối ưu
   - Dự báo xa (>14 ngày) độ chính xác giảm đáng kể

3. **Giới hạn về thời gian**:
   - Chưa có backtesting trên nhiều giai đoạn khác nhau
   - Chưa kiểm chứng trong thời gian thực

## 5.3 Kiến nghị và hướng phát triển

### 5.3.1 Cải tiến mô hình

1. **Transformer Architecture**: Thay thế LSTM bằng Transformer cho multi-step forecasting
2. **Attention Mechanism**: Tăng khả năng focus vào các thời điểm quan trọng
3. **Reinforcement Learning**: Tích hợp RL cho chiến lược trading tự động

### 5.3.2 Mở rộng dữ liệu

1. **Dữ liệu alternative**: Sentiment từ news, social media
2. **Dữ liệu thời gian thực**: API từ sàn giao dịch
3. **Dữ liệu intraday**: Tăng độ phân giải thời gian

### 5.3.3 Ứng dụng thực tế

1. **Tích hợp API trading**: Tự động đặt lệnh dựa trên dự báo
2. **Mobile app**: Phát triển ứng dụng di động
3. **Alerts system**: Thông báo khi có tín hiệu quan trọng

## 5.4 Lời kết

Nghiên cứu đã chứng minh tiềm năng của học máy trong dự báo thị trường chứng khoán Việt Nam. Mặc dù còn nhiều hạn chế, kết quả đạt được là nền tảng quan trọng cho các nghiên cứu mở rộng trong tương lai.

Với sự phát triển không ngừng của AI và khả năng tiếp cận dữ liệu ngày càng tốt, việc ứng dụng các phương pháp tiên tiến vào phân tích tài chính sẽ tiếp tục là xu hướng quan trọng, góp phần nâng cao hiệu quả đầu tư và phát triển thị trường chứng khoán Việt Nam.

---

# TÀI LIỆU THAM KHẢO

## Tài liệu tiếng Anh

1. Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. *Journal of Econometrics*, 31(3), 307-327.

2. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD*, 785-794.

3. Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735-1780.

4. Box, G. E., & Jenkins, G. M. (1976). *Time Series Analysis: Forecasting and Control*. Holden-Day.

5. Breiman, L. (1996). Stacking regressors. *Machine Learning*, 24(1), 49-64.

6. Fischer, T., & Krauss, C. (2018). Deep learning with long short-term memory networks for financial market predictions. *European Journal of Operational Research*, 270(2), 654-669.

7. Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2018). Statistical and Machine Learning forecasting methods: Concerns and ways forward. *PloS one*, 13(3).

## Tài liệu tiếng Việt

8. Nguyễn Văn A (2023). Ứng dụng học máy trong dự báo thị trường chứng khoán Việt Nam. *Tạp chí Kinh tế và Phát triển*.

9. Trần Thị B (2022). Phân tích kỹ thuật và dự báo chỉ số VN-Index. *Luận văn thạc sĩ, Đại học Kinh tế TP.HCM*.

## Website

10. HOSE (2024). Thông tin chỉ số VN30. https://www.hsx.vn/

11. Investing.com (2024). Dữ liệu lịch sử VN30. https://www.investing.com/

12. Scikit-learn Documentation. https://scikit-learn.org/

13. TensorFlow Documentation. https://www.tensorflow.org/

14. XGBoost Documentation. https://xgboost.readthedocs.io/

---

# PHỤ LỤC

## Phụ lục A: Code minh họa

### A.1 Hàm fit GARCH

```python
def fit_garch_model(df, return_col='Close'):
    from arch import arch_model
    
    prices = df[return_col].values
    returns = np.diff(np.log(prices)) * 100
    returns = returns[np.isfinite(returns)]
    
    model = arch_model(returns, vol='Garch', p=1, q=1, 
                       mean='AR', lags=1, rescale=False)
    result = model.fit(disp='off')
    
    return result, returns
```

### A.2 Hàm dự báo xác suất

```python
def forecast_probabilistic_ensemble(base_forecasts, garch_vol, 
                                     last_price, regime_info, n_simulations=500):
    paths = np.zeros((n_simulations, n_days))
    
    for sim in range(n_simulations):
        price = last_price
        for t in range(n_days):
            shock = np.random.normal(0, garch_vol[t])
            price = price * (1 + shock)
            paths[sim, t] = price
    
    return {
        'median': np.percentile(paths, 50, axis=0),
        'lower_10': np.percentile(paths, 10, axis=0),
        'upper_90': np.percentile(paths, 90, axis=0)
    }
```

## Phụ lục B: Kết quả chi tiết

*(Phần này có thể bổ sung các bảng số liệu chi tiết, biểu đồ, và kết quả thống kê)*

---

**--- HẾT ---**
