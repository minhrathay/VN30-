import pandas as pd
import numpy as np
import random

# Load VN30 data to get dates
df_vn30 = pd.read_csv('Dữ liệu Lịch sử VN 30.csv')
df_vn30['Date'] = pd.to_datetime(df_vn30['Ngày'], format='%d/%m/%Y')
dates = df_vn30['Date'].sort_values().values

# Generate SP500 Data
# Start around 4500, correlation with VN30 direction roughly
np.random.seed(42)
sp500_close = [4500]
for i in range(1, len(dates)):
    change = np.random.normal(0, 0.01) # 1% volatility
    sp500_close.append(sp500_close[-1] * (1 + change))

df_sp500 = pd.DataFrame({
    'Date': dates,
    'Close': sp500_close,
    'Open': sp500_close, # Simplify
    'High': sp500_close,
    'Low': sp500_close,
    'Volume': 1000000
})
# Save SP500 (standard format usually YYYY-MM-DD or same as input?)
# Let's use YYYY-MM-DD for external standard
df_sp500.to_csv('SP500.csv', index=False, date_format='%Y-%m-%d')

# Generate USD/VND Data
# Start 23500, slight upward drift
usdvnd_close = [23500]
for i in range(1, len(dates)):
    change = np.random.normal(0.0001, 0.002) # drift up
    usdvnd_close.append(usdvnd_close[-1] * (1 + change))

df_usdvnd = pd.DataFrame({
    'Date': dates,
    'Close': usdvnd_close
})
df_usdvnd.to_csv('USD_VND.csv', index=False, date_format='%Y-%m-%d')

print("Mock data generated: SP500.csv, USD_VND.csv")
