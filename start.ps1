$venvPath = ".\vn30_env"
$appPath = ".\app.py"

Write-Host "Starting VN30 Quant Terminal..." -ForegroundColor Green
& "$venvPath\Scripts\streamlit.exe" run $appPath --server.headless=true
