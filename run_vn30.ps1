$venvPath = "C:\Users\Admin\.gemini\antigravity\scratch\vn30_env"
$scriptPath = "C:\Users\Admin\.gemini\antigravity\scratch\vn30_forecast_fixed.py"
Write-Host "Starting VN30 Forecast..." -ForegroundColor Cyan
& "$venvPath\Scripts\python.exe" $scriptPath
