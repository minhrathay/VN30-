# Run the VN30 Dashboard
$env:PYTHONPATH = "C:\Users\Admin\.gemini\antigravity\scratch"
$venvPath = ".\vn30_env"
$appPath = ".\app.py"

Write-Host "Starting VN30 Dashboard..." -ForegroundColor Green
& "$venvPath\Scripts\streamlit.exe" run $appPath
