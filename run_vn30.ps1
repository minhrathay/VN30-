$venvPath = "C:\Users\Admin\.gemini\antigravity\scratch\vn30_env"

Write-Host "Starting VN30 Analytics Dashboard..." -ForegroundColor Cyan
Write-Host "Installing/Verifying Dependencies..." -ForegroundColor Gray
& "$venvPath\Scripts\pip" install -r requirements.txt
Write-Host "Launching App..." -ForegroundColor Green
& "$venvPath\Scripts\streamlit" run app.py
