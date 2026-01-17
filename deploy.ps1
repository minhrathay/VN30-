# Deploy to GitHub
# Make sure you have Git installed and you are logged in.

Write-Host "========================================"
Write-Host "VN30 Forecasting Dashboard - Deploy"
Write-Host "========================================"

# Update dependencies
Write-Host "`n[1] Checking dependencies..."
if (Test-Path ".\vn30_env\Scripts\pip.exe") {
    Write-Host "    Installing new ML dependencies..."
    .\vn30_env\Scripts\pip.exe install pmdarima optuna -q
    Write-Host "    Dependencies OK"
}

Write-Host "`n[2] Initializing Git Repository..."
Compare-Object -ErrorAction SilentlyContinue .gitignore .gitignore_backup
if (-not (Test-Path .gitignore)) {
    Write-Host "    Creating .gitignore..."
    Set-Content .gitignore "__pycache__/`n*.pyc`nvn30_env/`ntest_output*.txt`n*.log"
}

git init
git add .

# Commit with timestamp
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm"
$commitMsg = "ML Improvements: LSTM Attention, Optuna, Walk-Forward Validation [$timestamp]"
git commit -m $commitMsg

git branch -M main
git remote add origin https://github.com/minhrathay/VN30-.git 2>$null

Write-Host "`n[3] Pushing to GitHub..."
git push -u origin main

Write-Host "`n========================================"
Write-Host "Deployment Completed!"
Write-Host "========================================"
Pause
