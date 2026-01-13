# Deploy to GitHub
# Make sure you have Git installed and you are logged in.

Write-Host "Initializing Git Repository..."
diff -ErrorAction SilentlyContinue .gitignore .gitignore_backup
if (-not (Test-Path .gitignore)) {
    Write-Host "Creating .gitignore..."
    Set-Content .gitignore "__pycache__/`n*.pyc`nvn30_env/"
}

git init
git add .
git commit -m "Initial commit - VN30 Forecasting Dashboard"
git branch -M main
git remote add origin https://github.com/minhrathay/VN30-.git

Write-Host "Pushing to GitHub..."
git push -u origin main

Write-Host "Deployment Script Completed."
Pause
