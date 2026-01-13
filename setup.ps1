# VN30 Auto Setup - Simple Version

Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "  VN30 FORECAST - AUTO SETUP" -ForegroundColor Cyan  
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""

$pythonVersion = "3.11.9"
$downloadUrl = "https://www.python.org/ftp/python/$pythonVersion/python-$pythonVersion-amd64.exe"
$installerPath = "$env:TEMP\python-installer.exe"
$pythonPath = "$env:LOCALAPPDATA\Programs\Python\Python311"

# Step 1: Check Python 3.11
Write-Host "[1/5] Checking Python 3.11..." -ForegroundColor Yellow

if (Test-Path "$pythonPath\python.exe") {
    Write-Host "   OK: Python 3.11 found" -ForegroundColor Green
} else {
    Write-Host "   Downloading Python $pythonVersion..." -ForegroundColor Cyan
    
    try {
        Invoke-WebRequest -Uri $downloadUrl -OutFile $installerPath -UseBasicParsing
        Write-Host "   Download complete" -ForegroundColor Green
        
        Write-Host "   Installing Python 3.11..." -ForegroundColor Cyan
        Start-Process -FilePath $installerPath -ArgumentList "/quiet", "InstallAllUsers=0", "PrependPath=0" -Wait
        
        Write-Host "   Python 3.11 installed!" -ForegroundColor Green
        Remove-Item $installerPath -Force
    } catch {
        Write-Host "   ERROR: $_" -ForegroundColor Red
        exit 1
    }
}

# Step 2: Create virtual environment
Write-Host ""
Write-Host "[2/5] Creating virtual environment..." -ForegroundColor Yellow
$venvPath = "C:\Users\Admin\.gemini\antigravity\scratch\vn30_env"

if (Test-Path "$venvPath\Scripts\python.exe") {
    Write-Host "   Virtual environment exists" -ForegroundColor Green
} else {
    & "$pythonPath\python.exe" -m venv $venvPath
    Write-Host "   Virtual environment created" -ForegroundColor Green
}

# Step 3: Upgrade pip
Write-Host ""
Write-Host "[3/5] Upgrading pip..." -ForegroundColor Yellow
& "$venvPath\Scripts\python.exe" -m pip install --upgrade pip --quiet
Write-Host "   Pip upgraded" -ForegroundColor Green

# Step 4: Install packages
Write-Host ""
Write-Host "[4/5] Installing packages (5-10 minutes)..." -ForegroundColor Yellow
$reqFile = "C:\Users\Admin\.gemini\antigravity\scratch\requirements.txt"

if (Test-Path $reqFile) {
    & "$venvPath\Scripts\pip.exe" install -r $reqFile
    Write-Host "   Packages installed!" -ForegroundColor Green
} else {
    Write-Host "   ERROR: requirements.txt not found" -ForegroundColor Red
    exit 1
}

# Step 5: Create helper scripts
Write-Host ""
Write-Host "[5/5] Creating helper scripts..." -ForegroundColor Yellow

$runScript = @"
`$venvPath = "C:\Users\Admin\.gemini\antigravity\scratch\vn30_env"
`$scriptPath = "C:\Users\Admin\.gemini\antigravity\scratch\vn30_forecast_fixed.py"
Write-Host "Starting VN30 Forecast..." -ForegroundColor Cyan
& "`$venvPath\Scripts\python.exe" `$scriptPath
"@

Set-Content -Path "C:\Users\Admin\.gemini\antigravity\scratch\run_vn30.ps1" -Value $runScript
Write-Host "   Created run_vn30.ps1" -ForegroundColor Green

# Done
Write-Host ""
Write-Host "=====================================" -ForegroundColor Green
Write-Host "  SETUP COMPLETE!" -ForegroundColor Green
Write-Host "=====================================" -ForegroundColor Green
Write-Host ""
Write-Host "NEXT STEPS:" -ForegroundColor Yellow
Write-Host "1. Place CSV file: 'Du lieu Lich su VN 30.csv' in scratch folder" -ForegroundColor White
Write-Host "2. Run: .\run_vn30.ps1" -ForegroundColor White
Write-Host ""
