# Check if python is available
if (-not (Get-Command "python" -ErrorAction SilentlyContinue)) {
    Write-Host "錯誤: 找不到 Python。請確認您已安裝 Python 並將其加入 PATH 環境變數。" -ForegroundColor Red
    exit 1
}

# Create virtual environment if it doesn't exist
if (-not (Test-Path "venv")) {
    Write-Host "正在建立虛擬環境 (venv)..." -ForegroundColor Cyan
    python -m venv venv
} else {
    Write-Host "虛擬環境 (venv) 已存在。" -ForegroundColor Yellow
}

# Activate virtual environment
Write-Host "正在啟動虛擬環境..." -ForegroundColor Cyan
. .\venv\Scripts\Activate.ps1

# Upgrade pip
Write-Host "正在升級 pip..." -ForegroundColor Cyan
python -m pip install --upgrade pip

# Install GPU PyTorch (CUDA 11.8)
Write-Host "正在安裝 PyTorch (GPU/CUDA 11.8)..." -ForegroundColor Cyan
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install requirements
if (Test-Path "requirements.txt") {
    Write-Host "正在安裝其他相依套件 (requirements.txt)..." -ForegroundColor Cyan
    pip install -r requirements.txt
    Write-Host "✅ 安裝完成！" -ForegroundColor Green
} else {
    Write-Host "⚠️ 找不到 requirements.txt，跳過套件安裝。" -ForegroundColor Yellow
}

Write-Host "`nTo activate this environment in your terminal, run:" -ForegroundColor White
Write-Host "  .\venv\Scripts\Activate.ps1" -ForegroundColor Cyan
