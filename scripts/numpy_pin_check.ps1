# Validación del nuevo pin de numpy (priority 2): venv limpio + install desde
# requirements.txt local + pytest + diff de determinismo de los modelos clásicos.
$ErrorActionPreference = 'Stop'
$repo = "C:\Users\User\TesisProject"
$tmp = Join-Path $env:TEMP "tesis_numpy_check"
$sysPython = "C:\Users\User\AppData\Local\Programs\Python\Python311\python.exe"

if (Test-Path $tmp) { Remove-Item -Recurse -Force $tmp }
New-Item -ItemType Directory -Force $tmp | Out-Null

Write-Host "[1/5] venv limpio..."
& $sysPython -m venv (Join-Path $tmp ".venv")
$py = Join-Path $tmp ".venv\Scripts\python.exe"

Write-Host "[2/5] pip install -r requirements.txt (local)..."
& $py -m pip install --upgrade pip --quiet --disable-pip-version-check
& $py -m pip install -r (Join-Path $repo "requirements.txt") --quiet --disable-pip-version-check
if ($LASTEXITCODE -ne 0) { Write-Host "NUMPYCHECK_FAIL: pip install"; exit 1 }
& $py -c "import numpy; print('   numpy instalado:', numpy.__version__)"

Write-Host "[3/5] pytest tests/ con el nuevo numpy..."
Push-Location $repo
& $py -m pytest tests/ -q
$rcPytest = $LASTEXITCODE
Pop-Location
if ($rcPytest -ne 0) { Write-Host "NUMPYCHECK_FAIL: pytest"; exit 1 }

Write-Host "[4/5] Regenerando predicciones CLASICAS con el nuevo numpy..."
Push-Location $repo
& $py src\stages\10_baselines.py    | Out-Null
& $py src\stages\11_train_arima.py  | Out-Null
& $py src\stages\13_train_sarimax.py | Out-Null
Pop-Location

Write-Host "[5/5] Diff de determinismo (preds clasicas vs RC2 committeado)..."
Push-Location $repo
$diff = git status --short "outputs/preds/OOS/preds_T1_RW.csv" "outputs/preds/OOS/preds_T20_ARIMA.csv" "outputs/preds/OOS/preds_T20_ARIMAX.csv" "outputs/preds/OOS/preds_T20_SARIMAX.csv" "outputs/preds/OOS/preds_T5_ARIMA.csv"
$diffAll = git status --short "outputs/preds/OOS/"
git checkout -- "outputs/preds/OOS/" 2>$null
Pop-Location

if ([string]::IsNullOrWhiteSpace($diffAll)) {
    Write-Host "   Predicciones clasicas: IDENTICAS a RC2 (numpy 2.3.3 reproduce los numeros)"
    Write-Host "NUMPYCHECK_OK"
} else {
    Write-Host "   ATENCION: cambiaron predicciones:"
    Write-Host $diffAll
    Write-Host "NUMPYCHECK_NUMBERS_CHANGED"
    exit 2
}
