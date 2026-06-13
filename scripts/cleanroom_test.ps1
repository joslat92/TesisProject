# Prueba de fuego de reproducibilidad (Apéndice B): clon limpio + entorno
# desde cero + run_all --quick + comparación de clásicos contra RC2.
$ErrorActionPreference = 'Stop'
$repo = "C:\Users\User\TesisProject"
$tmp = Join-Path $env:TEMP "tesis_cleanroom_20260612"
$sysPython = "C:\Users\User\AppData\Local\Programs\Python\Python311\python.exe"

if (Test-Path $tmp) { Remove-Item -Recurse -Force $tmp }

Write-Host "[1/5] Clonando a $tmp ..."
git clone --quiet --branch reestructura-dic2025 $repo $tmp
if ($LASTEXITCODE -ne 0) { Write-Host "FALLO clone"; exit 1 }

Write-Host "[2/5] venv desde cero..."
& $sysPython -m venv (Join-Path $tmp ".venv")
if ($LASTEXITCODE -ne 0) { Write-Host "FALLO venv"; exit 1 }
$py = Join-Path $tmp ".venv\Scripts\python.exe"

Write-Host "[3/5] pip install -r requirements.txt (descarga torch CPU)..."
& $py -m pip install --upgrade pip --quiet --disable-pip-version-check
& $py -m pip install -r (Join-Path $tmp "requirements.txt") --quiet --disable-pip-version-check
if ($LASTEXITCODE -ne 0) { Write-Host "FALLO pip install"; exit 1 }
& $py -c "import torch, pandas, statsmodels; print('   deps OK: torch', torch.__version__)"

Write-Host "[4/5] run_all.py --quick en el clon..."
Push-Location $tmp
& $py run_all.py --quick
$rc = $LASTEXITCODE
Pop-Location
if ($rc -ne 0) { Write-Host "FALLO run_all --quick (rc=$rc)"; exit 1 }

Write-Host "[5/5] Comparando metricas CLASICAS del clon vs RC2 del repo..."
& $py -c @"
import pandas as pd, sys
clasicos = ['RW', 'ARIMA', 'ARIMAX', 'SARIMAX']
fallos = 0
for nombre, par in [
    ('metrics_OOS.csv', (r'$tmp\reports\data\metrics_OOS.csv', r'$repo\reports\data\metrics_OOS.csv')),
    ('metrics_OOS_2025.csv', (r'$tmp\reports\data\metrics_OOS_2025.csv', r'$repo\reports\data\metrics_OOS_2025.csv')),
]:
    a = pd.read_csv(par[0]); b = pd.read_csv(par[1])
    a = a[a.Model.isin(clasicos)].sort_values(['Horizon','Model']).reset_index(drop=True)
    b = b[b.Model.isin(clasicos)].sort_values(['Horizon','Model']).reset_index(drop=True)
    cols = [c for c in ['Horizon','Model','RMSE','MAE','MDA'] if c in a.columns and c in b.columns]
    iguales = a[cols].equals(b[cols])
    print(f'   {nombre}: clasicos clon == RC2 ->', iguales)
    if not iguales:
        fallos += 1
        diff = (a[cols].select_dtypes('number') - b[cols].select_dtypes('number')).abs().max()
        print(diff)
sys.exit(1 if fallos else 0)
"@
if ($LASTEXITCODE -ne 0) { Write-Host "FALLO comparacion clasicos"; exit 1 }

Write-Host "CLEANROOM_OK"
