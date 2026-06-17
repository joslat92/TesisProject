# Verificación final: run_all.py --quick en un clon LIMPIO de cierre-auditoria
# (deja el árbol de trabajo intacto; usa el numpy 2.3.3 ya pineado). Copia el
# log a logs/run_all_quick_2026-06-16.log del repo principal.
$ErrorActionPreference = 'Stop'
$repo = "C:\Users\User\TesisProject"
$tmp = Join-Path $env:TEMP "tesis_runall_quick"
$sysPython = "C:\Users\User\AppData\Local\Programs\Python\Python311\python.exe"
$logDst = Join-Path $repo "logs\run_all_quick_2026-06-16.log"

if (Test-Path $tmp) { Remove-Item -Recurse -Force $tmp }

Write-Host "[1/4] Clonando cierre-auditoria a $tmp ..."
git clone --quiet --branch cierre-auditoria $repo $tmp
if ($LASTEXITCODE -ne 0) { Write-Host "RUNALLQUICK_FAIL: clone"; exit 1 }

Write-Host "[2/4] venv + install (numpy 2.3.3 pineado)..."
& $sysPython -m venv (Join-Path $tmp ".venv")
$py = Join-Path $tmp ".venv\Scripts\python.exe"
& $py -m pip install --upgrade pip --quiet --disable-pip-version-check
& $py -m pip install -r (Join-Path $tmp "requirements.txt") --quiet --disable-pip-version-check
if ($LASTEXITCODE -ne 0) { Write-Host "RUNALLQUICK_FAIL: install"; exit 1 }

Write-Host "[3/4] run_all.py --quick (log -> $logDst)..."
Push-Location $tmp
& $py run_all.py --quick *>&1 | Tee-Object -FilePath $logDst
$rc = $LASTEXITCODE
Pop-Location

Write-Host "[4/4] Resultado:"
if (Select-String -Path $logDst -Pattern "RUN_ALL_OK" -Quiet) {
    Write-Host "RUNALLQUICK_OK"
} else {
    Write-Host "RUNALLQUICK_FAIL: sin RUN_ALL_OK (rc=$rc)"
    exit 1
}
