# Regeneración post-fix del bug y_true off-by-seq_len (2026-06-11).
# Borra artefactos LSTM obsoletos y re-corre WF, robustez 2025, multi-semilla
# y toda la cadena de evaluación/figuras. Para en el primer fallo.
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = ".venv\Scripts\python.exe"

Write-Host "[1/7] Borrando preds LSTM WF obsoletos..."
Remove-Item outputs\preds\WF\preds_T*LSTM*_block*.csv -Force

Write-Host "[2/7] Walk-forward (re-genera 336 archivos)..."
& $py src\stages\14_walkforward.py
if ($LASTEXITCODE -ne 0) { Write-Host "FALLO 14"; exit 1 }

Write-Host "[3/7] Robustez 2025..."
& $py src\stages\16_robustez_2025.py
if ($LASTEXITCODE -ne 0) { Write-Host "FALLO 16"; exit 1 }

Write-Host "[4/7] Multi-semilla (120 entrenamientos)..."
& $py src\stages\15_multiseed_lstm.py
if ($LASTEXITCODE -ne 0) { Write-Host "FALLO 15"; exit 1 }

Write-Host "[5/7] Metricas WF..."
& $py src\stages\24_wf_metrics.py
if ($LASTEXITCODE -ne 0) { Write-Host "FALLO 24"; exit 1 }

Write-Host "[6/7] Evaluacion consolidada (gate completo)..."
& $py src\stages\20_evaluate_stats.py
if ($LASTEXITCODE -ne 0) { Write-Host "FALLO 20"; exit 1 }

Write-Host "[7/7] Figuras canonicas..."
& $py src\stages\30_make_figures.py
if ($LASTEXITCODE -ne 0) { Write-Host "FALLO 30"; exit 1 }

Write-Host "RERUN_POSTFIX_OK"
