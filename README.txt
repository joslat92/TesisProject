# TesisProject – Series‑Time Forecasting with ARIMA & LSTM


---

## 1. Repository layout

```
.
├── data/                  # CSVs ready to use (main: df_final_ready_plus_vix.csv)
├── models/
│   ├── ARIMA/             # 53‑fold CV artefacts
│   ├── SARIMA/
│   ├── LSTM_Plain/        # baseline single‑feature LSTM
│   ├── LSTM_VIX/
│   ├── LSTM_PLAIN_TUNED/  # tuned hyper‑params
│   ├── LSTM_VIX_TUNED/
│   ├── LSTM_HYBRID/       # LSTM + AR residual model
│   └── archive_models.zip # heavy .keras weights (git‑LFS avoided)
├── outputs/
│   ├── tuning_log*.csv    # grid‑search logs
│   └── dm_results.csv     # Diebold‑Mariano pairwise table
├── figs/                  # final figures for the thesis
│   ├── error_boxplot_final.png
│   └── time_series_final.png
├── src/                   # all python scripts
│   ├── train_arima.py, train_sarima.py
│   ├── tune_lstm.py, tune_lstm_vix.py
│   ├── infer_lstm_cv.py
│   ├── dm_test.py
│   └── utils5_estable.py  # helper functions
├── requirements.txt
└── README.md             

---

## 2. Quick‑start

```bash
# clone & create env
git clone https://github.com/joslat92/TesisProject.git
cd TesisProject
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\Activate
pip install -r requirements.txt

# reproduce statistical baselines
python src/train_arima.py
python src/train_sarima.py

# hyper‑parameter search (≈30 min on CPU)
python src/tune_lstm.py         # plain LSTM
python src/tune_lstm_vix.py     # LSTM with VIX

# cross‑validated inference & CSV predictions
python src/infer_lstm_cv.py models/LSTM_PLAIN_TUNED   data/df_final_ready_plus_vix.csv Target_Price 40
python src/infer_lstm_cv.py models/LSTM_VIX_TUNED     data/df_final_ready_plus_vix.csv Target_Price 40

# Diebold‑Mariano comparison (all pairs)
python src/dm_test.py
```

---

## 3. Results snapshot (05 Jul 2025)

| Rank | Model                | DM vs. best | p‑value | Common obs |
|-----:|----------------------|------------:|--------:|-----------:|
| 1    | **LSTM_VIX_TUNED**   | –           | –       | 2 521 |
| 2    | LSTM_PLAIN_TUNED     | −6.09       | 1.1e‑9  | 2 521 |
| 3    | LSTM_HYBRID          | −9.60       | 0.0     | 2 421 |
| 4    | ARIMA                | −8.71       | 0.0     | 1 060 |
| …    | …                    |             |         |       |

Full table in *outputs/dm_results.csv*.

Figures:  
![Boxplot](figs/error_boxplot_final.png)  
![Last 250 days](figs/time_series_final.png)

---

## 4. Reproducibility notes
* Random seeds fixed in every script (`numpy`, `tensorflow`, `random`).
* Expanding‑window CV with 53 folds aligns all models.
* Heavy `.keras` weights are zipped to keep repo light (<1 MB); unzip if you need to re‑evaluate.

---

## 5. Next steps
1. **Document SARIMA‑VIX** training script and add metrics.
2. Final tidy‑up of `utils5_estable.py` → rename to `utils.py`.
3. Write discussion section in thesis: contrast DM results & practical significance.
4. Optional: export best model to ONNX for deployment demo.

---

## 6. License
MIT © 2025 José Agudelo

