# -*- coding: utf-8 -*-
"""
Rolling/expanding-window cross-validation para LSTM.
Genera predicciones alineadas con el esquema ARIMA/SARIMA:
    • 53 folds
    • ventana de test = 20 días (≈ 1 mes bursátil)
    • horizonte = 1 día, con predicción directa
Salida:
    models/LSTM_[Plain|VIX]_CV/
        ├─ fold_01_preds.csv  (Date,y_true,y_pred)
        ├─ …
        ├─ predictions_all_folds.csv
        └─ metrics.json       {"RMSE": … }
Uso:
    python src/train_lstm_cv.py --data data/df_final_ready_plus_vix.csv \
                                --cols Target_Price
    python src/train_lstm_cv.py --data data/df_final_ready_plus_vix.csv \
                                --cols Target_Price,VIX_Close
"""
import argparse, json, os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential

# -------------------------- helpers ----------------------------------- #
def make_windows(arr: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    """Convierte una matriz (n, k) en (n-L, L, k), (n-L,)"""
    X, y = [], []
    for i in range(len(arr) - lookback):
        X.append(arr[i : i + lookback])
        y.append(arr[i + lookback, 0])              # solo Target_Price
    return np.asarray(X), np.asarray(y)


def build_model(lookback: int, n_features: int) -> Sequential:
    model = Sequential(
        [
            LSTM(64, input_shape=(lookback, n_features)),
            Dense(1),
        ]
    )
    model.compile(loss="mse", optimizer="adam")
    return model


def rolling_folds(df: pd.DataFrame, lookback: int, test_window: int):
    """
    Genera índices (train_end, test_idx) para 53 folds con
    expanding train y ventana fija de test_window.
    """
    first_train_end = lookback + 160  # ~8 meses para la 1ª ventana
    train_end = first_train_end
    while train_end + test_window <= len(df):
        test_idx = df.index[train_end : train_end + test_window]
        yield df.index[:train_end], test_idx
        train_end += test_window  # expande 1 mes cada vez


# -------------------------- main -------------------------------------- #
def main(args):
    LOOKBACK = 40
    TEST_WINDOW = 20
    PATIENCE = 10
    EPOCHS = 100
    BATCH = 32

    all_cols = args.cols.split(",")
    out_dir = Path(f"models/{'LSTM_VIX_CV' if len(all_cols) > 1 else 'LSTM_Plain_CV'}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------- carga y pre-procesamiento ------------------------------- #
    df = (
        pd.read_csv(args.data, parse_dates=["Date"])
        .set_index("Date")
        [all_cols]
        .copy()
    )

    preds_global = []

    # -------- rolling-CV --------------------------------------------- #
    for fold, (train_idx, test_idx) in enumerate(
        rolling_folds(df, LOOKBACK, TEST_WINDOW), 1
    ):
        train_df, test_df = df.loc[train_idx], df.loc[test_idx]

        scaler = MinMaxScaler().fit(train_df.values)
        train_scaled = pd.DataFrame(
            scaler.transform(train_df.values),
            index=train_df.index,
            columns=all_cols,
        )
        test_scaled = pd.DataFrame(
            scaler.transform(test_df.values),
            index=test_df.index,
            columns=all_cols,
        )

        X_train, y_train = make_windows(train_scaled.values, LOOKBACK)

        model = build_model(LOOKBACK, len(all_cols))
        es = EarlyStopping(
            monitor="val_loss", patience=PATIENCE, restore_best_weights=True, verbose=0
        )
        model.fit(
            X_train,
            y_train,
            validation_split=0.1,
            epochs=EPOCHS,
            batch_size=BATCH,
            verbose=0,
            callbacks=[es],
        )

        # ---- predicción de cada día de la ventana test -------------- #
        for date in test_scaled.index:
            idx = df.index.get_loc(date)
            window = df.iloc[idx - LOOKBACK : idx][all_cols].values
            window_scaled = scaler.transform(window)
            y_pred = model.predict(window_scaled[np.newaxis, ...], verbose=0)[0, 0]
            preds_global.append(
                {"Date": date, "y_true": df.at[date, all_cols[0]], "y_pred": y_pred}
            )

        # opcional: guardar por fold
        pd.DataFrame(preds_global[-TEST_WINDOW:]).to_csv(
            out_dir / f"fold_{fold:02d}_preds.csv", index=False
        )
        print(f"Fold {fold:02d} completado – {len(preds_global)} preds acumuladas.")

    # --------‐ métricas globales & guardado -------------------------- #
    preds_df = pd.DataFrame(preds_global).sort_values("Date").reset_index(drop=True)
    preds_df.to_csv(out_dir / "predictions_all_folds.csv", index=False)

    rmse = np.sqrt(
        ((preds_df["y_true"] - preds_df["y_pred"]) ** 2).mean()
    )
    with open(out_dir / "metrics.json", "w") as fp:
        json.dump({"RMSE": float(rmse)}, fp, indent=2)

    print(f"✅  Terminado. RMSE global = {rmse:.6f}")
    print("📦  Archivos guardados en", out_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="CSV con la serie completa")
    p.add_argument(
        "--cols",
        required=True,
        help="Columnas de entrada separadas por coma (primera = Target_Price)",
    )
    args = p.parse_args()
    main(args)
