import glob, os, json, pathlib
import pandas as pd, numpy as np

pathlib.Path("tables").mkdir(exist_ok=True)

records = []
for csv in glob.glob("models/*/predictions_all_folds.csv"):
    model  = os.path.basename(os.path.dirname(csv))
    df     = pd.read_csv(csv)
    diff   = df["y_true"] - df["y_pred"]
    mae    = np.mean(np.abs(diff))
    rmse   = np.sqrt(np.mean(diff**2))
    rec    = {"model": model,
              "MAE":   round(mae, 4),
              "RMSE":  round(rmse, 4)}
    records.append(rec)
    # copia individual por si la quieres (opcional)
    with open(f"tables/{model}.json", "w") as fp:
        json.dump(rec, fp, indent=2)

summary = pd.DataFrame(records).sort_values("RMSE")
summary.to_csv("tables/table_7_1_metrics.csv", index=False)
print("\nTabla 7.1 – Métricas CV (ordenada por RMSE)")
print(summary.to_string(index=False))
