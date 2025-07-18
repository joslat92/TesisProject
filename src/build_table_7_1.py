# build_table_7_1.py  –– crea Tabla 7.1
import pandas as pd, numpy as np, glob, pathlib, sys
from tabulate import tabulate

records = []
for csv in glob.glob("models/*/predictions_all_folds.csv"):
    model = pathlib.Path(csv).parent.name
    df = pd.read_csv(csv)
    required = {"y_true","y_pred"}
    if not required.issubset(df.columns):
        print(f"⚠  {model} saltado (encabezados)")
        continue
    mae  = np.mean(np.abs(df.y_true - df.y_pred))
    rmse = np.sqrt(np.mean((df.y_true - df.y_pred)**2))
    records.append({"model":model,"MAE":mae,"RMSE":rmse})

if not records:
    sys.exit("❌  Ningún modelo válido encontrado")

table = (pd.DataFrame(records)
         .sort_values("RMSE")
         .round(4)
         .reset_index(drop=True))

# pinta y guarda
print("\nTabla 7.1 – Métricas globales de CV (ordenadas por RMSE)")
print(tabulate(table, headers="keys", tablefmt="github", showindex=False))

path_out = pathlib.Path("tables/table_7_1_metrics.csv")
path_out.parent.mkdir(exist_ok=True)
table.to_csv(path_out, index=False)
print("✓ Guardado", path_out)
