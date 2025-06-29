import pandas as pd, json, numpy as np, pathlib, sys
path = pathlib.Path("models/ARIMA/predictions_all_folds.csv")
df   = pd.read_csv(path)

# detecta automáticamente nombres de columnas habituales
y_true = df.filter(regex="true|actual|y").iloc[:,0]
y_pred = df.filter(regex="pred|hat").iloc[:,0]

rmse = float(np.sqrt(((y_true - y_pred)**2).mean()))
print("RMSE =", rmse)

# escribe el metrics.json
(path.parent / "metrics.json").write_text(
    json.dumps({"RMSE": rmse}, indent=2)
)
