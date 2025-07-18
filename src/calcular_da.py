import pandas as pd
import numpy as np
import pathlib

# Define las carpetas de los modelos que quieres evaluar
model_dirs = [
    "models/LSTM_HYBRID",
    "models/LSTM_PLAIN_TUNED",
    "models/LSTM_NO_VIX_TUNED",
    "models/ARIMA" 
    # Añade aquí los otros modelos que quieras incluir
]

results = []

for model_path_str in model_dirs:
    model_path = pathlib.Path(model_path_str)
    pred_file = model_path / "predictions_all_folds.csv"

    if not pred_file.exists():
        print(f"⚠️  Advertencia: No se encontró '{pred_file}'. Saltando este modelo.")
        continue

    try:
        df = pd.read_csv(pred_file)

        # Asegurar que las columnas existan
        if 'y_true' in df.columns and 'y_pred' in df.columns:
            r_true = df['y_true'].diff()
            r_pred = df['y_pred'] - df['y_true'].shift(1)

            # Calcular la precisión direccional (omitiendo el primer valor NaN)
            da = (np.sign(r_true.iloc[1:]) == np.sign(r_pred.iloc[1:])).mean()

            results.append({"Modelo": model_path.name, "Precisión Direccional (DA)": f"{da:.2%}"})
        else:
            print(f"⚠️  Advertencia: El archivo en '{pred_file}' no contiene las columnas 'y_true' y 'y_pred'.")

    except Exception as e:
        print(f"❌ Error procesando el archivo {pred_file}: {e}")

# Imprimir la tabla de resultados
if results:
    results_df = pd.DataFrame(results)
    print("\n--- Resultados de Precisión Direccional ---")
    print(results_df.to_string(index=False))
else:
    print("\nNo se pudieron calcular resultados. Verifica las rutas y los archivos CSV.")
