import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow import keras
import joblib
from pathlib import Path
import matplotlib.pyplot as plt

# --- Parámetros globales del experimento ---
N_STEPS = 40
TRAIN_END = "2023-12-31"
TEST_START = "2024-01-01"
DATA_PATH = "data/df_final_ready_plus_vix.csv"  # Asegúrate de que esta ruta sea correcta
MODEL_DIR = "models/LSTM_HYBRID_40"             # Carpeta del modelo a evaluar
SCALER_PATH = f"{MODEL_DIR}/scaler.pkl"        # Ruta al escalador
MODEL_PATH = f"{MODEL_DIR}/model.h5"           # Ruta al modelo

# --- Función para crear secuencias ---
def make_sequences(arr, n_steps):
    X, y = [], []
    for i in range(len(arr) - n_steps):
        X.append(arr[i:(i + n_steps)])
        y.append(arr[i + n_steps])
    return np.array(X), np.array(y)

# --- Función para precisión direccional ---
def directional_accuracy(y_true, y_pred, y_prev):
    true_move = np.sign(y_true - y_prev)
    pred_move = np.sign(y_pred - y_prev)
    return (true_move == pred_move).mean() * 100

# --- Script principal ---
print("--- Iniciando prueba de robustez para LSTM_HYBRID ---")

# 1. Cargar y filtrar datos
df = pd.read_csv(DATA_PATH, parse_dates=["Date"], index_col="Date")
train_df = df.loc[:TRAIN_END]
test_df = df.loc[TEST_START:]
print(f"Datos cargados. Train: {train_df.shape}, Test: {test_df.shape}")

# Asumimos que el modelo solo usa 'Target_Price'
train_vals = train_df["Target_Price"].values.reshape(-1, 1)
test_vals = test_df["Target_Price"].values.reshape(-1, 1)

# 2. Cargar el escalador y transformar los datos
scaler = joblib.load(SCALER_PATH)
train_scaled = scaler.transform(train_vals)
test_scaled = scaler.transform(test_vals)

# 3. Crear secuencias para el conjunto de prueba
# Se necesita un "seed" con los últimos N_STEPS del entrenamiento para predecir el primer punto del test
test_input_data = np.concatenate([train_scaled[-N_STEPS:], test_scaled])
X_test, y_test_scaled = make_sequences(test_input_data, N_STEPS)
print(f"Forma de X_test para predicción: {X_test.shape}")

# 4. Cargar modelo y predecir
model = keras.models.load_model(MODEL_PATH, compile=False)
y_pred_scaled = model.predict(X_test, verbose=0).ravel()

# 5. Desescalar los resultados
y_test_inv = scaler.inverse_transform(y_test_scaled.reshape(-1, 1)).ravel()
y_pred_inv = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

# 6. Calcular métricas
rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
mae = mean_absolute_error(y_test_inv, y_pred_inv)

# Para DA, necesitamos el valor previo al primer valor de y_test_inv
y_prev_vals = np.concatenate([train_df["Target_Price"].iloc[-1:], y_test_inv[:-1]])
da = directional_accuracy(y_test_inv, y_pred_inv, y_prev_vals)

print("\n--- Resultados de la Evaluación ---")
print(f"RMSE: {rmse:.4f}")
print(f"MAE:  {mae:.4f}")
print(f"DA:   {da:.2f}%")

# 7. Guardar resultados y gráfico
out_dir = Path("models/LSTM_HYBRID_WF_ROBUST")
out_dir.mkdir(parents=True, exist_ok=True)

results_df = pd.DataFrame({
    "Date": test_df.index[N_STEPS:],
    "y_true": y_test_inv,
    "y_pred": y_pred_inv
})
results_path = out_dir / "pred_walkforward.csv"
results_df.to_csv(results_path, index=False)
print(f"\n✅ Predicciones guardadas en: {results_path}")

# Generar gráfico
plt.figure(figsize=(9, 4))
plt.plot(results_df["Date"], results_df["y_true"], label="Real", lw=1.5, color="black")
plt.plot(results_df["Date"], results_df["y_pred"], label="Predicho (LSTM Hybrid)", alpha=0.8, linestyle='--')
plt.title("Prueba de Robustez Walk-Forward (2024-2025) – LSTM Hybrid")
plt.ylabel("Precio NASDAQ-100")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
fig_path = Path("figs/wf_hybrid_robust.png")
fig_path.parent.mkdir(exist_ok=True, parents=True)
plt.savefig(fig_path, dpi=300)
print(f"✅ Gráfico guardado en: {fig_path}")
plt.show()
