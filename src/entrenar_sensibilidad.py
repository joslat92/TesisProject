import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_squared_error
from pathlib import Path

# --- Función para crear secuencias ---
def make_sequences(series, n_steps):
    X, y = [], []
    for i in range(len(series) - n_steps):
        X.append(series[i : i + n_steps])
        y.append(series[i + n_steps])
    return np.array(X), np.array(y)

# --- Función para ejecutar un experimento completo ---
def run_experiment(n_steps, cfg):
    print(f"\n--- Iniciando experimento para N_STEPS = {n_steps} ---")

    # 1. Cargar y preparar los datos
    prices = pd.read_csv(cfg["data_path"])["Target_Price"].values
    X, y = make_sequences(prices, n_steps)

    # 2. Dividir en entrenamiento y prueba (80/20)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # 3. Construir el modelo (idéntico salvo por el input_shape)
    inputs = keras.Input(shape=(n_steps, 1))
    x = layers.LSTM(64, return_sequences=True)(inputs) # Asumiendo arquitectura base
    x = layers.LSTM(32)(x)
    outputs = layers.Dense(1)(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse")

    print(f"Modelo para N={n_steps} construido. Entrenando...")

    # 4. Entrenar el modelo
    es_callback = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    model.fit(
        X_train, y_train,
        validation_split=0.2, # Usa 20% del train set para validación interna
        epochs=200,
        batch_size=cfg["batch_size"],
        callbacks=[es_callback],
        verbose=2
    )

    # 5. Evaluar y mostrar RMSE
    y_pred = model.predict(X_test).ravel()
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"✅ Resultado para N={n_steps}  ->  RMSE={rmse:.4f}")

    # 6. Guardar los resultados
    out_dir = Path(f"models/LSTM_HYBRID_{n_steps}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Guardar predicciones para la tabla final
    pd.DataFrame({"y_true": y_test, "y_pred": y_pred}).to_csv(f"{out_dir}/predictions_all_folds.csv", index=False)

    # Guardar el modelo entrenado
    model.save(f"{out_dir}/model.h5")
    print(f"Modelo y predicciones para N={n_steps} guardados en '{out_dir}'.")
    return rmse

# --- Punto de entrada principal del script ---
if __name__ == '__main__':
    # Configuración principal del experimento
    config = {"data_path": "data/df_final_ready_plus_vix.csv", "batch_size": 64}

    # Ejecutar el experimento para cada valor de look-back
    for n_lookback in [30, 40, 50]:
        run_experiment(n_lookback, config)

    print("\n--- Todos los experimentos de sensibilidad completados. ---")
