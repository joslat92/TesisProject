import pandas as pd
import numpy as np
import yaml
import os
import sys
import copy
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.core.contract import ContractValidator

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class SimpleDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.linear(self.dropout(out[:, -1, :]))

def create_sequences(X_data, y_data, seq_len):
    """Crea ventanas de ``seq_len`` que TERMINAN en la fecha del target.

    La muestra etiquetada en la fila t contiene las features de
    t-seq_len+1, ..., t. De este modo la LSTM comparte el mismo conjunto de
    información hasta t que ARIMA/ARIMAX y el nivel base P_t usado para
    reconstruir el pronóstico.
    """
    xs, ys = [], []
    for end in range(seq_len - 1, len(X_data)):
        start = end - seq_len + 1
        xs.append(X_data[start:end + 1])
        ys.append(y_data[end])
    return np.array(xs), np.array(ys)

def load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def train_with_early_stopping(model, X_tr, y_tr, X_val, y_val, params):
    """
    Entrena con early stopping: val = último 20% del train (cronológico),
    paciencia sobre el MSE de validación, restaura los mejores pesos.
    """
    dl_train = DataLoader(SimpleDataset(X_tr, y_tr),
                          batch_size=params['batch_size'], shuffle=True)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    criterion = nn.MSELoss()

    best_val = np.inf
    best_state = copy.deepcopy(model.state_dict())
    patience_left = params['patience']

    for epoch in range(params['epochs']):
        model.train()
        for xb, yb in dl_train:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t).item()

        if val_loss < best_val - 1e-8:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_left = params['patience']
        else:
            patience_left -= 1
            if patience_left == 0:
                break

    model.load_state_dict(best_state)
    return model

def fit_predict(df, feature_cols, params, seed, h, price_col, train_end,
                pred_start, pred_end):
    """
    Entrena una variante LSTM para un horizonte y devuelve el frame de
    contrato (sin columna model/block). Reutilizable por OOS y walk-forward:
    - Scaler fit SOLO con Date <= train_end (IS o IS+bloques previos).
    - Purga anti-fuga: muestras de train cuyo target se realiza después de
      train_end quedan excluidas.
    - Early stopping: val = último 20% del train (cronológico) con embargo
      de h muestras entre train y val (targets solapados no cruzan el corte).
    """
    set_seeds(seed)
    seq_len = params['seq_len']
    target_col = f'Target_Ret_h{h}'

    df_train = df[df['Date'] <= train_end]
    scaler = StandardScaler()
    scaler.fit(df_train[feature_cols].values)

    full_feats = scaler.transform(df[feature_cols].values)
    full_targets = df[target_col].fillna(0).values

    X_all, y_all = create_sequences(full_feats, full_targets, seq_len)
    dates_all = df['Date'].iloc[seq_len - 1:].reset_index(drop=True)

    target_end_dates = df['Date'].shift(-h).iloc[seq_len - 1:].reset_index(drop=True)
    mask_train = target_end_dates <= train_end
    mask_pred = (dates_all >= pred_start) & (dates_all <= pred_end)

    X_train_full = X_all[mask_train]
    y_train_full = y_all[mask_train]

    n = len(X_train_full)
    n_val = int(params['val_frac'] * n)
    cut = n - n_val
    X_tr, y_tr = X_train_full[:cut - h], y_train_full[:cut - h]
    X_val, y_val = X_train_full[cut:], y_train_full[cut:]

    X_pred = X_all[mask_pred]
    # Mapeo a filas ORIGINALES del parquet: la muestra i del frame desplazado
    # (post-secuencias) corresponde a la fila i + seq_len - 1 del df. La
    # última fila de X_pred es, por contrato, la misma fecha de origen t.
    orig_rows = dates_all[mask_pred].index + seq_len - 1
    assert (df['Date'].iloc[orig_rows].values
            == dates_all[mask_pred].values).all(), \
        "Desalineación fila/fecha en la salida LSTM"
    y_pred_true = df[target_col].iloc[orig_rows].values
    price_pred_true = df[f'Target_Price_h{h}'].iloc[orig_rows].values
    price_base = df[price_col].iloc[orig_rows].values

    model = LSTMModel(len(feature_cols), params['hidden_dim'], params['dropout'])
    model = train_with_early_stopping(model, X_tr, y_tr, X_val, y_val, params)

    model.eval()
    with torch.no_grad():
        y_pred_ret = model(torch.tensor(X_pred, dtype=torch.float32)).numpy().flatten()

    df_out = pd.DataFrame()
    df_out['Date'] = dates_all[mask_pred].values
    df_out['h'] = h
    df_out['y_true_ret'] = y_pred_true
    df_out['y_true_level'] = price_pred_true
    df_out['y_pred_ret'] = y_pred_ret
    df_out['y_pred_level'] = price_base * np.exp(y_pred_ret)
    return df_out

def run_lstm():
    cfg = load_config()
    print(">>> [Fase 2] Entrenando variantes LSTM bajo Contrato...")

    horizons = cfg['features']['horizons']
    input_pattern = cfg['paths']['features_parquet_pattern']
    output_dir = cfg['paths']['preds_oos_dir']

    params = cfg['models']['params']['lstm']
    variants = params['variants']

    train_end = pd.Timestamp(cfg['data']['splits']['train_end'])
    oos_start = pd.Timestamp(cfg['data']['splits']['oos_start'])
    oos_end = pd.Timestamp(cfg['data']['splits']['oos_end'])

    for model_name, feature_cols in variants.items():
        print(f"   Variante {model_name} (features: {feature_cols})")
        for h in horizons:
            df = pd.read_parquet(input_pattern.format(h=h))
            df_out = fit_predict(
                df, feature_cols, params, cfg['project']['seed'], h,
                cfg['data']['target_col'], train_end, oos_start, oos_end
            )
            df_out.insert(2, 'model', model_name)

            filename = cfg['contract']['naming']['oos'].format(h=h, model=model_name)
            df_out.to_csv(os.path.join(output_dir, filename), index=False)
            print(f"      -> Generado {model_name} h={h}")

    print("   Validando artefactos LSTM...")
    ContractValidator().validate_all_outputs()

if __name__ == "__main__":
    run_lstm()
