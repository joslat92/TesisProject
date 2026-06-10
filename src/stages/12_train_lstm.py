import pandas as pd
import numpy as np
import yaml
import os
import sys
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
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.linear(out[:, -1, :])

def create_sequences(X_data, y_data, seq_len):
    xs, ys = [], []
    for i in range(len(X_data) - seq_len):
        xs.append(X_data[i : i+seq_len])
        ys.append(y_data[i+seq_len])
    return np.array(xs), np.array(ys)

def load_config():
    # AGREGAR EL PARÁMETRO encoding="utf-8"
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def run_lstm():
    cfg = load_config()
    print(">>> [Fase 2] Entrenando LSTM bajo Contrato...")
    set_seeds(cfg['project']['seed'])
    
    horizons = cfg['features']['horizons']
    model_name = "LSTM"
    
    input_pattern = cfg['paths']['features_parquet_pattern']
    output_dir = cfg['paths']['preds_oos_dir']
    
    # Params
    params = cfg['models']['params']['lstm']
    seq_len = params['seq_len']
    hidden_dim = params['hidden_dim']
    epochs = params['epochs']
    
    # Fechas
    train_end = pd.Timestamp(cfg['data']['splits']['train_end'])
    oos_start = pd.Timestamp(cfg['data']['splits']['oos_start'])
    oos_end = pd.Timestamp(cfg['data']['splits']['oos_end'])

    for h in horizons:
        print(f"   Procesando h={h}...")
        df = pd.read_parquet(input_pattern.format(h=h))
        
        # Features: Usamos 'ret_1d' como input principal (estacionario)
        # Podríamos agregar exógenas aquí según 'variants' del config, 
        # pero para cerrar el contrato usaremos LSTM Plain (solo retornos).
        feature_cols = ['ret_1d'] 
        target_col = f'Target_Ret_h{h}'
        
        # Scaler (Fit on Train ONLY)
        df_train = df[df['Date'] <= train_end].copy()
        scaler = StandardScaler()
        train_feats = df_train[feature_cols].values
        scaler.fit(train_feats)
        
        # Transform Full Dataset (para poder crear secuencias continuas)
        full_feats = scaler.transform(df[feature_cols].values)
        full_targets = df[target_col].fillna(0).values # Fillna 0 para el paso de create_seq (no se usa en test)
        
        # Create Sequences
        X_all, y_all = create_sequences(full_feats, full_targets, seq_len)
        
        # Alineación de índices: 
        # create_sequences reduce el tamaño en seq_len. 
        # La secuencia 'i' termina en el índice 'i + seq_len' del df original.
        # Esa fila 'i + seq_len' es la que estamos prediciendo.
        valid_indices = df.index[seq_len:]
        
        # Filtrar Train y OOS basado en índices de fecha
        # Necesitamos saber qué fechas corresponden a X_all
        dates_all = df['Date'].iloc[seq_len:].reset_index(drop=True)
        
        mask_train = dates_all <= train_end
        mask_oos = (dates_all >= oos_start) & (dates_all <= oos_end)
        
        X_train = X_all[mask_train]
        y_train = y_all[mask_train]
        
        X_oos = X_all[mask_oos]
        # Targets reales de OOS (para evaluar)
        y_oos_true = df.loc[dates_all[mask_oos].index, target_col].values
        price_oos_true = df.loc[dates_all[mask_oos].index, f'Target_Price_h{h}'].values
        price_base_oos = df.loc[dates_all[mask_oos].index, cfg['data']['target_col']].values
        
        # Training
        ds_train = SimpleDataset(X_train, y_train)
        dl_train = DataLoader(ds_train, batch_size=32, shuffle=True)
        
        model = LSTMModel(len(feature_cols), hidden_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        model.train()
        for epoch in range(epochs):
            for xb, yb in dl_train:
                optimizer.zero_grad()
                out = model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                
        # Prediction
        model.eval()
        with torch.no_grad():
            pred_tensor = model(torch.tensor(X_oos, dtype=torch.float32))
            y_pred_ret = pred_tensor.numpy().flatten()
            
        # Reconstrucción
        y_pred_level = price_base_oos * np.exp(y_pred_ret)
        
        # Output Contrato
        df_out = pd.DataFrame()
        df_out['Date'] = dates_all[mask_oos].values
        df_out['h'] = h
        df_out['model'] = model_name
        df_out['y_true_ret'] = y_oos_true
        df_out['y_true_level'] = price_oos_true
        df_out['y_pred_ret'] = y_pred_ret
        df_out['y_pred_level'] = y_pred_level
        
        filename = cfg['contract']['naming']['oos'].format(h=h, model=model_name)
        df_out.to_csv(os.path.join(output_dir, filename), index=False)
        print(f"   -> Generado LSTM h={h}")

    print("   Validando artefactos LSTM...")
    ContractValidator().validate_all_outputs()

if __name__ == "__main__":
    run_lstm()
