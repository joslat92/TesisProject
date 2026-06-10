import numpy as np
import pandas as pd

def calculate_metrics(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    mae = np.mean(np.abs(y_true - y_pred))
    smape = 100/len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
    
    # MDA (Mean Directional Accuracy)
    actual_diff = np.diff(y_true) > 0
    pred_diff = (y_pred[1:] - y_true[:-1]) > 0
    mda = np.mean(actual_diff == pred_diff)
    
    return {'RMSE': rmse, 'MAE': mae, 'sMAPE': smape, 'MDA': mda}
