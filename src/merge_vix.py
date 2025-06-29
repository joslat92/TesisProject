import pandas as pd
import os

# Asegura que la carpeta data existe
os.makedirs('data', exist_ok=True)

# Carga el NASDAQ
nasdaq = pd.read_csv('data/df_final_ready.csv', parse_dates=['Date']).set_index('Date')

# Carga el VIX (tiene 6 columnas), sin encabezados
vix = pd.read_csv('data/vix.csv', header=None, names=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])

# Limpia posibles espacios o caracteres invisibles en la columna Date
vix['Date'] = vix['Date'].astype(str).str.strip()

# Extrae solo 'Date' y 'Close', y renombra 'Close' a 'VIX_Close'
vix = vix[['Date', 'Close']]
vix.columns = ['Date', 'VIX_Close']
vix['Date'] = pd.to_datetime(vix['Date'], errors='coerce')  # errores se convierten en NaT
vix = vix.dropna(subset=['Date'])  # elimina filas donde la fecha no se pudo convertir
vix = vix.set_index('Date')

# Unimos y llenamos días festivos con el último valor válido
merged = nasdaq.join(vix, how='left').ffill()

# Guardamos el nuevo archivo
merged.to_csv('data/df_final_ready_plus_vix.csv')
print('✅ Archivo generado correctamente: data/df_final_ready_plus_vix.csv')
