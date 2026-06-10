import pandas as pd
import pandas_market_calendars as mcal

def get_trading_index(start_date, end_date):
    """Genera el índice de trading oficial de la NYSE[cite: 58]."""
    nyse = mcal.get_calendar('NYSE')
    schedule = nyse.schedule(start_date=start_date, end_date=end_date)
    return schedule.index

def align_to_trading_days(df):
    """Alinea el dataframe a los días de trading eliminando gaps[cite: 13, 43]."""
    # Asegurar formato ISO YYYY-MM-DD [cite: 47]
    df['Date'] = pd.to_datetime(df['Date']).dt.normalize()
    
    trading_days = get_trading_index(df['Date'].min(), df['Date'].max())
    trading_df = pd.DataFrame({'Date': trading_days})
    
    # Inner-join estricto para evitar fuga de futuro [cite: 13, 43]
    aligned_df = pd.merge(trading_df, df, on='Date', how='inner')
    
    return aligned_df.sort_values('Date')
