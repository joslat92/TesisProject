import numpy as np
from statsmodels.stats.sandwich_covariance import cov_hac_simple

def diebold_mariano_test(y_true, y_pred1, y_pred2, h=1):
    """
    Calcula el estadístico DM para comparar dos modelos.
    h: horizonte de pronóstico (T+h).
    """
    e1 = (y_true - y_pred1)**2
    e2 = (y_true - y_pred2)**2
    d = e1 - e2
    
    # Media de la diferencia de pérdidas
    d_mean = np.mean(d)
    T = len(d)
    
    # Ajuste de Newey-West para la varianza (HAC)
    # Según contrato: lag h-1 
    import statsmodels.api as sm
    X = np.ones(T)
    model = sm.OLS(d, X).fit(cov_type='HAC', cov_kwds={'maxlags': h-1})
    
    dm_stat = model.tvalues.iloc[0]
    p_value = model.pvalues.iloc[0]
    return dm_stat, p_value
