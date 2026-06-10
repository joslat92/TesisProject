import statsmodels.api as sm
import pandas as pd

def mincer_zarnowitz_regression(y_true, y_pred):
    """
    Regresión MZ: y_t = alpha + beta * y_hat_t + epsilon [cite: 16]
    Evalúa H0: alpha=0, beta=1 para calibración[cite: 107].
    """
    X = sm.add_constant(y_pred)
    # Uso de errores robustos HC3 para heterocedasticidad [cite: 59, 379]
    model = sm.OLS(y_true, X).fit(cov_type='HC3')
    
    return {
        'alpha': model.params.iloc[0],  # Cambiado a .iloc para evitar warnings
        'beta': model.params.iloc[1],
        'p_alpha': model.pvalues.iloc[0],
        'p_beta': model.pvalues.iloc[1],
        'r2': model.rsquared
    }
