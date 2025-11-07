import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def generate_data(structure_type: str, n_samples: int = 300) -> pd.DataFrame:
    """
    Generates data for the three fundamental 3-node SCMs.
    Uses simple linear models with non-Gaussian noise to make dependencies clear.
    """
    if structure_type == 'chain':
        # X -> Z -> Y
        n_x = np.random.uniform(-2, 2, n_samples)
        n_z = np.random.uniform(-1, 1, n_samples)
        n_y = np.random.uniform(-1, 1, n_samples)
        
        x = n_x
        z = 1.5 * x + n_z
        y = 1.5 * z + n_y
        
    elif structure_type == 'fork':
        # X <- Z -> Y
        n_z = np.random.uniform(-2, 2, n_samples)
        n_x = np.random.uniform(-1, 1, n_samples)
        n_y = np.random.uniform(-1, 1, n_samples)
        
        z = n_z
        x = 1.5 * z + n_x
        y = 1.5 * z + n_y
        
    elif structure_type == 'collider':
        # X -> Z <- Y
        n_x = np.random.uniform(-2, 2, n_samples)
        n_y = np.random.uniform(-2, 2, n_samples)
        n_z = np.random.uniform(-1, 1, n_samples)
        
        x = n_x
        y = n_y
        z = 1.5 * x + 1.5 * y + n_z
        
    else:
        raise ValueError("Unknown structure type specified.")
        
    return pd.DataFrame({'X': x, 'Y': y, 'Z': z})

def get_residuals(df: pd.DataFrame, var_to_regress: str, conditioning_var: str) -> np.ndarray:
    """
    Calculates the residuals of var_to_regress ~ conditioning_var.
    This is used to "condition on" the conditioning_var.
    """
    model = LinearRegression()
    X = df[[conditioning_var]]
    y = df[var_to_regress]
    model.fit(X, y)
    
    predictions = model.predict(X)
    residuals = y - predictions
    return residuals
