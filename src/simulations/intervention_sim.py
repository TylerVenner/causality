import pandas as pd
import numpy as np

# Noise is a fixed parameter of the model, not a user input.
NOISE_STD = 1.5

def generate_observational_data(n_samples: int = 1000, slope: float = 2.0) -> pd.DataFrame:
    """
    Generates observational data from the ground truth LINEAR SCM: X -> Y.
    Noise standard deviation is fixed.
    """
    n_x = np.random.normal(loc=0, scale=1, size=n_samples)
    n_y = np.random.normal(loc=0, scale=NOISE_STD, size=n_samples)
    
    x = n_x
    y = slope * x + n_y
    
    df = pd.DataFrame({'X': x, 'Y': y})
    return df


def perform_intervention(var_name: str, value: float, n_samples: int = 1000, slope: float = 2.0) -> pd.DataFrame:
    """
    Performs a hard intervention on a variable in the LINEAR SCM.
    Noise standard deviation is fixed.
    """
    if var_name.upper() == 'X':
        x_intervened = np.full(n_samples, value)
        n_y = np.random.normal(loc=0, scale=NOISE_STD, size=n_samples)
        y_post_intervention = slope * x_intervened + n_y
        
        df = pd.DataFrame({'X_intervened': x_intervened, 'Y_post_intervention': y_post_intervention})
        
    elif var_name.upper() == 'Y':
        y_intervened = np.full(n_samples, value)
        n_x = np.random.normal(loc=0, scale=1, size=n_samples)
        x_post_intervention = n_x
        
        df = pd.DataFrame({'X_post_intervention': x_post_intervention, 'Y_intervened': y_intervened})
        
    else:
        raise ValueError("var_name must be 'X' or 'Y'")
        
    return df