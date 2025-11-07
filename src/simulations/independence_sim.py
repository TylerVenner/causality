# src/simulations/invariance_sim.py

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def generate_data(environment: str, n_samples: int = 200) -> pd.DataFrame:
    """
    Generates data for the Fertilizer -> Crop Yield SCM.
    The distribution of the cause (Fertilizer) changes based on the environment,
    but the mechanism (how Yield responds to Fertilizer) remains the same.
    
    Args:
        environment (str): Either "Small Farms" or "Industrial Farms".
        n_samples (int): The number of data points to generate.
        
    Returns:
        pd.DataFrame: A DataFrame with 'Fertilizer' and 'Crop_Yield'.
    """
    # Ground Truth SCM:
    # F := N_F
    # Y := 5 * F + 20 + N_Y
    
    if environment == "Small Farms":
        # Low-mean, low-variance fertilizer application
        n_f = np.random.uniform(low=1, high=4, size=n_samples)
    elif environment == "Industrial Farms":
        # High-mean, high-variance fertilizer application
        n_f = np.random.uniform(low=5, high=10, size=n_samples)
    else:
        raise ValueError("Unknown environment specified.")
        
    # The physical mechanism is INVARIANT across environments
    n_y = np.random.normal(loc=0, scale=8, size=n_samples)
    
    # Structural Assignments
    fertilizer = n_f
    crop_yield = 5 * fertilizer + 20 + n_y
    
    return pd.DataFrame({'Fertilizer': fertilizer, 'Crop_Yield': crop_yield})


def fit_and_get_equation(df: pd.DataFrame, cause_col: str, effect_col: str) -> str:
    """
    Fits a linear regression model and returns the equation as a string.
    """
    model = LinearRegression()
    X = df[[cause_col]]
    y = df[effect_col]
    model.fit(X, y)
    
    slope = model.coef_[0]
    intercept = model.intercept_
    
    return f"{effect_col} ≈ {slope:.2f} * {cause_col} + {intercept:.2f}"

def generate_lingam_data(n_samples: int = 1000) -> pd.DataFrame:
    """
    Generates data from a Linear Non-Gaussian Acyclic Model (LiNGAM).
    
    SCM:
    N_X ~ Uniform(-2, 2)
    N_Y ~ Exponential(1)
    X := N_X
    Y := 2*X + N_Y
    """
    # Non-Gaussian noise terms
    n_x = np.random.uniform(low=-2, high=2, size=n_samples)
    n_y = np.random.exponential(scale=1, size=n_samples)
    
    # Structural Assignments
    x = n_x
    y = 2 * x + n_y
    
    return pd.DataFrame({'X': x, 'Y': y})

def fit_and_get_residuals(df: pd.DataFrame, cause_col: str, effect_col: str) -> pd.Series:
    """
    Fits a linear regression model and returns the residuals (estimated noise).
    """
    model = LinearRegression()
    X = df[[cause_col]]
    y = df[effect_col]
    model.fit(X, y)
    
    predictions = model.predict(X)
    residuals = y - predictions
    return residuals

#############################

def generate_ambiguous_gaussian_data(n_samples: int = 1000) -> pd.DataFrame:
    """
    Generates data for an $A -> B$ SCM with Gaussian noise,
    which is ambiguous to noise-based methods.
    
    SCM:
    N_A ~ N(0, 1)
    N_B ~ N(0, 1)
    A := N_A
    B := 1.5*A + N_B
    """
    n_a = np.random.normal(loc=0, scale=1, size=n_samples)
    n_b = np.random.normal(loc=0, scale=1, size=n_samples)
    
    a = n_a
    b = 1.5 * a + n_b
    
    return pd.DataFrame({'A': a, 'B': b})

def generate_interventional_gaussian_data(n_samples: int = 1000) -> pd.DataFrame:
    """
    Generates data from an intervened SCM: do(A := N(5, 1))
    The mechanism for B remains invariant.
    
    SCM:
    N_A ~ N(5, 1)  (This is the intervention)
    N_B ~ N(0, 1)  (This mechanism is unchanged)
    A := N_A
    B := 1.5*A + N_B
    """
    # Intervention on A: change its noise distribution
    n_a = np.random.normal(loc=5, scale=1, size=n_samples)
    
    # Mechanism for B is invariant
    n_b = np.random.normal(loc=0, scale=1, size=n_samples)
    
    a = n_a
    b = 1.5 * a + n_b
    
    return pd.DataFrame({'A': a, 'B': b})
