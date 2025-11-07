# src/simulations/confounding_vs_mediation_sim.py

import pandas as pd
import numpy as np

def generate_confounding_data(n_samples: int = 500) -> pd.DataFrame:
    """
    Generates data for a confounding scenario: Z -> X and Z -> Y.
    
    SCM:
    Z (Holiday_Season) := Bernoulli(0.2)
    X (Ad_Spend)       := 20*Z + N_X
    Y (Sales)          := 50*Z + 2*X + N_Y
    """
    # Z is a confounder (e.g., 1 if holiday season, 0 otherwise)
    z_holiday_season = np.random.binomial(1, 0.2, n_samples)
    
    # Noise terms
    n_x = np.random.normal(5, 2, n_samples)
    n_y = np.random.normal(50, 5, n_samples)
    
    # X (Ad Spend) is influenced by the holiday season
    x_ad_spend = 20 * z_holiday_season + n_x
    
    # Y (Sales) is strongly influenced by the holiday season and weakly by ad spend
    y_sales = 50 * z_holiday_season + 2 * x_ad_spend + n_y
    
    df = pd.DataFrame({
        'Ad_Spend': x_ad_spend,
        'Sales': y_sales,
        'Holiday_Season': z_holiday_season
    })
    return df


def generate_mediation_data(n_samples: int = 500) -> pd.DataFrame:
    """
    Generates data for a mediation scenario: X -> Z -> Y.
    
    SCM:
    X (Ad_Spend)       := N_X
    Z (Website_Clicks) := 10*X + N_Z
    Y (Sales)          := 5*Z + N_Y
    """
    # Noise terms
    n_x = np.random.uniform(1, 10, n_samples)
    n_z = np.random.normal(10, 5, n_samples)
    n_y = np.random.normal(20, 10, n_samples)
    
    # X (Ad Spend) is the initial cause
    x_ad_spend = n_x
    
    # Z (Website Clicks) is caused by Ad Spend
    z_website_clicks = 10 * x_ad_spend + n_z
    
    # Y (Sales) is caused by Website Clicks
    y_sales = 5 * z_website_clicks + n_y
    
    df = pd.DataFrame({
        'Ad_Spend': x_ad_spend,
        'Website_Clicks': z_website_clicks,
        'Sales': y_sales
    })
    return df