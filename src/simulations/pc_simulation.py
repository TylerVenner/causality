import pandas as pd
import numpy as np

import pandas as pd
import numpy as np

def generate_diamond_data(n_samples: int = 500) -> pd.DataFrame:
    """
    Generates data from a 4-variable 'Diamond' graph:
    A -> B
    A -> C
    B -> D
    C -> D
    
    SCM with GAUSSIAN noise (for pedagogical demonstration of ambiguity):
    A := N_A ~ N(0, 1)
    B := 1.0*A + N_B, N_B ~ N(0, 1)
    C := -1.5*A + N_C, N_C ~ N(0, 1)
    D := 2.0*B - 1.0*C + N_D, N_D ~ N(0, 1)
    
    With Gaussian noise, both the PC algorithm (Markov equivalence) 
    and noise-based methods (LiNGAM) fail to find edge directions.
    """
    
    rng = np.random.default_rng(1)
    # Gaussian noise makes the problem maximally ambiguous
    n_a = np.random.normal(0, 1, n_samples)
    n_b = np.random.normal(0, 1, n_samples)
    n_c = np.random.normal(0, 1, n_samples)
    n_d = np.random.normal(0, 1, n_samples)
    
    A = n_a
    B = 1.0 * A + n_b
    C = -1.5 * A + n_c
    D = 2.0 * B - 1.0 * C + n_d
    
    return pd.DataFrame({
        'A': A,
        'B': B,
        'C': C,
        'D': D
    })


def generate_diamond_interventional_data(n_samples: int = 500) -> pd.DataFrame:
    """
    Intervention: do(A := N(5, 1))
    
    Key: We change A's distribution (shift mean to 5), but keep 
    the mechanism noise distributions GAUSSIAN and INVARIANT.
    
    The asymmetry comes from the SHIFTED mean, not from changing
    to a different noise type.
    """
    
    rng = np.random.default_rng(1)
    
    # INTERVENTION: Shift A's distribution
    n_a = np.random.uniform(3, 7, n_samples)  # shift to uniform

    
    # INVARIANCE: Same Gaussian noise for mechanisms
    n_b = np.random.normal(0, 1, n_samples)
    n_c = np.random.normal(0, 1, n_samples)
    n_d = np.random.normal(0, 1, n_samples)
    
    A = n_a
    B = 1.0 * A + n_b  # Mechanism stays the same
    C = -1.5 * A + n_c
    D = 2.0 * B - 1.0 * C + n_d
    
    return pd.DataFrame({
        'A': A,
        'B': B,
        'C': C,
        'D': D
    })


def get_ground_truth_graph() -> str:
    """
    Returns the Graphviz DOT string for the true 'Diamond' graph.
    """
    return """
    digraph {
        rankdir=LR;
        node [shape=circle, style="filled", fillcolor=lightblue];
        A -> B;
        A -> C;
        B -> D;
        C -> D;
    }
    """