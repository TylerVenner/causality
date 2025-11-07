import pandas as pd
import numpy as np
import graphviz

def generate_m_graph_data(n_samples: int = 2000) -> pd.DataFrame:
    """
    Generates data from a "Bow-Tie" style graph with two hidden variables.
    
    This graph has:
    - A hidden variable H1 affecting A
    - A hidden confounder H2 affecting D and E
    - A causal chain (A -> B -> C)
    - A collider at E (C -> E <- H2)
    
    True SCM:
    H1 := N_H1  (Hidden Variable 1)
    H2 := N_H2  (Hidden Confounder 2)
    A := 2.0*H1 + N_A
    B := 1.5*A + N_B
    C := 1.0*B + N_C
    D := 1.5*H2 + N_D
    E := 1.5*H2 + 2.0*C + N_E
    
    Key features for PC Algorithm failure:
    - D <- H2 -> E: A classic hidden confounder.
    - C -> E <- H2: This, combined with the above, creates a v-structure
      signature between observed nodes C and D, with E as the collider.
      (C -> E <- H2 -> D).
    - PC will (incorrectly) orient D -> E as part of the C -> E <- D v-structure.
    """
    # Hidden variables
    h1 = np.random.normal(0, 1, n_samples)
    h2 = np.random.normal(0, 1, n_samples)
    
    # Exogenous noises for observed vars
    n_a = np.random.normal(0, 0.5, n_samples)
    n_b = np.random.normal(0, 0.5, n_samples)
    n_c = np.random.normal(0, 0.5, n_samples)
    n_d = np.random.normal(0, 0.5, n_samples)
    n_e = np.random.normal(0, 0.5, n_samples)
    
    # Structural assignments
    a = 2.0 * h1 + n_a
    b = 1.5 * a + n_b
    c = 1.0 * b + n_c
    d = 1.5 * h2 + n_d
    e = 1.5 * h2 + 2.0 * c + n_e
    
    # We only return the *observed* variables
    return pd.DataFrame({
        'A': a, 
        'B': b, 
        'C': c, 
        'D': d, 
        'E': e
    })


def get_m_graph_ground_truth_dot() -> graphviz.Digraph:
    """
    Returns the Graphviz object for the *true* underlying SCM,
    including the hidden nodes H1 and H2.
    """
    dot = graphviz.Digraph(comment="True SCM with Hidden Variables")
    dot.attr(rankdir='LR')
    dot.attr('node', shape='circle', style='filled', fillcolor='lightblue')
    
    # Style the hidden nodes
    dot.node('H1', style='filled,dashed', fillcolor='lightgrey', fontcolor='darkgrey')
    dot.node('H2', style='filled,dashed', fillcolor='lightgrey', fontcolor='darkgrey')
    
    # Causal edges
    dot.edge('H1', 'A')
    dot.edge('A', 'B')
    dot.edge('B', 'C')
    dot.edge('C', 'E')
    dot.edge('H2', 'D')
    dot.edge('H2', 'E')
    
    return dot


def get_fci_correct_output_dot() -> graphviz.Digraph:
    """
    Returns the Graphviz object for the *correct* Partial Ancestral Graph (PAG)
    that the FCI algorithm would discover.
    
    Notation:
    A -> B -> C  (Correctly oriented causal chain)
    D <-> E      (Bi-directed edge, correctly flags H2 as a hidden confounder)
    C o-> E      (Circle-tail arrow. Means C is an ancestor of E, but the
                 algorithm is uncertain about the tail. It can't distinguish
                 C -> E from a confounded C <-> E, because of the
                 confounding environment around E.)
    """
    dot = graphviz.Digraph(comment="Correct FCI Output (PAG)")
    dot.attr(rankdir='LR')
    dot.attr('node', shape='circle', style='filled', fillcolor='lightblue')
    
    # Clear causal chain (FCI gets this right)
    dot.edge('A', 'B', dir='forward', arrowhead='normal')
    dot.edge('B', 'C', dir='forward', arrowhead='normal')
    
    # D <-> E (bi-directed arrow for confounding)
    dot.edge('D', 'E', dir='both', arrowhead='normal', arrowtail='normal')
    
    # C o-> E (circle on tail because the algorithm detects confounding
    # around E and remains skeptical about the C-E edge tail)
    dot.edge('C', 'E', dir='forward', arrowhead='normal', arrowtail='odot')
    
    return dot