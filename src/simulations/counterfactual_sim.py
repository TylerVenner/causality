def solve_for_nb(T: int, B: int) -> int:
    """
    Solves for the exogenous noise N_B given the observed T and B.
    This is the 'Abduction' step.
    
    SCM equation for B: B = T * N_B + (1-T) * (1-N_B)
    """
    if T == 1:
        # If T=1, equation simplifies to B = N_B
        return B
    elif T == 0:
        # If T=0, equation simplifies to B = 1 - N_B
        return 1 - B
    else:
        raise ValueError("Treatment T must be 0 or 1")


def calculate_counterfactual_outcome(deduced_nb: int, counterfactual_T: int) -> int:
    """
    Calculates the counterfactual outcome B' given the deduced N_B and the
    new counterfactual action for T. This is the 'Prediction' step.
    
    SCM equation for B: B' = T' * N_B + (1-T') * (1-N_B)
    """
    return counterfactual_T * deduced_nb + (1 - counterfactual_T) * (1 - deduced_nb)