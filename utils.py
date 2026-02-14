"""
Utility functions for HMB contraception modeling
"""
import numpy as np


def logistic(instance, uids, pars, intercept_scale=None):
    """
    Calculate logistic regression probabilities.

    Computes P(Y=1) = 1 / (1 + exp(-(intercept + β₁X₁ + β₂X₂ + ...)))
    where intercept = -log(1/base - 1).

    Args:
        instance: The class instance containing state attributes (e.g., Menstruation connector).
                 State attributes are accessed via getattr(instance, term)[uids].
        uids: Array of unique IDs for which to calculate probabilities.
        pars: Parameters object (dict-like) containing:
              - 'base': Baseline probability when all covariates are 0
              - Other keys: Covariate names matching instance state attributes,
                           with values as regression coefficients
        intercept_scale: Optional array or scalar to multiply the intercept by.
                        Useful for individual-level heterogeneity. If provided,
                        should be length len(uids) or a scalar.

    Returns:
        Array of probabilities (length len(uids)) calculated via logistic regression

    Example:
        >>> pars = {'base': 0.5, 'anemic': 1.0, 'pain': 0.5}
        >>> # For someone with anemia and pain:
        >>> # intercept = -log(1/0.5-1) = 0
        >>> # rhs = 0 + 1.0*1 + 0.5*1 = 1.5
        >>> # P = 1/(1+exp(-1.5)) ≈ 0.82
    """
    intercept = -np.log(1/pars.base-1)
    rhs = np.full_like(uids, fill_value=intercept, dtype=float)
    if intercept_scale is not None:
        rhs *= intercept_scale

    # Add all covariates
    for term, val in pars.items():
        if term != 'base':
            rhs += val * getattr(instance, term)[uids]

    # Calculate the probability
    return 1 / (1+np.exp(-rhs))
