
import numpy as np
from utils.MNL import *
from tqdm import tqdm

from non_adaptive_algorithm.ilsr import NormOfDifferenceTest, log_transform, exp_transform

# ===================================  CHOIX CODE   =========================================================

def _mm(
    n_items: int,
    data,
    initial_params,
    alpha: float,
    max_iter: int,
    tol: float,
    mm_fun,
    return_anyway = True,
):
    """
    Iteratively refine MM estimates until convergence.

    Raises
    ------
    RuntimeError
        If the algorithm does not converge after `max_iter` iterations.
    """
    if initial_params is None:
        params = np.zeros(n_items)
    else:
        params = initial_params
    converged = NormOfDifferenceTest(tol=tol, order=1)
    for _ in tqdm(range(max_iter), disable=False):
        nums, denoms = mm_fun(n_items, data, params)
        params = log_transform((nums + alpha) / (denoms + alpha))
        if converged(params):
            return params, True
    
    if return_anyway:
        return params, False
    raise RuntimeError("Did not converge after {} iterations".format(max_iter))


def _mm_pairwise(
    n_items: int,
    data,
    params,
):
    """Inner loop of MM algorithm for pairwise data."""
    weights = exp_transform(params)

    wins = np.zeros(n_items, dtype=float)
    denoms = np.zeros(n_items, dtype=float)

    # data columns
    i = data[:, 0].astype(int)
    j = data[:, 1].astype(int)
    win_i = data[:, 2]
    win_j = data[:, 3]

    # wins accumulation
    np.add.at(wins, i, win_i)
    np.add.at(wins, j, win_j)

    # denom accumulation
    val = (win_i + win_j) / (weights[i] + weights[j])
    np.add.at(denoms, i, val)
    np.add.at(denoms, j, val)

    """
    #python code... this is slow
    for i, j, win_i, win_j in data:
        wins[i] += win_i
        wins[j] += win_j
        val = (win_i + win_j) / (weights[i] + weights[j])
        denoms[i] += val
        denoms[j] += val    
    """
    #original choix code: must repeat pairs
    #for winner, loser in data:
    #    wins[winner] += 1.0
    #    val = 1.0 / (weights[winner] + weights[loser])
    #    denoms[winner] += val
    #    denoms[loser] += val

    return wins, denoms

def mm_pairwise(
    n_items: int,
    data,
    initial_params = None,
    alpha: float = 0.0,
    max_iter: int = 10000,
    tol: float = 1e-8,
    return_anyway = True,
):
    """Compute the ML estimate of model parameters using the MM algorithm.

    This function computes the maximum-likelihood (ML) estimate of model
    parameters given pairwise-comparison data (see :ref:`data-pairwise`), using
    the minorization-maximization (MM) algorithm [Hun04]_, [CD12]_.

    If ``alpha > 0``, the function returns the maximum a-posteriori (MAP)
    estimate under a (peaked) Dirichlet prior. See :ref:`regularization` for
    details.

    Parameters
    ----------
    n_items : int
        Number of distinct items.
    data : list of lists
        Pairwise-comparison data.
    initial_params : array_like, optional
        Parameters used to initialize the iterative procedure.
    alpha : float, optional
        Regularization parameter.
    max_iter : int, optional
        Maximum number of iterations allowed.
    tol : float, optional
        Maximum L1-norm of the difference between successive iterates to
        declare convergence.

    Returns
    -------
    params : numpy.ndarray
        The ML estimate of model parameters.
    """
    return _mm(n_items, data, initial_params, alpha, max_iter, tol, _mm_pairwise, return_anyway=return_anyway)

# ============================================================================================

def mmAlgorithm(n, data, max_iter=10000, initial_params=None, tol=1e-9):
    theta, converged = mm_pairwise(n, data, alpha=1e-5, max_iter=max_iter, initial_params=initial_params, tol=tol)
    Mout = MNL(None, theta, logweights=True)
    return Mout, converged