#import choix 
import math
import numpy as np
from utils.MNL import *
from tqdm import tqdm


# ============ FROM CHOIX LIBRARY ===============
import scipy.linalg as spl
import abc
import warnings
from collections.abc import Callable

from evaluate.worst_pair import *

def log_transform(weights):
    """Transform weights into centered log-scale parameters."""
    params = np.log(weights)
    return params - params.mean()

def exp_transform(params):
    """Transform parameters into exp-scale weights."""
    params = np.asarray(params)
    weights = np.exp(params - np.mean(params))
    return (len(weights) / weights.sum()) * weights

def _init_lsr(
    n_items: int,
    alpha: float,
    initial_params,
):
    """Initialize the LSR Markov chain and the weights."""
    if initial_params is None:
        weights = np.ones(n_items)
    else:
        weights = exp_transform(initial_params)
    chain = alpha * np.ones((n_items, n_items), dtype=float)
    return weights, chain

class ConvergenceTest(metaclass=abc.ABCMeta):
    """Abstract base class for convergence tests.

    Convergence tests should implement a single function, `__call__`, which
    takes a parameter vector and returns a boolean indicating whether or not
    the convergence criterion is met.
    """

    @abc.abstractmethod
    def __call__(self, params, update: bool = True) -> bool:
        """Test whether convergence criterion is met.

        The parameter `update` controls whether `params` should replace the
        previous parameters (i.e., modify the state of the object).
        """

class NormOfDifferenceTest(ConvergenceTest):
    """Convergence test based on the norm of the difference vector.

    This convergence test computes the difference between two successive
    parameter vectors, and declares convergence when the norm of this
    difference vector (normalized by the number of items) is below `tol`.
    """

    def __init__(self, tol: float = 1e-8, order: int = 1):
        self._tol = tol
        self._ord = order
        self._prev_params = None

    def __call__(self, params, update: bool = True) -> bool:
        params = np.asarray(params) - np.mean(params)
        if self._prev_params is None:
            if update:
                self._prev_params = params
            return False
        dist = np.linalg.norm(self._prev_params - params, ord=self._ord)
        if update:
            self._prev_params = params
        return bool(dist <= self._tol * len(params))
    
class WorstPairTest(ConvergenceTest):
    """Convergence test based on the worst pair between two successively learned MNLs.
    """

    def __init__(self, tol: float = 1e-8):
        self._tol = tol
        self._prev_params = None

    def __call__(self, params, update: bool = True) -> bool:
        params = np.asarray(params) - np.mean(params)
        if self._prev_params is None:
            if update:
                self._prev_params = params
            return False

        M1 = MNL(None, params, logweights=True)
        M2 = MNL(None, self._prev_params, logweights=True)
        _, dist, _ = worstPairApproximate(M1, M2, 0.02, max_iterations_in_pruned_search_timesn = 8)
        
        print('worst pair distance: ', dist)
 
        if update:
            self._prev_params = params
        
        return bool(dist <= self._tol)


def _ilsr(
    fun,
    params,
    max_iter: int,
    tol: float,
    return_anyway: bool = False,
    save_iterations: bool = False,
    worst_pair_test: bool = False,
    min_iter=-1,
):
    """Iteratively refine LSR estimates until convergence.

    Raises
    ------
    RuntimeError
        If the algorithm does not converge after ``max_iter`` iterations.
    """
    if worst_pair_test:
        converged = WorstPairTest(tol=tol)
    else:  
        converged = NormOfDifferenceTest(tol, order=1)
    all_params = []
    for curr_iter in tqdm(range(max_iter)): #range(max_iter): 
        params = fun(params)
        if save_iterations:
            all_params.append(params[:])
        if curr_iter >= min_iter and converged(params):
            if save_iterations:
                return all_params
            return params
    if return_anyway:
        if save_iterations:
            return all_params
        return params
    raise RuntimeError("Did not converge after {} iterations".format(max_iter))

def statdist(generator):
    """Compute the stationary distribution of a Markov chain.

    Parameters
    ----------
    generator : array_like
        Infinitesimal generator matrix of the Markov chain.

    Returns
    -------
    dist : numpy.ndarray
        The unnormalized stationary distribution of the Markov chain.

    Raises
    ------
    ValueError
        If the Markov chain does not have a unique stationary distribution.
    """
    generator = np.asarray(generator)
    n = generator.shape[0]
    with warnings.catch_warnings():
        # The LU decomposition raises a warning when the generator matrix is
        # singular (which it, by construction, is!).
        warnings.filterwarnings("ignore")
        lu, piv = spl.lu_factor(generator.T, check_finite=False)
    # The last row contains 0's only.
    left = lu[:-1, :-1]
    right = -lu[:-1, -1]
    # Solves system `left * x = right`. Assumes that `left` is
    # upper-triangular (ignores lower triangle).
    try:
        res = spl.solve_triangular(left, right, check_finite=False)
    except:  # noqa: E722
        # Ideally we would like to catch `spl.LinAlgError` only, but there seems
        # to be a bug in scipy, in the code that raises the LinAlgError (!!).
        raise ValueError(
            "stationary distribution could not be computed. "
            "Perhaps the Markov chain has more than one absorbing class?"
        )
    res = np.append(res, 1.0)
    return (n / res.sum()) * res

def lsr_pairwise_dense(
    comp_mat,
    alpha: float = 0.0,
    initial_params = None,
):
    """Compute the LSR estimate of model parameters given dense data.

    This function implements the Luce Spectral Ranking inference algorithm
    [MG15]_ for dense pairwise-comparison data.

    The data is described by a pairwise-comparison matrix ``comp_mat`` such
    that ``comp_mat[i,j]`` contains the number of times that item ``i`` wins
    against item ``j``.

    In comparison to :func:`~choix.lsr_pairwise`, this function is particularly
    efficient for dense pairwise-comparison datasets (i.e., containing many
    comparisons for a large fraction of item pairs).

    The argument ``initial_params`` can be used to iteratively refine an
    existing parameter estimate (see the implementation of
    :func:`~choix.ilsr_pairwise` for an idea on how this works). If it is set
    to `None` (the default), the all-ones vector is used.

    The transition rates of the LSR Markov chain are initialized with
    ``alpha``. When ``alpha > 0``, this corresponds to a form of regularization
    (see :ref:`regularization` for details).

    Parameters
    ----------
    comp_mat : np.array
        2D square matrix describing the pairwise-comparison outcomes.
    alpha : float, optional
        Regularization parameter.
    initial_params : array_like, optional
        Parameters used to build the transition rates of the LSR Markov chain.

    Returns
    -------
    params : np.array
        An estimate of model parameters.
    """
    n_items = comp_mat.shape[0]
    ws, chain = _init_lsr(n_items, alpha, initial_params)
    denom = np.tile(ws, (n_items, 1))
    chain += comp_mat.T / (denom + denom.T)
    chain -= np.diag(chain.sum(axis=1))
    return log_transform(statdist(chain))

def ilsr_pairwise_dense(
    comp_mat,
    alpha: float = 0.0,
    initial_params = None,
    max_iter: int = 100,
    tol: float = 1e-8,
    return_anyway: bool = False,
    save_iterations: bool = False,
    worst_pair_test: bool = False,
    min_iter = -1,
):
    """Compute the ML estimate of model parameters given dense data.

    This function computes the maximum-likelihood (ML) estimate of model
    parameters given dense pairwise-comparison data.

    The data is described by a pairwise-comparison matrix ``comp_mat`` such
    that ``comp_mat[i,j]`` contains the number of times that item ``i`` wins
    against item ``j``.

    In comparison to :func:`~choix.ilsr_pairwise`, this function is
    particularly efficient for dense pairwise-comparison datasets (i.e.,
    containing many comparisons for a large fraction of item pairs).

    The transition rates of the LSR Markov chain are initialized with
    ``alpha``. When ``alpha > 0``, this corresponds to a form of regularization
    (see :ref:`regularization` for details).

    Parameters
    ----------
    comp_mat : np.array
        2D square matrix describing the pairwise-comparison outcomes.
    alpha : float, optional
        Regularization parameter.
    initial_params : array_like, optional
        Parameters used to initialize the iterative procedure.
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

    def fun(params):
        return lsr_pairwise_dense(comp_mat=comp_mat, alpha=alpha, initial_params=params)

    return _ilsr(fun, initial_params, max_iter, tol, return_anyway=return_anyway, save_iterations=save_iterations, worst_pair_test=worst_pair_test, min_iter=min_iter)
# ==========================================================

def ILSR_(M, budgetQueries, num_samples_per_pair=1, save_iterations=False):
    num_slates = math.ceil(budgetQueries / num_samples_per_pair)
    dataset = M.generatePairwiseDataset(num_slates, num_samples_per_pair)
    return ILSR(M.getNumberOfItems(), dataset, save_iterations=save_iterations), num_slates * num_samples_per_pair

def ILSR(n, dataset, save_iterations=False, initial_params=None, tol=1e-8, worst_pair_test = False, min_iter=-1, alpha=1e-5):
    idx1 = dataset[:,0]
    idx2 = dataset[:,1]
    first_item_wins = dataset[:,2]
    second_item_wins = dataset[:,3]

    matrix = np.zeros((n, n))
    matrix[idx1, idx2] = first_item_wins
    matrix[idx2, idx1] = second_item_wins
    
    theta = ilsr_pairwise_dense(matrix, alpha=alpha, return_anyway=True, save_iterations=save_iterations, initial_params=initial_params, 
                                tol=tol, worst_pair_test=worst_pair_test, min_iter=min_iter)

    if save_iterations:
        outs = []
        for t in theta:
            outs.append(MNL(None, t.tolist(), logweights=True))
        return outs
    
    return MNL(None, theta.tolist(), logweights=True)
    