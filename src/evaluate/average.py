from utils.func import *
from evaluate.evaluate import *
import numpy as np
import math

def _findLocalPairs(v, w, alpha, K):
    sort_idx = np.argsort(v)
    v_sorted = v[sort_idx]
    w_sorted = w[sort_idx]

    n = v_sorted.size
    i_list = []
    j_list = []

    num_pairs_considered = 0

    for i in range(n):
        v_i = v_sorted[i]
        # upper bound on v_j
        upper = alpha * v_i

        # because v_sorted is non-decreasing, find the last index
        # where v_j <= upper using searchsorted
        j_max = np.searchsorted(v_sorted, upper, side="right") - 1

        # we only allow j > i, and within [i+1, i + K]
        start = i + 1
        if j_max < start:
            continue  # no valid j for this i
        
        end_idx = min(j_max + 1, start + K)
        cand_js = np.arange(start, end_idx)

        num_pairs_considered += end_idx - start

        i_list.append(np.full(cand_js.shape, i, dtype=int))
        j_list.append(cand_js)

    if not i_list:
        # no pairs at all
        return np.array([]), np.array([]), 0

    i_idx = np.concatenate(i_list)
    j_idx = np.concatenate(j_list)

    dv_pairs = v_sorted[i_idx] - v_sorted[j_idx]
    dw_pairs = w_sorted[i_idx] - w_sorted[j_idx]

    return dv_pairs, dw_pairs, num_pairs_considered


def _computerErrors(M1, M2, rng, items, eps, delta, force_exact, buckets, all_pairs = True, alpha=1.1, K=100, fixed_number_queries = None):
    n = M1.getNumberOfItems()
    if M2.getNumberOfItems() != n:
        raise Exception(f"M1 and M2 must have same number of items: {n} {M2.getNumberOfItems()}")
    
    if items is None:
        items = list(range(n))
    n = len(items)
    
    if n < 2:
        return np.array([]), 0
    
    M1.convertToLogWeights()
    M2.convertToLogWeights()

    v = np.array([M1.weights[i] for i in items])
    w = np.array([M2.weights[i] for i in items])

    if all_pairs:
        if (n <= 15000 or force_exact) and (fixed_number_queries is None):
            # compute all pairs

            # Pairwise differences: dv[i, j] = v_i - v_j, same for dw
            dv = v[:, None] - v[None, :]
            dw = w[:, None] - w[None, :]

            # Use only i < j (upper triangle, excluding diagonal)
            iu, ju = np.triu_indices(n, k=1)
            dv_pairs = dv[iu, ju]
            dw_pairs = dw[iu, ju]

            num_pairs_considered = n * (n-1) // 2
        else:
            #sample some pairs
            if fixed_number_queries is None:
                M = int(math.ceil(2 * (2 / (eps * eps) * np.log(2 / delta))))
                if buckets is not None:
                    M = max(M, int(math.ceil(2 * buckets / (eps * eps))))
            else:
                M = fixed_number_queries
            
            i = rng.integers(0, n, size=M)
            j = rng.integers(0, n - 1, size=M)
            # shift j >= i to avoid j == i
            mask = j >= i
            j[mask] += 1  # now j in [0, n) and j != i

            dv_pairs = v[i] - v[j]
            dw_pairs = w[i] - w[j]

            num_pairs_considered = M
    else:
        # we will consider only pairs of items that are "close" to each other
        dv_pairs, dw_pairs, num_pairs_considered = _findLocalPairs(v, w, alpha, K)
    
    if num_pairs_considered < 1:
        return np.array([]), 0

    # Compute sigmoid of differences
    sv = npStableSigmoid(dv_pairs)
    sw = npStableSigmoid(dw_pairs)

    # err(i,j) = 2 * |sv - sw|
    err_pairs = 2.0 * np.abs(sv - sw)

    return err_pairs, int(num_pairs_considered)

def avg_median_error_pairs_d1(M1, M2, rng, items = None, eps=.01, delta=.01, force_exact=False):
    err_pairs, _ = _computerErrors(M1, M2, rng, items=items, eps=eps, delta=delta, force_exact=force_exact, buckets=None)
    avg_err = err_pairs.mean()
    median_err = np.median(err_pairs)

    return avg_err, median_err

def error_distribution_pairs_d1(M1, M2, rng, items = None, eps=.01, delta=.01, force_exact=False, buckets=100, alpha=1.1, K=100, all_pairs=True, fixed_number_queries=None):
    err_pairs, num_pairs_considered = _computerErrors(M1, M2, rng, items=items, eps=eps, delta=delta, force_exact=force_exact, buckets=buckets, 
                                                     alpha=alpha, K=K, all_pairs=all_pairs, fixed_number_queries=fixed_number_queries)

    print('max err:', err_pairs.max())
    bin_edges = np.linspace(err_pairs.min(), err_pairs.max(), buckets + 1, endpoint=True)

    # Histogram counts per bucket
    counts, _ = np.histogram(err_pairs, bins=bin_edges)

    # Normalize to get fractions (probabilities) per bucket
    fractions = counts.astype(np.float64) / err_pairs.size

    return bin_edges, fractions, num_pairs_considered

def computePairwiseErrors(M1, M2, rng, items = None, eps=.01, delta=.01, force_exact=False, buckets=100, alpha=1.1, K=100, all_pairs=True, fixed_number_queries=None):
    return _computerErrors(M1, M2, rng, items=items, eps=eps, delta=delta, force_exact=force_exact, buckets=buckets, 
                            alpha=alpha, K=K, all_pairs=all_pairs, fixed_number_queries=fixed_number_queries)
