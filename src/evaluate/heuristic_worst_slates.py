from evaluate.worst_pair import *
from evaluate.evaluate import *
import math
import numpy as np

# ================= greedy extension =============

def _extend_slate_greedy(M1, M2, slate, norm):
    n = M1.getNumberOfItems()
    best_err = None
    best_candidate = None
    slate_set = set(slate)
    for i in range(n):
        if i in slate_set:
            continue
        candidate = slate + [i]
        new_error = getSlateError(M1, M2, candidate, norm=norm)
        if best_err is None or new_error > best_err:
            best_err = new_error
            best_candidate = candidate
    
    return best_candidate, best_err

def _shrink_slate_greedy(M1, M2, slate, norm):
    best_err = None
    best_candidate = None
    for i in slate:
        candidate = [x for x in slate if x != i]
        new_error = getSlateError(M1, M2, candidate, norm=norm)
        if best_err is None or new_error > best_err:
            best_err = new_error
            best_candidate = candidate
    
    return best_candidate, best_err

def greedy_extension(M1, M2, eps_threshold=0.001, min_num_iter = 5, max_iter_mul=5, start_slate=None, d1=True):
    norm = 1 if d1 else np.inf
    if start_slate is None:
        curr_slate, error, _ = worstPairApproximate(M1, M2, 0.01, max_iterations_in_pruned_search_timesn=8)
        if not d1:
            error /= 2
    else:
        error = getSlateError(M1, M2, start_slate, norm=norm)
        curr_slate = start_slate
    curr_slate = list(curr_slate)

    n = M1.getNumberOfItems()
    if M2.getNumberOfItems() != n:
        raise Exception(f"M1 and M2 must have same number of items: {n} != {M2.getNumberOfItems()}")    
    
    max_iter = math.ceil(np.log(n)) * max_iter_mul
    for curr_iter in range(max_iter):
        best_delta = 0
        # increase slate first, then try to shrink
        for round in ['extend', 'shrink']:
            if round == 'extend':
                new_slate, new_error = _extend_slate_greedy(M1, M2, curr_slate, norm=norm)
            else:
                if len(curr_slate) <= 3:
                    break
                new_slate, new_error = _shrink_slate_greedy(M1, M2, curr_slate, norm=norm)
            if new_slate is None or len(new_slate) <= 2:
                break

            delta = new_error - error
            best_delta += max(delta, 0)
            if delta <= 0:
                break 

            error = new_error
            curr_slate = new_slate

        if best_delta <= eps_threshold and curr_iter >= min_num_iter:
            break
    
    return curr_slate, error

# ============== EXHAUSTIVE SEARCH ==========
def worst_error_on_filtered_MNL(M1, M2, num_items=14, d1=True):
    if not M1.isLogWeight() or not M2.isLogWeight():
        raise Exception(f"MNLs must be in log-weight: {M1.isLogWeight() }, {M2.isLogWeight()}")
    
    n = M1.getNumberOfItems()
    delta = sorted([(M1.weights[i] - M2.weights[i], i) for i in range(n)])
    num_items = min(num_items, n)
    left = num_items // 2
    right = num_items - left
    items = [v[1] for v in delta[:left] + delta[-right:]]
    
    return computeWorstErrorExactly(M1, M2, items=items, d1=d1)


# ================ PUT TOGETHER MULTIPLE HEURISTICS ===============

def replace_slate(S1, err1, S2, err2):
    if S1 is None:
        return S2, err2, True
    if S2 is None:
        return S1, err1, False
    
    if err1 >= err2:
        return S1, err1, False
    return S2, err2, True

def heuristic_worst_slate(M1, M2, d1=True, max_iter_mul=5):
    n = M1.getNumberOfItems()
    if M2.getNumberOfItems() != n:
        raise Exception(f"M1 and M2 must have same number of items: {n} != {M2.getNumberOfItems()}")   

    candidate, err_candidate = greedy_extension(M1, M2, d1=d1, max_iter_mul=max_iter_mul)

    candidate, err_candidate, replaced = replace_slate(candidate, err_candidate, *worst_error_on_filtered_MNL(M1, M2, d1=d1, num_items=14))

    if replaced:
        candidate, err_candidate, _ = replace_slate(candidate, err_candidate, *greedy_extension(M1, M2, start_slate=candidate, d1=d1, max_iter_mul=max_iter_mul))

    return candidate, err_candidate