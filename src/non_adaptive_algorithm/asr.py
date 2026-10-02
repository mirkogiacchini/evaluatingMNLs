import numpy as np
from utils.MNL import *
from scipy.sparse import coo_matrix

def ASR(n, dataset, initial_params=None, lam=1e-5, tol=1e-8, max_iter = 10000, min_iter=0):
    idx1 = dataset[:,0]
    idx2 = dataset[:,1]
    first_item_wins = dataset[:,2]
    second_item_wins = dataset[:,3]

    if initial_params is None:
        params = np.ones(n) / n
    else:
        params = np.array(initial_params)
    
    prev_params = None

    # === compute D
    L = first_item_wins + second_item_wins
    s1 = np.bincount(idx1, weights=L, minlength=n)
    s2 = np.bincount(idx2, weights=L, minlength=n)

    c1 = np.bincount(idx1, minlength=n)
    c2 = np.bincount(idx2, minlength=n)

    d = (s1 + s2) / 2 + lam * (c1 + c2)


    # ==== compute P
    pij = (second_item_wins + lam) / (2 * d[idx1])
    pji = (first_item_wins + lam) / (2 * d[idx2])

    tot_win = np.bincount(idx1, weights=first_item_wins, minlength=n) + np.bincount(idx2, weights=second_item_wins, minlength=n)
 
    pii = tot_win / (2 * d) + lam * (c1 + c2) / (2 * d)
    iind = np.arange(n)

    i_ind = np.concatenate([idx1, idx2, iind])
    j_ind = np.concatenate([idx2, idx1, iind])
    p_ind = np.concatenate([pij, pji, pii])


    P = coo_matrix((p_ind, (i_ind, j_ind)), shape=(n, n))
    P = P.tocsr()

    curr_iter = 0
    while ((prev_params is None or np.linalg.norm(prev_params - params_log, ord=1) > tol * n) and curr_iter < max_iter) or curr_iter < min_iter:
        #print('curr iter:', curr_iter)
        prev_params = np.array(np.log(params))
        prev_params -= np.mean(prev_params)
        
        params = P.T @ params
        params /= np.sum(params)

        curr_iter += 1

        params_log = np.log(params)
        params_log -= np.mean(params_log)

    params /= d
    params /= np.sum(params)

    return MNL(None, params.tolist(), logweights=False)