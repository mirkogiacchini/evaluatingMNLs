from utils.MNL import *
from utils.func import *

def build_distortion_RMQ(wv):
    arr = []
    for i in range(len(wv)-1):
        distortion = wv[i][0] - wv[i][1] + wv[i+1][1] - wv[i+1][0]
        arr.append(distortion)
    for i in range(1, len(arr)):
        arr[i] += arr[i-1]
    return RMSQ(arr)

def query_distortion_RMQ(distortion_rmq, l, r):
    """returns maximum distortion between items in arr[l:r+1]"""
    assert l <= r, f"{l} > {r}"
    if l == r:
        return 1
    max_distortion = distortion_rmq.range_max(l, r-1) 
    min_distortion = distortion_rmq.range_min(l, r-1)
    if l >= 1:
        max_distortion -= distortion_rmq.arr[l-1]
        min_distortion -= distortion_rmq.arr[l-1]
    
    worst_distortion = max(abs(max_distortion), abs(min_distortion))
    if max_distortion >= 0 and min_distortion < 0:
        worst_distortion = max_distortion - min_distortion
    
    worst_distortion = np.exp(min(worst_distortion, 100)) #cap before exponentiation --> pass from log-space to multiplicative error
    return worst_distortion

def obtain_ub_errors(i, j, wv, distortion_rmq, w_nse, v_nse, minIdxRi, minHatIdxRi):
    """compute errors eps1, ..., eps4, mu by using arr[i:j+1] as Head and arr[j+1:] as Tail"""
    n = len(wv)
    mu = query_distortion_RMQ(distortion_rmq, i, j)
    assert mu>=1, f"Error, mu<1: {mu}"

    wi = wv[i][0]
    vi = wv[i][1]

    eps1, eps2, eps3, eps4 = 0, 0, 0, 0
    if j+1 < n:
        eps1 = w_nse.suffix_norm_sum_exp(j+1, minIdxRi)
        eps2 = v_nse.suffix_norm_sum_exp(j+1, minHatIdxRi)

        wT = w_nse.suffix_max(j+1)
        if wi >= wT - 1:
            eps3 = np.exp(wT - wi)
    
        vT = v_nse.suffix_max(j+1)
        if vi >= vT - 1:
            eps4 = np.exp(vT - vi)

    eps5 = 2
    #if eps1 < 1:
    #    eps5 = max(1 - 1/(mu * (1+eps2)), mu / (1-eps1) - 1)
    psi1 = max(1 - 1/(mu * (1+eps2)), (1+eps1)*mu - 1)
    psi2 = max(1 - 1/(mu * (1+eps1)), (1+eps2)*mu - 1)
    #psi3 = 2 * max(1 - 1/(mu * (1+eps1)), 1 - 1/(mu * (1+eps2)))
    psi3 = min(2, mu * (1 + min(eps1, eps2))) * max(1 - 1/(mu * (1+eps1)), 1 - 1/(mu * (1+eps2)))
    eps5 = min(psi1, psi2, psi3)

    return eps1, eps2, eps3, eps4, eps5

def build_clusters(wv, jump_factor = 1.5):
    clusters = []
    curr_idx = 0
    curr_min_w = 0
    curr_min_v = 0
    jump_factor = np.log(jump_factor)

    for i in range(1, len(wv)):
        distortion = wv[curr_idx][0] - wv[i][0]
        assert distortion >= 0, f"Error: {distortion}" #wv must be sorted by decreasing w
        if distortion > jump_factor:
            clusters.append( (curr_idx, curr_min_w, curr_min_v) )
            curr_idx, curr_min_w, curr_min_v = i, i, i
        else:
            if wv[i][0] < wv[curr_min_w][0]:
                curr_min_w = i
            if wv[i][1] < wv[curr_min_v][1]:
                curr_min_v = i

    clusters.append( (curr_idx, curr_min_w, curr_min_v) )
    return clusters

def _upperbound_dist(M1, M2, d1, eps_tol = 0.001, min_iter = 5):
    if M1.getNumberOfItems() != M2.getNumberOfItems():
        raise ValueError("MNLs must have same size")
    M1.convertToLogWeights()
    M2.convertToLogWeights()
    n = M1.getNumberOfItems()
    wv = sorted([(M1.weights[i], M2.weights[i]) for i in range(n)], reverse=True)

    distortion_rmq = build_distortion_RMQ(wv)
    w_nse = SuffixNormSumExp([w[0] for w in wv])
    v_nse = SuffixNormSumExp([v[1] for v in wv])
    eps = 0

    clusters = build_clusters(wv)
    clusters.append( (len(wv), -1, -1) )

    for i in range(len(clusters)-1):
        eps1 = 1
        eps_prime = 2 if d1 else 1
        for j in range(i+1, len(clusters)):
            eps1, eps2, eps3, eps4, eps5 = obtain_ub_errors(clusters[i][0], clusters[j][0] - 1, wv, distortion_rmq, w_nse, v_nse, clusters[i][1], clusters[i][2])

            if d1:
                eps_prime = min(eps_prime, eps1 + eps2 + eps5)
            else:
                eps_prime = min(eps_prime, max(eps3, eps4, eps5))
            
            if eps1 + eps2 <= eps_tol and j - i >= min_iter:
                break
        
        eps = max(eps, eps_prime)

    return eps

def error_upperbound(M1, M2, d1=True):
    merr = min(_upperbound_dist(M1, M2, d1), _upperbound_dist(M2, M1, d1))
    if not d1:
        return min(merr, error_upperbound(M1, M2, d1=True) / 2)
    return merr