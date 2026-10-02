from utils.MNL import *
import heapq, math
from utils.func import *
import numpy as np

def getPairL1Error(M1, M2, i, j):
    d1 = M1.winningDistribution([i,j])
    d2 = M2.winningDistribution([i,j])
    return abs(d1[0] - d2[0]) + abs(d1[1] - d2[1])

def worstPairPrunedSearch(M1, M2, max_iterations = None):
    M1.convertToLogWeights()
    M2.convertToLogWeights()

    n = M1.getNumberOfItems()
    c = [M1.weights[i] - M2.weights[i] for i in range(n)]
    K = sorted(range(n), key=lambda i: c[i])
    
    def key_value(i, j):
        return 2 * math.tanh(abs(c[K[i]] - c[K[j]]) / 4.0)
    
    heap = [(-key_value(0, n-1), 0, n-1)]
    visited = set([(0, n-1)])
    max_dist = -1
    max_pair = None
    num_iter = 0
    while heap:
        neg_ub, l, r = heapq.heappop(heap)
        if -neg_ub <= max_dist:
            continue
        dist = getPairL1Error(M1, M2, K[l], K[r]) 
        if dist > max_dist:
            max_dist = dist
            max_pair = (min(K[l], K[r]), max(K[l], K[r]))
        if l == r-1:
            continue
        if (l+1, r) not in visited:
            visited.add((l+1, r))
            ub_child = key_value(l+1, r)
            if ub_child > max_dist:
                heapq.heappush(heap, (-ub_child, l+1, r))
        if (l, r-1) not in visited:
            visited.add((l, r-1))
            ub_child = key_value(l, r-1)
            if ub_child > max_dist:
                heapq.heappush(heap, (-ub_child, l, r-1))
        
        num_iter += 1
        if max_iterations is not None and num_iter >= max_iterations:
            #check if it's optimal or not
            optimal = True
            while heap:
                neg_ub, _, _ = heapq.heappop(heap)
                if -neg_ub > max_dist:
                    optimal = False
                    break
            return max_dist, max_pair, optimal

    return max_dist, max_pair, True


def worstPairApproximate(M1, M2, epsilon, max_iterations_in_pruned_search_timesn = -1):
    '''
    returns: approximate worst pair, the error induced by the approximate worst pair, and an upper bound on the worst possible error on pairs
    Note: if the algorithm can certify that the pair is the worst one, the upper bound is equal to the error induced by this pair
    '''
    epsilon /= 2

    M1.convertToLogWeights()
    M2.convertToLogWeights()

    if max_iterations_in_pruned_search_timesn > 0:
        max_iter = M1.getNumberOfItems() * max_iterations_in_pruned_search_timesn
        error, worst_pair, exact_error = worstPairPrunedSearch(M1, M2, max_iterations=max_iter)
        if exact_error:
            return worst_pair, error, error
    else:
        worst_pair, error = (0,1), getPairL1Error(M1, M2, 0,1)
    
    n = M1.getNumberOfItems()
    sorted_idx = sorted(range(n), key=lambda x: M1.weights[x])

    WV = [(M1.weights[i], M2.weights[i]) for i in sorted_idx]
    wv_index = {i: sorted_idx[i] for i in range(n)}

    # filter WV
    WV_filtered = [WV[0]]
    wv_filter_idx = {0: wv_index[0]}
    for i in range(1, len(WV)):
        assert WV_filtered[-1][0] <= WV[i][0]
        if WV_filtered[-1][1] >= WV[i][1]:
            continue
        wv_filter_idx[len(WV_filtered)] = wv_index[i]
        WV_filtered.append(WV[i])

    # fix i and find the approximate best j
    eps1 = 1 / math.ceil(1 / epsilon)
    for i in range(n-1, -1, -1):
        #print('current error:', error)
        max_alpha = min(1 - eps1, stableSigmoidFloat(WV[i][0] - WV_filtered[0][0]))
        
        mul = int(math.ceil(max_alpha / eps1))
        while mul >= 0 and 2 * (mul+1) * eps1 > error:
            #fix a threshold, note that we can do some pruning (eg, (mul+1) * eps1 is an upper bound on the l_infty error we'll find)
            alpha = mul * eps1

            if mul == 0:
                j = len(WV_filtered) - 1
            else:
                threshold = np.log(1 - alpha) - np.log(alpha) + WV[i][0]
                j = binary_search_UB(WV_filtered, threshold, index=0)
            
            if j >= 0 and wv_index[i] == wv_filter_idx[j]:
                # same item, take the previous one
                j -= 1
            if j >= 0:
                assert wv_index[i] != wv_filter_idx[j]
                wij = stableSigmoidFloat(WV[i][0] - WV_filtered[j][0])
                vij = stableSigmoidFloat(WV[i][1] - WV_filtered[j][1])
                assert wij >= alpha, f"{wij} < {alpha}"
                new_error = 2 * (wij - vij) 
                if new_error > error:
                    error = new_error
                    worst_pair = (wv_index[i], wv_filter_idx[j])
            
            mul -= 1

    # multiply by 2 for d_1 instead of d_infty
    return worst_pair, error, error + 2 * epsilon