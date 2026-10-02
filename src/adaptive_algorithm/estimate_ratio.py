from utils.MNL import *
import numpy as np
from math import ceil as ceiling

def fastEstimateRatio(M:MNL,i:int,j:int, alpha:float,epsilon:float,delta:float)->tuple[int,int]:
    c = alpha/(alpha+1)

    lnDelta = np.log(6 / delta)
    m = ceiling(np.log(6/delta)/(c * (epsilon * epsilon))) 
 
    batch_size = min(10, m)
    mi, mj, t = 0, 0, 0

    while t < m:
        t += batch_size
        sample = M.maxSampleManyPair(i, j, batch_size)
        mi += sample[0]
        mj += sample[1]
        
        et = math.sqrt((0.5/t)*np.log(4 * t * t / delta))
        p_i = mi/t
        p_j = mj/t

        ub_sigma = min(1/4, p_i + et, p_j+et)
        lambda_t = (1/ (3*t)) * np.log(t * t /delta) + (1/(2*t)) * \
            math.sqrt((4/9)*(np.log(t * t/delta)) + 8*t*ub_sigma* np.log(t * t/delta))
        et = min(et,lambda_t)

        if p_i + et < c: 
            return (0, -1)
        if p_j + et < c:
            return(-1, 0)
        
        if mj != 0 and mi != 0:
            ub_ij_ratio = (p_i + et) / (p_j - et)
            lb_ij_ratio = (p_i - et) / (p_j + et)
            ub_ji_ratio = (p_j + et) / (p_i - et)
            lb_ji_ratio = (p_j - et) / (p_i + et)
            if (1-epsilon) * ub_ij_ratio <= mi / mj <= (1+epsilon) * lb_ij_ratio and \
                (1-epsilon) * ub_ji_ratio <= mj / mi <= (1+epsilon) * lb_ji_ratio:
                return (mi, mj)

        #update maximum number of queries based on the estimate of the ratio
        if min(p_i - et, p_j - et) > c:
            m = min(m, ceiling(lnDelta / (min(p_i - et, p_j - et) * (epsilon * epsilon))))

        batch_size = min(batch_size * 2, m - t)

    if p_i < c/2:
        return (0, -1)
    if p_j < c/2:
        return (-1, 0)
    return (mi, mj)
