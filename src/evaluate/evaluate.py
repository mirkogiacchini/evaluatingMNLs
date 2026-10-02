from utils.MNL import *
from scipy.special import softmax
from adaptive_algorithm.lobster_forest import *
import itertools
import numpy as np
from typing import List

def getSlateError(M1:MNL, M2:MNL, S:List[int], norm=1)->float:
    # Computes the ell_1 distance between the winning distributions induced on S by M1 and M2 respectively

    # assert all([isinstance(x,int) for x in S])
    # assert min(S)>=0

    w1 = M1.winningDistribution(S)
    w2 = M2.winningDistribution(S)

    return np.linalg.norm(w1 - w2, ord=norm)

def fullSlateError(M1, M2, norm=1):
    return getSlateError(M1, M2, list(range(M1.getNumberOfItems())), norm=norm)

def mg15_ERMS(M1, M2):
    '''
    error based on the norm of the MNLs, used in the work of: 
    https://proceedings.neurips.cc/paper_files/paper/2015/file/2a38a4a9316c49e5a833517c45d31070-Paper.pdf
    '''
    w1 = np.array(M1.weights)
    w2 = np.array(M2.weights)
    if not M1.isLogWeight():
        w1 = np.log(w1)
    w1 -= np.mean(w1)
    if not M2.isLogWeight():
        w2 = np.log(w2)
    w2 -= np.mean(w2)

    #print('w1:', w1)
    #print('w2:', w2)
    return np.linalg.norm(w1 - w2, ord=2) / np.sqrt(M1.getNumberOfItems())
    
def noh12_normalizedNorm(M1, M2):
    '''
    error based on the norm used here:
    https://papers.nips.cc/paper_files/paper/2012/file/9adeb82fffb5444e81fa0ce8ad8afe7a-Paper.pdf
    '''
    w1 = np.array(M1.weights)
    w2 = np.array(M2.weights)
    if M1.isLogWeight():
        w1 = softmax(w1)
    else:
        w1 /= np.sum(w1)

    if M2.isLogWeight():
        w2 = softmax(w2)
    else:
        w2 /= np.sum(w2)

    return np.linalg.norm(w1 - w2, ord=2) / np.linalg.norm(w1, ord=2)

def computeWorstErrorExactly(M1:MNL, M2:MNL, items=None, d1=True) -> float:
    # computes the maximum ell1 error when using M2 to estimate M1 (or viceversa)
    assert M1.getNumberOfItems() == M2.getNumberOfItems()
    norm = 1 if d1 else np.inf
    if items == None:
        items = list(range(M1.getNumberOfItems()))
    
    err = getSlateError(M1, M2, [items[0],items[1]], norm=norm)
    worst_slate = [items[0],items[1]]
    for r in range(2, len(items) + 1):
        for slate in itertools.combinations(items, r):
            new_err = getSlateError(M1, M2, slate, norm=norm)
            if new_err > err:
                err = new_err
                worst_slate = slate
    return worst_slate, err
