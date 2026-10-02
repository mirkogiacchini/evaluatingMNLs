from utils.MNL import *
from math import log as ln
import numpy as np
from dataclasses import dataclass
from adaptive_algorithm.estimate_ratio import *


class Cluster:
    def __init__(self,center=None, elements=None):
        self.center = center # int
        self.elements = elements # List[int]
        self.ratios = {} # Dict{(int,int):(int,int)} #ratio saved as (numerator, denominator)

    def getSize(self):
        return len(self.elements)
    
    def getCenter(self):
        return self.center

    def getRatio(self, i, j):
        return self.ratios[(i,j)]

    def getItems(self):
        return self.elements

class EstimationTree:
    def __init__(self, logweights):
        self.logweights = logweights
        self.ratios = {} #Dict{(int,int):(int,int)}
        self.eval_ratios = {} # Dict{(int,int):float} # ratio evaluted to numerator / denominator
        self.root = -1 #int
        self.adj_list = {} #Dict{int: list[int]}
    
    def addNeighbor(self, i, j, ratio, lb_ratio=None, ub_ratio=None):
        if i not in self.adj_list:
            self.adj_list[i] = []
        self.adj_list[i].append(j)
        self.ratios[(i,j)] = ratio

        if self.logweights:
            eval_r = np.log(ratio[0]) - np.log(ratio[1])
        else:
            eval_r = ratio[0] / ratio[1]

        if lb_ratio is not None:
            eval_r = max(eval_r, lb_ratio)
        if ub_ratio is not None:
            eval_r = min(eval_r, ub_ratio)

        self.eval_ratios[(i,j)] = eval_r 

    def getNeighborhood(self, v):
        if v not in self.adj_list:
            return []
        return self.adj_list[v]

    def getRatio(self, i, j):
        return self.ratios[(i,j)]
    
    def getEvaluatedRatio(self, i, j):
        return self.eval_ratios[(i,j)]

    def setRoot(self, root):
        self.root = root
    
    def getRoot(self):
        return self.root

    def isInLogWeights(self):
        return self.isInLogWeights

class EstimationLobster(EstimationTree):
    def __init__(self, logweights):
        super().__init__(logweights)
        self.backbone = []
    
    def appendBackboneItem(self, i):
        self.backbone.append(i)

    def getBackbone(self):
        return self.backbone

def noisyRecursiveQuickSort(M:MNL, S:list, alpha:float, epsilon:float, delta:float,qs_rnd,fast=False)->list[Cluster]:
    if len(S) == 0:
        return []
    
    n = M.n
    c = qs_rnd.choice(S)
    #print(f'Noisy clustering, current list: {S} ; current center: {c}')
    L, C, R = [], Cluster(c,[c]), []
    for s in S:
        if s==c:
            continue
        # Compute r(c,s)
        (mc, ms) = fastEstimateRatio(M, c, s, alpha, epsilon, delta/(6 * n)) # might need n**2 to have guarantee

        if mc == 0:
            R.append(s)
        elif mc==-1:
            L.append(s)
        else:
            C.elements.append(s)
            C.ratios[(c,s)] = (mc,ms)
            #C.ratios[(s,c)] = (ms,mc)
        
    return noisyRecursiveQuickSort(M, L, alpha, epsilon, delta, qs_rnd, fast) + [C] + noisyRecursiveQuickSort(M, R, alpha, epsilon, delta, qs_rnd, fast)

def noisyQuickSort(M:MNL, alpha, epsilon:float, delta:float,qs_rnd,fast=False)->list[int]:
    return noisyRecursiveQuickSort(M, [i for i in range(M.getNumberOfItems())],alpha, epsilon, delta, qs_rnd, fast)

def quicksortClustering(M : MNL, alpha:float, epsilon:float, delta:float, qs_rnd)->list[Cluster]:
    return noisyRecursiveQuickSort(M, list(range(M.getNumberOfItems())), alpha, epsilon, delta, qs_rnd, True)
    
def buildEstimationForest(M : MNL, qs_rnd, epsilon:float, delta:float, alpha:float = 0.5, logweights:bool = True, verbose=False) -> list[EstimationTree]:
    M.resetQueries()
    #eps1 = (1+epsilon)**0.2 - 1
    eps1 = epsilon 

    n = M.getNumberOfItems()
    clusters = quicksortClustering(M, alpha, eps1, delta, qs_rnd)
    
    item_to_cluster = {}
    for it,cluster in enumerate(clusters):
        for item in cluster.getItems():
            item_to_cluster[item] = it

    if verbose:
        print(f'Clustering done. Total queries to far: {M.totalQueriesMade()}')
        print(f'Number of clusters: {len(clusters)}')
        print(f'Cluster sizes: {[cluster.getSize() for cluster in clusters]}')

    T = len(clusters)
    Z = []
    beta = []
    for i in range(T):
        if len(Z) == 0:
            Z.append(clusters[i].getSize())
        else:
            Z.append(alpha * Z[-1] + clusters[i].getSize())
        
        beta.append( epsilon / Z[-1] ) #might break theoretical guarantees


    i = T - 1
    j = T - 2
    current_center = clusters[i].getCenter()
    forest = [EstimationLobster(logweights)]
    forest[-1].setRoot(current_center)
    forest[-1].appendBackboneItem(current_center)

    for v in clusters[i].getItems():
        if v != current_center:
            forest[-1].addNeighbor(current_center, v, clusters[i].getRatio(current_center, v))

    while j >= 0:
        if verbose:
            print(f'//Building Forest// i: {i} ; j: {j} ; beta_j: {beta[j]}')
        assert item_to_cluster[current_center] != item_to_cluster[clusters[j].getCenter()]
        r_ij = (0, -1)
        retry = 5
        while r_ij == (0, -1) and retry > 0:
            retry -= 1 
            r_ij = fastEstimateRatio(M, clusters[i].getCenter(), clusters[j].getCenter(), beta[j], eps1, delta / (6 * T)) #6*n
            # if r_ij == (0, -1), then estimateRatio has failed: try again

        if r_ij != (-1, 0): # i not "infinitely larger than" j
            center_j = clusters[j].getCenter()
            
            cc = item_to_cluster[current_center]
            cj = item_to_cluster[center_j]
            ub_r, lb_r = None, None
            if logweights:
                if cc > cj:
                    lb_r = - abs(cc - cj) * ln(alpha)
                else:
                    ub_r = abs(cc - cj) * ln(alpha)
            else:
                if cc > cj:
                    lb_r = (1/alpha)**abs(cc - cj)
                else:
                    ub_r = (alpha)**abs(cc - cj) 
            forest[-1].addNeighbor(current_center, center_j, r_ij, ub_ratio=ub_r, lb_ratio=lb_r)

            for v in clusters[j].getItems():
                if v != center_j:
                    forest[-1].addNeighbor(center_j, v, clusters[j].getRatio(center_j, v))
            j -= 1
        elif i == j+1:
            i = j
            j -= 1
            current_center = clusters[i].getCenter()
            forest.append(EstimationLobster(logweights))
            forest[-1].setRoot(current_center)
            forest[-1].appendBackboneItem(current_center)

            for v in clusters[i].getItems():
                if v != current_center:
                    forest[-1].addNeighbor(current_center, v, clusters[i].getRatio(current_center, v))
        else:
            i = j+1
            current_center = clusters[i].getCenter()
            forest[-1].appendBackboneItem(current_center)

    if verbose:
        print(f'Forest computed. Total queries: {M.totalQueriesMade()}')
        print(f'#connected components: {len(forest)}')
        #print(f'Roots: {[f.getRoot() for f in forest]}')
        #print(f'Item-to-cluster: {item_to_cluster}')
    
    return forest, item_to_cluster

def generateWeightsForTree(tree, current_vertex, item_to_cluster, alpha, weights, log_weights):
    #print(f'Exploring vertex: {current_vertex}')
    for neigh in tree.getNeighborhood(current_vertex):
        
        ratio_cn = tree.getEvaluatedRatio(current_vertex, neigh)

        if log_weights:
            w_neigh = weights[current_vertex] - ratio_cn
        else:
            w_neigh = weights[current_vertex] / ratio_cn
        
        weights[neigh] = w_neigh

        generateWeightsForTree(tree, neigh, item_to_cluster, alpha, weights, log_weights)
        
def generateWeights(forest, item_to_cluster, epsilon, alpha, log_weights = False):
    n = len(item_to_cluster)
    weights = {}
    max_weight = -1
    lnNeps = 1 + ln(n) - ln(epsilon)
    last_cluster = -1
    for i in range(len(forest)-1, -1, -1):
        assert item_to_cluster[forest[i].getRoot()] > last_cluster
        last_cluster = item_to_cluster[forest[i].getRoot()]
    
        new_weights = {}
        new_weights[forest[i].getRoot()] = (0 if log_weights else 1)
        generateWeightsForTree(forest[i], forest[i].getRoot(), item_to_cluster, alpha, new_weights, log_weights=log_weights)
        
        min_weight = min(new_weights.values())

        if max_weight > -1:
            for item in new_weights.keys():
                if log_weights:
                    new_weights[item] += max_weight - min_weight + lnNeps
                else:
                    new_weights[item] *= max_weight / min_weight * (2 * n / epsilon)
            
        for item,w in new_weights.items():
            weights[item] = w
            max_weight = max(max_weight, w)

    #print(f'Final weights: {weights}')
    listWeights = [weights[i] for i in range(n)]
    return MNL(None, listWeights, logweights=log_weights)


def lobsterForestAlgorithm(M : MNL, qs_rnd, epsilon:float, delta:float, alpha:float = 0.5, verbose=False) -> MNL:
    estimationForest, itemsToCluster = buildEstimationForest(M, qs_rnd, epsilon, delta, alpha=alpha, logweights=M.isLogWeight(), verbose=verbose)
    weights = generateWeights(estimationForest, itemsToCluster, epsilon, alpha, log_weights=M.isLogWeight())
    return weights, estimationForest, itemsToCluster