import math, json
import numpy as np
from utils.func import *

class SDF:
    # subset distribution family
    def __init__(self, rnd, name="Unkown MNL"):
        if rnd is None:
            self.rnd = np.random.default_rng()
        else:
            self.rnd = rnd
        self.name = name

    def maxSample(self, slate : list[int]) -> int:
        pass

    def maxSampleManyPair(self, i : int, j : int, N : int) -> tuple[int, int]:
        pass 

    def winningDistribution(self, slate : list[int])->np.array:
        pass 

    def __str__(self):
        return self.name

    def short_name(self):
        return self.name

class MNL(SDF):
    def __init__(self, rnd, weights, logweights=True, name="unkown-MNL"):
        super().__init__(rnd, name)
        self.n = len(weights)
        self.weights = weights 
        self.logweights = logweights
        self.total_queries = 0
    
    def convertToLogWeights(self):
        if not self.logweights:
            self.weights = [np.log(w) for w in self.weights]
            self.logweights = True
        self.rescaleWeights()
    
    def convertToExponentiatedWeights(self):
        if self.logweights:
            self.weights = [np.exp(w) for w in self.weights]
            self.logweights = False
        self.rescaleWeights()

    def accessExplicitWeight(self, i):
        return self.weights[i]

    def isLogWeight(self):
        return self.logweights

    def getNumberOfItems(self)->int:
        return self.n

    def resetQueries(self):
        self.total_queries = 0
    
    def totalQueriesMade(self):
        return self.total_queries

    def dumpWeights(self, path):
        with open(path, 'w') as f:
            json.dump(self.toDict(), f)

    def toDict(self):
        return {'weights': self.weights, 'logweights': self.logweights}
    
    @staticmethod
    def fromDict(d):
        return MNL(None, d['weights'], logweights=d['logweights'])
    
    def rescaleWeights(self):
        m = min(self.weights)
        if self.logweights:
            self.weights = [w - m for w in self.weights]
        else:
            self.weights = [w / m for w in self.weights]
    
    def maxSample(self, slate : list[int]):
        #might not be numerically stable when logweights=True
        self.total_queries += 1

        if not self.logweights:
            tot_weight = sum([self.weights[i] for i in slate])
        else:
            tot_weight = sum([math.exp(self.weights[i]) for i in slate])
        
        rnd_val = self.rnd.random()
        for i in range(len(slate)-1):
            vi = slate[i]
            if not self.logweights:
                p = self.weights[vi] / tot_weight
            else:
                p = math.exp(self.weights[vi]) / tot_weight
            if p >= rnd_val:
                return vi
            rnd_val -= p 
        return slate[-1]
    
    def maxSampleManyPair(self, i : int, j : int, N : int) -> tuple[int, int]:
        #takes N samples on the slate (i,j)
        #returns (#times i won, #times j won)
        self.total_queries += N

        if not self.logweights:
            p = self.weights[i] / (self.weights[i] + self.weights[j])
        else:
            if self.weights[j] - self.weights[i] >= 700: #prevents overflow
                p = 0
            else:
                p = 1 / (1 + np.exp(self.weights[j] - self.weights[i]))
        wins_i = self.rnd.binomial(N, p) # sample a binomial distribution with head probability p
        return (wins_i, N - wins_i)

    def winningDistribution(self, slate : list[int])->np.array:
        '''
            Currently the winning distribution is given as a numpy array
            where the ith entry is the winning probability of the element slate[i]
        '''
        assert all([isinstance(i,int) and (i>=0) and (i<self.n) for i in slate])

        if not self.logweights:
            w = np.array([self.weights[i] for i in slate])
            return w / np.sum(w)

        log_weights = np.array([self.weights[i] for i in slate])
        shifted = log_weights - np.max(log_weights)
        exp_shifted = np.exp(shifted)
        return exp_shifted / np.sum(exp_shifted)
        
        #return np.exp(log_weights) / np.sum(np.exp(log_weights))
        # tot_weight = sum([math.exp(self.weights[i]) for i in slate])
        # return [math.exp(self.weights[i]) / tot_weight for i in slate]

    def generatePairwiseDataset(self, num_slates, num_samples_per_pair):
        true_weights = np.array(self.weights)
        if not self.isLogWeight():
            true_weights = np.log(true_weights)

        n = self.getNumberOfItems()
        num_pairs = n * (n - 1) // 2
        samples_per_pair = self.rnd.multinomial(num_slates, np.ones(num_pairs) / num_pairs)
        samples_per_pair *= num_samples_per_pair
        
        idx1, idx2 = np.triu_indices(n, k=1)
        filter = (samples_per_pair != 0)
        idx1, idx2, samples_per_pair = idx1[filter], idx2[filter], samples_per_pair[filter]
        
        logits = true_weights[idx1] - true_weights[idx2]
        winProb = npStableSigmoid(logits)

        first_item_wins = self.rnd.binomial(samples_per_pair, winProb)
        second_item_wins = samples_per_pair - first_item_wins

        #return idx1, idx2, first_item_wins, second_item_wins
        return np.column_stack([idx1, idx2, first_item_wins, second_item_wins])

    def generateSparsePairwiseDataset(self, num_slates, num_samples_total):      
        # this is more efficient than the other sampling algorithm
        # however, it can generate duplicate slates in the dataset.
        # If the dataset is large, there will be only a few duplicated  

        # choose slates
        _left = self.rnd.integers(self.n, size = num_slates)
        _right = self.rnd.integers(1, self.n, size = num_slates)
        _right = (_left + _right) % self.n

        idx1 = np.minimum(_left, _right) 
        idx2 = np.maximum(_left, _right)

        # compute winning probabilities
        true_weights = np.array(self.weights)
        if not self.isLogWeight():
            true_weights = np.log(true_weights)

        logits = true_weights[idx1] - true_weights[idx2]
        winProb = npStableSigmoid(logits)

        # choose number of samples per slates (balanced)
        base = num_samples_total // num_slates
        rem = num_samples_total % num_slates

        # counts: base everywhere, +1 for 'rem' randomly chosen slates
        counts = np.full(num_slates, base, dtype=np.int64)
        if rem:
            extra_ix = self.rnd.choice(num_slates, size=rem, replace=False)
            counts[extra_ix] += 1

        # generate dataset
        first_item_wins = self.rnd.binomial(counts, winProb)
        second_item_wins = counts - first_item_wins

        assert np.sum(counts) == num_samples_total

        return np.column_stack([idx1, idx2, first_item_wins, second_item_wins])
        
    def __str__(self):
        return f"MNL : {self.name}\n {self.n} items\n {self.total_queries=}"