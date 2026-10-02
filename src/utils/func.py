import numpy as np

def npStableSigmoid(x):
    out = np.empty_like(x, dtype=np.float64)
    pos = (x >= 0)
    out[pos]  = 1.0 / (1.0 + np.exp(-x[pos]))
    expx      = np.exp(x[~pos])
    out[~pos] = expx / (1.0 + expx)
    return out

def stableSigmoidFloat(x):
    if x >= 0:
        return 1.0 / (1.0 + np.exp(-x))
    expx = np.exp(x)
    return expx / (1.0 + expx)
    
def binary_search_UB(vt, value, index=0):
    l, r = 0, len(vt)
    while l < r:
        m = (l + r) // 2
        if vt[m][index] > value:
            r = m
        else:
            l = m+1
    return l-1

class RMSQ:
    # credits: ChatGPT 5.1 (tested and double-checked)
    """
    Range Minimum/Maximum Query on a static array.
    - Preprocessing: O(n log n)
    - Query: O(1)
    Indices are 0-based, queries are inclusive: [l, r].
    """

    def __init__(self, arr):
        if not arr:
            raise ValueError("Array must be non-empty")
        self.n = len(arr)
        self.log = [0] * (self.n + 1)
        for i in range(2, self.n + 1):
            self.log[i] = self.log[i // 2] + 1

        K = self.n.bit_length()    # ~ floor(log2(n)) + 1

        # st_min[k][i] = min on interval [i, i + 2^k - 1]
        # st_max[k][i] = max on interval [i, i + 2^k - 1]
        self.st_min = [[0] * self.n for _ in range(K)]
        self.st_max = [[0] * self.n for _ in range(K)]

        # k = 0: intervals of length 1
        for i in range(self.n):
            self.st_min[0][i] = arr[i]
            self.st_max[0][i] = arr[i]

        # build for k > 0
        k = 1
        while (1 << k) <= self.n:
            length = 1 << k
            half = length >> 1
            for i in range(self.n - length + 1):
                self.st_min[k][i] = min(self.st_min[k - 1][i],
                                        self.st_min[k - 1][i + half])
                self.st_max[k][i] = max(self.st_max[k - 1][i],
                                        self.st_max[k - 1][i + half])
            k += 1

        # prefix-sum
        self.prefix_sum = [0]
        for v in arr:
            self.prefix_sum.append(self.prefix_sum[-1] + v)
        self.arr = arr

    def range_min(self, l, r):
        """Return min(arr[l:r+1])."""
        if not (0 <= l <= r < self.n):
            raise IndexError("Invalid range")
        length = r - l + 1
        k = self.log[length]
        return min(self.st_min[k][l],
                   self.st_min[k][r - (1 << k) + 1])

    def range_max(self, l, r):
        """Return max(arr[l:r+1])."""
        if not (0 <= l <= r < self.n):
            raise IndexError("Invalid range")
        length = r - l + 1
        k = self.log[length]
        return max(self.st_max[k][l],
                   self.st_max[k][r - (1 << k) + 1])

    def range_sum(self, l, r):
        """Return sum(arr[l:r+1])."""
        if not (0 <= l <= r < self.n):
            raise IndexError("Invalid range")
        return self.prefix_sum[r+1] - self.prefix_sum[l]
    
class SuffixNormSumExp:
    def __init__(self, arr, default_infty = 2):
        self.n = len(arr)
        self._suffix_max = list(arr)
        for i in range(len(arr)-2, -1, -1):
            self._suffix_max[i] = max(self._suffix_max[i], self._suffix_max[i+1])

        self.normalized_sum_exp = [1]
        self.currMax = [self._suffix_max[self.n-1]]
        for i in range(self.n-2, -1, -1):
            newMax = self._suffix_max[i]
            s = self.normalized_sum_exp[-1] * np.exp(self.currMax[-1] - newMax)
            s += np.exp(arr[i] - newMax)
            self.normalized_sum_exp.append(s)
            self.currMax.append(newMax)

        self.normalized_sum_exp.reverse()
        self.currMax.reverse()
        self.arr = arr
        self.default_infty = default_infty

    def suffix_max(self, i):
        """returns max(arr[i:n])"""
        if not (0 <= i < self.n):
            raise IndexError("Invalid range")
        return self._suffix_max[i]

    def suffix_norm_sum_exp(self, i, j):
        """returns sum(exp(arr[i:n])) / exp(arr[j])"""
        mx = self.currMax[i]
        if mx - self.arr[j] >= 600: # exponential unstable if self.arr[j] < mx, so assume the worst
            return self.default_infty
        return self.normalized_sum_exp[i] * np.exp(mx - self.arr[j]) 