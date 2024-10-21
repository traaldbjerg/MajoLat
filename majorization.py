import numpy as np

class ProbVector(): # notations from Cicalese and Vaccaro 2002
    
    tolerance = 1e-25
    
    __slots__ = ('dim', 'probs') # avoids overhead of dictionary
    
    def __init__(self, probs):
        #if sum(probs) != 1: # does floating point error mess this up ?
        #    raise ValueError("Probabilities must sum to 1")
        for i in range(len(probs)):
            if probs[i] < 0:
                if probs[i] > -ProbVector.tolerance: # might mess with fringe cases which might be interesting :(
                    probs[i] = 0
                else:
                    raise ValueError("Probabilities must be nonnegative")
        self.dim = len(probs)
        self.probs = np.array(np.sort(probs)[::-1]) # decreasing order
    
    def __repr__(self):
        return "ProbVector({})".format(self.probs)
    
    def __str__(self):
        return "ProbVector({})".format(self.probs)
    
    def __len__(self):
        return self.dim
    
    def __iter__(self):
        return iter(self.probs)
    
    def __next__(self):
        return next(self.probs)
    
    def getProbs(self):
        return self.probs

    def majorizes(self, other):
        p = self
        q = other
        dim_diff = len(self) - len(other)
        if dim_diff > 0:
            q = ProbVector(np.append(other.getProbs(), [0]*dim_diff))
        elif dim_diff < 0:
            p = ProbVector(np.append(self.getProbs(), [0]*-dim_diff))
        return all([p.getProbs()[:i+1].sum() >= q.getProbs()[:i+1].sum() # +1 because end index is exclusive
                    for i in range(p.dim)]) # true only if p majorizes q
        
    def __eq__(self, other):
        return all(self.getProbs() == other.getProbs())
    
    def __ne__(self, other):
        return not self == other
    
    def __lt__(self, other):
        return other.majorizes(self)
    
    def __gt__(self, other):
        return self.majorizes(other)

    def __add__(self, other): # meet of two probability vectors, notations + algorithm from Cicalese and Vaccaro 2002
        a = np.array([]) # coefficients of alpha(p, q) in the text
        p = self
        q = other
        dim_diff = len(self) - len(other)
        if dim_diff > 0:
            q = ProbVector(np.append(other.getProbs(), [0]*dim_diff))
        elif dim_diff < 0:
            p = ProbVector(np.append(self.getProbs(), [0]*-dim_diff))        
        p_sum = 0
        q_sum = 0
        a_sum = 0
        for i in range(self.dim):
            p_sum += p.getProbs()[i]
            q_sum += q.getProbs()[i]
            a_i = min(p_sum, q_sum) - a_sum # where a_i is the ith element of the meet of p and q
            a = np.append(a, a_i) # not very clean but numpy allocates its arrays in a contiguous block of memory so no real alternative
            a_sum += a_i
        return ProbVector(a) # returns the glb
    
    def __mul__(self, other): # join of two probability vectors, notations + algorithm from Cicalese and Vaccaro 2002
        b = np.array([]) # coefficients of beta(p, q) in the text
        p = self
        q = other
        dim_diff = len(self) - len(other)
        if dim_diff > 0:
            q = ProbVector(np.append(other.getProbs(), [0]*dim_diff))
        elif dim_diff < 0:
            p = ProbVector(np.append(self.getProbs(), [0]*-dim_diff))        
        p_sum = 0
        q_sum = 0
        b_sum = 0
        for i in range(self.dim):
            p_sum += p.getProbs()[i]
            q_sum += q.getProbs()[i]
            b_i = max(p_sum, q_sum) - b_sum
            b = np.append(b, b_i)
            b_sum += b_i
        # the components of beta are not necessarily in decreasing order at this stage, so we need to smoothe out beta to find the lowest lorenz curve that majorizes it
        for j in range(1, self.dim): # j+1 in {2, n} in the paper but 0-indexed
            if b[j] > b[j-1]: # in the case of a concave dent in the lorenz curve
                i = j - 1 # initialization
                s = b[j] + b[j-1]
                a = s/(j-i+1)
                while i > 0 and a > b[i-1]: # should stop correctly for i = 0 because the second condition would not be evaluated
                    s += b[i-1]
                    a = s/(j-i+1)
                    i -= 1
                for k in range(i, j+1):
                    b[k] = a
        return ProbVector(b) # returns the lub
    
    def meet(self, other):
        return self + other
    
    def join(self, other):
        return self * other
    
    