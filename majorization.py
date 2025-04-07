import numpy as np
import math

class ProbVector(): # notations from Cicalese and Vaccaro 2002
    
    tolerance = 1e-12
    
    __slots__ = ('dim', 'probs') # avoids overhead of dictionary
    
    def __init__(self, probs):
        #if sum(probs) != 1: # does floating point error mess this up ?
        #    raise ValueError("Probabilities must sum to 1")
        norm_1 = np.linalg.norm(probs, ord=1)
        for i in range(len(probs)):
            if probs[i] < 0:
                if probs[i] > -ProbVector.tolerance: # might mess with fringe cases which might be interesting :(
                    probs[i] = 0
                else:
                    raise ValueError("Probabilities must be nonnegative")
        self.dim = len(probs)
        # normalize probabilities
        self.probs = np.array(np.sort(probs)[::-1])/norm_1 # decreasing order
    
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
        switch = True
        sum_p = 0
        sum_q = 0
        for i in range(p.dim):
            sum_p += p.getProbs()[i]
            sum_q += q.getProbs()[i]
            #print(sum_p, sum_q) # debug
            if sum_p - sum_q < -ProbVector.tolerance: # this sucks
                switch = False
                break
        return switch
    
    def majorizes_debug(self, other):
        p = self
        q = other
        dim_diff = len(self) - len(other)
        if dim_diff > 0:
            q = ProbVector(np.append(other.getProbs(), [0]*dim_diff))
        elif dim_diff < 0:
            p = ProbVector(np.append(self.getProbs(), [0]*-dim_diff))
        print(p)
        print(q)
        switch = True
        sum_p = 0
        sum_q = 0
        for i in range(p.dim):
            sum_p += p.getProbs()[i]
            sum_q += q.getProbs()[i]
            print(sum_p, sum_q)
            if sum_p < sum_q:
                switch = False
        return switch
            
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
        for i in range(p.dim):
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
        for i in range(p.dim):
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
                while i > 0:
                    if a > b[i-1]: # should stop correctly for i = 0 because the second condition would not be evaluated
                        s += b[i-1]
                        i -= 1 # could also do this at the end but then need to change the formula for a
                        a = s/(j-i+1)
                    else:
                        break # not super clean but this way I'm sure the conditions are executed correctly
                for k in range(i, j+1):
                    b[k] = a
        return ProbVector(b) # returns the lub
    
    def meet(self, other):
        return self + other
    
    def join(self, other):
        return self * other
    
    def __sub__(self, other): # entropic distance as defined in Cicalese and Vaccaro 2013
        return self.entropy() + other.entropy() - 2*(self * other).entropy() # d(x, y) in the paper 
    
    
    
class BistochMatrix(): # useful for degrading 2 vectors with the same bistochastic matrix and seeing what happens
    
    __slots__ = ('array', 'dims')
     
    def __init__(self, array=None, dims=None, rand_combs=None):
        if array is None: # generate random bistochastic matrix using Birkhoff-von Neumann theorem
            if dims is None:
                raise ValueError("Need to specify dimensions")
            if rand_combs is None:
                rand_combs = math.ceil(dims/2) # default value, mixing but not too mixing
            self.array = np.zeros((dims, dims))
            weights = np.random.rand(rand_combs) # rand_combs is a very important parameter because it allows to control how mixing the final matrix will be
            weights /= np.sum(weights)
            for i in range(rand_combs):
                perm = np.eye(dims)
                np.random.shuffle(perm)
                self.array += weights[i]*perm
            #print(self.array)
        else:
            self.array = array
        self.dims = (len(self.array), len(self.array[0]))
        
        #check if bistochastic
        if not self.isBistochastic():
            raise ValueError("Matrix is not bistochastic")
        
    def __repr__(self):
        return "BistochMatrix({})".format(self.array)
    
    def __str__(self):
        return "BistochMatrix({})".format(self.array)
    
    def __len__(self):
        return self.dims
    
    def __iter__(self):
        return iter(self.array)
    
    def __next__(self):
        return next(self.array)
    
    def getMatrix(self):
        return self.array
    
    def isBistochastic(self):
        #return np.all(np.sum(self.array, axis=0) == 1) and np.all(np.sum(self.array, axis=1) == 1)
        return True # for now, probably need to implement a tolerance
        
    def __mul__(self, other):
        if isinstance(other, BistochMatrix):
            return BistochMatrix(np.dot(self.array, other.getMatrix()))
        elif isinstance(other, ProbVector):
            return ProbVector(np.dot(self.array, other.getProbs()))
        


### GENERAL PURPOSE FUNCTIONS ###
  
def entropy(v):
    return -sum([p*np.log2(p) for p in v.getProbs() if p != 0]) # ln instead of log2

def guessing_entropy(v):
    return sum((i+1)* v.getProbs()[i] for i in range(len(v))) # + 1 because of 0-indexing

def renyi_entropy(v, alpha):
    if alpha == 0: # hartley entropy
        return np.log2(len(v))
    elif alpha == 1: # shannon entropy
        return entropy(v)
    elif alpha == np.inf: # min-entropy
        return -np.log2(max(v.getProbs()))
    else: # general renyi entropy
        return 1/(1-alpha)*np.log2(sum([p**alpha for p in v.getProbs()])) # ln instead of log2

def mutual_information(p, q):
    return entropy(p) + entropy(q) - entropy(p + q) # as defined in Cicalese and Vaccaro 2002

def relative_entropy(p, q): # as defined in Thomas and Cover 2006
    dim_diff = len(p) - len(q)
    p_new = p
    q_new = q
    if dim_diff > 0:
        q_new = ProbVector(np.append(q.getProbs(), [0]*dim_diff))
    elif dim_diff < 0:
        p_new = ProbVector(np.append(p.getProbs(), [0]*-dim_diff))        

    res = 0
    for i in range(len(p_new)):
        if p_new.getProbs()[i] > 0 and q_new.getProbs()[i] == 0:
            return np.inf
        elif p_new.getProbs()[i] == 0 and q_new.getProbs()[i] == 0:
            pass
        elif p_new.getProbs()[i] == 0 and q_new.getProbs()[i] > 0:
            pass
        else:
            res += p_new.getProbs()[i]*np.log2(p_new.getProbs()[i]/q_new.getProbs()[i])
    return res

def d(p,q): # as defined in Cicalese and Vaccaro 2013
    return entropy(p) + entropy(q) - 2*entropy(p * q)

def d_prime(p, q):
    return 2 * entropy(p + q) - entropy(p) - entropy(q) # quasi-distance going trough the meet instead of the join

def D(p, q): # as defined in Cicalese and Vaccaro 2013
    n = max(len(p), len(q))
    return 2/n * (2 * guessing_entropy(p + q) - guessing_entropy(p) - guessing_entropy(q))

def S(p, q, alpha = 1): # basically just supermodularity
    return (- renyi_entropy(p, alpha) - renyi_entropy(q, alpha) + renyi_entropy(p * q, alpha) + renyi_entropy(p + q, alpha))

def d_subadd(p, q, alpha = 1):
    return - renyi_entropy(p + q, alpha) + renyi_entropy(p, alpha) + renyi_entropy(q, alpha)

def E_plus(p, q): # see theorem 2
    return d_prime(p, p + q)

def E_minus(p, q): # see theorem 1
    return d(p, p * q)

def E_t(p, q):
    return E_plus(p, q) - E_minus(p, q)

