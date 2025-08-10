import math
import numpy as np
import matplotlib.pyplot as plt
import itertools


class ProbVector(): # notations from Cicalese and Vaccaro 2002
    
    tolerance = 1e-12
    
    __slots__ = ('dim', 'probs') # avoids overhead of dictionary
    
    def __init__(self, probs, rearrange=True, normalize=True):
        #if sum(probs) != 1: # does floating point error mess this up ?
        #    raise ValueError("Probabilities must sum to 1")
        norm_1 = np.linalg.norm(probs, ord=1)
        for i in range(len(probs)):
            if probs[i] < 0:
                if probs[i] > -ProbVector.tolerance: # might mess with fringe cases
                    probs[i] = 0
                else:
                    raise ValueError("Probabilities must be nonnegative")
        self.dim = len(probs)
        # normalize and rearrange probabilities
        if rearrange:
            probs = np.sort(probs)[::-1]
        self.probs = np.array(probs)
        if normalize:
            self.probs = self.probs/norm_1

    
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
    
    def getArray(self):
        return self.probs

    def majorizes(self, other):
        p = self
        q = other
        dim_diff = len(self) - len(other)
        if dim_diff > 0:
            q = ProbVector(np.append(other.getArray(), [0]*dim_diff))
        elif dim_diff < 0:
            p = ProbVector(np.append(self.getArray(), [0]*-dim_diff))
        switch = True
        sum_p = 0
        sum_q = 0
        for i in range(p.dim):
            sum_p += p.getArray()[i]
            sum_q += q.getArray()[i]
            #print(sum_p, sum_q) # debug
            if sum_p - sum_q < -ProbVector.tolerance: # this sucks
                switch = False
                break
        return switch
            
    def __eq__(self, other):
        return all(self.getArray() == other.getArray())
    
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
            q = ProbVector(np.append(other.getArray(), [0]*dim_diff))
        elif dim_diff < 0:
            p = ProbVector(np.append(self.getArray(), [0]*-dim_diff))
        p_sum = 0
        q_sum = 0
        a_sum = 0
        for i in range(p.dim):
            p_sum += p.getArray()[i]
            q_sum += q.getArray()[i]
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
            q = ProbVector(np.append(other.getArray(), [0]*dim_diff))
        elif dim_diff < 0:
            p = ProbVector(np.append(self.getArray(), [0]*-dim_diff))        
        p_sum = 0
        q_sum = 0
        b_sum = 0
        for i in range(p.dim):
            p_sum += p.getArray()[i]
            q_sum += q.getArray()[i]
            b_i = max(p_sum, q_sum) - b_sum
            b = np.append(b, b_i)
            b_sum += b_i
        # the components of beta are not necessarily in decreasing order at this stage, so we need to smoothe out beta to find the lowest lorenz curve that still majorizes both p and q
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
    
    def __sub__(self, other): # entropic distance as defined in Cicalese, Gargano and Vaccaro 2013
        return self.entropy() + other.entropy() - 2*(self * other).entropy() # d(x, y) in the paper 
    
    
 
    
class BistochMatrix(): # useful for degrading 2 vectors with the same bistochastic matrix
    
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
    
    def getArray(self):
        return self.array
    
    def isBistochastic(self):
        return np.all(1 - 1e-12 <= np.sum(self.array, axis=0) <= 1 + 1e-12) and np.all(1 - 1e-12 <= np.sum(self.array, axis=1) <= 1 + 1e-12) # tolerances
        
    def __mul__(self, other):
        if isinstance(other, BistochMatrix):
            return BistochMatrix(np.matmul(self.array, other.getArray()))
        elif isinstance(other, ProbVector):
            return ProbVector(np.matmul(self.array, other.getArray()))
        


### GENERAL PURPOSE FUNCTIONS ###
  
def entropy(v):
    return -sum([p*np.log2(p) for p in v.getArray() if p != 0]) # ln instead of log2

def guessing_entropy(v):
    return sum((i+1)* v.getArray()[i] for i in range(len(v))) # + 1 because of 0-indexing

def renyi_entropy(v, alpha):
    if alpha == 0: # hartley entropy
        return np.log2(len(v))
    elif alpha == 1: # shannon entropy
        return entropy(v)
    elif alpha == np.inf: # min-entropy
        return -np.log2(max(v.getArray()))
    else: # general renyi entropy
        return 1/(1-alpha)*np.log2(sum([p**alpha for p in v.getArray()])) # ln instead of log2

def mutual_information(p, q):
    return entropy(p) + entropy(q) - entropy(p + q) # as defined in Cicalese and Vaccaro 2002

def relative_entropy(p, q): # as defined in Thomas and Cover 2006
    dim_diff = len(p) - len(q)
    p_new = p
    q_new = q
    if dim_diff > 0:
        q_new = ProbVector(np.append(q.getArray(), [0]*dim_diff))
    elif dim_diff < 0:
        p_new = ProbVector(np.append(p.getArray(), [0]*-dim_diff))        

    res = 0
    for i in range(len(p_new)):
        if p_new.getArray()[i] > 0 and q_new.getArray()[i] == 0:
            return np.inf
        elif p_new.getArray()[i] == 0 and q_new.getArray()[i] == 0:
            pass
        elif p_new.getArray()[i] == 0 and q_new.getArray()[i] > 0:
            pass
        else:
            res += p_new.getArray()[i]*np.log2(p_new.getArray()[i]/q_new.getArray()[i])
    return res

def d(p,q): # as defined in Cicalese and Vaccaro 2013
    return entropy(p) + entropy(q) - 2*entropy(p * q)

def d_prime(p, q):
    return 2 * entropy(p + q) - entropy(p) - entropy(q) # quasi-distance going through the meet instead of the join

def D(p, q): # as defined in Cicalese and Vaccaro 2013
    n = max(len(p), len(q))
    return 2/n * (2 * guessing_entropy(p + q) - guessing_entropy(p) - guessing_entropy(q))

def S(p, q, alpha = 1): # basically just supermodularity but with renyi entropies
    return (- renyi_entropy(p, alpha) - renyi_entropy(q, alpha) + renyi_entropy(p * q, alpha) + renyi_entropy(p + q, alpha))

def d_subadd(p, q, alpha = 1):
    return - renyi_entropy(p + q, alpha) + renyi_entropy(p, alpha) + renyi_entropy(q, alpha)

def E_past(p, q): # see theorem 2, E^- in the text
    return d_prime(p, p + q)

def E_future(p, q): # see theorem 1, E^+ in the text
    return d(p, p * q)

def F(p, q):
    return E_future(p, q) - E_past(p, q)

def G(p, q):
    return E_future(p, q) * E_past(p, q)



def remove_majorized(bank): # remove states that add nothing but unnecessary terms in the sums for majorization cone calculations
    b = bank
    index_record = []
    for i in range(1,len(b)):
        for j in range(i-1): # avoid comparing index i to index i or we would delete all states
            if b[i] < b[j]:
                index_record.append(j) # jth state majorizes and so is below on the lattice
            elif b[i] > b[j]: # if several copies of same state, elif only removes one of the pair -> only one is left in the final bank
                index_record.append(i)
    index_record = sorted(set(index_record), reverse=True) # removes duplicates indexes and reverse order to avoid indexerror on pops
    for i in index_record:
        #print(i)
        b.pop(i)
    return b

def unique_entropy(p, bank, reduce=True): # unique accessible future volume from p and not from the rest of the state bank, using ansatz H(p) instead of convex polytope volume computations
    for state in bank:
        if state < p: # save unnecessary computation time
            return 0
    entropy_sum = entropy(p)
    if reduce:
        b = remove_majorized(bank)
    else: # mostly debug purposes
        b = bank
    for i in range(len(b)): # inefficient, should be optimized (reuse previous joins instead of computing them from scratch every single time ?)
        for combination in itertools.combinations(b, i + 1): # generate all the combinations of length i+1 (0-indexing)
            joined_state = combination[0]
            for state in combination: # generate the joined_state
                joined_state = joined_state * state
            entropy_sum += ((-1) ** (i+1)) * entropy(p * joined_state)
    return entropy_sum

def construct_concatenated(p, q): # hypothesis test for the alternative supermodularity/subadditivity proof
    b = np.array([]) # coefficients of beta(p, q) in the text
    p_copy = p
    q_copy = q
    dim_diff = len(p_copy) - len(q_copy)
    if dim_diff > 0:
        q_copy = ProbVector(np.append(q_copy.getArray(), [0]*dim_diff))
    elif dim_diff < 0:
        p_copy = ProbVector(np.append(p_copy.getArray(), [0]*-dim_diff))        
    p_sum = 0
    q_sum = 0
    b_sum = 0
    for i in range(p_copy.dim):
        p_sum += p_copy.getArray()[i]
        q_sum += q_copy.getArray()[i]
        b_i = max(p_sum, q_sum) - b_sum
        b = np.append(b, b_i)
        b_sum += b_i
    print(b)
    A = ProbVector(np.hstack([p.getArray(), q.getArray()]), normalize=False) # sum is 2
    B = ProbVector(np.hstack([(p+q).getArray(), b]), normalize=False) # b is already only an array
    return A, B




# display tools
def plot_lorenz_curves(*prob_vectors, labels=None, colors=None, markers=None, title="Lorenz Curves",
                       linestyles=None, figsize=(6,6)):
    """
    Plot the Lorenz curves for one or more ProbVector instances.

    Parameters:
        *prob_vectors: Variable number of ProbVector instances.
        labels (list of str, optional): labels for the probability vectors.
        colors (list of str, optional): colors for the probability vectors.
        markers (list of str, optional): markers for the probability vector
        title (str, optional): title of the plot.
        figsize (tuple, optional): Size of the plot.
    """

    if labels is None:
        labels = [f"Vector {i+1}" for i in range(len(prob_vectors))]
    
    if colors is None:
        colors = ['b'] * len(prob_vectors)

    if markers is None:
        markers = ['o'] * len(prob_vectors)

    if linestyles is None:
        linestyles = ['solid'] * len(prob_vectors)

    plt.figure(figsize=figsize)
    
    for pv, label, color, marker, linestyle in zip(prob_vectors, labels, colors, markers, linestyles):
        if type(pv) == ProbVector: # in case we want to plot something that is not in non-increasing order
            p = np.array(pv.getArray())
        else:
            p = pv
        #p_sorted = np.sort(p)[::-1]
        cumulative = np.insert(np.cumsum(p), 0, 0)  # prepend 0
        n = len(p)
        x = np.linspace(0, n, n+1)  # normalized x-axis
        plt.plot(x, cumulative, marker=marker, label=label, color=color, linestyle=linestyle)

    #plt.plot([0, 1], [0, 1], 'k--', label="Perfect equality")  # 45-degree line
    plt.xlabel("Component")
    plt.ylabel("Cumulative probability")
    plt.title(title)
    plt.legend()
    plt.xticks(x)
    plt.grid(True)
    plt.xlim(0, n)
    plt.ylim(0, 1)
    plt.show()
