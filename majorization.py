import math
import numpy as np
import matplotlib.pyplot as plt
import itertools


class ProbVector(): # notations from Cicalese and Vaccaro 2002
    """Class for probability vectors. Implements several methods for majorization-related applications. Most notably,
       < and > are overriden to mean 'is majorized by' and 'majorizes', respectively., and + and * are overriden to mean
       'meet' and 'join', respectively. Other useful functions can be used on them, such as incomparability monotones or the
       uniqueness entropy.

       Parameters:
       probs (ArrayLike): array of probabilites to convert into a ProbVector object. Must sum to 1
                          (with a tolerance for floating point error).
       rearrange (bool): if True, sorts the entries of probs in nonincreasing order. Defaults to True. 
       normalize (bool): if True, will normalize any input array to 1, and then use it as probs. Defaults to True. 
    """
    
    tolerance = 1e-12
    
    __slots__ = ('dim', 'probs') # avoids overhead of dictionary
    
    def __init__(self, probs, rearrange=True, normalize=True):
        #if sum(probs) != 1: # does floating point error mess this up ?
        #    raise ValueError("Probabilities must sum to 1")
        norm_1 = np.linalg.norm(probs, ord=1)
        for i in range(len(probs)):
            if probs[i] < 0:
                if probs[i] > -ProbVector.tolerance: # might mess with fringe cases ?
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
    
    def getArray(self): # for compatibility with BistochMatrix but __getitem__ and __setitem__ are enough to retrieve and manipulate the array otherwise
        return self.probs
    
    def __getitem__(self, index):
        return self.probs[index]
    
    def __setitem__(self, index, value):
        self.probs[index] = value

    def majorizes(self, other):
        """Method that handles the majorization relation. Implements Eq. (1.2) from the manuscript.

        Args:
            other (ProbVector): the ProbVector to compare self to.

        Returns:
            switch (bool): whether self majorizes other or not
        """
        p = self
        q = other
        dim_diff = len(self) - len(other)
        if dim_diff > 0: # handle differing input dimensions
            q = ProbVector(np.append(other, [0]*dim_diff))
        elif dim_diff < 0:
            p = ProbVector(np.append(self, [0]*-dim_diff))
        switch = True
        sum_p = 0
        sum_q = 0
        for i in range(p.dim):
            sum_p += p[i]
            sum_q += q[i]
            #print(sum_p, sum_q) # debug
            if sum_p - sum_q < -ProbVector.tolerance:
                switch = False
                break
        return switch
            
    def __eq__(self, other):
        return all(self == other)
    
    def __ne__(self, other):
        return not self == other
    
    def __lt__(self, other): # defines <
        return other.majorizes(self)
    
    def __gt__(self, other): # defines >
        return self.majorizes(other)

    def __add__(self, other):
        """Method that handles the construction of the meet of two probability vectors. Notations and algorithm from Cicalese and Vaccaro 2002, which is described in Section 1.3.3 in the manuscript.

        Args:
            other (ProbVector): the second distribution with which to construct the meet.

        Returns:
            meet (ProbVector): the meet of self and other.
        """
        a = np.array([]) # coefficients of alpha(p, q) in the text
        p = self
        q = other
        dim_diff = len(self) - len(other)
        if dim_diff > 0:
            q = ProbVector(np.append(other, [0]*dim_diff))
        elif dim_diff < 0:
            p = ProbVector(np.append(self, [0]*-dim_diff))
        p_sum = 0
        q_sum = 0
        a_sum = 0
        for i in range(p.dim):
            p_sum += p[i]
            q_sum += q[i]
            a_i = min(p_sum, q_sum) - a_sum # where a_i is the ith element of the meet of p and q
            a = np.append(a, a_i) # not very clean but numpy allocates its arrays in a contiguous block of memory so no real alternative
            a_sum += a_i
        meet = ProbVector(a)
        return meet # returns the glb
    
    def __mul__(self, other):
        """Method that handles the construction of the join of two probability vectors. Notations and algorithm from Cicalese and Vaccaro 2002, which is described in Section 1.3.4 in the manuscript.

        Args:
            other (ProbVector): the second distribution with which to construct the join.

        Returns:
            join (ProbVector): the join of self and other.
        """
        b = np.array([]) # coefficients of beta(p, q) in the text
        p = self
        q = other
        dim_diff = len(self) - len(other)
        if dim_diff > 0:
            q = ProbVector(np.append(other, [0]*dim_diff))
        elif dim_diff < 0:
            p = ProbVector(np.append(self, [0]*-dim_diff))        
        p_sum = 0
        q_sum = 0
        b_sum = 0
        for i in range(p.dim):
            p_sum += p[i]
            q_sum += q[i]
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
                        break # not super clean but this way the conditions are executed correctly
                for k in range(i, j+1):
                    b[k] = a
        join = ProbVector(b)
        return join # returns the lub
    
    def meet(self, other): # alias
        return self + other
    
    def join(self, other): # alias
        return self * other
    
    def __sub__(self, other):
        """Method that computes the entropic distance from Cicalese, Gargano and Vaccaro 2013 (cf. Section 1.4.3 in the manuscript).

        Args:
            other (ProbVector): the endpoint distribution of the segment to measure.

        Returns:
            (float): computed distance.
        """
        return self.entropy() + other.entropy() - 2*(self * other).entropy() # d(x, y) in the paper 
    
    def __truediv__(self, other):
        res = []
        for i in range(self.dim):
            res.append(self[i]/other)
        res = ProbVector(res, normalize=False, rearrange=False)
        return res
    


class StochMatrix():
    """Class for stochastic matrices. Implements several methods for majorization-related applications. Mostly useful 
       for degrading two different probability vectors and comparing the final result, or checking monotonicity results.    

       Parameters:
       array (ArrayLike): array of coefficients to convert into a BistochMatrix object. Rows must sum to
                          1 (with a tolerance). Defaults to None.
       dims (int): The size of the matrix, i.e. output object will be of dimension dims x dims. Defaults to None. 
       rand_combs (int):  specifies the number of permutation matrices to randomly generate and mix together to produce
                          the final bistochastic matrix (cf. Theorem 1.1 in the manuscript). Very important parameter,
                          allows to control how mixing the final matrix will be. Defaults to None, which leads to dims/2
                          being selected if left at None.
    """
    
    __slots__ = ('array', 'dims')
    
    def __init__(self, array=None, dims=None):
        if array is None: # generate random stochastic matrix
            if dims is None:
                raise ValueError("Need to specify dimensions")
            elif isinstance(dims, int):
                self.array = np.zeros((dims, dims))
            else:
                self.array = np.zeros((dims[0], dims[1]))
            for row in range(dims[0]):
                self.array[row] = np.random.dirichlet(np.ones(dims[1]))
            #print(self.array)
        else:
            self.array = array
        self.dims = (len(self.array), len(self.array[0]))
        
        #check if stochastic
        if not self.isStochastic():
            raise ValueError("Matrix is not stochastic")
        
    def __repr__(self):
        return "StochMatrix({})".format(self.array)
    
    def __str__(self):
        return "StochMatrix({})".format(self.array)
    
    def __len__(self):
        return self.dims
    
    def __iter__(self):
        return iter(self.array)
    
    def __next__(self):
        return next(self.array)
    
    def __getitem__(self, index):
        return self.array[index]
    
    def __setitem__(self, index, value):
        self.array[index] = value
    
    def getArray(self):
        return self.array
    
    def isStochastic(self): # checks sum of rows equal 1 with a tolerance for floating point error
        return np.logical_and(1 - 1e-12 <= np.sum(self.array, axis=1), np.sum(self.array, axis=1) <= 1 + 1e-12).all() # tolerances
        
    def __mul__(self, other): # implements matrix multiplication, might be refactored without the getArray() method but not crucial right now
        if isinstance(other, StochMatrix):
            return StochMatrix(np.matmul(self.array, other.getArray()))
        elif isinstance(other, ProbVector):
            return ProbVector(np.matmul(self.array, other.getArray()))
        else:
            print("Warning: non-supported type multiplied with {}".format(str(self)))

    
    
class BistochMatrix(StochMatrix):
    """Class for bistochastic matrices. Implements several methods for majorization-related applications. Mostly useful 
       for degrading two different probability vectors and comparing the final result, or checking monotonicity results.    

       Parameters:
       array (ArrayLike): array of coefficients to convert into a BistochMatrix object. Rows and columns must sum to
                          1 (with a tolerance). Defaults to None.
       dims (int): The size of the matrix, i.e. output object will be of dimension dims x dims. Defaults to None. 
       rand_combs (int):  specifies the number of permutation matrices to randomly generate and mix together to produce
                          the final bistochastic matrix (cf. Theorem 1.1 in the manuscript). Very important parameter,
                          allows to control how mixing the final matrix will be. Defaults to None, which leads to dims/2
                          being selected if left at None. Remark: should be set manually for very low dimensions
    """
    
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
                perm = np.eye(dims) # generate the identity
                np.random.shuffle(perm) # shuffle the identity to get a permutation matrix
                self.array += weights[i]*perm # mix the permutation matrices together
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
    
    def isBistochastic(self): # checks sum of rows and sum of columns both equal 1 with a tolerance for floating point error
        return np.logical_and(1 - 1e-12 <= np.sum(self.array, axis=0), np.sum(self.array, axis=1) <= 1 + 1e-12).all() and self.isStochastic() # tolerances
        
    def __mul__(self, other): # implements matrix multiplication, might be refactored without the getArray() method but not crucial right now
        if isinstance(other, BistochMatrix):
            return BistochMatrix(np.matmul(self.array, other.getArray()))
        elif isinstance(other, StochMatrix):
            return StochMatrix(np.matmul(self.array, other.getArray()))
        elif isinstance(other, ProbVector):
            return ProbVector(np.matmul(self.array, other.getArray()))
        else:
            print("Warning: non-supported type multiplied with {}".format(str(self)))
        


### GENERAL PURPOSE FUNCTIONS ###
  
def entropy(v):
    """Implements the Shannon entropy of a probability distribution.

    Args:
        v (ProbVector or ArrayLike): distribution to compute the entropy of.

    Returns:
        (float): value of the Shannon entropy of v, in bits
    """
    return -sum([p*np.log2(p) for p in v if p != 0]) # ln instead of log2

def renyi_entropy(v, alpha):
    """Implements the Rényi entropy of arbitrary order (cf. Section 1.2.3).

    Args:
        v (ProbVector): distribution to compute the Rényi entropy of.
        alpha (float): order parameter of the Rényi entropy. Some  of the limits (e.g. alpha = 1) are coded explicitly.

    Returns:
        (float): value of the Rényi entropy of v of order alpha.
    """
    if alpha == 0: # hartley entropy
        return np.log2(len(v))
    elif alpha == 1: # shannon entropy
        return entropy(v)
    elif alpha == np.inf: # min-entropy
        return -np.log2(max(v))
    else: # general renyi entropy
        return 1/(1-alpha)*np.log2(sum([p**alpha for p in v]))

def mutual_information(p, q):
    """Implements the lattice-based mutual entropy analogy from Cicalese and Vaccaro 2002 (not in the manuscript).

    Args:
        p (ProbVector): first distribution of the pair of which to compute the mutual entropy of
        q (ProbVector): second distribution of the pair of which to compute the mutual entropy of

    Returns:
        (float): value of the mutual information
    """
    return entropy(p) + entropy(q) - entropy(p + q) # as defined in Cicalese and Vaccaro 2002

def relative_entropy(p, q):
    """Implements relative entropy as defined in Thomas and Cover 2006 (cf. Definition B.3).

    Args:
        p (ProbVector): first distribution, to be compared to q
        q (ProbVector): second distribution, to which p is compared

    Returns:
        (float): value of D(p || q).
    """
    dim_diff = len(p) - len(q)
    p_new = p
    q_new = q
    if dim_diff > 0:
        q_new = ProbVector(np.append(q, [0]*dim_diff))
    elif dim_diff < 0:
        p_new = ProbVector(np.append(p, [0]*-dim_diff))        

    res = 0
    for i in range(len(p_new)):
        if p_new[i] > 0 and q_new[i] == 0:
            return np.inf
        elif p_new[i] == 0 and q_new[i] == 0:
            pass
        elif p_new[i] == 0 and q_new[i] > 0:
            pass
        else:
            res += p_new[i]*np.log2(p_new[i]/q_new[i])
    return res


# quantities from Chapter 1
def d(p,q):
    """Implements the entropic distance from Cicalese, Gargano and Vaccaro 2013 (cf. Def. 1.14 in the manuscript)

    Args:
        p (ProbVector): startpoint distribution of the distance to be computed.
        q (ProbVector): endpoint distribution of the distance to be computed.

    Returns:
        (float): value of the entropic distance from p to q.
    """
    return entropy(p) + entropy(q) - 2*entropy(p * q)

def d_prime(p, q):
    """Implements the meet-based entropic quasidistance (cf. Def. 1.15 in the manuscript)

    Args:
        p (ProbVector): startpoint distribution of the distance to be computed.
        q (ProbVector): endpoint distribution of the distance to be computed.

    Returns:
        (float): value of the entropic quasidistance from p to q.
    """
    return 2 * entropy(p + q) - entropy(p) - entropy(q) # quasi-distance going through the meet instead of the join

# function for Chapter 3
def construct_concatenated(p, q):
    """Implements the ordered vector of two probability vectors (cf. Def. 3.1 in the manuscript), used to numerically
       test the majorization precursor postulated for supermodularity (cf. Conjecture 3.1).

    Args:
        p (ProbVector): first distribution to concatenate.
        q (_type_): second distribution to concatenate.

    Returns:
        (ProbVector): ordered concatenation of p and q. Not normalized to 1.
    """
    b = np.array([]) # coefficients of beta(p, q) in the text
    p_copy = p
    q_copy = q
    dim_diff = len(p_copy) - len(q_copy)
    if dim_diff > 0:
        q_copy = ProbVector(np.append(q_copy, [0]*dim_diff))
    elif dim_diff < 0:
        p_copy = ProbVector(np.append(p_copy, [0]*-dim_diff))        
    p_sum = 0
    q_sum = 0
    b_sum = 0
    for i in range(p_copy.dim):
        p_sum += p_copy[i]
        q_sum += q_copy[i]
        b_i = max(p_sum, q_sum) - b_sum
        b = np.append(b, b_i)
        b_sum += b_i
    print(b)
    A = ProbVector(np.hstack([p, q]), normalize=False) # sum is 2
    B = ProbVector(np.hstack([(p+q), b]), normalize=False) # b is already only an array
    return A, B

# quantities from Chapter 4
def E_future(p, q):
    """Implements the future incomparability monotone between two distributions on the lattice (cf. Def. 4.1 in the manuscript).

    Args:
        p (ProbVector): probe state.
        q (ProbVector): reference state.

    Returns:
        (float): value of the future incomparability of p relative to q
    """
    return d(p, p * q) # see Theorem 4.1

def E_past(p, q):
    """Implements the past incomparability monotone between two distributions on the lattice (cf. Def. 4.2 in the manuscript).

    Args:
        p (ProbVector): probe state.
        q (ProbVector): reference state.

    Returns:
        (float): value of the past incomparability of p relative to q
    """
    return d_prime(p, p + q) # see Theorem 4.2
                             # also equal to d(p, p + q) but d' is slightly cheaper for computation time (because of meet in def)
                            

def F(p, q):
    """Implements the distance-like incomparability fucntion between two distributions on the lattice
       (cf. Def. 4.3 in the manuscript).

    Args:
        p (ProbVector): probe state.
        q (ProbVector): reference state.

    Returns:
        (float): value of the distance-like incomparability of p relative to q
    """
    return E_future(p, q) - E_past(p, q)

def G(p, q):
    """Implements the area-like incomparability function between two distributions on the lattice
       (cf. Def. 4.4 in the manuscript).

    Args:
        p (ProbVector): probe state.
        q (ProbVector): reference state.

    Returns:
        (float): value of the area-like incomparability of p relative to q
    """
    return E_future(p, q) * E_past(p, q)



# functions from Chapter 5
def unique_entropy(p, bank, reduce=True):
    """Implements the uniqueness entropy of p relative to a bank (cf. Def. 5.1 in the manuscript)

    Args:
        p (ProbVector): probe state.
        bank (list(ProbVector)): set of reference states to compare p to.
        reduce (bool: whether to remove trivial terms in the calculation (cf. property 4 of uniqueness entropy).
                      Defaults to True.

    Returns:
        (float): value of the uniqueness entropy of p relative to the bank 
    """
    for state in bank:
        if state < p: # save unnecessary computation time
            return 0
    entropy_sum = entropy(p)
    if reduce:
        b = remove_majorizers(bank)
    else: # mostly debug purposes
        b = bank
    for i in range(len(b)): # inefficient, should be optimized (reuse previous joins instead of computing them from scratch every single time ?)
        for combination in itertools.combinations(b, i + 1): # generate all the combinations of length i+1 (0-indexing)
            joined_state = combination[0]
            for state in combination: # generate the joined_state
                joined_state = joined_state * state
            entropy_sum += ((-1) ** (i+1)) * entropy(p * joined_state)
    return entropy_sum

def remove_majorizers(bank):
    """Function that automatically removes the states that only add computation time but do not change the final value of
       the uniqueness entropy (cf. property 4, Section 5.1.3). To be used in unique_entropy.

    Args:
        bank (list of ProbVector): list of states in the bank, from which to find and remove the majorizers.

    Returns:
        b (list of ProbVector): list of states left in the bank after removal of redundant states.
    """
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



# miscellaneous functions
def guessing_entropy(v):
    """Implements the guessing entropy of a probability distribution (see Cicalese, Gargano and Vaccaro 2013),
       mostly for a Gini index.

    Args:
        v (ProbVector): distribution to compute the guessing entropy of.

    Returns:
        (float): value of the guessing entropy of v.
    """
    return sum((i+1)* v[i] for i in range(len(v))) # + 1 because of 0-indexing

def D(p, q): # as defined in Cicalese and Vaccaro 2013
    n = max(len(p), len(q))
    return 2/n * (2 * guessing_entropy(p + q) - guessing_entropy(p) - guessing_entropy(q))

def S(p, q, alpha = 1): # basically just supermodularity but with renyi entropies, note that using S on concatenations can yield negative values as * and + automatically renormalize their output
    return (- renyi_entropy(p, alpha) - renyi_entropy(q, alpha) + renyi_entropy(p * q, alpha) + renyi_entropy(p + q, alpha))

def d_subadd(p, q, alpha = 1):
    return - renyi_entropy(p + q, alpha) + renyi_entropy(p, alpha) + renyi_entropy(q, alpha)

def concatenate(p, q, rearrange=False, normalize=False):
    r = ProbVector(np.hstack([p, q]), rearrange=rearrange, normalize=normalize)
    return r

def TV(p, q): # total variation distance of two distributions
    res = 1/2*sum([abs(p[i] - q[i]) for i in range(min(len(p), len(q)))]) + sum([p[i] if len(q) < len(p) else q[i] for i in range(min(len(p), len(q)), max(len(p), len(q)))])

def split_resource(d, f, split):
    res = abs(f(d[:split]) - f(d[split:]))
    return res



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
            p = np.array(pv)
        else:
            p = pv
        #p_sorted = np.sort(p)[::-1]
        cumulative = np.insert(np.cumsum(p), 0, 0) # prepend 0
        n = len(p)
        x = np.linspace(0, n, n+1)  # normalized x-axis
        plt.plot(x, cumulative, marker=marker, label=label, color=color, linestyle=linestyle)

    plt.xlabel("Component")
    plt.ylabel("Cumulative probability")
    plt.title(title)
    plt.legend()
    plt.xticks(x)
    plt.grid(True)
    plt.xlim(0, n)
    plt.ylim(0, 1)
    plt.show()
