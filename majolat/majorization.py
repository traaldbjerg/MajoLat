"""
Core majorization theory classes and operations.

This module implements probability vectors and (bi)stochastic matrices
for majorization lattice operations.
"""

import numpy as np


class ProbVector():  # notations from Cicalese and Vaccaro 2002
    """Class for probability vectors. Implements several methods for majorization-related applications. Most notably,
       < and > are overriden to mean 'is majorized by' and 'majorizes', respectively., and + and * are overriden to mean
       'meet' and 'join', respectively. Other useful functions can be used on them, such as monotones.

       Parameters:
       probs (ArrayLike): array of probabilites to convert into a ProbVector object. Must sum to 1
                          (with a tolerance for floating point error).
       rearrange (bool): if True, sorts the entries of probs in nonincreasing order. Defaults to True.
       normalize (bool): if True, will normalize any input array to 1, and then use it as probs. Defaults to True.
    """

    tolerance = 1e-12

    __slots__ = ('dim', 'probs')  # avoids overhead of dictionary

    def __init__(self, probs, rearrange=True, normalize=True):
        norm_1 = np.linalg.norm(probs, ord=1)
        for i in range(len(probs)):
            if probs[i] < 0:
                if probs[i] > -ProbVector.tolerance:  # might mess with fringe cases ?
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

    def getArray(self):  # for compatibility with BistochMatrix but __getitem__ and __setitem__ are enough to retrieve and manipulate the array otherwise
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
        if dim_diff > 0:  # handle differing input dimensions
            q = ProbVector(np.append(other, [0]*dim_diff))
        elif dim_diff < 0:
            p = ProbVector(np.append(self, [0]*-dim_diff))
        switch = True
        sum_p = 0
        sum_q = 0
        for i in range(p.dim):
            sum_p += p[i]
            sum_q += q[i]
            if sum_p - sum_q < -ProbVector.tolerance:
                switch = False
                break
        return switch

    def __eq__(self, other):
        return all(self == other)

    def __ne__(self, other):
        return not self == other

    def __lt__(self, other):  # defines <
        return other.majorizes(self)

    def __gt__(self, other):  # defines >
        return self.majorizes(other)

    def __add__(self, other):
        """Method that handles the construction of the meet of two probability vectors. Notations and algorithm from Cicalese and Vaccaro 2002, which is described in Section 1.3.3 in the manuscript.

        Args:
            other (ProbVector): the second distribution with which to construct the meet.

        Returns:
            meet (ProbVector): the meet of self and other.
        """
        a = np.array([])  # coefficients of alpha(p, q) in the text
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
            a_i = min(p_sum, q_sum) - a_sum  # where a_i is the ith element of the meet of p and q
            a = np.append(a, a_i)  # not very clean but numpy allocates its arrays in a contiguous block of memory so no real alternative
            a_sum += a_i
        meet = ProbVector(a)
        return meet  # returns the glb

    def __mul__(self, other):
        """Method that handles the construction of the join of two probability vectors. Notations and algorithm from Cicalese and Vaccaro 2002, which is described in Section 1.3.4 in the manuscript.

        Args:
            other (ProbVector): the second distribution with which to construct the join.

        Returns:
            join (ProbVector): the join of self and other.
        """
        b = np.array([])  # coefficients of beta(p, q) in the text
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
        for j in range(1, self.dim):  # j+1 in {2, n} in the paper but 0-indexed
            if b[j] > b[j-1]:  # in the case of a concave dent in the lorenz curve
                i = j - 1  # initialization
                s = b[j] + b[j-1]
                a = s/(j-i+1)
                while i > 0:
                    if a > b[i-1]:  # should stop correctly for i = 0 because the second condition would not be evaluated
                        s += b[i-1]
                        i -= 1  # could also do this at the end but then need to change the formula for a
                        a = s/(j-i+1)
                    else:
                        break  # not super clean but this way the conditions are executed correctly
                for k in range(i, j+1):
                    b[k] = a
        join = ProbVector(b)
        return join  # returns the lub

    def meet(self, other):  # alias
        return self + other

    def join(self, other):  # alias
        return self * other

    def __sub__(self, other):
        """Method that computes the entropic distance from Cicalese, Gargano and Vaccaro 2013 (cf. Section 1.4.3 in the manuscript).

        Args:
            other (ProbVector): the endpoint distribution of the segment to measure.

        Returns:
            (float): computed distance.
        """
        from .utils import entropy
        return entropy(self) + entropy(other) - 2*entropy(self * other)  # d(x, y) in the paper

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
        if array is None:  # generate random stochastic matrix
            if dims is None:
                raise ValueError("Need to specify dimensions")
            elif isinstance(dims, int):
                self.array = np.zeros((dims, dims))
            else:
                self.array = np.zeros((dims[0], dims[1]))
            for row in range(dims[0]):
                self.array[row] = np.random.dirichlet(np.ones(dims[1]))
        else:
            self.array = array
        self.dims = (len(self.array), len(self.array[0]))

        # check if stochastic
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

    def isStochastic(self):  # checks sum of rows equal 1 with a tolerance for floating point error
        return np.logical_and(1 - 1e-12 <= np.sum(self.array, axis=1), np.sum(self.array, axis=1) <= 1 + 1e-12).all()  # tolerances

    def __mul__(self, other):  # implements matrix multiplication, might be refactored without the getArray() method but not crucial right now
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
        if array is None:  # generate random bistochastic matrix using Birkhoff-von Neumann theorem
            if dims is None:
                raise ValueError("Need to specify dimensions")
            if rand_combs is None:
                import math
                rand_combs = math.ceil(dims/2)  # default value, mixing but not too mixing
            self.array = np.zeros((dims, dims))
            weights = np.random.rand(rand_combs)  # rand_combs is a very important parameter because it allows to control how mixing the final matrix will be
            weights /= np.sum(weights)
            for i in range(rand_combs):
                perm = np.eye(dims)  # generate the identity
                np.random.shuffle(perm)  # shuffle the identity to get a permutation matrix
                self.array += weights[i]*perm  # mix the permutation matrices together
        else:
            self.array = array
        self.dims = (len(self.array), len(self.array[0]))

        # check if bistochastic
        if not self.isBistochastic():
            raise ValueError("Matrix is not bistochastic")

    def __repr__(self):
        return "BistochMatrix({})".format(self.array)

    def __str__(self):
        return "BistochMatrix({})".format(self.array)

    def isBistochastic(self):  # checks sum of rows and sum of columns both equal 1 with a tolerance for floating point error
        return np.logical_and(1 - 1e-12 <= np.sum(self.array, axis=0), np.sum(self.array, axis=1) <= 1 + 1e-12).all() and self.isStochastic()  # tolerances

    def __mul__(self, other):  # implements matrix multiplication, might be refactored without the getArray() method but not crucial right now
        if isinstance(other, BistochMatrix):
            return BistochMatrix(np.matmul(self.array, other.getArray()))
        elif isinstance(other, StochMatrix):
            return StochMatrix(np.matmul(self.array, other.getArray()))
        elif isinstance(other, ProbVector):
            return ProbVector(np.matmul(self.array, other.getArray()))
        else:
            print("Warning: non-supported type multiplied with {}".format(str(self)))
