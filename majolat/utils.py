"""
Utility functions for entropy, distance measures, and visualization.

This module contains various entropy measures, incomparability functions,
and plotting tools that work with ProbVector objects.
"""

import numpy as np
import matplotlib.pyplot as plt
import itertools
from .majorization import ProbVector


### ENTROPY AND INFORMATION MEASURES ###

def entropy(v):
    """Implements the Shannon entropy of a probability distribution.

    Args:
        v (ProbVector or ArrayLike): distribution to compute the entropy of.

    Returns:
        (float): value of the Shannon entropy of v, in bits
    """
    return -sum([p*np.log2(p) for p in v if p != 0])


def renyi_entropy(v, alpha):
    """Implements the Rényi entropy of arbitrary order (cf. Section 1.2.3).

    Args:
        v (ProbVector): distribution to compute the Rényi entropy of.
        alpha (float): order parameter of the Rényi entropy. Some of the limits (e.g. alpha = 1) are coded explicitly.

    Returns:
        (float): value of the Rényi entropy of v of order alpha.
    """
    if alpha == 0:  # hartley entropy
        return np.log2(len(v))
    elif alpha == 1:  # shannon entropy
        return entropy(v)
    elif alpha == np.inf:  # min-entropy
        return -np.log2(max(v))
    else:  # general renyi entropy
        return 1/(1-alpha)*np.log2(sum([p**alpha for p in v]))


def mutual_information(p, q):
    """Implements the lattice-based mutual entropy analogy from Cicalese and Vaccaro 2002 (not in the manuscript).

    Args:
        p (ProbVector): first distribution of the pair of which to compute the mutual entropy of
        q (ProbVector): second distribution of the pair of which to compute the mutual entropy of

    Returns:
        (float): value of the mutual information
    """
    return entropy(p) + entropy(q) - entropy(p + q)  # as defined in Cicalese and Vaccaro 2002


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



### SUPERMODULARITY ###

# see A. Américo, M. H. R. Khouzani, and P. Malacaria, Channel-Supermodular Entropies: Order Theory and an Application to Query Anonymization, Entropy 24, 39 (2022)

def cc_entropy(wrap, phi, v): # core-concave entropy
    return wrap(phi(v))

def ar_entropy(p, alpha): # alpha-Arimoto-Renyi entropy
    return alpha/(1 - alpha) * np.log2(np.linalg.norm(p, ord=alpha))

def hr_entropy(p, alpha): # alpha-Hayashi-Renyi entropy
    return 1/(1 - alpha) * np.log2(np.sum([pi**alpha for pi in p]))

def tsallis_entropy(p, alpha): # Tsallis entropy
    return 1/(alpha - 1) * (1 - np.sum([pi**alpha for pi in p]))

def sm_entropy(p, alpha, beta): # Sharma-Mittal entropy
    return 1/(beta - 1) * (1 - (np.sum([pi**alpha for pi in p])))**((1 - beta)/(1 - alpha))



### DISTANCE MEASURES ###

def d(p, q):
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
    return 2 * entropy(p + q) - entropy(p) - entropy(q)  # quasi-distance going through the meet instead of the join


### INCOMPARABILITY MEASURES ###

def E_future(p, q):
    """Implements the future incomparability monotone between two distributions on the lattice (cf. Def. 4.1 in the manuscript).

    Args:
        p (ProbVector): probe state.
        q (ProbVector): reference state.

    Returns:
        (float): value of the future incomparability of p relative to q
    """
    return d(p, p * q)  # see Theorem 4.1


def E_past(p, q):
    """Implements the past incomparability monotone between two distributions on the lattice (cf. Def. 4.2 in the manuscript).

    Args:
        p (ProbVector): probe state.
        q (ProbVector): reference state.

    Returns:
        (float): value of the past incomparability of p relative to q
    """
    return d_prime(p, p + q)  # see Theorem 4.2
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


### UNIQUENESS ENTROPY ###

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
        if state < p:  # save unnecessary computation time
            return 0
    entropy_sum = entropy(p)
    if reduce:
        b = remove_majorizers(bank)
    else:  # mostly debug purposes
        b = bank
    for i in range(len(b)):  # inefficient, should be optimized (reuse previous joins instead of computing them from scratch every single time ?)
        for combination in itertools.combinations(b, i + 1):  # generate all the combinations of length i+1 (0-indexing)
            joined_state = combination[0]
            for state in combination:  # generate the joined_state
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
    for i in range(1, len(b)):
        for j in range(i-1):  # avoid comparing index i to index i or we would delete all states
            if b[i] < b[j]:
                index_record.append(j)  # jth state majorizes and so is below on the lattice
            elif b[i] > b[j]:  # if several copies of same state, elif only removes one of the pair -> only one is left in the final bank
                index_record.append(i)
    index_record = sorted(set(index_record), reverse=True)  # removes duplicates indexes and reverse order to avoid indexerror on pops
    for i in index_record:
        b.pop(i)
    return b


### MISCELLANEOUS FUNCTIONS ###

def construct_concatenated(p, q):
    """Implements the ordered vector of two probability vectors (cf. Def. 3.1 in the manuscript), used to numerically
       test the majorization precursor postulated for supermodularity (cf. Conjecture 3.1).

    Args:
        p (ProbVector): first distribution to concatenate.
        q (_type_): second distribution to concatenate.

    Returns:
        (ProbVector): ordered concatenation of p and q. Not normalized to 1.
    """
    b = np.array([])  # coefficients of beta(p, q) in the text
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
    A = ProbVector(np.hstack([p, q]), normalize=False)  # sum is 2
    B = ProbVector(np.hstack([(p+q), b]), normalize=False)  # b is already only an array
    return A, B


def guessing_entropy(v):
    """Implements the guessing entropy of a probability distribution, models the average number of tries needed for guessing a secret with an optimal strategy
       (i.e. guessing the most likely possibility first, then the second etc.)

    Args:
        v (ProbVector): distribution to compute the guessing entropy of.

    Returns:
        (float): value of the guessing entropy of v.
    """
    return sum((i+1)* v[i] for i in range(len(v)))  # + 1 because of 0-indexing


def D(p, q):  # as defined in Cicalese and Vaccaro 2013
    n = max(len(p), len(q))
    return 2/n * (2 * guessing_entropy(p + q) - guessing_entropy(p) - guessing_entropy(q))


def S(p, q, alpha=1):  # basically just supermodularity but with renyi entropies, note that using S on concatenations can yield negative values as * and + automatically renormalize their output
    return (- renyi_entropy(p, alpha) - renyi_entropy(q, alpha) + renyi_entropy(p * q, alpha) + renyi_entropy(p + q, alpha))


def d_subadd(p, q, alpha=1):
    return - renyi_entropy(p + q, alpha) + renyi_entropy(p, alpha) + renyi_entropy(q, alpha)


def concatenate(p, q, rearrange=False, normalize=False):
    r = ProbVector(np.hstack([p, q]), rearrange=rearrange, normalize=normalize)
    return r


def TV(p, q):  # total variation distance of two distributions
    res = 1/2*sum([abs(p[i] - q[i]) for i in range(min(len(p), len(q)))]) + sum([p[i] if len(q) < len(p) else q[i] for i in range(min(len(p), len(q)), max(len(p), len(q)))])
    return res


def split_resource(d, f, split):
    res = abs(f(d[:split]) - f(d[split:]))
    return res


### VISUALIZATION TOOLS ###

def plot_lorenz_curves(*prob_vectors, labels=None, colors=None, markers=None, title="Lorenz Curves",
                       linestyles=None, figsize=(6, 6)):
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
        if type(pv) == ProbVector:  # in case we want to plot something that is not in non-increasing order
            p = np.array(pv)
        else:
            p = pv
        cumulative = np.insert(np.cumsum(p), 0, 0)  # prepend 0
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
