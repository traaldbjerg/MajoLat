from majolat import (
    ProbVector, renyi_entropy, vidal_probability, tsallis_entropy, relative_entropy, concatenate, entropy, W_divergence
)
import numpy as np
import matplotlib
from tqdm import tqdm
import itertools



def generate_attempts(dimensions = 5, tries = 10000, ratio=0, comp=0, n=3, hypothesis=True):
    """Implements a generic hypothesis test through statistical sampling. The code should be tweaked manually to change the hypothesis.

    Args:
        dimensions (int, optional): _description_. Defaults to 5.
        tries (int, optional): _description_. Defaults to 10000.
        ratio (int, optional): _description_. Defaults to 0.
        comp (int, optional): _description_. Defaults to 0.
        bank_size (int, optional): _description_. Defaults to 4.
        hypothesis (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    record_1 = 0
    record_2 = 0

    def phi(x): # concave function
        return - x ** 2
    def phi_prime(x): # derivative of the concave function
        return - 2 * x
    def F(p): # sum-concave function
        return np.sum([phi(pi) for pi in p])

    for _ in tqdm(range(tries), desc="Testing hypothesis"):
        switch = True
        while switch: # prevent comparability
            p = ProbVector(np.random.dirichlet(np.ones(dimensions))) # uniform over k-1 simplex
            q = ProbVector(np.random.dirichlet(np.ones(dimensions))) # uniform over k-1 simplex
            if (p < q or p > q):
                switch = False
        m = p + q
        j = p - q
        t = p * q
        m.rearrange()
        j.rearrange()
        t.rearrange()
        #for alpha in range(11, 99):
            #alpha = alpha / 10
            #if W_divergence(m, t, alpha) > 2**alpha * W_divergence(concatenate(m, ProbVector([1, 0]), rearrange=True)/2,
            #                                                            concatenate(p, q, rearrange = True)/2, alpha):
            #    print(alpha)
            #    print(p)
            #    print(q)
            #    print(W_divergence(m, t, alpha))
            #    print(2 ** alpha * W_divergence(concatenate(m, ProbVector([1, 0])/2, rearrange=True),
            #                                                            concatenate(p, q, rearrange = True)/2, alpha))
            #    print("Hypo false")
            #    record_1 += 1
            #if tsallis_entropy(p, alpha) + tsallis_entropy(q, alpha) - tsallis_entropy(m, alpha) < 2 ** alpha * W_divergence(concatenate(m, ProbVector([1, 0]), rearrange=True)/2, concatenate(p, q, rearrange=True)/2, alpha):
            #if renyi_entropy(p, alpha) + renyi_entropy(q, alpha) - renyi_entropy(m, alpha) < np.log2(np.e) * W_divergence(m, t, alpha): # nats or bits ?
            #    print("oh no")
            #    record_1 += 1
            #if tsallis_entropy(m, alpha) + tsallis_entropy(j, alpha) - tsallis_entropy(p, alpha) - tsallis_entropy(q, alpha) + 1e-12 < 2 ** alpha * W_divergence(concatenate(p, q, rearrange=True)/2, concatenate(m, j, rearrange=True)/2, alpha):
                #print(tsallis_entropy(m, alpha))
                #print(alpha)
                #print(2 ** alpha * W_divergence(concatenate(p, q, rearrange=True)/2, concatenate(m, j, rearrange=True)/2, alpha))
                #record_1 += 1
            #print(2 * relative_entropy(concatenate(p, q, rearrange=True)/2,
            #                                                                            concatenate(m, j, rearrange=True)/2)/entropy(m))
        if F(p) + F(q) - F(m) - F(ProbVector([1, 0])) < W_divergence(concatenate(m, ProbVector([1, 0]), rearrange=True), concatenate(p, q, rearrange=True), phi=phi, phi_prime=phi_prime):
            #print(p)
            #print(q)
            #print(m)
            #print(F(p) + F(q) - F(m) - F(ProbVector([1, 0])))
            #print(W_divergence(concatenate(m, ProbVector([1, 0]), rearrange=True), concatenate(p, q, rearrange=True), phi=phi, phi_prime=phi_prime))
            record_1 += 1
        #if F(m) + F(j) - F(p) - F(q) < W_divergence(concatenate(p, q, rearrange=True), concatenate(m, j, rearrange=True), phi=phi, phi_prime=phi_prime):
        #    print(F([1])) # debug
        #    print(F(m))
        #    print(F(p))
        #    print(W_divergence(concatenate(p, q, rearrange=True), concatenate(m, j, rearrange=True), phi=phi, phi_prime=phi_prime))
        #    record_1 += 1
        #if not (concatenate(m, j, rearrange=True) < concatenate(p, q, rearrange=True)):
        #    print("oh no")
        #if not (concatenate(m, ProbVector([1, 0]), rearrange=True) > concatenate(p, q, rearrange=True)):
        #    print("oh no")
        #if F(p) < F(q) + W_divergence(q, p, phi=phi, phi_prime=phi_prime):
        #    print("oh no")
    
    print(record_1)
    #print(record_2)
    return hypothesis, comp

def generate_bank(dims, total, ocr=0, distribution=None):
    if distribution == None:
        distribution = np.ones(dims) # sample uniformly
    b = []
    for i in range(total - ocr): # number of normal states
        b.append(ProbVector(np.random.dirichlet(distribution)))
    for _ in range(ocr): # number of jokers
        b.append(ProbVector([1/dims for _ in range(dims)]))
    return b

if __name__ == "__main__":
    dimensions = 5
    tries = 10000
    ratio = 0
    comp = 0
    n = 7
    hypothesis = True

    hypothesis, comp = generate_attempts(dimensions=dimensions, tries=tries, ratio=ratio, comp=comp, n=n, hypothesis=hypothesis)
    print(hypothesis)