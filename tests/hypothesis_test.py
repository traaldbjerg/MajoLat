from majolat import (
    ProbVector, renyi_entropy, vidal_probability
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
    record = 0
    for _ in tqdm(range(tries), desc="Testing hypothesis"):
        switch = True
        while switch: # prevent comparability
            p = ProbVector(np.random.dirichlet(np.ones(dimensions))) # uniform over k-1 simplex
            q = ProbVector(np.random.dirichlet(np.ones(dimensions))) # uniform over k-1 simplex
            if not (p < q or p > q) and p[-1] > q[-1]: # tensoring can only help if last proba is not the bottleneck
                switch = False
        res_direct = vidal_probability(p, q)
        res_multi = vidal_probability(p**n, (p+q) * p**(n-1))
        #print(vidal_probability(p * p, (p + q) * (p - q)) * vidal_probability((p + q) * (p - q), q * q))
        res_supermod = vidal_probability(p**n, (p + q) * (p - q) * p**(n-2))
        #vidal_probability(p**n, (p + q) * p**(n-1))
        if res_multi < res_supermod - 1e-12:
            print(res_direct)
            print(res_multi)
            print(res_supermod)
            record += 1
        
    print(record)
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
    dimensions = 4
    tries = 10000
    ratio = 0
    comp = 0
    n = 7
    hypothesis = True

    hypothesis, comp = generate_attempts(dimensions=dimensions, tries=tries, ratio=ratio, comp=comp, n=n, hypothesis=hypothesis)
    print(hypothesis)