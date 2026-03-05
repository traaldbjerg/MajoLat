from majolat import (
    ProbVector, concatenate, vidal_probability
)
import numpy as np
import matplotlib
from tqdm import tqdm
import itertools



def generate_attempts(dimensions = 5, tries = 10000, ratio=0, comp=0, bank_size=4, hypothesis=True):
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
            if not (p < q or p > q):
                switch = False
        A = concatenate(p, p, normalize=True, rearrange=True)
        B = concatenate(p + q, p * q, normalize=True, rearrange=True) # m + j
        C = concatenate(p, q, normalize=True, rearrange=True) # p + q
        
        if vidal_probability(A, B) < vidal_probability(A, C) - 1e-12:
            print(f"Obtaining B from A: {vidal_probability(A, B)}; obtaining C from A: {vidal_probability(A, C)}")
        #print(f"Ratio: {vidal_probability(p, q)/vidal_probability(A, B)}; without EPR: {vidal_probability(p, q)}; with EPR: {vidal_probability(A, B)}")
        #if vidal_probability(p, q) > vidal_probability(A, B) + 1e-12:
        #    print(vidal_probability(A, C))
        #    print(vidal_probability(A, B))
        #    print(p)
        #    print(q)
        #    print(A)
        #    print(B)
        #    print(C)
        #    print("rip")
        
    #print(record)
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
    tries = 100000
    ratio = 0
    comp = 0
    bank_size = 5
    hypothesis = True

    hypothesis, comp = generate_attempts(dimensions=dimensions, tries=tries, ratio=ratio, comp=comp, bank_size=bank_size, hypothesis=hypothesis)
    print(hypothesis)