from majolat import (
    ProbVector, concatenate, S, relative_entropy, tsallis_entropy
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
        # code to tweak depending on hypothesis
        #for beta in range(1, 1000, 10):
        #    pass
        #for alpha in range(1, 1000, 10):
        #    pass
        for alpha in range(1001, 2000, 10):
            alpha /= 1000
            value = tsallis_entropy(p+q, alpha) + tsallis_entropy(p*q, alpha) - tsallis_entropy(p, alpha) - tsallis_entropy(q, alpha)
            if value < -1e-12:
                print(p)
                print(q)
                print(tsallis_entropy(p, alpha))
                print(tsallis_entropy(q, alpha))
                #print(alpha)
                #print(beta)
                print(value)
                hypothesis = False
                    
        
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
    dimensions = 5
    tries = 10000
    ratio = 0
    comp = 0
    bank_size = 5
    hypothesis = True

    hypothesis, comp = generate_attempts(dimensions=dimensions, tries=tries, ratio=ratio, comp=comp, bank_size=bank_size, hypothesis=hypothesis)
    print(hypothesis)