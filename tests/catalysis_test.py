from majolat import (
    ProbVector, renyi_entropy, vidal_probability
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
    vidal_direct_average = 0
    vidal_double_average = 0
    vidal_supermod_average = 0
    direct_better = 0
    direct_worse = 0
    double_better = 0
    double_worse = 0
    supermod_better = 0
    supermod_worse = 0
    for _ in tqdm(range(tries), desc="Testing hypothesis"):
        switch = True
        while switch: # prevent comparability
            p = ProbVector(np.random.dirichlet(np.ones(dimensions))) # uniform over k-1 simplex
            q = ProbVector(np.random.dirichlet(np.ones(dimensions))) # uniform over k-1 simplex
            if not (p < q or p > q):
                switch = False
            
        #for alpha in range(1, 9999):
        #    alpha = alpha / 10000
        direct_proba = vidal_probability(p, q)
        double_proba = vidal_probability(p * p, p * q)
        supermod_proba = vidal_probability(p * p, (p + q) * (p - q))
        vidal_direct_average += direct_proba
        vidal_double_average += double_proba
        vidal_supermod_average += supermod_proba
        if direct_proba > double_proba + 1e-12 and direct_proba > supermod_proba + 1e-12:
            direct_better += 1
            if double_proba > supermod_proba + 1e-12:
                supermod_worse += 1
            else:
                double_worse += 1
        elif double_proba > direct_proba + 1e-12 and double_proba > supermod_proba + 1e-12:
            double_better += 1
            if direct_proba > supermod_proba + 1e-12:
                supermod_worse += 1
            else:
                direct_worse += 1
        elif supermod_proba > direct_proba + 1e-12 and supermod_proba > double_proba + 1e-12:
            supermod_better += 1
            if direct_proba > double_proba + 1e-12:
                double_worse += 1
            else:
                direct_worse += 1
        
    vidal_direct_average /= tries
    vidal_double_average /= tries
    vidal_supermod_average /= tries
    print(vidal_direct_average)
    print(vidal_double_average)
    print(vidal_supermod_average)
    print(direct_better)
    print(direct_worse)
    print(double_better)
    print(double_worse)
    print(supermod_better)
    print(supermod_worse)
            #print("Counterexample found:")
            #print(p)
            #print(q)
            #print(p + q)
            #print(p - q)
            #print(vidal_probability(p * p, p * q))
            #print(vidal_probability(p * p, (p + q) * (p - q)))
        #        print(alpha)
        #        switch = True
        #if switch:
    #record += 1
        #    switch = False
    #print(record)
        
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
    dimensions = 3
    tries = 10000
    ratio = 0
    comp = 0
    bank_size = 5
    hypothesis = True

    hypothesis, comp = generate_attempts(dimensions=dimensions, tries=tries, ratio=ratio, comp=comp, bank_size=bank_size, hypothesis=hypothesis)
    print(hypothesis)