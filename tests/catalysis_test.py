from majolat import (
    ProbVector, renyi_entropy, vidal_probability
)
import numpy as np
import matplotlib
from tqdm import tqdm
import itertools



def generate_attempts(dimensions = 5, tries = 10000, ratio=0, comp=0, n=4, hypothesis=True):
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
    vidal_serge_average = 0
    vidal_multi_average = 0
    vidal_multi_serge_average = 0
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
                
        direct_proba = vidal_probability(p, q)
        serge_proba = vidal_probability(p, p + q)
        multi_proba = vidal_probability(p ** n, (p ** (n-1)) * (q))
        multi_serge_proba = vidal_probability(p ** n, (p ** (n-1)) * (p+q))
        supermod_proba = vidal_probability(p ** n, (p + q) * (p - q) * (p ** (n-2)))
        vidal_direct_average += direct_proba
        vidal_serge_average += serge_proba
        vidal_multi_average += multi_proba
        vidal_multi_serge_average += multi_serge_proba
        vidal_supermod_average += supermod_proba
        if direct_proba > multi_proba + 1e-12 and direct_proba > supermod_proba + 1e-12:
            direct_better += 1
            if multi_proba > supermod_proba + 1e-12:
                supermod_worse += 1
            else:
                double_worse += 1
        elif multi_proba > direct_proba + 1e-12 and multi_proba > supermod_proba + 1e-12:
            double_better += 1
            if direct_proba > supermod_proba + 1e-12:
                supermod_worse += 1
            else:
                direct_worse += 1
        elif supermod_proba > direct_proba + 1e-12 and supermod_proba > multi_proba + 1e-12:
            supermod_better += 1
            if direct_proba > multi_proba + 1e-12:
                double_worse += 1
            else:
                direct_worse += 1
        
    vidal_direct_average /= tries
    vidal_serge_average /= tries
    vidal_multi_average /= tries
    vidal_multi_serge_average /= tries
    vidal_supermod_average /= tries
    print(vidal_direct_average)
    print(vidal_serge_average)
    print(vidal_multi_average)
    print(vidal_multi_serge_average)
    print(vidal_supermod_average)
    #print(direct_better)
    #print(direct_worse)
    #print(double_better)
    #print(double_worse)
    #print("SUPERMOD")
    #print(supermod_better)
    #print(supermod_worse)
    
    return hypothesis, comp

if __name__ == "__main__":
    dimensions = 4
    tries = 1000
    ratio = 0
    comp = 0
    n = 5
    hypothesis = True

    hypothesis, comp = generate_attempts(dimensions=dimensions, tries=tries, ratio=ratio, comp=comp, n=n, hypothesis=hypothesis)
    print(hypothesis)