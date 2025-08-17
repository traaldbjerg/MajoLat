import majorization as mj
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
        p = mj.ProbVector(np.random.dirichlet(np.ones(dimensions))) # uniform over k-1 simplex
        bank = generate_bank(dimensions, bank_size)
        # code to tweak depending on hypothesis
        unique_volume = mj.unique_entropy(p, bank)
        D = mj.BistochMatrix(dims=dimensions)
        degraded_bank = bank
        degraded_bank[0] = D * degraded_bank[0] # order of states in bank does not matter so no need to randomize the index
        degraded_unique_volume = mj.unique_entropy(p, degraded_bank) # should be smaller
        if unique_volume < degraded_unique_volume - 1e-12:
            hypothesis = False
            print(unique_volume)
            print(degraded_unique_volume)
            print(bank[0])
            print(degraded_bank[0])
            for state in bank:
                print(state)
            break
    #print(record)
    return hypothesis, comp

def generate_bank(dims, total, ocr=0, distribution=None):
    if distribution == None:
        distribution = np.ones(dims) # sample uniformly
    b = []
    for i in range(total - ocr): # number of normal states
        b.append(mj.ProbVector(np.random.dirichlet(distribution)))
    for _ in range(ocr): # number of jokers
        b.append(mj.ProbVector([1/dims for _ in range(dims)]))
    return b

if __name__ == "__main__":
    dimensions = 8
    tries = 10000
    ratio = 0
    comp = 0
    bank_size = 5
    hypothesis = True

    hypothesis, comp = generate_attempts(dimensions=dimensions, tries=tries, ratio=ratio, comp=comp, bank_size=bank_size, hypothesis=hypothesis)
    print(hypothesis)
    
if __name__ == "__main__":
    hypothesis, comp = generate_attempts()
    print(hypothesis)