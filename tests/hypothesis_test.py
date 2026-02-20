from majolat import (
    ProbVector, concatenate, E_future, E_past
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
            if (E_future(p, q) >= 0.25 and E_past(p, q) >= 0.25):
                switch = False
        A = concatenate(p, q, rearrange=True)
        B = concatenate(p + q, p * q, rearrange=True)
        print("ok1")
        
        for attempt in range(tries):
            ### generate r and s
            switch1 = True
            while switch1:
                switch2 = True
                while switch2:
                    r = ProbVector(np.random.dirichlet(np.ones(dimensions))) # uniform over k-1 simplex
                    if (p > r):
                        switch2 = False
                switch3 = True
                while switch3:
                    s = ProbVector(np.random.dirichlet(np.ones(dimensions))) # uniform over k-1 simplex
                    if (q < s):
                        switch3 = False
                C = concatenate(r, s, rearrange=True)
                #print("ok1?")
                #print("attempt")
                if (A > C):
                    switch1 = False
            print("ok2")
                    
            ### test hypothesis
            
            if (C > B):
                print("Hypothesis is false")
                print(p)
                print(q)
                print(p + q)
                print(p * q)
                print(r)
                print(s)
                    
        
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
    tries = 100
    ratio = 0
    comp = 0
    bank_size = 5
    hypothesis = True

    hypothesis, comp = generate_attempts(dimensions=dimensions, tries=tries, ratio=ratio, comp=comp, bank_size=bank_size, hypothesis=hypothesis)
    print(hypothesis)