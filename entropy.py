import numpy as np

def shannon_entropy(p):
    return -sum([p_i*np.log(p_i) for p_i in p if p_i != 0])