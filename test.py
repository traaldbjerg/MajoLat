#import os
#os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'

import majorization as mj
import numpy as np
import matplotlib
from alive_progress import alive_bar
import itertools

#matplotlib.use('TkAgg')

#generate lots of probability vectors and check whether S(p || q) >= S(Dp || q) for all D, p, q

dimensions = 8
tries = 10000
ratio = 0
comp = 0
bank_size = 5
hypothesis = True

def generate_attempts(dimensions = 5, tries = 10000, ratio=0, comp=0, bank_size=4, hypothesis=True):
    with alive_bar(tries, title="Testing hypothesis", calibrate=100) as bar:
        record = 0
        for _ in range(tries):
            p = mj.ProbVector(np.random.rand(dimensions))
            bank = generate_bank(bank_size, dimensions)
            unique_volume = mj.unique_entropy(p, bank)
            D = mj.BistochMatrix(dims=dimensions)
            degraded_bank = bank
            degraded_bank[0] = D * degraded_bank[0] # order of states in bank does not matter so no need to randomize the index
            #upgraded_unique_volume = mj.unique_entropy(D*p, bank) # should be larger
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
            bar()
    #print(record)
    return hypothesis, comp

def generate_bank(bank_size, dimensions):
    bank = []
    for _ in range(bank_size): # fill the bank
        q = mj.ProbVector(np.random.rand(dimensions))
        bank.append(q)
    return bank

    

#print("RATIO: {}".format(ratio/tries))
#print("INCOMPARABILITY RATIO: {}".format(comp/tries))

#A = mj.BistochMatrix(dims=4)
#
##print(A)
#
#
#
#p = mj.ProbVector([0.6, 0.2, 0.2])
#q = mj.ProbVector([0.45, 0.45, 0.1])
#beta = p*q
#w = A*v
#x = A*u
#
#print(mj.S(u, v))
#print(mj.S(x, v))
##print(mj.S(w, x))

#mj.plot_lorenz_curves(p, q, beta, labels=['p', 'q', "beta(p, q)"], markers=['o', 'x', ''],
#                       colors=['blue', 'orange', 'lightgreen'], linestyles=['solid', 'solid', 'dashed'],
#                       figsize=(6, 4))


hypothesis, comp = generate_attempts(dimensions=dimensions, tries=tries, ratio=ratio, comp=comp, bank_size=bank_size, hypothesis=hypothesis)

print(hypothesis)
#print(comp)

