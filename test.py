#import os
#os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'

import majorization as mj
import numpy as np
import matplotlib
from alive_progress import alive_bar
import itertools

#matplotlib.use('TkAgg')

#generate lots of probability vectors and check whether S(p || q) >= S(Dp || q) for all D, p, q

dimensions = 5
tries = 10000
ratio = 0
comp = 0
bank_size = 10
hypothesis = True

def generate_attempts(dimensions = 5, tries = 10000, ratio=0, comp=0, bank_size=4, hypothesis=True):
    with alive_bar(tries, title="Testing hypothesis", calibrate=100) as bar:
        for i in range(tries):
            p = mj.ProbVector(np.random.rand(dimensions))
            bank = generate_bank(bank_size, dimensions) # fill the bank with random states
            partial_comp = 0
            for q in bank: # not necessary, just testing
                if (p > q or p < q):
                    partial_comp += 1
            if partial_comp == dimensions:
                comp += 1
        #    mat = mj.BistochMatrix(dims=dimensions)
        #    if not mj.S(p, q) <= (mj.S(mat * p, q) + 1e-12):
        #        print(p)
        #        print(q)
        #        print(p * q)
        #        print(mat*p)
        #        print(mat)
        #        print(mj.S(p, q))
        #        print(mj.S(mat*p, mat*q))
        #        print("")
        #        ratio += 1
            #unique_volume = mj.E_u_future(p, bank=bank, reduce=False)
            unique_volume_reduced = mj.E_u_future(p, bank=bank, reduce=True)
            #print("{}, {}".format(unique_volume, unique_volume_reduced))
            #if not unique_volume >= -1e-12:
            #    hypothesis = False
            #if not (abs(unique_volume) + 1e-12 >= abs(unique_volume_reduced) and abs(unique_volume) - 1e-12 <= abs(unique_volume_reduced)):
            #    hypothesis = False
            #print(unique_volume)
            bar()
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


#hypothesis, comp = generate_attempts(dimensions=dimensions, tries=tries, ratio=ratio, comp=comp, bank_size=bank_size, hypothesis=hypothesis)

#print(hypothesis)
#print(comp)

for size in range(bank_size):
    print("size:{}".format(size))
    bank = generate_bank(size, dimensions)
    combs = 1
    for i in range(size):
        for combination in itertools.combinations(bank, i+1):
            combs +=1
    print(combs)

