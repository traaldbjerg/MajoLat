import os
os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'

import majorization as mj
import numpy as np
import matplotlib

matplotlib.use('TkAgg')

#generate lots of probability vectors and check whether S(p || q) >= S(Dp || q) for all D, p, q

#dimensions = 8
#tries = 10000
#ratio = 0
#comp = 0

#for i in range(tries):
#    p = mj.ProbVector(np.random.rand(dimensions))
#    q = mj.ProbVector(np.random.rand(dimensions))
#    if not (p > q or p < q):
#        comp += 1
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
        
#print("RATIO: {}".format(ratio/tries))
#print("INCOMPARABILITY RATIO: {}".format(comp/tries))

#A = mj.BistochMatrix(dims=4)
#
##print(A)
#
#
#
p = mj.ProbVector([0.6, 0.15, 0.15, 0.1])
q = mj.ProbVector([0.5, 0.25, 0.20, 0.05])
beta = [0.6, 0.175, 0.175, 0.05]
beta_prime = mj.ProbVector([0.6, 0.2, 0.15, 0.05])
#w = A*v
#x = A*u
#
#print(mj.S(u, v))
#print(mj.S(x, v))
##print(mj.S(w, x))

mj.plot_lorenz_curves(p, q, beta_prime, beta, labels=['p', 'q', "beta'(p, q)", "beta''(p, q)"], markers=['o', 'x', '', ''],
                       colors=['blue', 'orange', 'red', 'gray',], linestyles=['solid', 'solid', 'dashed', 'dashed'],
                       figsize=(6, 4))
