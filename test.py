import majorization as mj
import numpy as np

#generate lots of probability vectors and check whether S(p || q) >= S(Dp || q) for all D, p, q

dimensions = 10
tries = 100000
ratio = 0
comp = 0

for i in range(tries):
    p = mj.ProbVector(np.random.rand(dimensions))
    q = mj.ProbVector(np.random.rand(dimensions))
    if not (p > q or p < q):
        comp += 1
        mat = mj.BistochMatrix(dims=dimensions)
        if not mj.E_minus(p, q) >= (mj.E_minus(p, mat*q) +S 1e-12): # this should work but I don't get why not
            print(p)
            print(q)
            print(mat*q)
            print(mat)
            print(mj.E_minus(p, q))
            print(mj.E_minus(p, mat*q))
            print("")
            ratio += 1
            
print("RATIO: {}".format(ratio/tries))
print("INCOMPARABILITY RATIO: {}".format(comp/tries))

#A = mj.BistochMatrix(dims=4)
#
##print(A)
#
#
#
#v = mj.ProbVector([0.1, 0.2, 0.3, 0.4])
#u = mj.ProbVector([0.5, 0.2, 0.15, 0.15])
#w = A*v
#x = A*u
#
#print(mj.S(u, v))
#print(mj.S(x, v))
##print(mj.S(w, x))

