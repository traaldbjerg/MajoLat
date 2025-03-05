import majorization as mj
import numpy as np

#generate lots of probability vectors and check whether S(p || q) >= S(Dp || q) for all D, p, q

dimensions = 8
tries = 10000
ratio = 0
comp = 0

for i in range(tries):
    p = mj.ProbVector(np.random.rand(dimensions))
    q = mj.ProbVector(np.random.rand(dimensions))
    if (p > q or p < q):
        comp += 1
        mat = mj.BistochMatrix(dims=dimensions, rand_combs=2)
        if not mj.S(p, q) >= (mj.S(mat * p, q) - 1e-12):
            print(p)
            print(q)
            print(p * q)
            print(mat*p)
            print(mat)
            print(mj.S(p, q))
            print(mj.S(mat*p, q))
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

