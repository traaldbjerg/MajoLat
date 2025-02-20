import majorization as mj
import numpy as np
import entropy as en
#import qutip as q

#q = mj.ProbVector([0.6, 0.4])
#p = mj.ProbVector([0.7, 0.29, 0.01])
#
#pp = mj.ProbVector([0.7, 0.15, 0.15])
#ppp = mj.ProbVector([0.6, 0.39, 0.01])
#
#print(mj.E_up(p, q))
#print(mj.E_up(pp, q))
#print(mj.E_up(ppp, q))



#meet = a + b
#join = a * b


#hypothesis = mj.d(b, meet)
#print(hypothesis)

#for i in range(1000):
#    for j in range(1000 - i):
#        s = mj.ProbVector([0.6 - (i+j)/10000, 0.4 + i/10000, j/10000])
#        #print(mj.d(b, s))
#        if mj.d(b + s, b) > mj.d(b, s):
#            print("Counterexample found: ")
#            print(s)
#            print(mj.d(b, s))
#            break

#print(meet)
#print(join)

#for alpha in range(max_alpha):
#    dist_to_comp = mj.d_comp(a, b, alpha)
#    print("Distance to comparability with alpha = {} : {}".format(alpha, dist_to_comp))
#
#for alpha in range(max_alpha):
#    dist_to_subadd = mj.d_subadd(a, b, alpha)
#    print("Distance to subadd with alpha = {} : {}".format(alpha, dist_to_subadd))

#print("Relative entropy of A with respect to B: ", mj.relative_entropy(a, b))
#print("Relative entropy of B with respect to A: ", mj.relative_entropy(b, a))
#print("Relative entropy of A with respect to meet: ", mj.relative_entropy(a, meet))
#print("Relative entropy of B with respect to meet: ", mj.relative_entropy(b, meet))
#print("Relative entropy of A with respect to join: ", mj.relative_entropy(a, join))
#print("Relative entropy of B with respect to join: ", mj.relative_entropy(b, join))
#print("Relative entropy of meet with respect to join: ", mj.relative_entropy(meet, join))
#print("Relative entropy of join with respect to meet: ", mj.relative_entropy(join, meet))

#bell = q.bell_state()
#rho = bell * bell.dag()
#print(rho.eigenenergies())
#rho = q.rand_dm(4, dims=[[2, 2], [2, 2]])
#alpha = 1

#print(rho)
#print(rho.ptrace(0))
#print(rho.ptrace(1))

#print(mj.renyi_entropy(mj.ProbVector(rho.eigenenergies()), alpha))
#print(mj.renyi_entropy(mj.ProbVector(rho.ptrace(0).eigenenergies()), alpha))
#print(mj.renyi_entropy(mj.ProbVector(rho.ptrace(1).eigenenergies()), alpha))

#comp_count = 0

#for i in range(1000):


#print(comp_count)



A = mj.BistochMatrix(dims=10)

print(A)