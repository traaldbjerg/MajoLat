import majorization as mj
import numpy as np
import entropy as en
import qutip as q

a = mj.ProbVector([0.6, 0.2, 0.2])
b = mj.ProbVector([0.5, 0.4, 0.075, 0.025])
meet = a + b
join = a * b
max_alpha = 125

print(meet)
print(join)

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

#rho = q.rand_dm(4, dims=[[2, 2], [2, 2]])
#alpha = 1

#print(rho)
#print(rho.ptrace(0))
#print(rho.ptrace(1))

#print(mj.renyi_entropy(mj.ProbVector(rho.eigenenergies()), alpha))
#print(mj.renyi_entropy(mj.ProbVector(rho.ptrace(0).eigenenergies()), alpha))
#print(mj.renyi_entropy(mj.ProbVector(rho.ptrace(1).eigenenergies()), alpha))

