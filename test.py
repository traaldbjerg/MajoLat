import majorization as mj
import numpy as np
import entropy as en
import qutip as q

a = mj.ProbVector([0.6, 0.15, 0.15, 0.1])
b = mj.ProbVector([0.5, 0.25, 0.2, 0.04, 0.01])

print("Relative entropy of A with respect to B: ", mj.relative_entropy(a, b))
print("Relative entropy of B with respect to A: ", mj.relative_entropy(b, a))

