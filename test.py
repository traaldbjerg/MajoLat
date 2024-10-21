import majorization as mj
import numpy as np
import entropy as en
import qutip as q

a = mj.ProbVector([0.6, 0.15, 0.15, 0.1])
b = mj.ProbVector([0.5, 0.25, 0.2, 0.05])

#print("\nMajorization calculations:\n")
#
#print(a)
#print(b)
#print(a.majorizes(b))
#print(b.majorizes(a))
#
#meet = a + b
#join = a * b
#
#print(meet)
#print(join)
#
#print(meet > a)
#print(meet > b)
#print(meet < a)
#print(meet < b)
#print(join > a)
#print(join > b)
#print(join < a)
#print(join < b)
#
#print("\nEntropy calculations:\n")
#
#H_a = en.shannon_entropy(a)
#H_b = en.shannon_entropy(b)
#H_meet = en.shannon_entropy(meet)
#H_join = en.shannon_entropy(join)
#
#print(H_a)
#print(H_b)
#print(H_meet)
#print(H_join)
##supermodularity test
#print(H_meet + H_join >= H_a + H_b)
##subadditivity test
#print(H_meet <= H_a + H_b)

# test with quantum states

ket_a1 = q.basis(2, 0) # |0>
ket_a2 = q.basis(2, 1) # |1>
ket_a = 1/np.sqrt(2) * (ket_a1 + ket_a2)
rho_a = ket_a * ket_a.dag()

rho_b = q.fock_dm(2, 1) # density matrix for |1><1|

rho_ab1 = q.tensor(rho_a, rho_b) # density matrix for (1/2|0><0| + 1/2|1><1|) x |1><1|

rho_ab2 = q.tensor(q.fock_dm(2, 0), q.fock_dm(2, 1)) # density matrix for |0><0| x |1><1|

ket_a3 = q.basis(2, 0) # |1>
ket_a4 = q.basis(2, 1) # |0>
ket_b3 = q.basis(2, 0) # |0>
ket_b4 = q.basis(2, 1) # |1>
ket_b = 1/np.sqrt(2) * (ket_b3 - ket_b4)
ket_a = 1/np.sqrt(2) * (ket_a3 + ket_a4)
rho_a = ket_a * ket_a.dag()
rho_b = ket_b * ket_b.dag()

rho_ab3 = q.tensor(rho_a, rho_b) # density matrix for |1><1| x |0><0|

rho_ab = 0.5 * rho_ab1 + 0.3 * rho_ab2 + 0.2 * rho_ab3

eig_ab = np.linalg.eigvals(rho_ab)

rho_a = rho_ab.ptrace(0) # partial trace of rho_ab over the second subsystem, 0 means the first subsystem is kept
rho_b = rho_ab.ptrace(1) # partial trace of rho_ab over the first subsystem, 1 means the second subsystem is kept

eig_a = np.linalg.eigvals(rho_a)
eig_b = np.linalg.eigvals(rho_b)

print(eig_ab)
print(eig_a)
print(eig_b)

ab = mj.ProbVector(eig_ab)
a = mj.ProbVector(eig_a)
b = mj.ProbVector(eig_b)

print("\nMajorization calculations:\n")

print("Comparable ?")
print(a > b or b > a)

print(a > ab)
print(b > ab)

p = a + b
q = a * b

print(p > ab)
print(q > ab)