import qutip as q
import numpy as np

#rho_a = 1/2 * q.fock_dm(2, 0) + 1/2 * q.fock_dm(2, 1) # density matrix for 1/2|0><0| + 1/2|1><1|
#superposed state of |0> and |1> as density matrix ?
ket_a1 = q.basis(2, 0) # |0>
ket_a2 = q.basis(2, 1) # |1>
ket_a = 1/np.sqrt(2) * (ket_a1 - ket_a2) # (|0> + |1>)/sqrt(2)
rho_a = ket_a * ket_a.dag() # density matrix for (|0> + |1>)(<0| + <1>)/2

rho_b = q.fock_dm(2, 1) # density matrix for |1><1|

rho_ab = q.tensor(rho_a, rho_b) # density matrix for (1/2|0><0| + 1/2|1><1|) x |1><1|

print(rho_ab)

rho_a = rho_ab.ptrace(0) # partial trace of rho_ab over the second subsystem, 0 means the first subsystem is kept
rho_b = rho_ab.ptrace(1) # partial trace of rho_ab over the first subsystem, 1 means the second subsystem is kept

print(np.linalg.eigvals(rho_a))
print(np.linalg.eigvals(rho_b))

