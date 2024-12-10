import qutip as q
import numpy as np
import scipy as sp

def quantum_renyi_entropy(rho, base=np.e, alpha=1):
    """
    Compute the Renyi entropy of a quantum state rho.
    """
    if alpha == 1:
        return q.entropy.entropy_vn(rho, base)
    else:
        return 1/(1-alpha) * np.log(np.trace(sp.linalg.fractional_matrix_power(rho, alpha)))
    
