import qutip as q
import numpy as np
import majorization as mj
import entropy as en

def random_mixed_state(dim):
    """
    Generates a random density matrix (mixed state) in the given dimension.
    """
    return q.rand_dm(dim)

def random_separable_mixed_state(dim1, dim2, num_terms=5):
    """
    Generate a random separable mixed state in the tensor product space of dim1 x dim2.
    This state is created as a convex combination of tensor products of mixed states.
    
    Parameters:
    dim1: Dimension of subsystem 1 (e.g., qubit = 2)
    dim2: Dimension of subsystem 2 (e.g., qubit = 2)
    num_terms: Number of product states to sum over
    
    Returns:
    A separable density matrix as a Qobj object.
    """
    state = q.Qobj(np.zeros([dim1 * dim2, dim1 * dim2]), dims=[[dim1, dim2], [dim1, dim2]])
    
    # Generate random probabilities for the convex combination
    probs = np.random.dirichlet(np.ones(num_terms))

    for _ in range(num_terms):
        # Random mixed state for system 1 and system 2
        rho1 = random_mixed_state(dim1)
        rho2 = random_mixed_state(dim2)
        
        # Tensor product of mixed states
        product_state = q.tensor(rho1, rho2)
        
        # Add the weighted sum of the tensor products to form the separable state
        state += probs[_] * product_state

    return state

# Example usage
num_states = 10000  # Number of random separable states to generate
dim1, dim2 = 7,7  # Dimensions for two qubits (change as needed)

separable_mixed_states = [random_separable_mixed_state(dim1, dim2) for _ in range(num_states)] # state generation is the costly part of the program

true_count = 0
false_count = 0
comparable_count = 0
majo_by_meet_count = 0

# Printing out some details about the generated states
for i, state in enumerate(separable_mixed_states):
    # create P_n objects
    a = mj.ProbVector(state.ptrace(0).eigenenergies())
    b = mj.ProbVector(state.ptrace(1).eigenenergies())
    ab = mj.ProbVector(state.eigenenergies())
    if (a > b or b > a): # if the 2 are comparable the rest is not interesting
        comparable_count += 1
        #print("This should always happen in dim 2")
        #a.majorizes_debug(b)
    else: # interesting case where meet and join is not trivial
        #print("This should never happen in dim 2")
        #a.majorizes_debug(b)
        # check if the state is majorized by the reduced states
        if (a > ab and b > ab):
            true_count += 1
        else: # should never happen, would indicate a bug in the code
            false_count += 1
            print("What ???")
            #print(state)
            #print(state.ptrace(0))
            #print(state.ptrace(1))
            print(ab)
            print(a)
            print(b)
            a.majorizes_debug(ab)
            b.majorizes_debug(ab)
            break
        # create meet
        meet = a + b
        if (meet > ab):
            majo_by_meet_count += 1

print("Number of comparable states: {}".format(comparable_count))
print("Number of bugs: {}".format(false_count))
print("Number that verify conjecture: {}".format(majo_by_meet_count + comparable_count))

