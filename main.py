import qutip as q
import numpy as np
import majorization as mj
import qentropy

def random_mixed_state(dim, dims=None):
    """
    Generates a random density matrix (mixed state) in the given dimension.
    """
    return q.rand_dm(dim, dims=dims)

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

def random_entangled_state(dim1, dim2):
    """
    Generate a random entangled mixed state in the tensor product space of dim1 x dim2.
    This state is created as a convex combination of tensor products of mixed states.
    
    Parameters:
    dim1: Dimension of subsystem 1 (e.g., qubit = 2)
    dim2: Dimension of subsystem 2 (e.g., qubit = 2)
    
    Returns:
    An entangled density matrix as a Qobj object.
    """
    
    a = 0
    b = 0
    ab = 0
    meet = 1
    while meet > ab: # only true if the state is separable, so we need to loop until we get an entangled state
        state = q.rand_dm(dim1 * dim2, dims=[[dim1, dim2], [dim1, dim2]]) # the tensor structure is set at the end so this is actually fine and doesn't create a product state
        ab = mj.ProbVector(state.eigenenergies())
        a = mj.ProbVector(state.ptrace(0).eigenenergies())
        b = mj.ProbVector(state.ptrace(1).eigenenergies())
        meet = a + b

    return state

num_states = 1000 # Number of random separable states to generate
dim1, dim2 = 2,3  # Dimensions for two qubits (change as needed)

# very dumb way to generate states, hogs a ton of memory for no reason, just generate the states one at a time
#states = [random_separable_mixed_state(dim1, dim2) for _ in range(num_states)] # state generation is the costly part of the program
#states = [random_entangled_state(dim1, dim2) for _ in range(num_states)]

true_count = 0
false_count = 0
comparable_count = 0
majo_by_meet_count = 0
alpha = 1
max_alpha = 100

# Printing out some details about the generated states
for i in range(num_states):
    # create P_n objects
    state = random_entangled_state(dim1, dim2)
    a = mj.ProbVector(state.ptrace(0).eigenenergies())
    b = mj.ProbVector(state.ptrace(1).eigenenergies())
    ab = mj.ProbVector(state.eigenenergies())
    if (mj.d_comp(a, b) > 0): # if the 2 are comparable the rest is not interesting
        comparable_count += 1
        #if (mj.d_comp(ab, a) > 10**-12 and mj.d_comp(ab, b) > 10**-12):
            #print("State: {}".format(i))
        meet = a + b # analoguous to joint entropy of X and Y
        join = a * b
            #print(state)
            #print(state.ptrace(0))
            #print(state.ptrace(1))
        print(a)
        print(b)
            #print(ab)
            #print(meet)
            #print(join)
            #print(mj.d_comp(ab, meet))
            #print("Distance to comparability for A and B:")
            #print(mj.d_comp(ab, a))
            #print(mj.d_comp(ab, b))
        #print(mj.d_comp(a, b))
        
        # check renyi entropies
        #if mj.renyi_entropy(ab, alpha) >= mj.renyi_entropy(meet, alpha):
        for i in range(max_alpha):
            if mj.d_subadd(a, b, i) < 0:
                print("Subadditivity does not hold for alpha = {}, ".format(i))
            if mj.d_comp(a, b, i) < 0:
                print("Supermodularity does not hold for alpha = {}".format(i))
                #false_count += 1
        #print("Renyi entropy of meet: {}".format(mj.renyi_entropy(meet, alpha)))
        #print("Renyi entropy of A: {}".format(mj.renyi_entropy(a, alpha)))
        #print("Renyi entropy of B: {}".format(mj.renyi_entropy(b, alpha)))
        #print("Renyi entropy of AB: {}".format(mj.renyi_entropy(ab, alpha)))
        #print("Renyi entanglement entropy of A: {}".format(qentropy.quantum_renyi_entropy(state.ptrace(0), alpha=alpha)))
        #print("Renyi entanglement entropy of B: {}".format(qentropy.quantum_renyi_entropy(state.ptrace(1), alpha=alpha)))
        #print("Renyi entanglement entropy of AB: {}".format(qentropy.quantum_renyi_entropy(state, alpha=alpha)))
        #else:
        #    false_count += 1
            
print("Number of comparable states: {}".format(comparable_count))
#print("Number of bugs: {}".format(false_count))
#print("Number that verify conjecture: {}".format(majo_by_meet_count + comparable_count))

# Information theory by Cover and Thomas
# Faut vraiment lire l'article d'Hiroshima
