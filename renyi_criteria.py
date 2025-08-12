import qutip as q
import numpy as np
import majorization as mj
import csv
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# functions needed for the test (except the first one)
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
    probs = np.random.dirichlet(np.ones(num_terms)) # uniform over k-1 simplex

    for _ in range(num_terms):
        # Random mixed state for system 1 and system 2
        rho1 = q.rand_dm(dim1)
        rho2 = q.rand_dm(dim2)
        
        # Tensor product of mixed states
        product_state = q.tensor(rho1, rho2)
        
        # Add the weighted sum of the tensor products to form the separable state
        state += probs[_] * product_state

    return state

def random_strongly_entangled_state(dim1, dim2, entropic=False):
    """
    Generate a random strongly entangled mixed state in the tensor product space of dim1 x dim2.
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
    meet = 1 # just to pass the initial test, then probvectors
    if not entropic: # use majorization criterion
        while meet > ab: # only false if the state is entangled (note that lightly entangled states are not detected here and will never be outputted by this function)
            state = q.rand_dm(dim1*dim2, dims=[[dim1, dim2], [dim1, dim2]]) # the tensor structure is set at the end so this is actually fine and doesn't create a product state
            ab = mj.ProbVector(state.eigenenergies())
            a = mj.ProbVector(state.ptrace(0).eigenenergies())
            b = mj.ProbVector(state.ptrace(1).eigenenergies())
            meet = a + b
    else: # use entropic criterion:
        while mj.entropy(meet) < mj.entropy(ab): # only false if the state is entangled (note that lightly entangled states are not detected here and will never be outputted by this function)
            state = q.rand_dm(dim1*dim2, dims=[[dim1, dim2], [dim1, dim2]]) # the tensor structure is set at the end so this is actually fine and doesn't create a product state
            ab = mj.ProbVector(state.eigenenergies())
            a = mj.ProbVector(state.ptrace(0).eigenenergies())
            b = mj.ProbVector(state.ptrace(1).eigenenergies())
            meet = a + b

    return state

def compare_renyi_criteria(num_states=10000, dim1=2, dim2=2, max_alpha=10, alpha_step=0.1): # see section 4.1.3
    Path.mkdir(Path("results"), exist_ok=True)
    csv_path = Path("results/renyi_comparison_{}_{}_{}_{}.csv".format(dim1, dim2, max_alpha, alpha_step))
    png_path = Path("results/renyi_comparison_{}_{}_{}_{}.png".format(dim1, dim2, max_alpha, alpha_step))

    
    max_alpha = int(round(max_alpha/alpha_step))
    results_new = np.zeros(max_alpha)
    results_old = np.zeros(max_alpha)
    for i in tqdm(range(num_states), desc="Comparing Renyi criteria - dims {} x {}".format(dim1, dim2)):
        state = random_strongly_entangled_state(dim1, dim2, entropic=False) # use majorization criterion, and so all strongly entangled
        a = mj.ProbVector(state.ptrace(0).eigenenergies())
        b = mj.ProbVector(state.ptrace(1).eigenenergies())
        ab = mj.ProbVector(state.eigenenergies())
        
        for alpha in range(max_alpha):
            old_criterion = mj.renyi_entropy(ab, alpha*alpha_step) < max(mj.renyi_entropy(a, alpha*alpha_step), mj.renyi_entropy(b, alpha*alpha_step))
            new_criterion = mj.renyi_entropy(ab, alpha*alpha_step) < mj.renyi_entropy(a+b, alpha*alpha_step)
            if new_criterion: # if the new can tell it is entangled
                results_new[alpha] += 1
            if old_criterion: # if the old can't
                    results_old[alpha] += 1
    
    results_new[:] /= num_states
    results_old[:] /= num_states
    with open(file=csv_path, mode="w") as f:
        writer = csv.writer(f)
        writer.writerow(results_new)
        writer.writerow(results_old)

    # plot results
    fig, ax = plt.subplots()
    x = np.linspace(0, max_alpha*alpha_step, max_alpha)
    ax.plot(x, results_new, color="blue", label="Detection ratio by new entropic criterion") # plot number of only new detections
    ax.plot(x, results_old, color="orange", label="Detection ratio by old entropic criterion") # plot number of only new detections
    
    ax.set_xlabel("Alpha")
    ax.set_ylabel("Detection ratio")
    ax.set_title("Entanglement detection as a function of alpha for dimensions {} x {}".format(dim1, dim2))
    ax.legend()
    ax.tick_params(which='major', width=1.00, length=5)
    ax.tick_params(which='minor', width=0.75, length=2.5)
    ax.xaxis.set_major_locator(ticker.AutoLocator())
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(True)
    ax.set_xlim(0, max_alpha*alpha_step)
    ax.set_ylim(0, 1)
    plt.savefig(png_path)
    #plt.show()


# execution
if __name__ == "__main__":
    #dimension_list = [(2, 2), (2, 3), (3, 3), (3, 4), (3, 5), (3, 6)]#, (4, 4), (4, 6), (4, 8)] # dimensions to test, eigenvalue calculations quickly become expensive
    dimension_list = [(4, 4)]
    num_states = 10000
    max_alpha = 30
    alpha_step = 0.2
    for dims in dimension_list: # loop over dimensions of interest
        compare_renyi_criteria(num_states, dims[0], dims[1], max_alpha=max_alpha, alpha_step=alpha_step)