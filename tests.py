import qutip as q
import numpy as np
import majorization as mj
import csv
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

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
    probs = np.random.dirichlet(np.ones(num_terms)) # uniform over k-1 simplex

    for _ in range(num_terms):
        # Random mixed state for system 1 and system 2
        rho1 = random_mixed_state(dim1)
        rho2 = random_mixed_state(dim2)
        
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

    with open(file=csv_path, mode="w") as f:
        max_alpha = int(round(max_alpha/alpha_step))
        results_diff = np.zeros(max_alpha)
        results_total = np.zeros(max_alpha)
        for i in tqdm(range(num_states), desc="Comparing Renyi criteria - dims {} x {}".format(dim1, dim2)):
            state = random_strongly_entangled_state(dim1, dim2, entropic=False) # use majorization criterion, and so all strongly entangled
            a = mj.ProbVector(state.ptrace(0).eigenenergies())
            b = mj.ProbVector(state.ptrace(1).eigenenergies())
            ab = mj.ProbVector(state.eigenenergies())
            
            for alpha in range(max_alpha):
                old_criterion = mj.renyi_entropy(ab, alpha*alpha_step) < max(mj.renyi_entropy(a, alpha*alpha_step), mj.renyi_entropy(b, alpha*alpha_step))
                new_criterion = mj.renyi_entropy(ab, alpha*alpha_step) < mj.renyi_entropy(a+b, alpha*alpha_step)
                if new_criterion: # if the new can tell it is entangled
                    results_total[alpha] +=1
                    if not old_criterion: # if the old can't
                        results_diff[alpha] += 1
        
        results_diff[:] /= num_states
        results_total[:] /= num_states
        writer = csv.writer(f)
        writer.writerow(results_diff)
        writer.writerow(results_total)

        # plot results
        fig, ax = plt.subplots()
        x = np.linspace(0, max_alpha*alpha_step, max_alpha)
        ax.plot(x, results_diff, color="blue", label="Detection ratio only by new entropic criterion") # plot number of only new detections
        ax.plot(x, results_total, color="orange", label="Detection ratio by both entropic criteria") # plot number of only new detections
        
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


def LOCC_target_game(dims, bank, alpha=0, beta=1, targets=[]): # see definition 6.5
    successes = 0
    if targets == []:
        targets = [mj.ProbVector(np.random.dirichlet(np.ones(dims)))]
        #print(target_list)
        endless = True
    else:
        endless = False
    while targets != []:
        # step 1 of algo -- eliminate all non-majorized states
        #print(1)
        can_reach = [] # index list
        for i in range(len(bank)):
            if bank[i] < targets[-1]:
                can_reach.append(i)
        if can_reach == []:
            break # game is over if target non-reachable
        #print(can_reach)
        # step 2 of algo -- eliminate all non-minimal states and copies of minimal states
        #print(2)
        index_record = [] # keep track of all majorized states to remove them
        for i in range(1,len(can_reach)):
            for j in range(i-1): # avoid comparing index i to index i or we would delete all states
                if bank[can_reach[i]] < bank[can_reach[j]]:
                    index_record.append(i) # jth state majorizes and so is below on the lattice
                elif bank[can_reach[i]] > bank[can_reach[j]]: # if several copies of same state, elif only removes the last of the comparison -> only one is left in the final bank
                    index_record.append(j)
        index_record = sorted(set(index_record), reverse=True) # removes duplicates indexes and reverse order to avoid indexerror on pops
        lowest_reach = can_reach # index list too
        for i in index_record:
            lowest_reach.pop(i)
        #print(lowest_reach)
        # step 3 of algo -- compute weighted loss function
        #print(3)
        a = []
        b = []
        c = []
        for i in range(len(lowest_reach)):
            if beta == 0: # save on computation time
                a.append(alpha * mj.entropy(bank[lowest_reach[i]])) # loss function
            else: # entropy is not that heavy computationally so whatever
                a.append(alpha * mj.entropy(bank[lowest_reach[i]]) + beta * mj.unique_entropy(bank[lowest_reach[i]], [bank[_] for _ in lowest_reach if _ != lowest_reach[i]])) # loss function
            b.append(0)
            for j in range(len(can_reach)): # redundancy factor
                if bank[can_reach[j]] < bank[lowest_reach[i]]:
                    b[i] += 1
            c.append(a[i]/b[i]) # weighted loss function
        # step 4 -- construct target if possible
        #print(4)
        index_min = np.argmin(c) # find index of least valuable state
        bank.pop(lowest_reach[index_min]) # pop the state used to construct the target
        targets.pop() # pop the last target
        if endless:
            targets = [mj.ProbVector(np.random.dirichlet(np.ones(dims)))]
        successes += 1

    return successes

def generate_bank(dims, total, ocr=0, distribution=None):
    if distribution == None:
        distribution = np.ones(dims) # sample uniformly
    b = []
    for i in range(total - ocr): # number of normal states
        b.append(mj.ProbVector(np.random.dirichlet(distribution)))
    for _ in range(ocr): # number of jokers
        b.append(mj.ProbVector([1/dims for _ in range(dims)]))
    return b

def generate_targets(dims, total, distribution=None):
    if distribution == None:
        distribution = np.ones(dims) # sample uniformly
    t = []
    for i in range(total): # number of normal states
        t.append(mj.ProbVector(np.random.dirichlet(distribution)))
    return t