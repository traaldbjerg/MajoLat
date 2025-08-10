import majorization as mj
import numpy as np
from tqdm import tqdm
import copy

# functions needed for the game
def generate_bank(dims, total, ocr=0, distribution=None):
    if distribution == None:
        distribution = np.ones(dims) # sample uniformly
    b = []
    for i in range(total - ocr): # number of normal states
        b.append(mj.ProbVector(np.random.dirichlet(distribution)))
    for _ in range(ocr): # number of jokers
        b.append(mj.ProbVector([1/dims for _ in range(dims)]))
    return b

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
        can_reach = [] # index list
        for i in range(len(bank)):
            if bank[i] < targets[-1]:
                can_reach.append(i)
        if can_reach == []:
            break # game is over if target non-reachable
        #print(can_reach)
        # step 2 of algo -- eliminate all non-minimal states and copies of minimal states
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
        index_min = np.argmin(c) # find index of least valuable state
        bank.pop(lowest_reach[index_min]) # pop the state used to construct the target
        targets.pop() # pop the last target
        if endless:
            targets = [mj.ProbVector(np.random.dirichlet(np.ones(dims)))]
        successes += 1

    return successes


# execution
if __name__ == "__main__":
    tries = 1000
    entropic_successes = 0
    unique_entropy_successes = 0

    for _ in tqdm(range(tries), desc="Comparing strategies"):
        dims = 3 * 3

        bank = generate_bank(dims, 95, 5)
        bank_copy = copy.deepcopy(bank)
        targets = generate_bank(dims, 100)
        targets_copy = copy.deepcopy(targets)
        entropic_successes = LOCC_target_game(dims, bank, alpha=1, beta=0, targets=targets) # try entropic strategy
        # get back to the copy to revert the pops
        bank = bank_copy
        targets = targets_copy
        unique_entropy_successes = LOCC_target_game(dims, bank, alpha=0, beta=1, targets=targets) # try unique entropy strategy

    entropic_successes /= 500
    unique_entropy_successes /= 500
    print(entropic_successes)
    print(unique_entropy_successes)