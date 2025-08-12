import majorization as mj
import numpy as np
from tqdm import tqdm
import copy
import csv
from pathlib import Path
import matplotlib.pyplot as plt

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

def LOCC_target_game(dims, bank, alpha=0, targets=[], distribution=None): # see definition 6.5
    successes = 0
    indexes = [_ for _ in range(len(bank))]
    target_indexes = [_ for _ in range(len(targets))]
    if target_indexes == []:
        targets = [mj.ProbVector(np.random.dirichlet(distribution))]
        target_indexes = [0]
        #print(target_list)
        endless = True
    else:
        endless = False
    while target_indexes != []:
        # step 1 of algo -- eliminate all non-majorized states
        can_reach = [] # index list
        for i in indexes:
            if bank[i] < targets[target_indexes[-1]]:
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
        index_record = sorted(set(index_record), reverse=True) # removes duplicate indexes and reverse order to avoid indexerror on pops
        lowest_reach = can_reach # index list too
        for i in index_record:
            lowest_reach.pop(i)
        #print(lowest_reach)
        # step 3 of algo -- compute weighted loss function
        a = []
        b = []
        c = []
        current_index = 0 # not very clean but prevents having bank[lowest_reach[i]] everywhere so more readable
        for i in lowest_reach:
            if alpha == 1: # save on computation time
                a.append(mj.entropy(bank[i])) # loss function
            else: # entropy is not that heavy computationally so not the end of the world to include it if alpha = 0
                a.append(alpha * mj.entropy(bank[i]) + (1 - alpha) * mj.unique_entropy(bank[i],
                                                                                    [bank[_] for _ in indexes if _ != i and not bank[_] < bank[i]])) # loss function
            b.append(0)
            for j in can_reach: # redundancy factor
                if bank[j] < bank[i]:
                    b[current_index] += 1
            c.append(a[current_index]/b[current_index]) # weighted loss function
            current_index += 1
        # step 4 -- construct target if possible
        index_min = np.argmin(c) # find index of least valuable state
        #print(indexes)
        #print(lowest_reach)
        #print(index_min)
        indexes.remove(lowest_reach[index_min]) # pop the state used to construct the target
        target_indexes.pop() # pop the last target
        if endless:
            targets = [mj.ProbVector(np.random.dirichlet(distribution))] # regenerate another target
            target_indexes = [0]
        successes += 1

    return successes


# execution
if __name__ == "__main__":
    tries = 100
    entropic_successes = 0
    unique_entropy_successes = 0
    bank_size = 10
    ocr = 1
    dims = 4
    skew = 2
    target_distribution = [skew]
    target_distribution.extend([1/skew for _ in range(dims-1)]) # skew targets toward bottom of simplex
    #print(target_distribution)
    step = 0.01
    successes = np.array([0 for _ in range(int(round(1/step) + 1))], dtype=np.float64)
    #print(successes)

    Path.mkdir(Path("results"), exist_ok=True)
    csv_path = Path("results/locc_game_{}_{}_{}_{}_{}_{}.csv".format(tries, dims, bank_size, ocr, step, skew))
    png_path = Path("results/locc_game_{}_{}_{}_{}_{}_{}.png".format(tries, dims, bank_size, ocr, step, skew))

    for game in tqdm(range(tries), desc="Comparing strategies"): 
        bank = generate_bank(dims, bank_size, ocr)
        targets = generate_bank(dims, bank_size, distribution=target_distribution)
        #bank_copy = copy.deepcopy(bank)
        #targets_copy = copy.deepcopy(targets)
        for alpha in range(int(round(1/step) + 1)):
            successes[alpha] += LOCC_target_game(dims, bank, alpha=alpha*step, targets=targets) # try the different strategies
            #bank = copy.deepcopy(bank_copy) # revert to old state (not very clean, probably more efficient to only keep track of indices in LOCC_target_game)
            #targets = copy.deepcopy(targets_copy)

    successes[:] /= tries
    #print(successes)
    with open(file=csv_path, mode="w") as f:
        writer = csv.writer(f)
        writer.writerow(successes)

    # plot results
    fig, ax = plt.subplots()
    x = np.linspace(0, 1, int(round(1/step) + 1))
    ax.plot(x, successes, color="blue") # plot number of only new detections
    
    ax.set_xlabel("Alpha")
    ax.set_ylabel("Average number of targets constructed")
    ax.set_title("Average number of targets constructed as a function of the strategy parameter alpha")
    #ax.legend()
    #ax.tick_params(which='major', width=1.00, length=5)
    #ax.tick_params(which='minor', width=0.75, length=2.5)
    #ax.xaxis.set_major_locator(ticker.AutoLocator())
    #ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(True)
    ax.set_xlim(0, 1)
    #ax.set_ylim(0, 1)
    plt.savefig(png_path)
    plt.show()