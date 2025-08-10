import majorization as mj
from tests import *
import copy

dimension_list = [(2, 2), (2, 3), (3, 3), (3, 4), (3, 5), (3, 6)]#, (4, 4), (4, 6), (4, 8)] # dimensions to test, eigenvalue calculations quickly become expensive

#for dims in dimension_list: # loop over dimensions of interest
#    compare_renyi_criteria(10000, dims[0], dims[1], max_alpha=30, alpha_step=0.2)

tries = 500
entropic_successes = 0
unique_entropy_successes = 0

for _ in tqdm(range(tries), desc="Comparing strategies"):
    dims = 3 * 3

    bank = generate_bank(dims, 50, 5)
    bank_copy = copy.deepcopy(bank)
    targets = generate_targets(dims, 50)
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