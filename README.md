# Majolat: Majorization and Quantum Information Tools

A Python library for working with majorization theory, probability vectors, (bi)stochastic matrices, and quantum entanglement transformations (SLOCC protocols).

## Project Structure

```
majolat/
├── majolat/              # Main package
│   ├── __init__.py      # Public API
│   ├── majorization.py  # Core majorization classes (ProbVector, StochMatrix, BistochMatrix)
│   ├── quantum.py       # Quantum information tools (SLOCC)
│   └── utils.py         # Entropy, distances, incomparability measures, plotting
├── examples/            # Example scripts
│   ├── slocc_example.py
│   └── slocc_analysis.py
├── tests/               # Test files
│   └── test_slocc.py
├── docs/                # Documentation
│   └── SLOCC_README.md
├── pyproject.toml       # Package configuration
└── README.md           # This file
```

## Installation and Dependencies

This project uses [NumPy](https://numpy.org/), [Matplotlib](https://matplotlib.org/), and [QuTiP](https://qutip.org/).

### For Development

From the project root directory, simply set PYTHONPATH to use the library:

```bash
# Add to PYTHONPATH for the current session
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or run scripts directly with PYTHONPATH
PYTHONPATH=. python examples/slocc_example.py
PYTHONPATH=. python tests/test_slocc.py
```

### System Installation (Optional)

If you prefer to install the package:

```bash
pip install .
```

Then you can import directly without PYTHONPATH:

```python
from majolat import ProbVector, SLOCC, entropy
```

## Quick Start

```python
from majolat import ProbVector, SLOCC, entropy

# Create probability vectors
p = ProbVector([0.7, 0.2, 0.1])
q = ProbVector([0.5, 0.3, 0.2])

# Majorization operations
meet = p + q  # Greatest lower bound (meet)
join = p * q  # Least upper bound (join)
majorizes = p > q  # Check if p majorizes q

# Entropy measures
h = entropy(p)
print(f"Entropy: {h:.4f} bits")

# SLOCC protocols for quantum entanglement transformations
initial = ProbVector([0.7, 0.2, 0.1])
target = ProbVector([0.5, 0.3, 0.2])
slocc = SLOCC(initial, target)

print(f"Success probability: {slocc.get_success_probability():.4f}")
print(f"Failure Schmidt coeffs: {slocc.get_failure_schmidt()}")
```

## Usage

The library revolves around the `ProbVector` class. To construct a `ProbVector`, simply feed an array to the constructor, and it will interpret the array as the probability distribution of the `ProbVector`. By default, it will normalize the array to have a sum equal to 1, and will rearrange the components to be in nonincreasing order.

The usefulness of the `ProbVector` class is in its implementation of methods for majorization-related tasks. Let `p` and `q` be `ProbVector`s, then: 
- `p < q` returns `True` if `p` is majorized by `q`, and `False` otherwise.
- `p > q` returns `True` if `p` majorizes `q`, and `False` otherwise.
- `p + q` returns a new `ProbVector` which is the meet of `p` and `q`.
- `p * q` returns a new `ProbVector` which is the join of `p` and `q`.
- `p - q` returns the value of the entropic distance $d(p, q)$ from Cicalese, Gargano and Vaccaro 2013 (cf. Section 1.4.3) as a float.

A class `BistochMatrix` is also defined, which is useful to apply bistochastic degradations to `ProbVectors` and check monotonicity results.

Most quantities defined in the MSc Thesis "Majorization lattice in the theory of entanglement" are also defined in the `majorization` module, and can be called. List of functions (non-exhaustive):

- `entropy(p)`: returns the Shannon entropy of `p`as a float.
- `renyi_entropy(p, alpha)`: returns the Rényi entropy of `p` of order `alpha` as a float.
- `E_future(p, q)`: returns the value of the future incomparability $E^+(p, \parallel q)$ of the probe state `p` to the reference state `q` (cf. Definition 4.1) as a float.
- `E_past(p, q)`: returns the value of the past incomparability $E^-(p, \parallel q)$ of the probe state `p` to the reference state `q` (cf. Definition 4.2)as a float.
- `unique_entropy(p, bank)`: returns the value of the uniqueness entropy of `p` relative to the `bank` as a float (cf. Definition 5.1) as a float.
- `construct_concatenated(p, q)`: returns a non-normalized `ProbVector` which is the ordered concatenation of `p` and `q`. Useful to test the postulated majorization precursor from Conjecture 3.1.

Finally, a couple pre-made scripts are also available in the `examples/` and the `tests/` folders. Note that the libraries QuTip, matplotlib and tqdm are only used here. They are thus not necessary to run the `majorization` module itself.

- `hypothesis_test.py`: implements a generic hypothesis test, which seeks a counterexample with statistical sampling. In the default state, the code seeks a counterexample to the bank monotonicity of the uniqueness entropy, but the code should be tweaked manually for another hypothesis test.
- `renyi_criteria.py`: implements the Rényi criterion test to compare with the meet from Section 4.1.3 by sampling entangled state representations. The parameters can be tweaked at the bottom of the file, after the `if __name__ == "__main__"` test.
- `locc_game.py`: implements the mixed RSSS from Definition 5.4 and compares the different strategies for different values of $\alpha$ by sampling banks uniformly on the $\Delta_{d-1}$ simplex and lists of successive targets to construct. The distribution from which the targets are sampled on the simplex is a Dirichlet distribution, which can be skewed with the parameter `skew` towards the bottom of the simplex, which means sampling less entangled states on average. A value of `skew` of 1 yields the uniform distribution, which is not recommended as all strategies perform poorly at such values and are thus difficult to compare. The parameters can be tweaked at the bottom of the file, after the `if __name__ == "__main__"` test.