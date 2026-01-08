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
├── tests/               # Test files
├── docs/                # Documentation
├── pyproject.toml       # Package configuration
└── README.md           # This file
```

## Installation and Dependencies

This project uses [NumPy](https://numpy.org/), [Matplotlib](https://matplotlib.org/), and [QuTiP](https://qutip.org/) [1] and [uv](https://docs.astral.sh/uv/) for dependency management. To install the package (in editable mode by default) run:

```bash
uv sync
```

This will:
- Install Python 3.9 (required for qutip 4.x compatibility)
- Create a virtual environment in `.venv/`
- Install the package in editable mode along with all dependencies

To use the library, you then need to activate the environment:

```bash
# Activate the virtual environment
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate  # On Windows

# Now you can use the library
python examples/slocc_example.py
python tests/test_slocc.py
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

The usefulness of the `ProbVector` class is in its implementation of methods for majorization-related tasks (see Ref. [2] for definitions and entropic properties and Ref. [3] for applications to entanglement conversion). Let `p` and `q` be `ProbVector`s, then: 
- `p < q` returns `True` if `p` is majorized by `q`, and `False` otherwise.
- `p > q` returns `True` if `p` majorizes `q`, and `False` otherwise.
- `p + q` returns a new `ProbVector` which is the meet of `p` and `q`.
- `p * q` returns a new `ProbVector` which is the join of `p` and `q`.
- `p - q` returns the value of the entropic distance $d(p, q)$ from Ref. [4] as a float.

A class `BistochMatrix` is also defined, which is useful to apply bistochastic degradations to `ProbVectors` and check monotonicity results.

Most quantities defined in the MSc Thesis "Majorization lattice in the theory of entanglement" [5] are also defined in the `majorization` submodule, and can be called. Non-exhaustive list of functions (section and equation numbers from Ref. [5]):

- `entropy(p)`: returns the Shannon entropy of `p`as a float.
- `renyi_entropy(p, alpha)`: returns the Rényi entropy of `p` of order `alpha` as a float.
- `E_future(p, q)`: returns the value of the future incomparability $E^+(p, \parallel q)$ of the probe state `p` to the reference state `q` (cf. Definition 4.1) as a float.
- `E_past(p, q)`: returns the value of the past incomparability $E^-(p, \parallel q)$ of the probe state `p` to the reference state `q` (cf. Definition 4.2) as a float.
- `unique_entropy(p, bank)`: returns the value of the uniqueness entropy of `p` relative to the `bank` as a float (cf. Definition 5.1) as a float.
- `construct_concatenated(p, q)`: returns a non-normalized `ProbVector` which is the ordered concatenation of `p` and `q`. Useful to test the postulated majorization precursor from Conjecture 3.1.

Finally, a couple pre-made scripts are also available in the `examples/` and the `tests/` folders. Note that the libraries QuTip, matplotlib and tqdm are only used here. They are thus not necessary to run the `majorization` module itself.

- `hypothesis_test.py`: implements a generic hypothesis test, which seeks a counterexample with statistical sampling. In the default state, the code seeks a counterexample to the bank monotonicity of the uniqueness entropy, but the code should be tweaked manually for another hypothesis test.
- `renyi_criteria.py`: implements the Rényi criterion test to compare with the meet from Section 4.1.3 by sampling entangled state representations. The parameters can be tweaked at the bottom of the file, after the `if __name__ == "__main__"` test.
- `locc_game.py`: implements the mixed RSSS from Definition 5.4 and compares the different strategies for different values of $\alpha$ by sampling banks uniformly on the $\Delta_{d-1}$ simplex and lists of successive targets to construct. The distribution from which the targets are sampled on the simplex is a Dirichlet distribution, which can be skewed with the parameter `skew` towards the bottom of the simplex, which means sampling less entangled states on average. A value of `skew` of 1 yields the uniform distribution, which is not recommended as all strategies perform poorly at such values and are thus difficult to compare. The parameters can be tweaked at the bottom of the file, after the `if __name__ == "__main__"` test.

## References

[1] J. R. Johansson, P. D. Nation, and F. Nori, QuTiP: An open-source Python framework for the dynamics of open quantum systems, Comput. Phys. Comm. 183, 1760 (2012).

[2] F. Cicalese and U. Vaccaro, Supermodularity and subadditivity properties of the entropy on the majorization lattice, IEEE Trans. Inform. Theory 48, 933 (2002).

[3] M. A. Nielsen and G. Vidal, Majorization and the interconversion of bipartite states, Quant. Inf. Comput. 1, 76 (2001).

[4] F. Cicalese, L. Gargano, and U. Vaccaro, Information Theoretic Measures of Distances and Their Econometric Applications, in 2013 IEEE International Symposium on Information Theory (2013), pp. 409–413.

[5] A. Stévins, Majorization lattice in the theory of quantum entanglement, Master thesis, Université libre de Bruxelles, 2025.
