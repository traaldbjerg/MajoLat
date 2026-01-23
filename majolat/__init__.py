"""
Majolat: Majorization and Quantum Information Tools

A library for working with majorization theory, probability vectors,
(bi)stochastic matrices, and quantum entanglement transformations.

Core classes:
- ProbVector: Probability distributions with majorization operations
- StochMatrix: Stochastic matrices
- BistochMatrix: Bistochastic matrices
- SLOCC: Stochastic LOCC protocols for quantum entanglement

Utility functions:
- Entropy measures (Shannon, Rényi, relative, mutual)
- Distance functions (d, d_prime)
- Incomparability measures (E_future, E_past, F, G)
- Uniqueness entropy and related functions
- Visualization tools (plot_lorenz_curves)
"""

__version__ = "0.1.0"

# Core majorization classes
from .majorization import ProbVector, StochMatrix, BistochMatrix

# Quantum tools
from .quantum import SLOCC

# Entropy and information measures
from .utils import (
    entropy,
    renyi_entropy,
    mutual_information,
    relative_entropy,
    ar_entropy,
    hr_entropy,
    tsallis_entropy,
    sm_entropy,
    cc_entropy,
)

# Distance measures
from .utils import d, d_prime

# Incomparability measures
from .utils import E_future, E_past, F, G

# Uniqueness entropy
from .utils import unique_entropy, remove_majorizers

# Miscellaneous utility functions
from .utils import (
    construct_concatenated,
    guessing_entropy,
    D,
    S,
    d_subadd,
    concatenate,
    TV,
    split_resource,
)

# Visualization
from .utils import plot_lorenz_curves

# Define public API
__all__ = [
    # Core classes
    'ProbVector',
    'StochMatrix',
    'BistochMatrix',
    'SLOCC',

    # Entropy measures
    'entropy',
    'renyi_entropy',
    'mutual_information',
    'relative_entropy',

    # Distance measures
    'd',
    'd_prime',

    # Incomparability measures
    'E_future',
    'E_past',
    'F',
    'G',

    # Uniqueness entropy
    'unique_entropy',
    'remove_majorizers',

    # Miscellaneous
    'construct_concatenated',
    'guessing_entropy',
    'D',
    'S',
    'd_subadd',
    'concatenate',
    'TV',
    'split_resource',

    # Visualization
    'plot_lorenz_curves',
]
