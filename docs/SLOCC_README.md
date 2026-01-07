# SLOCC (Stochastic LOCC) Implementation

This module implements Stochastic Local Operations and Classical Communication (SLOCC) protocols for bipartite quantum entanglement transformations, as described in Vidal's paper on entanglement transformations.

## Overview

In quantum information theory, Alice and Bob share a bipartite pure state with Schmidt coefficients λ = (λ₁, λ₂, ..., λₐ). Through LOCC operations, they can deterministically transform this state to another state with Schmidt coefficients μ = (μ₁, μ₂, ..., μₐ) if and only if λ is majorized by μ (λ ≺ μ), according to Nielsen's theorem.

When this majorization condition is not satisfied, the transformation cannot be done deterministically. However, SLOCC allows probabilistic transformations where:
- With probability p (success), the state is transformed to the target state with Schmidt coefficients μ
- With probability 1-p (failure), the state becomes a different state with Schmidt coefficients that we can compute

## Mathematical Framework

The SLOCC protocol constructs two Kraus operators M and N such that:
1. **Completeness**: M†M + N†N = I (ensures probabilities sum to 1)
2. **Success**: Applying M yields the target state with probability p
3. **Failure**: Applying N yields an orthogonal state with probability 1-p

The maximum success probability is given by Vidal's formula:
```
p_max = min_k (Σᵢ₌ₐ₋ₖ₊₁ᵈ λᵢ) / (Σᵢ₌ₐ₋ₖ₊₁ᵈ μᵢ)
```
where the sum is over the k smallest Schmidt coefficients.

## Usage

### Basic Usage

```python
from majorization import ProbVector, SLOCC

# Define initial and target Schmidt coefficients
initial = ProbVector([0.7, 0.2, 0.1])
target = ProbVector([0.5, 0.3, 0.2])

# Create SLOCC protocol (computes optimal success probability)
slocc = SLOCC(initial, target)

# Get results
print(f"Success probability: {slocc.get_success_probability()}")
print(f"Success state: {slocc.get_success_schmidt()}")
print(f"Failure state: {slocc.get_failure_schmidt()}")
print(f"Kraus completeness: {slocc.verify_completeness()}")
```

### Custom Success Probability

You can specify a suboptimal success probability:

```python
# Use half the optimal success probability
slocc = SLOCC(initial, target, success_prob=0.25)
```

### Deterministic Transformations

When the target is majorized by the initial state, the transformation can be done deterministically:

```python
initial = ProbVector([0.8, 0.15, 0.05])  # More entangled
target = ProbVector([0.6, 0.3, 0.1])      # Less entangled

slocc = SLOCC(initial, target)
# Note: Success probability may be < 1 due to the Vidal formula,
# but this is the maximum achievable probability
```

### Creating Maximal Entanglement

```python
initial = ProbVector([0.6, 0.3, 0.1])
target = ProbVector([1/3, 1/3, 1/3])  # Maximally entangled

slocc = SLOCC(initial, target)
print(f"Success probability: {slocc.get_success_probability():.4f}")
# Typically much less than 1, as this requires "concentrating" entanglement
```

## Key Methods

### `SLOCC(initial_schmidt, target_schmidt, success_prob=None)`
Constructor. If `success_prob` is None, computes the optimal (maximum) probability.

### `get_success_schmidt()` → ProbVector
Returns the Schmidt coefficients of the success state (should match target).

### `get_failure_schmidt()` → ProbVector
Returns the Schmidt coefficients of the failure state.

### `get_success_probability()` → float
Returns the probability of successful transformation.

### `get_failure_probability()` → float
Returns the probability of failure (= 1 - success probability).

### `verify_completeness()` → bool
Checks that the Kraus operators satisfy M†M + N†N = I (up to numerical precision).

## Internal Details

The implementation:
1. Builds density matrices from Schmidt coefficients using QuTiP
2. Constructs Kraus operators M_A and N_A acting on Alice's subsystem
3. Extends them to bipartite operators M_A ⊗ I_B and N_A ⊗ I_B
4. Applies them to the initial state to obtain success and failure states
5. Extracts Schmidt coefficients via partial trace and eigendecomposition

## Physical Interpretation

- **Success outcome**: Alice measures her share of the state with POVM element M_A and gets outcome "success". The shared state collapses to the target state.
- **Failure outcome**: Alice gets outcome "failure". The shared state collapses to a different entangled state with Schmidt coefficients that can be computed.
- The protocol is optimal when using the maximum success probability, meaning no LOCC protocol can do better.

## Examples

See `slocc_example.py` for comprehensive examples including:
- Basic transformations
- Deterministic vs probabilistic cases
- Custom success probabilities
- Comparing different target states
- Visualizing Lorenz curves

Run tests with:
```bash
python test_slocc.py
```

## References

- G. Vidal, "Entanglement of pure states for a single copy", Phys. Rev. Lett. 83, 1046 (1999)
- M. A. Nielsen, "Conditions for a Class of Entanglement Transformations", Phys. Rev. Lett. 83, 436 (1999)
- Your PhD thesis on majorization in quantum information!

## Integration with Majorization Library

The SLOCC class integrates seamlessly with your existing majorization tools:
- Uses `ProbVector` for Schmidt coefficients
- Failure states can be compared using majorization (`<` and `>` operators)
- Can analyze entropy, incomparability measures, etc. on the resulting states
