"""
Example script demonstrating SLOCC (Stochastic LOCC) protocols.

This script shows how to:
1. Create a SLOCC protocol with initial and target Schmidt coefficients
2. Extract the Schmidt coefficients of both success and failure states
3. Verify the physical consistency of the protocol
4. Compare different transformations
"""

import numpy as np
from majolat import ProbVector, SLOCC, plot_lorenz_curves

def example_basic_slocc():
    """Basic example: try to transform from a less entangled to a more entangled state."""
    print("=" * 70)
    print("Example 1: Basic SLOCC transformation")
    print("=" * 70)

    # Initial state: less entangled (more uniform Schmidt coefficients)
    initial = ProbVector([0.6, 0.3, 0.1])

    # Target state: more entangled (more peaked Schmidt coefficients)
    target = ProbVector([0.8, 0.15, 0.05])

    # Create SLOCC protocol (optimal success probability computed automatically)
    slocc = SLOCC(initial, target)

    print(slocc)
    print()

    # Access the results
    print(f"Success state Schmidt coefficients: {slocc.get_success_schmidt()}")
    print(f"Failure state Schmidt coefficients: {slocc.get_failure_schmidt()}")
    print(f"Success probability: {slocc.get_success_probability():.6f}")
    print(f"Failure probability: {slocc.get_failure_probability():.6f}")
    print()

    # Verify the protocol is physically consistent
    print(f"Kraus operators complete: {slocc.verify_completeness()}")
    print()

def example_deterministic_transformation():
    """Example with deterministic transformation (target majorizes initial)."""
    print("=" * 70)
    print("Example 2: Deterministic LOCC transformation")
    print("=" * 70)

    # Initial state: more entangled
    initial = ProbVector([0.8, 0.15, 0.05])

    # Target state: less entangled (majorized by initial)
    target = ProbVector([0.6, 0.3, 0.1])

    # This should give success probability = 1.0
    slocc = SLOCC(initial, target)

    print(slocc)
    print()

    print(f"Since target is majorized by initial, success probability = {slocc.get_success_probability():.6f}")
    print(f"This is a deterministic LOCC transformation (Nielsen's theorem).")
    print()

def example_with_custom_success_prob():
    """Example with manually specified (suboptimal) success probability."""
    print("=" * 70)
    print("Example 3: SLOCC with custom success probability")
    print("=" * 70)

    initial = ProbVector([0.7, 0.2, 0.1])
    target = ProbVector([0.9, 0.08, 0.02])

    # First compute optimal
    slocc_optimal = SLOCC(initial, target)
    print(f"Optimal success probability: {slocc_optimal.get_success_probability():.6f}")
    print()

    # Now use suboptimal probability
    custom_prob = 0.5
    slocc_custom = SLOCC(initial, target, success_prob=custom_prob)

    print(f"Custom success probability: {slocc_custom.get_success_probability():.6f}")
    print(f"Failure state Schmidt coefficients: {slocc_custom.get_failure_schmidt()}")
    print()

def example_comparing_failure_states():
    """Compare how failure states depend on target state choice."""
    print("=" * 70)
    print("Example 4: Comparing different target states")
    print("=" * 70)

    # Same initial state
    initial = ProbVector([0.5, 0.3, 0.15, 0.05])

    # Different target states
    target1 = ProbVector([0.7, 0.2, 0.08, 0.02])
    target2 = ProbVector([0.9, 0.06, 0.03, 0.01])

    slocc1 = SLOCC(initial, target1)
    slocc2 = SLOCC(initial, target2)

    print(f"Target 1: {target1}")
    print(f"  Success prob: {slocc1.get_success_probability():.6f}")
    print(f"  Failure Schmidt: {slocc1.get_failure_schmidt()}")
    print()

    print(f"Target 2: {target2}")
    print(f"  Success prob: {slocc2.get_success_probability():.6f}")
    print(f"  Failure Schmidt: {slocc2.get_failure_schmidt()}")
    print()

    # Compare using majorization
    if slocc1.get_failure_schmidt() < slocc2.get_failure_schmidt():
        print("Target 1 leads to a failure state majorized by target 2's failure state")
    elif slocc1.get_failure_schmidt() > slocc2.get_failure_schmidt():
        print("Target 2 leads to a failure state majorized by target 1's failure state")
    else:
        print("The failure states are incomparable under majorization")
    print()

def example_visualize_lorenz_curves():
    """Visualize the Lorenz curves for initial, target, success, and failure states."""
    print("=" * 70)
    print("Example 5: Visualizing Lorenz curves")
    print("=" * 70)

    initial = ProbVector([0.6, 0.25, 0.1, 0.05])
    target = ProbVector([0.8, 0.12, 0.05, 0.03])

    slocc = SLOCC(initial, target)

    print(f"Initial: {initial}")
    print(f"Target:  {target}")
    print(f"Success: {slocc.get_success_schmidt()}")
    print(f"Failure: {slocc.get_failure_schmidt()}")
    print(f"Success probability: {slocc.get_success_probability():.6f}")
    print()

    # Plot Lorenz curves
    plot_lorenz_curves(
        initial,
        target,
        slocc.get_success_schmidt(),
        slocc.get_failure_schmidt(),
        labels=["Initial", "Target", "Success", "Failure"],
        colors=["blue", "green", "red", "orange"],
        markers=["o", "s", "^", "v"],
        title="SLOCC Protocol: Lorenz Curves"
    )

def example_maximal_entanglement():
    """Try to generate a maximally entangled state from a less entangled one."""
    print("=" * 70)
    print("Example 6: Attempting to create maximal entanglement")
    print("=" * 70)

    # Start with moderately entangled state
    initial = ProbVector([0.5, 0.3, 0.15, 0.05])

    # Target: maximally entangled state (uniform Schmidt coefficients)
    target = ProbVector([0.25, 0.25, 0.25, 0.25])

    slocc = SLOCC(initial, target)

    print(slocc)
    print()

    print("Note: Creating maximal entanglement from less entangled states")
    print("is impossible deterministically, but can be done stochastically!")
    print()

if __name__ == "__main__":
    # Run all examples
    example_basic_slocc()
    example_deterministic_transformation()
    example_with_custom_success_prob()
    example_comparing_failure_states()
    example_maximal_entanglement()

    # Uncomment to visualize (requires display)
    # example_visualize_lorenz_curves()

    print("=" * 70)
    print("All examples completed!")
    print("=" * 70)
