"""
Analysis script for exploring SLOCC protocols and their failure states.

This script helps analyze how the failure state depends on the choice of
target state and success probability.
"""

from majolat import ProbVector, SLOCC, entropy
import numpy as np
import matplotlib.pyplot as plt


def analyze_failure_vs_success_prob(initial, target):
    """
    Analyze how the failure state changes as we vary the success probability.
    """
    # Compute optimal success probability
    slocc_opt = SLOCC(initial, target)
    p_max = slocc_opt.get_success_probability()

    print(f"Initial state: {initial}")
    print(f"Target state:  {target}")
    print(f"Max success probability: {p_max:.4f}\n")

    # Try different success probabilities
    probabilities = np.linspace(0.1 * p_max, p_max, 10)
    failure_entropies = []
    failure_largest_coeffs = []

    print("Success Prob | Failure Schmidt | Failure Entropy")
    print("-" * 60)

    for p in probabilities:
        slocc = SLOCC(initial, target, success_prob=p)
        failure_schmidt = slocc.get_failure_schmidt()
        failure_entropy = entropy(failure_schmidt)

        failure_entropies.append(failure_entropy)
        failure_largest_coeffs.append(failure_schmidt[0])

        print(f"{p:12.4f} | {str(failure_schmidt):25} | {failure_entropy:.4f}")

    print()

    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(probabilities, failure_entropies, 'o-', linewidth=2, markersize=8)
    ax1.set_xlabel('Success Probability', fontsize=12)
    ax1.set_ylabel('Failure State Entropy (bits)', fontsize=12)
    ax1.set_title('Failure State Entropy vs Success Probability', fontsize=14)
    ax1.grid(True, alpha=0.3)

    ax2.plot(probabilities, failure_largest_coeffs, 's-', color='red', linewidth=2, markersize=8)
    ax2.set_xlabel('Success Probability', fontsize=12)
    ax2.set_ylabel('Largest Schmidt Coefficient', fontsize=12)
    ax2.set_title('Failure State: Largest Coefficient vs Success Probability', fontsize=14)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def compare_multiple_targets(initial, targets):
    """
    Compare SLOCC protocols with the same initial state but different targets.
    """
    print(f"Initial state: {initial}\n")
    print("Target | Success Prob | Failure State | Failure Entropy")
    print("-" * 80)

    results = []
    for target in targets:
        slocc = SLOCC(initial, target)
        p_s = slocc.get_success_probability()
        failure_schmidt = slocc.get_failure_schmidt()
        failure_entropy = entropy(failure_schmidt)

        results.append({
            'target': target,
            'p_success': p_s,
            'failure_schmidt': failure_schmidt,
            'failure_entropy': failure_entropy
        })

        print(f"{str(target):25} | {p_s:12.4f} | {str(failure_schmidt):25} | {failure_entropy:.4f}")

    print()

    # Analyze majorization relationships among failure states
    print("Majorization relations among failure states:")
    for i, res_i in enumerate(results):
        for j, res_j in enumerate(results):
            if i < j:
                if res_i['failure_schmidt'] < res_j['failure_schmidt']:
                    print(f"  Failure_{i+1} ≺ Failure_{j+1} (target {i+1} failure is majorized by target {j+1} failure)")
                elif res_i['failure_schmidt'] > res_j['failure_schmidt']:
                    print(f"  Failure_{i+1} ≻ Failure_{j+1} (target {i+1} failure majorizes target {j+1} failure)")
                else:
                    print(f"  Failure_{i+1} ~ Failure_{j+1} (incomparable)")


def explore_entanglement_concentration():
    """
    Explore entanglement concentration: trying to create more peaked
    (more entangled) states from less peaked ones.
    """
    print("Entanglement Concentration Analysis")
    print("=" * 70)

    # Start with a relatively flat distribution
    initial = ProbVector([0.4, 0.3, 0.2, 0.1])

    # Try increasingly peaked target distributions
    targets = [
        ProbVector([0.5, 0.3, 0.15, 0.05]),
        ProbVector([0.6, 0.25, 0.1, 0.05]),
        ProbVector([0.7, 0.2, 0.07, 0.03]),
        ProbVector([0.8, 0.12, 0.05, 0.03]),
    ]

    print(f"Initial state: {initial}")
    print(f"Initial entropy: {entropy(initial):.4f} bits\n")

    success_probs = []
    target_entropies = []

    for i, target in enumerate(targets, 1):
        slocc = SLOCC(initial, target)
        p_s = slocc.get_success_probability()
        target_entropy = entropy(target)

        success_probs.append(p_s)
        target_entropies.append(target_entropy)

        print(f"Target {i}: {target}")
        print(f"  Entropy: {target_entropy:.4f} bits")
        print(f"  Success probability: {p_s:.4f}")
        print(f"  Failure state: {slocc.get_failure_schmidt()}")
        print()

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(target_entropies, success_probs, 'o-', linewidth=2, markersize=10)
    plt.xlabel('Target State Entropy (bits)', fontsize=12)
    plt.ylabel('Maximum Success Probability', fontsize=12)
    plt.title('Entanglement Concentration:\nSuccess Probability vs Target Entropy', fontsize=14)
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Deterministic (p=1)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def optimal_failure_state_analysis():
    """
    For a given initial state and target, analyze the optimal failure state
    and its properties.
    """
    print("Optimal Failure State Analysis")
    print("=" * 70)

    initial = ProbVector([0.6, 0.25, 0.1, 0.05])
    target = ProbVector([0.8, 0.12, 0.05, 0.03])

    slocc = SLOCC(initial, target)

    print(f"Initial:  {initial} | Entropy: {entropy(initial):.4f}")
    print(f"Target:   {target} | Entropy: {entropy(target):.4f}")
    print(f"Success probability: {slocc.get_success_probability():.4f}")
    print()

    success = slocc.get_success_schmidt()
    failure = slocc.get_failure_schmidt()

    print(f"Success:  {success} | Entropy: {entropy(success):.4f}")
    print(f"Failure:  {failure} | Entropy: {entropy(failure):.4f}")
    print()

    # Check majorization relations
    print("Majorization relations:")
    if initial > success:
        print(f"  Initial ≻ Success ✓")
    if initial > failure:
        print(f"  Initial ≻ Failure ✓")

    if success < failure:
        print(f"  Success ≺ Failure")
    elif success > failure:
        print(f"  Success ≻ Failure")
    else:
        print(f"  Success ~ Failure (incomparable)")

    # Compute incomparability measures if available
    try:
        from majolat import E_future, E_past, F, G

        print("\nIncomparability measures (Initial vs Success):")
        print(f"  E_future: {E_future(initial, success):.4f}")
        print(f"  E_past:   {E_past(initial, success):.4f}")
        print(f"  F:        {F(initial, success):.4f}")
        print(f"  G:        {G(initial, success):.4f}")

        print("\nIncomparability measures (Initial vs Failure):")
        print(f"  E_future: {E_future(initial, failure):.4f}")
        print(f"  E_past:   {E_past(initial, failure):.4f}")
        print(f"  F:        {F(initial, failure):.4f}")
        print(f"  G:        {G(initial, failure):.4f}")
    except ImportError:
        pass


if __name__ == "__main__":
    # Example 1: Vary success probability
    print("\n" + "=" * 70)
    print("Analysis 1: Failure state vs success probability")
    print("=" * 70 + "\n")

    initial = ProbVector([0.6, 0.25, 0.1, 0.05])
    target = ProbVector([0.4, 0.3, 0.2, 0.1])
    analyze_failure_vs_success_prob(initial, target)

    # Example 2: Compare multiple targets
    print("\n" + "=" * 70)
    print("Analysis 2: Multiple target states")
    print("=" * 70 + "\n")

    initial = ProbVector([0.5, 0.3, 0.15, 0.05])
    targets = [
        ProbVector([0.6, 0.25, 0.1, 0.05]),
        ProbVector([0.4, 0.35, 0.15, 0.1]),
        ProbVector([0.7, 0.2, 0.07, 0.03]),
    ]
    compare_multiple_targets(initial, targets)

    # Example 3: Entanglement concentration
    print("\n" + "=" * 70)
    print("Analysis 3: Entanglement concentration")
    print("=" * 70 + "\n")
    explore_entanglement_concentration()

    # Example 4: Detailed analysis of optimal failure state
    print("\n" + "=" * 70)
    print("Analysis 4: Optimal failure state properties")
    print("=" * 70 + "\n")
    optimal_failure_state_analysis()
