"""
Quantum information tools for entanglement transformations.

This module implements SLOCC (Stochastic LOCC) protocols and other
quantum-specific operations using Schmidt coefficients.
"""

import numpy as np
import qutip as qt
from .majorization import ProbVector


class SLOCC():
    """Class for Stochastic LOCC (SLOCC) protocols. Given an initial pure state with Schmidt coefficients
       and a target 'success' state with desired Schmidt coefficients, this class constructs the Kraus operators
       for the measurement that either projects onto the success state (with some probability) or onto a failure state.

       The protocol works as follows:
       1. Start with initial state |ψ⟩ with Schmidt decomposition Σ λ_i |ii⟩
       2. Apply local operations to attempt transformation to target state |φ⟩ with Schmidt coeffs μ_i
       3. Perform a final measurement with two outcomes:
          - Success (Kraus operator M): projects onto target state with probability p_success
          - Failure (Kraus operator N): projects onto orthogonal complement with probability 1 - p_success

       Parameters:
       initial_schmidt (ProbVector): Schmidt coefficients of the initial bipartite pure state
       target_schmidt (ProbVector): Schmidt coefficients of the desired success state
       success_prob (float, optional): Desired success probability. If None, will be computed optimally.
    """

    __slots__ = ('initial_schmidt', 'target_schmidt', 'success_prob',
                 'kraus_M', 'kraus_N', 'initial_state', 'success_state', 'failure_state',
                 'failure_schmidt', 'dim')

    def __init__(self, initial_schmidt, target_schmidt, success_prob=None):
        # Store Schmidt coefficients as ProbVectors
        self.initial_schmidt = initial_schmidt if isinstance(initial_schmidt, ProbVector) else ProbVector(initial_schmidt)
        self.target_schmidt = target_schmidt if isinstance(target_schmidt, ProbVector) else ProbVector(target_schmidt)

        # Pad to same dimension if needed
        self.dim = max(len(self.initial_schmidt), len(self.target_schmidt))
        if len(self.initial_schmidt) < self.dim:
            self.initial_schmidt = ProbVector(np.append(self.initial_schmidt.probs,
                                                       [0]*(self.dim - len(self.initial_schmidt))))
        if len(self.target_schmidt) < self.dim:
            self.target_schmidt = ProbVector(np.append(self.target_schmidt.probs,
                                                      [0]*(self.dim - len(self.target_schmidt))))

        # Compute or validate success probability
        if success_prob is None:
            self.success_prob = self._compute_optimal_success_prob()
        else:
            self.success_prob = success_prob
            if not self._validate_success_prob():
                raise ValueError(f"Success probability {success_prob} is not achievable for this transformation")

        # Build quantum states and Kraus operators
        self.initial_state = self._build_density_matrix(self.initial_schmidt)
        self._construct_kraus_operators()

        # Apply Kraus operators to get success and failure states
        self.success_state = self._apply_kraus(self.kraus_M)
        self.failure_state = self._apply_kraus(self.kraus_N)

        # Extract Schmidt coefficients of failure state
        self.failure_schmidt = self._extract_schmidt_coeffs(self.failure_state)

    def _build_density_matrix(self, schmidt_coeffs):
        """Construct density matrix for bipartite pure state from Schmidt coefficients.

        For a bipartite pure state |ψ⟩ = Σ √λ_i |i⟩_A ⊗ |i⟩_B,
        the density matrix is ρ = |ψ⟩⟨ψ|
        """
        dim = len(schmidt_coeffs)

        # Create the pure state vector in the computational basis
        state_vector = np.zeros(dim * dim, dtype=complex)
        for i in range(dim):
            # Index i in the bipartite Hilbert space corresponds to |i⟩_A ⊗ |i⟩_B
            # which is at position i*dim + i in the flattened basis
            state_vector[i * dim + i] = np.sqrt(schmidt_coeffs[i])

        # Convert to QuTiP ket and create density matrix
        psi = qt.Qobj(state_vector, dims=[[dim, dim], [1, 1]])
        rho = psi * psi.dag()

        return rho

    def _compute_optimal_success_prob(self):
        """Compute the maximum achievable success probability for the transformation.

        For LOCC transformations, Nielsen's theorem states that |ψ⟩ → |φ⟩ is possible
        iff λ is majorized by μ (λ ≺ μ). For SLOCC, we can achieve any transformation
        with some probability.

        Following Vidal's paper, the maximum success probability is given by:
        p_max = min_k (Σ_{i=d-k+1}^{d} λ_i) / (Σ_{i=d-k+1}^{d} μ_i)

        This iterates over cumulative sums from smallest to largest Schmidt coefficient.
        """
        # If target majorizes initial, deterministic transformation is possible
        if self.target_schmidt > self.initial_schmidt:
            return 1.0

        # Otherwise compute maximum achievable probability
        # Iterate from smallest to largest (reverse order since ProbVectors are sorted descending)
        max_prob = np.inf
        sum_initial = 0
        sum_target = 0

        for i in range(self.dim - 1, -1, -1):  # Iterate backwards (smallest to largest)
            sum_initial += self.initial_schmidt[i]
            sum_target += self.target_schmidt[i]

            if sum_target > 1e-12:  # Avoid division by zero
                ratio = sum_initial / sum_target
                max_prob = min(max_prob, ratio)

        # Ensure we don't exceed probability 1
        return min(max_prob, 1.0)

    def _validate_success_prob(self):
        """Check if the requested success probability is achievable."""
        max_prob = self._compute_optimal_success_prob()
        return self.success_prob <= max_prob + 1e-10  # Small tolerance for numerical errors

    def _construct_kraus_operators(self):
        """Construct the Kraus operators M (success) and N (failure) for the SLOCC protocol.

        For a bipartite state |ψ⟩ = Σ √λ_i |i⟩_A|i⟩_B in Schmidt basis,
        we want to transform it to |φ⟩ = Σ √μ_i |i⟩_A|i⟩_B with some probability p.

        Alice applies a local POVM {M_A, N_A} where:
        - M_A = Σ_i √(p μ_i/λ_i) |i⟩⟨i|  (success measurement)
        - N_A satisfies M_A† M_A + N_A† N_A = I_A (completeness on Alice's space)

        The corresponding Kraus operators on the full bipartite space are M_A ⊗ I_B and N_A ⊗ I_B.
        """
        dim = self.dim

        # Construct M_A (diagonal in Schmidt basis)
        M_A = np.zeros((dim, dim), dtype=complex)
        for i in range(dim):
            if self.initial_schmidt[i] > 1e-12:
                M_A[i, i] = np.sqrt(self.success_prob * self.target_schmidt[i] / self.initial_schmidt[i])

        # Construct N_A from completeness: M_A† M_A + N_A† N_A = I
        M_A_dag_M_A = M_A.conj().T @ M_A
        N_A_dag_N_A = np.eye(dim) - M_A_dag_M_A

        # Ensure positive semidefinite (fix numerical errors)
        eigenvals, eigenvecs = np.linalg.eigh(N_A_dag_N_A)
        eigenvals = np.maximum(eigenvals, 0)
        N_A = eigenvecs @ np.diag(np.sqrt(eigenvals)) @ eigenvecs.conj().T

        # Tensor with identity on Bob's side: M_A ⊗ I_B and N_A ⊗ I_B
        I_B = np.eye(dim)
        M_bipartite = np.kron(M_A, I_B)
        N_bipartite = np.kron(N_A, I_B)

        # Convert to QuTiP objects with correct dimensions
        self.kraus_M = qt.Qobj(M_bipartite, dims=[[dim, dim], [dim, dim]])
        self.kraus_N = qt.Qobj(N_bipartite, dims=[[dim, dim], [dim, dim]])

    def _apply_kraus(self, kraus_op):
        """Apply a Kraus operator to the initial state and return the resulting (unnormalized) state.

        The resulting state is: ρ' = K ρ K†
        Then normalize: ρ_out = ρ' / Tr(ρ')
        """
        rho_prime = kraus_op * self.initial_state * kraus_op.dag()
        trace = rho_prime.tr()

        if trace < 1e-12:
            # Return zero state if trace is too small
            return rho_prime

        return rho_prime / trace

    def _extract_schmidt_coeffs(self, rho):
        """Extract Schmidt coefficients from a bipartite density matrix.

        For a bipartite state ρ_AB, the Schmidt coefficients are the eigenvalues
        of the reduced density matrix ρ_A = Tr_B(ρ_AB).
        """
        # Take partial trace over Bob's system (second subsystem)
        rho_A = rho.ptrace(0)

        # Get eigenvalues (these are the Schmidt coefficients squared)
        eigenvals = rho_A.eigenenergies()

        # Sort in descending order and ensure non-negative
        schmidt_coeffs = np.sort(np.abs(eigenvals))[::-1]

        # Normalize (should already be normalized, but just in case)
        schmidt_coeffs = schmidt_coeffs / np.sum(schmidt_coeffs)

        return ProbVector(schmidt_coeffs)

    def get_success_schmidt(self):
        """Return Schmidt coefficients of the success state."""
        return self._extract_schmidt_coeffs(self.success_state)

    def get_failure_schmidt(self):
        """Return Schmidt coefficients of the failure state."""
        return self.failure_schmidt

    def get_success_probability(self):
        """Return the success probability of the protocol."""
        # Can also be computed from the trace: Tr(M ρ M†)
        prob = (self.kraus_M * self.initial_state * self.kraus_M.dag()).tr()
        return np.real(prob)

    def get_failure_probability(self):
        """Return the failure probability of the protocol."""
        return 1.0 - self.get_success_probability()

    def verify_completeness(self):
        """Verify that the Kraus operators satisfy the completeness relation: M†M + N†N = I."""
        M_dag_M = self.kraus_M.dag() * self.kraus_M
        N_dag_N = self.kraus_N.dag() * self.kraus_N
        completeness = M_dag_M + N_dag_N
        identity = qt.qeye(completeness.dims[0])

        difference = (completeness - identity).norm()
        return difference < 1e-8

    def __repr__(self):
        return (f"SLOCC(initial={self.initial_schmidt}, target={self.target_schmidt}, "
                f"p_success={self.success_prob:.4f})")

    def __str__(self):
        return (f"SLOCC Protocol:\n"
                f"  Initial Schmidt coeffs: {self.initial_schmidt}\n"
                f"  Target Schmidt coeffs:  {self.target_schmidt}\n"
                f"  Success probability:    {self.success_prob:.4f}\n"
                f"  Failure Schmidt coeffs: {self.failure_schmidt}\n"
                f"  Kraus completeness:     {self.verify_completeness()}")
