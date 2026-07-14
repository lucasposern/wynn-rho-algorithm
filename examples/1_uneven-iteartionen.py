import numpy as np
import matplotlib.pyplot as plt
from math import pi, log


def wynn_rho(S, Kmax):
    """Compute the ρ-sequence from the given recurrence."""
    N = len(S)
    rho = np.full((N, Kmax + 2), np.nan)
    rho[:, 0] = 0.0
    rho[:, 1] = S

    for k in range(1, Kmax + 1):
        j = k + 1
        for n in range(N - k):
            diff = rho[n + 1, j - 1] - rho[n, j - 1]
            if np.abs(diff) < 1e-20:
                rho[n, j] = np.nan
            else:
                rho[n, j] = rho[n + 1, j - 2] + k / diff
    return rho


def analyze_sequence(sequence, true_value, title, Kmax=6):
    """Analyze a sequence with the ρ-algorithm and create plots."""
    N = len(sequence)
    rho_table = wynn_rho(sequence, Kmax)

    # Determine the highest even and odd stages
    even_k = Kmax if Kmax % 2 == 0 else Kmax - 1
    odd_k = Kmax - 1 if Kmax % 2 == 0 else Kmax

    # Extract the ρ-sequences for even and odd stages
    rho_even = rho_table[:, even_k + 1]
    rho_odd = rho_table[:, odd_k + 1]

    # Valid indices
    valid_even = np.arange(N - even_k)
    valid_odd = np.arange(N - odd_k)

    # Create figures
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle(f'Analysis for: {title}', fontsize=16)

    # Plot 1: value comparison
    ax1.plot(np.arange(N), sequence, 'mo-', markersize=4, label='Original sequence')
    ax1.plot(valid_even, rho_even[valid_even], 'b-', linewidth=2,
             label=f'ρ (k={even_k}, even)')
    ax1.plot(valid_odd, rho_odd[valid_odd], 'g-', linewidth=2,
             label=f'ρ (k={odd_k}, odd)')
    ax1.axhline(y=true_value, color='r', linestyle='--', label='Limit')
    ax1.set_xlabel('n')
    ax1.set_ylabel('Value')
    ax1.set_title('Value comparison')
    ax1.legend()
    ax1.grid(True)

    # Plot 2: error comparison
    error_original = np.abs(sequence - true_value)
    error_even = np.abs(rho_even[valid_even] - true_value)
    error_odd = np.abs(rho_odd[valid_odd] - true_value)

    ax2.semilogy(np.arange(N), error_original, 'mo-', markersize=4,
                 label='Sequence error')
    ax2.semilogy(valid_even, error_even, 'b-', linewidth=2,
                 label=f'Error ρ (k={even_k})')
    ax2.semilogy(valid_odd, error_odd, 'g-', linewidth=2,
                 label=f'Error ρ (k={odd_k})')
    ax2.set_xlabel('n')
    ax2.set_ylabel('Absolute error (log)')
    ax2.set_title('Error comparison')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    return rho_table


# Example 1: Riemann zeta function ζ(2) = π²/6
N1 = 50
n_vals1 = np.arange(1, N1 + 1, dtype=float)
S_zeta = np.cumsum(1 / n_vals1 ** 2)
zeta_2 = pi ** 2 / 6

'''
# Example 2: Alternating harmonic series (ln(2))
N2 = 50
n_vals2 = np.arange(1, N2 + 1, dtype=float)
S_alt = np.cumsum((-1) ** (n_vals2 + 1) / n_vals2)
ln2 = log(2)
'''
'''
# Example 3: Asymptotic expansion
theta = -4
c0, c1, c2 = 1.0, 0.5, 0.1
N3 = 50
n_vals3 = np.arange(1, N3 + 1, dtype=float)
S_asym = n_vals3 ** theta * (c0 + c1 / n_vals3 + c2 / n_vals3 ** 2)
'''

# Run the analyses
print("Analyzing zeta sequence...")
rho_zeta = analyze_sequence(S_zeta, zeta_2, "Zeta sequence for ζ(2)")

print("\nAnalyzing alternating harmonic series...")
rho_alt = analyze_sequence(S_alt, ln2, "Alternating harmonic series")

print("\nAnalyzing asymptotic sequence...")
rho_asym = analyze_sequence(S_asym, 0, f"Asymptotic sequence (θ={theta}, c₀={c0})")

# Additional plot: comparison of convergence speed
plt.figure(figsize=(12, 8))

# Error for the highest stages of the zeta sequence
error_zeta_orig = np.abs(S_zeta - zeta_2)
error_zeta_even = np.abs(rho_zeta[:, 7][:len(S_zeta) - 6] - zeta_2)  # k=6
plt.semilogy(np.arange(N1), error_zeta_orig, 'm-', label='Zeta sequence error')
plt.semilogy(np.arange(6, N1), error_zeta_even, 'b-', label='Zeta ρ (k=6)')

'''
# Error for the alternating series
error_alt_orig = np.abs(S_alt - ln2)
error_alt_odd = np.abs(rho_alt[:, 6][:len(S_alt) - 5] - ln2)  # k=5
plt.semilogy(np.arange(N2), error_alt_orig, 'c-', label='Alt. sequence error')
plt.semilogy(np.arange(5, N2), error_alt_odd, 'g-', label='Altern. ρ (k=5)')
'''
'''
# Error for the asymptotic sequence
error_asym_orig = np.abs(S_asym - 0)
error_asym_even = np.abs(rho_asym[:, 7][:len(S_asym) - 6] - 0)  # k=6
plt.semilogy(np.arange(N3), error_asym_orig, 'r-', label='Asympt. sequence error')
plt.semilogy(np.arange(6, N3), error_asym_even, 'y-', label='Asympt. ρ (k=6)')
'''

plt.xlabel('n')
plt.ylabel('Absolute error (log)')
plt.title('Comparison of convergence speeds')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()