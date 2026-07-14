import numpy as np
import matplotlib.pyplot as plt


def wynn_rho_classic(S, Kmax):
    """Classical Wynn ρ-algorithm"""
    N = len(S)
    rho = np.full((N, Kmax + 2), np.nan)
    rho[:, 0] = 0.0  # k = -1
    rho[:, 1] = S  # k = 0

    for k in range(1, Kmax + 1):
        j = k + 1
        for n in range(N - k):
            diff = rho[n + 1, j - 1] - rho[n, j - 1]
            if np.abs(diff) < 1e-20:
                rho[n, j] = np.nan
            else:
                rho[n, j] = rho[n + 1, j - 2] + k / diff
    return rho


def wynn_rho_modified(S, Kmax, gamma, p):
    """Modified ρ(γ,p)-algorithm following the new definition"""
    N = len(S)
    # Kmax+2 columns: k=-1,0,1,...,Kmax
    rho = np.full((N, Kmax + 2), np.nan)
    rho[:, 0] = 0.0  # k = -1
    rho[:, 1] = S  # k = 0

    # Compute the coefficients C_k^(γ,p) for k=0 to Kmax-1
    C = []
    for k in range(0, Kmax):
        if k % 2 == 0:  # even index (2m)
            m = k // 2
            C_val = -gamma + m / p
        else:  # odd index (2m+1)
            m = (k - 1) // 2
            C_val = -gamma + m / p + 1
        C.append(C_val)

    # Run the modified algorithm
    for k in range(0, Kmax):
        for n in range(N - k - 1):
            diff = rho[n + 1, k + 1] - rho[n, k + 1]
            if np.abs(diff) < 1e-20:
                rho[n, k + 2] = np.nan
            else:
                rho[n, k + 2] = rho[n + 1, k] + C[k] / diff
    return rho


def compare_algorithms(sequence, true_value, title, k_value, gamma, p):
    """Compare the original sequence, classical ρ and modified ρ(γ,p) for a fixed k"""
    N = len(sequence)

    # Compute both algorithms
    rho_classic = wynn_rho_classic(sequence, k_value)
    rho_modified = wynn_rho_modified(sequence, k_value, gamma, p)

    # Valid indices for the chosen k
    valid_n_classic = np.arange(N - k_value)
    valid_n_modified = np.arange(N - k_value - 1)

    # Plot of the values
    plt.figure(figsize=(12, 8))

    # Original sequence
    plt.plot(np.arange(N), sequence, 'm-', linewidth=2, label='Original sequence')

    # Classical ρ-algorithm
    if len(valid_n_classic) > 0:
        plt.plot(valid_n_classic, rho_classic[valid_n_classic, k_value + 1], 'b-',
                 label=f'Classical ρ (k={k_value})')

    # Modified ρ(γ,p)-algorithm
    if len(valid_n_modified) > 0:
        plt.plot(valid_n_modified, rho_modified[valid_n_modified, k_value + 1], 'g-',
                 label=f'Modified ρ(γ={gamma},p={p}) (k={k_value})')

    # Limit (falls gegeben)
    if true_value is not None:
        plt.axhline(y=true_value, color='r', linestyle='--', label='Limit')

    plt.xlabel('n')
    plt.ylabel('Value')
    plt.title(f'Comparison for: {title}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Table output
    print("\n" + "=" * 120)
    print(f"Comparison for k={k_value}, γ={gamma}, p={p}: {title}")
    print("=" * 120)

    print(f"{'n':<5} {'Original':>12} {'Klass. ρ':>12} {'Mod. ρ(γ,p)':>12}")
    print("-" * 45)

    max_rows = min(N, 15)
    for n in range(max_rows):
        orig_val = sequence[n]

        # Classical ρ value
        if n < len(valid_n_classic):
            classic_val = rho_classic[n, k_value + 1]
        else:
            classic_val = np.nan

        # Modified ρ(γ,p) value
        if n < len(valid_n_modified):
            modified_val = rho_modified[n, k_value + 1]
        else:
            modified_val = np.nan

        # Formatted output
        print(f"{n:<5} {orig_val:12.8f} ", end="")

        if np.isnan(classic_val):
            print(f"{'NaN':>12} ", end="")
        else:
            print(f"{classic_val:12.8f} ", end="")

        if np.isnan(modified_val):
            print(f"{'NaN':>12}")
        else:
            print(f"{modified_val:12.8f}")


# ===================================================================
# CONFIGURATION - SET THE EXAMPLE AND PARAMETERS HERE
# ===================================================================
example = 3  # Change this number (1-5)
k_value = 4  # Desired k for the comparison (typically 2,4,6)
gamma = 3.0  # γ value for the modified algorithm
p = 100.0  # p value for the modified algorithm

if example == 1:
    # Geometric series
    N = 30
    n_vals = np.arange(N)
    sequence = np.cumsum(0.5 ** n_vals)
    true_value = 2.0
    title = "Geometric series ∑(0.5)^k"

elif example == 2:
    # Linear sequence
    N = 30
    n_vals = np.arange(N)
    sequence = (2 + 3 * n_vals) / (1 + n_vals)
    true_value = 3.0
    title = "Sequence sₙ = (2 + 3n)/(1 + n)"

elif example == 3:
    # Alternating harmonic series
    N = 30
    n_vals = np.arange(N)
    sequence = np.cumsum([(-1) ** k / (k + 1) for k in n_vals])
    true_value = np.log(2)
    title = "Alternating harmonic series ∑(-1)^k/(k+1)"

elif example == 4:
    # Harmonic series (divergent)
    N = 30
    n_vals = np.arange(1, N + 1)
    sequence = np.cumsum(1 / n_vals)
    true_value = None
    title = "Harmonic series ∑(1/n)"

elif example == 5:
    # Constant sequence
    N = 30
    sequence = np.ones(N) * 5.0
    true_value = 5.0
    title = "Constant sequence (5.0)"

else:
    raise ValueError("Invalid example. Please choose 1, 2, 3, 4, or 5.")

# Run the comparison
compare_algorithms(sequence, true_value, title, k_value, gamma, p)