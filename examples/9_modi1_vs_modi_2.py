import numpy as np
import matplotlib.pyplot as plt
from scipy.special import zeta


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


def wynn_rho_gamma(S, Kmax, gamma):
    """ρ(γ)-algorithm with a γ parameter"""
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
                rho[n, j] = rho[n + 1, j - 2] + (k - gamma) / diff
    return rho


def wynn_rho_gamma_p(S, Kmax, gamma, p):
    """ρ(γ,p)-algorithm with γ and p parameters"""
    N = len(S)
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
        j = k + 2  # target column for ρ^{(n)}_{k+1}
        for n in range(N - k - 1):
            diff = rho[n + 1, k + 1] - rho[n, k + 1]
            if np.abs(diff) < 1e-20:
                rho[n, j] = np.nan
            else:
                rho[n, j] = rho[n + 1, k] + C[k] / diff
    return rho


def compare_algorithms(sequence, true_value, title, k_value, gamma, p):
    """Compare different ρ-algorithms with additional error analysis"""
    N = len(sequence)

    # Compute all algorithms
    rho_classic = wynn_rho_classic(sequence, k_value)
    rho_gamma = wynn_rho_gamma(sequence, k_value, gamma)
    rho_gamma_p = wynn_rho_gamma_p(sequence, k_value, gamma, p)

    # Valid indices for the chosen k
    valid_n_classic = np.arange(N - k_value)
    valid_n_gamma = np.arange(N - k_value)
    valid_n_gamma_p = np.arange(N - k_value - 1)

    # Plot of the values
    plt.figure(figsize=(14, 8))

    # Original sequence
    plt.plot(np.arange(N), sequence, 'm-', linewidth=2, label='Original sequence')

    # Classical ρ-algorithm
    if len(valid_n_classic) > 0:
        plt.plot(valid_n_classic, rho_classic[valid_n_classic, k_value + 1], 'b-',
                 label=f'Classical ρ (k={k_value})')

    # ρ(γ)-algorithm
    if len(valid_n_gamma) > 0:
        plt.plot(valid_n_gamma, rho_gamma[valid_n_gamma, k_value + 1], 'g-',
                 label=f'ρ(γ={gamma}) (k={k_value})')

    # ρ(γ,p)-algorithm
    if len(valid_n_gamma_p) > 0:
        plt.plot(valid_n_gamma_p, rho_gamma_p[valid_n_gamma_p, k_value + 1], 'c-',
                 label=f'ρ(γ={gamma},p={p}) (k={k_value})')

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

    # Table output of the values
    print("\n" + "=" * 120)
    print(f"Values for k={k_value}, γ={gamma}, p={p}: {title}")
    print("=" * 120)

    print(f"{'n':<5} {'Original':>12} {'Klass. ρ':>12} {'ρ(γ)':>12} {'ρ(γ,p)':>12}")
    print("-" * 60)

    max_rows = min(N, 30)
    # Store the values for the error table
    values = []

    for n in range(max_rows):
        orig_val = sequence[n]

        # Classical ρ value
        if n < len(valid_n_classic):
            classic_val = rho_classic[n, k_value + 1]
        else:
            classic_val = np.nan

        # ρ(γ) value
        if n < len(valid_n_gamma):
            gamma_val = rho_gamma[n, k_value + 1]
        else:
            gamma_val = np.nan

        # ρ(γ,p) value
        if n < len(valid_n_gamma_p):
            gamma_p_val = rho_gamma_p[n, k_value + 1]
        else:
            gamma_p_val = np.nan

        # Store the values for the error table
        values.append((orig_val, classic_val, gamma_val, gamma_p_val))

        # Formatted output
        print(f"{n:<5} {orig_val:12.8f} ", end="")

        if np.isnan(classic_val):
            print(f"{'NaN':>12} ", end="")
        else:
            print(f"{classic_val:12.8f} ", end="")

        if np.isnan(gamma_val):
            print(f"{'NaN':>12} ", end="")
        else:
            print(f"{gamma_val:12.8f} ", end="")

        if np.isnan(gamma_p_val):
            print(f"{'NaN':>12}")
        else:
            print(f"{gamma_p_val:12.8f}")

    # Additional error-analysis table (only if the limit is known)
    if true_value is not None:
        print("\n" + "=" * 120)
        print(f"Error deviations from the limit {true_value:.8f}")
        print("=" * 120)

        print(f"{'n':<5} {'Original':>12} {'Klass. ρ':>12} {'ρ(γ)':>12} {'ρ(γ,p)':>12}")
        print("-" * 60)

        for n, (orig_val, classic_val, gamma_val, gamma_p_val) in enumerate(values):
            orig_error = abs(orig_val - true_value) if not np.isnan(orig_val) else np.nan

            # Classical ρ error
            if not np.isnan(classic_val):
                classic_error = abs(classic_val - true_value)
            else:
                classic_error = np.nan

            # ρ(γ) error
            if not np.isnan(gamma_val):
                gamma_error = abs(gamma_val - true_value)
            else:
                gamma_error = np.nan

            # ρ(γ,p) error
            if not np.isnan(gamma_p_val):
                gamma_p_error = abs(gamma_p_val - true_value)
            else:
                gamma_p_error = np.nan

            # Formatted output
            print(f"{n:<5} ", end="")

            if np.isnan(orig_error):
                print(f"{'NaN':>12} ", end="")
            else:
                print(f"{orig_error:12.8f} ", end="")

            if np.isnan(classic_error):
                print(f"{'NaN':>12} ", end="")
            else:
                print(f"{classic_error:12.8f} ", end="")

            if np.isnan(gamma_error):
                print(f"{'NaN':>12} ", end="")
            else:
                print(f"{gamma_error:12.8f} ", end="")

            if np.isnan(gamma_p_error):
                print(f"{'NaN':>12}")
            else:
                print(f"{gamma_p_error:12.8f}")


# ===================================================================
# CONFIGURATION - EXAMPLES AND PARAMETERS
# ===================================================================
example = 6  # Example number: 3 old, 6 zeta
k_value = 4  # Desired k for the comparison
gamma = 0.9 # γ value for both !!! 0.9 !!!
p = 10.0  # p value for the ρ(γ,p)-algorithm

# Parameters for the zeta function (relevant only for example 6)
s_value = 1.1  # s > 1

if example == 1:
    # Geometric series
    N = 30
    n_vals = np.arange(N)
    sequence = np.cumsum(0.5 ** n_vals)
    true_value = 2.0
    title = "Geometric series ∑(0.5)^k"

elif example == 2:
    # Rational sequence
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
    true_value = None  # No limit, hence no error table
    title = "Harmonic series ∑(1/n)"

elif example == 5:
    # Constant sequence
    N = 30
    sequence = np.ones(N) * 5.0
    true_value = 5.0
    title = "Constant sequence (5.0)"

elif example == 6:
    # Riemann zeta function
    N = 30
    n_vals = np.arange(1, N + 1)

    # Compute the partial sums of the zeta function
    sequence = np.cumsum(1 / (n_vals ** s_value))

    # Compute the true limit with scipy (for s > 1)
    true_value = zeta(s_value, 1)
    title = f"Riemann zeta function ζ({s_value})"

else:
    raise ValueError("Invalid example. Please choose 1, 2, 3, 4, 5, or 6.")

# Run the comparison
compare_algorithms(sequence, true_value, title, k_value, gamma, p)