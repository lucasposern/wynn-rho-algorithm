import numpy as np
import matplotlib.pyplot as plt
from scipy.special import zeta  # For exact zeta function values


def wynn_rho(S, Kmax):
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


def wynn_epsilon(S, Kmax):
    N = len(S)
    eps = np.full((N, Kmax + 2), np.nan)
    eps[:, 0] = 0.0  # k = -1
    eps[:, 1] = S  # k = 0

    for k in range(1, Kmax + 1):
        j = k + 1
        for n in range(N - k):
            diff = eps[n + 1, j - 1] - eps[n, j - 1]
            if np.abs(diff) < 1e-20:
                eps[n, j] = np.nan
            else:
                eps[n, j] = eps[n + 1, j - 2] + 1 / diff
    return eps


def format_value(val):
    """Format numeric values for table output"""
    if np.isnan(val):
        return "      NaN"
    return f"{val:10.8f}"


def print_combined_tables(rho_table, eps_table, title, max_rows=15):
    N, cols = rho_table.shape
    Kmax = cols - 2

    print("\n" + "=" * 120)
    print(f"Combined results for: {title}")
    print("=" * 120)

    # Build the column headers
    rho_headers = [f"ρ-k={k}" for k in range(-1, Kmax + 1)]
    eps_headers = [f"ε-k={k}" for k in range(-1, Kmax + 1)]

    # Print the header for both tables
    header = "n      " + "".join([f"{h:^14}" for h in rho_headers]) + "   |   " + "".join(
        [f"{h:^14}" for h in eps_headers])
    separator = "-" * len(header)

    print(header)
    print(separator)

    # Print the data rows
    for n in range(min(N, max_rows)):
        # ρ-algorithm values
        rho_str = ""
        for j in range(cols):
            val = rho_table[n, j]
            rho_str += f"{format_value(val):>14}"

        # ε-algorithm values
        eps_str = ""
        for j in range(cols):
            val = eps_table[n, j]
            eps_str += f"{format_value(val):>14}"

        # Combined row
        print(f"{n:<6} {rho_str}   |   {eps_str}")

    if N > max_rows:
        print(f"\n... (showing only the first {max_rows} of {N} rows)")


def compare_algorithms(sequence, true_value, title):
    N = len(sequence)
    Kmax = 6

    # Compute both algorithms
    rho_table = wynn_rho(sequence, Kmax)
    eps_table = wynn_epsilon(sequence, Kmax)

    # Combined table output
    print_combined_tables(rho_table, eps_table, title)

    # Valid indices for k=6
    valid_n = np.arange(N - Kmax)

    # Plot for both algorithms with the original sequence
    plt.figure(figsize=(12, 8))

    # Add the original sequence
    plt.plot(np.arange(N), sequence, 'm-', linewidth=2, label='Original sequence')

    # Algorithmen
    plt.plot(valid_n, rho_table[valid_n, Kmax + 1], 'b-', label='Wynn-ρ (k=6)')
    plt.plot(valid_n, eps_table[valid_n, Kmax + 1], 'g-', label='Wynn-ε (k=6)')

    # Limit
    plt.axhline(y=true_value, color='r', linestyle='--', label='Limit')

    plt.xlabel('n')
    plt.ylabel('Value')
    plt.title(f'Comparison for: {title}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Custom analysis of the zeta function with a variable exponent
def zeta_analysis(s_values):
    """Analyze the zeta function for several exponents"""
    for s in s_values:
        print("\n" + "=" * 120)
        print(f"Example: Zeta series for ζ({s})")
        print("=" * 120)

        N = 100
        n_vals_zeta = np.arange(1, N + 1)

        # Compute partial sums
        S_zeta = np.cumsum(1 / n_vals_zeta ** s)

        # Compute the reference value
        if s > 1:
            # Exact value for s > 1
            zeta_s = zeta(s, 1)
        else:
            # For s <= 1 we use a long sum as an approximation
            N_ref = 1000000
            zeta_s = np.sum(1 / np.arange(1, N_ref) ** s)

        compare_algorithms(S_zeta, zeta_s, f"Zeta series (s={s})")


# Custom values for the zeta function
exponents = [4, 3, 2, 1.5 ,1.1]  # Any exponents can be entered here

# Run the analysis for all desired exponents
zeta_analysis(exponents)