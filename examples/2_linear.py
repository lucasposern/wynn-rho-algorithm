import numpy as np
import matplotlib.pyplot as plt


def wynn_rho(S, Kmax):
    """Compute Wynn's ρ-algorithm"""
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


def aitken_delta2(S):
    """Compute Aitken's δ²-algorithm"""
    N = len(S)
    aitken = np.full(N, np.nan)

    for n in range(N - 2):
        numerator = (S[n + 1] - S[n]) ** 2
        denominator = S[n + 2] - 2 * S[n + 1] + S[n]

        if np.abs(denominator) < 1e-12:
            aitken[n] = np.nan
        else:
            aitken[n] = S[n] - numerator / denominator

    return aitken


def format_value(val):
    """Format numeric values for table output"""
    if np.isnan(val):
        return "      NaN"
    return f"{val:10.8f}"


def print_combined_tables(rho_table, aitken, title, max_rows=15):
    N, cols = rho_table.shape
    Kmax = cols - 2

    print("\n" + "=" * 100)
    print(f"Comparison of ρ-algorithm and Aitken δ²-algorithm for: {title}")
    print("=" * 100)

    # Build the column headers
    rho_headers = [f"ρ-k={k}" for k in range(-1, Kmax + 1)]
    headers = rho_headers + ["Aitken δ²"]

    # Print the header
    header = "n      " + "".join([f"{h:^14}" for h in headers])
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

        # Aitken δ² value
        aitken_val = aitken[n] if n < len(aitken) else np.nan
        aitken_str = f"{format_value(aitken_val):>14}"

        print(f"{n:<6} {rho_str} {aitken_str}")

    if N > max_rows:
        print(f"\n... (showing only the first {max_rows} of {N} rows)")


def compare_algorithms(sequence, true_value, title):
    N = len(sequence)
    Kmax = 6

    # Compute both algorithms
    rho_table = wynn_rho(sequence, Kmax)
    aitken = aitken_delta2(sequence)

    # Table output
    print_combined_tables(rho_table, aitken, title)

    # Valid indices for k=2 (ρ) and Aitken
    valid_n_rho2 = np.arange(N - 2)  # For k=2
    valid_n_aitken = np.arange(N - 2)  # Aitken needs 3 values

    # Plot of the algorithms and the original sequence
    plt.figure(figsize=(12, 8))

    # Original sequence
    plt.plot(np.arange(N), sequence, 'm-', linewidth=2, label='Original sequence')

    # ρ-algorithm (k=2)
    plt.plot(valid_n_rho2, rho_table[valid_n_rho2, 3], 'b-', label='Wynn-ρ (k=2)')

    # Aitken δ²-algorithm
    plt.plot(valid_n_aitken, aitken[valid_n_aitken], 'g-', label='Aitken δ²')

    # Limit
    plt.axhline(y=true_value, color='r', linestyle='--', label='Limit')

    plt.xlabel('n')
    plt.ylabel('Value')
    plt.title(f'Comparison for: {title}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Error plot
    plt.figure(figsize=(12, 8))

    # Error of the original sequence
    error_sequence = np.abs(sequence - true_value)
    plt.semilogy(np.arange(N), error_sequence, 'm-', label='Error of the original sequence')

    # Errors of the algorithms
    error_rho2 = np.abs(rho_table[valid_n_rho2, 3] - true_value)
    plt.semilogy(valid_n_rho2, error_rho2, 'b-', label='Wynn-ρ (k=2)')

    error_aitken = np.abs(aitken[valid_n_aitken] - true_value)
    plt.semilogy(valid_n_aitken, error_aitken, 'g-', label='Aitken δ²')

    plt.xlabel('n')
    plt.ylabel('Absolute error (log)')
    plt.title(f'Error comparison for: {title}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Example 1: Linearly convergent sequence s_n = 3 - (0.9)^n
print("\n" + "=" * 100)
print("Example: Linearly convergent sequence s_n = 3 - (0.9)^n")
print("=" * 100)
N = 30
n_vals = np.arange(N)
sequence_linear = 3 - (0.9) ** n_vals
true_value_linear = 3.0
compare_algorithms(sequence_linear, true_value_linear, "Linearly convergent sequence: s_n = 3 - (0.9)^n")

'''
# Example 2: Geometric series
print("\n" + "=" * 100)
print("Example: Geometric series (converges to 2)")
print("=" * 100)
N_geo = 30
n_geo = np.arange(N_geo)
S_geo = np.cumsum(0.5 ** n_geo)  # Sum of (1/2)^k for k=0 to n
true_value_geo = 2.0
compare_algorithms(S_geo, true_value_geo, "Geometric series: ∑(0.5)^k")
'''
'''
# Example 3: Slowly convergent series
print("\n" + "=" * 100)
print("Example: Slowly convergent series s_n = ∑ 1/k(k+1) (converges to 1)")
print("=" * 100)
N_slow = 30
n_slow = np.arange(1, N_slow + 1)
# Partial sums of the series ∑_{k=1}^n 1/(k(k+1)) = 1 - 1/(n+1)
S_slow = np.cumsum(1 / (n_slow * (n_slow + 1)))
true_value_slow = 1.0
compare_algorithms(S_slow, true_value_slow, "Slowly convergent series: ∑ 1/(k(k+1))")
'''