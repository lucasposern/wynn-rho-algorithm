import numpy as np
import matplotlib.pyplot as plt
from math import isfinite, log


def wynn_rho(S, Kmax):
    """Classical Wynn ρ-algorithm"""
    N = len(S)
    rho = np.full((N, Kmax + 2), np.nan)
    rho[:, 0] = 0.0  # k = -1
    rho[:, 1] = S  # k = 0

    for k in range(1, Kmax + 1):
        j = k + 1
        for n in range(N - k):
            diff = rho[n + 1, j - 1] - rho[n, j - 1]
            if np.abs(diff) < 1e-20 or not isfinite(diff):
                rho[n, j] = np.nan
            else:
                rho[n, j] = rho[n + 1, j - 2] + k / diff
    return rho


def format_value(val):
    """Format numeric values for table output"""
    if np.isnan(val):
        return "      NaN"
    return f"{val:10.8f}"


def print_rho_table(rho_table, title, max_rows=20):
    N, cols = rho_table.shape
    Kmax = cols - 2

    print("\n" + "=" * 100)
    print(f"Results for: {title}")
    print("=" * 100)

    headers = [f"ρ-k={k}" for k in range(-1, Kmax + 1)]
    header = "n      " + "".join([f"{h:^14}" for h in headers])
    separator = "-" * len(header)

    print(header)
    print(separator)

    for n in range(min(N, max_rows)):
        row_str = ""
        for j in range(cols):
            val = rho_table[n, j]
            row_str += f"{format_value(val):>14}"
        print(f"{n:<6} {row_str}")

    if N > max_rows:
        print(f"\n... (showing only the first {max_rows} of {N} rows)")


def analyze_divergent_sequence(sequence, title, Kmax=6, limit=None):
    N = len(sequence)
    rho_table = wynn_rho(sequence, Kmax)

    # Table output for the first 20 rows
    print_rho_table(rho_table, title)

    # Plot mit Limit
    plt.figure(figsize=(12, 8))
    plt.plot(np.arange(N), sequence, 'm-', linewidth=2, label=title)

    # ρ-algorithm approximations
    for k, color in zip([2, 4, 6], ['b', 'g', 'r']):
        valid_n = np.arange(N - k)
        if len(valid_n) > 0:
            plt.plot(valid_n, rho_table[valid_n, k + 1], color + '-',
                     label=f'ρ (k={k})')

    # Limit anzeigen
    if limit is not None:
        plt.axhline(y=limit, color='k', linestyle='--',
                    label=f'Limit = {limit:.8f}')

    plt.xlabel('n')
    plt.ylabel('Value')
    plt.title(f'ρ-algorithm on: {title}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Prime generation
def generate_primes(n):
    sieve = np.ones(n + 1, dtype=bool)
    sieve[0] = sieve[1] = False
    for i in range(2, int(np.sqrt(n)) + 1):
        if sieve[i]:
            sieve[i * i::i] = False
    return np.flatnonzero(sieve)


# SELECT THE DESIRED EXAMPLE HERE (1-4)
example_number = 2 # Change this number between 1 and 4

# Common parameters
N = 50  # Number of terms

if example_number == 1:
    # Harmonic series (diverges logarithmically)
    title = "Harmonic series H_n"
    sequence = np.cumsum(1 / np.arange(1, N + 1))
    limit = None  # Diverges, no limit

elif example_number == 2:
    # Euler-Mascheroni constant
    title = "H_n - ln(n) = ∑ 1/k - ln(n) → γ (Euler-Mascheroni)"
    n_vals = np.arange(1, N + 1)
    H_n = np.cumsum(1 / n_vals)
    sequence = H_n - np.log(n_vals)
    limit = 0.5772156649  # Euler-Mascheroni constant


elif example_number == 3:
    # Meissel-Mertens constant
    title = "Sum_{p prim} 1/p - ln(ln(n)) → M"
    primes = generate_primes(1000000)[:N]  # First N primes
    prime_reciprocals = 1 / primes
    cumulative_sum = np.cumsum(prime_reciprocals)
    sequence = cumulative_sum - np.log(np.log(primes))
    limit = 0.2614972128  # Meissel-Mertens constant

elif example_number == 4:
    # Slowly converging series
    title = "Sum_{k=2}^n 1/(k·ln(k)·ln(ln(k)))"
    k_vals = np.arange(2, N + 2)
    terms = 1 / (k_vals * np.log(k_vals) * np.log(np.log(k_vals)))
    sequence = np.cumsum(terms)
    limit = None  # Known limit is non-trivial

else:
    raise ValueError("Invalid example number. Please choose 1-4.")

# Run the analysis
print("\n" + "=" * 100)
print(f"Example {example_number}: {title}")
print("=" * 100)
analyze_divergent_sequence(sequence, title, limit=limit)