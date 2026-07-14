import numpy as np
import matplotlib.pyplot as plt


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
            if np.abs(diff) < 1e-20:
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

    # Build the column headers
    headers = [f"ρ-k={k}" for k in range(-1, Kmax + 1)]
    header = "n      " + "".join([f"{h:^14}" for h in headers])
    separator = "-" * len(header)

    print(header)
    print(separator)

    # Print the data rows
    for n in range(min(N, max_rows)):
        row_str = ""
        for j in range(cols):
            val = rho_table[n, j]
            row_str += f"{format_value(val):>14}"
        print(f"{n:<6} {row_str}")

    if N > max_rows:
        print(f"\n... (showing only the first {max_rows} of {N} rows)")


def analyze_divergent_sequence(sequence, title, Kmax=6):
    N = len(sequence)
    rho_table = wynn_rho(sequence, Kmax)

    # Table output
    print_rho_table(rho_table, title)

    # Valid indices for different k values
    valid_n_k2 = np.arange(N - 2)  # For k=2
    valid_n_k4 = np.arange(N - 4)  # For k=4
    valid_n_k6 = np.arange(N - 6)  # For k=6

    # Plot of the sequence and the ρ-approximations
    plt.figure(figsize=(12, 8))

    # Divergent sequence
    plt.plot(np.arange(N), sequence, 'm-', linewidth=2, label=title)

    # ρ-algorithm approximations for different k values
    if len(valid_n_k2) > 0:
        plt.plot(valid_n_k2, rho_table[valid_n_k2, 3], 'b-', label='ρ (k=2)')
    if len(valid_n_k4) > 0:
        plt.plot(valid_n_k4, rho_table[valid_n_k4, 5], 'g-', label='ρ (k=4)')
    if len(valid_n_k6) > 0:
        plt.plot(valid_n_k6, rho_table[valid_n_k6, 7], 'r-', label='ρ (k=6)')

    plt.xlabel('n')
    plt.ylabel('Value')
    plt.title(f'ρ-algorithm on: {title}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot of the differences between successive approximations
    plt.figure(figsize=(12, 8))

    if len(valid_n_k2) > 1:
        diff_k2 = np.abs(rho_table[1:len(valid_n_k2) + 1, 3] - rho_table[:len(valid_n_k2), 3])
        plt.semilogy(valid_n_k2[1:], diff_k2, 'b-', label='Differenz k=2')

    if len(valid_n_k4) > 1:
        diff_k4 = np.abs(rho_table[1:len(valid_n_k4) + 1, 5] - rho_table[:len(valid_n_k4), 5])
        plt.semilogy(valid_n_k4[1:], diff_k4, 'g-', label='Differenz k=4')

    if len(valid_n_k6) > 1:
        diff_k6 = np.abs(rho_table[1:len(valid_n_k6) + 1, 7] - rho_table[:len(valid_n_k6), 7])
        plt.semilogy(valid_n_k6[1:], diff_k6, 'r-', label='Differenz k=6')

    plt.xlabel('n')
    plt.ylabel('|ρₙ₊₁ - ρₙ| (log)')
    plt.title(f'Stability analysis: {title}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot of the absolute values for high k values
    plt.figure(figsize=(12, 8))

    if len(valid_n_k6) > 0:
        plt.plot(valid_n_k6, np.abs(rho_table[valid_n_k6, 7]), 'ro-', label='|ρ(k=6)|')

    if len(valid_n_k4) > 0:
        plt.plot(valid_n_k4, np.abs(rho_table[valid_n_k4, 5]), 'go-', label='|ρ(k=4)|')

    plt.xlabel('n')
    plt.ylabel('Absolute value')
    plt.title(f'Absolute values of the approximations: {title}')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.tight_layout()
    plt.show()


# ===================================================================
# SELECT THE DESIRED EXAMPLE HERE
# ===================================================================
example_number = 7  # Change this number between 1 and 6

# Common parameters
N = 30  # Number of terms
n_vals = np.arange(N)

if example_number == 1:
    # Harmonic series (diverges logarithmically)
    title = "Harmonic series ∑(1/n)"
    sequence = np.cumsum(1 / (n_vals[1:] + 1))  # Starts at n=1

elif example_number == 2:
    # Sum of square roots (diverges faster)
    title = "∑√n"
    sequence = np.cumsum(np.sqrt(n_vals[1:] + 1))  # Starts at n=1

elif example_number == 3:
    # Alternating divergent series (diverges)
    title = "Alternating series (-1)^n * n"
    sequence = np.array([(-1) ** n * n for n in n_vals])

elif example_number == 4:
    # Exponentially growing sequence (diverges)
    title = "Exponential growth (1.5^n)"
    sequence = 1.5 ** n_vals

elif example_number == 5:
    # Linear sequence (diverges)
    title = "Linear sequence n"
    sequence = n_vals

elif example_number == 6:
    # Alternating harmonic series (converges)
    title = "Alternating harmonic series ∑(-1)^n/n"
    sequence = np.cumsum([(-1) ** k / (k + 1) for k in n_vals])

elif example_number == 7:
    # Oscillating sequence (-1)^n (diverges)
    title = "Oscillating sequence (-1)^n"
    sequence = (-1)**n_vals

else:
    raise ValueError("Invalid example number. Please choose a number between 1 and 6.")

# Run the analysis
print("\n" + "=" * 100)
print(f"Example {example_number}: {title}")
print("=" * 100)
analyze_divergent_sequence(sequence, title)