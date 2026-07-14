# Wynn's ρ-Algorithm

Seminar project on convergence acceleration (B.Sc. Mathematics, TU Dresden).

Implementation and analysis of Wynn's ρ-algorithm for accelerating the convergence of numerical
sequences and series. The write-up derives the algorithm, proves when it accelerates a sequence
and when it cannot, and compares it against Aitken's Δ² method and Wynn's ε-algorithm on linear,
alternating, logarithmic and slowly convergent test sequences. Two modifications, the ρ(γ) and
ρ(γ,p) algorithms, extend it to cases where the classical version stalls.

## Write-up

The paper exists in two languages, LaTeX sources and compiled PDFs alike:

- Deutsch — [`docs/wynn_rho_de.pdf`](docs/wynn_rho_de.pdf)
- English — [`docs/wynn_rho_en.pdf`](docs/wynn_rho_en.pdf)

Both versions share the figures in `docs/figures/` and the bibliography `docs/quellen.bib`.
The figures come from the scripts in `examples/`.

## Results in short

- The ρ-algorithm accelerates a sequence `s_n ~ s + n^θ(c₀ + c₁/n + …)` exactly when θ is a
  negative integer.
- It cannot accelerate linear convergence; Aitken's Δ² is the right tool there.
- For alternating series such as the Leibniz series, the ε-algorithm wins.
- Its strength is logarithmic convergence. The Euler–Mascheroni constant comes out exact to eight
  digits at recursion step k=6, depth n=2.
- ζ(1.1): the classical algorithm still carries an error above 5 at depth n=23, while ρ(γ=0.9)
  reaches ≈ 10⁻⁸. Choose γ carefully, since small deviations destroy the acceleration.

## Examples

```bash
python examples/1_uneven-iteartionen.py   # uneven iteration counts
python examples/2_linear.py               # linear convergence: ρ vs Aitken Δ²
python examples/3_alternating.py          # alternating series: ρ vs ε
python examples/4_logarithmic.py          # logarithmic convergence
python examples/5_slow_zeta.py            # slowly convergent zeta series
python examples/6__divergent.py           # divergent sequences
python examples/7_modification1.py        # ρ(γ)
python examples/8_modification2.py        # ρ(γ,p)
python examples/9_modi1_vs_modi_2.py      # comparison of both modifications
```

Requires `numpy` and `matplotlib`.

## Overleaf

The `docs/` sources are synced with the Overleaf project of the same name, which compiles both
documents.
