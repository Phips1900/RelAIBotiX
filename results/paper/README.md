# Frozen publication results

These tables contain the 31-experiment result set produced on 2026-09-14 with the
`additive_normalized` exposure model and the component probabilities in
`configs/robots`.

- Internal DTMC solver: enabled
- PRISM 4.10.1: exact verification passed for all 31 experiments
- Storm 1.14.0: exact verification passed for all 31 experiments
- Sensitivity perturbation: 10x, one component at a time
- Input integrity: SHA-256 values are stored in `configs/experiments/paper.json`

The CSV is the machine-readable authoritative table. The Markdown and LaTeX files
are renderings of the same rows. Full generated model files and HDF5 data are
release assets rather than Git source files.

## Supplementary sorting variations

[Real-Franka sorting with changed objects and switched boxes](sorting_variations/README.md)
contains five additional comparisons with the standard sorting baselines, including
run-selection rules, numerical results, skill contributions, and solver verification.
The supplementary analysis distinguishes task success from modeled hardware reliability
and is separate from the frozen 31-experiment table above.
