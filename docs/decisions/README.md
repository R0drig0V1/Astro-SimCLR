<!-- Index of architecture decision records. -->
# Architecture decision records

One file per decision, `NNNN-title.md`, with Context / Decision / Consequences.

| # | Decision |
|---|----------|
| [0001](0001-torch-wheels-and-accelerators.md) | Torch wheels and accelerators: PyPI torch by default, conflicting `cpu` / `cu126` extras, torch 2.14 pin, sync discipline |
| [0002](0002-nt-xent-denominator.md) | NT-Xent denominator: the legacy `1 + Σ_neg exp` (close to DCL) by default, the paper's formula behind `nt_xent_legacy_denominator: false` |
