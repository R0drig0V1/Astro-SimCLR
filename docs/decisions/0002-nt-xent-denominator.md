<!-- ADR 0002: which denominator NT_Xent uses by default. -->
# 0002. NT-Xent denominator

- **Status:** accepted (2026-11-07, phase 1, task 1.3; decided by the user)
- **Checked by:** `tests/test_losses.py` (both formulas against an independent reference;
  the default against the legacy fixtures)

## Context

SimCLR (Chen et al. 2020) defines, for a view `a` with positive `p` and the other
2(B − 1) views of the batch as negatives, with `s(a, b) = cos(z_a, z_b) / T`:

```
paper:   loss_a = −s(a, p) + log( exp(s(a, p)) + Σ_neg exp(s(a, n)) )
```

The legacy `NT_Xent` (`losses/losses.py:121-124`) builds the cross-entropy logits as
`[0, negatives...]` and subtracts the positives afterwards, so the positive term in the
denominator is `exp(0) = 1`:

```
legacy:  loss_a = −s(a, p) + log( 1 + Σ_neg exp(s(a, n)) )
```

Every SimCLR number from the legacy code (the 2022 confusion-matrix figures, any future
reference run R3) was trained with the legacy formula. Changing it changes the optimisation
problem, not only the logged value.

The legacy variant is close to the **decoupled contrastive loss** (DCL, Yeh et al. 2021,
arXiv:2110.06848), which removes the positive from the denominator:
`−s(a, p) + log Σ_neg exp(s(a, n))`. The legacy formula only adds the constant 1 inside the
log, which is negligible next to the 2(B − 1) negative terms at the batch sizes used
(150–550). Like DCL, its gradient on the positive is the full `−∇s(a, p)`, not the
`−(1 − softmax_p)·∇s(a, p)` of the paper, so it keeps pulling positives together even when
they already dominate the softmax.

## Decision

1. `astro_simclr.losses.NT_Xent` takes `legacy_denominator: bool = True`. **The legacy
   formula is the default**, so ported runs stay comparable with the legacy results.
2. `legacy_denominator=False` gives the paper's formula.
3. The flag is the run setting `nt_xent_legacy_denominator` (`config.Settings`, default
   `true`), so a config file or `--set nt_xent_legacy_denominator=false` selects it, and the
   resolved settings of a run record which formula was used.
4. The return values stay the legacy ones: `(loss, pos, neg)`, with `pos` and `neg` the mean
   cosine similarities of the positive and negative pairs.

### Rejected

- **Paper formula as the default:** correct by the paper, but it silently breaks the
  comparison with every legacy SimCLR result, and no reference run with it exists yet.
- **Keep only the legacy formula:** the paper's formula is the standard one, and comparing
  the two is a natural experiment; a flag costs one `masked_fill`.
- **A constructor argument only, not a setting:** the choice would then live in code, and
  the run's recorded settings would not say which loss was trained.

## Consequences

- Loss values differ between the two modes and are not comparable: the legacy loss can be
  negative (with B = 1 it is `−s(a, p)`), the paper's is always ≥ 0 (with B = 1 it is 0).
- Results trained with `nt_xent_legacy_denominator: false` are a new experiment and need
  their own reference.
- Calling the legacy variant "SimCLR" in reports is slightly inaccurate; reports should name
  the formula (or cite DCL) when it matters.
