<!-- Index of architecture decision records. -->
# Architecture decision records

One file per decision, `NNNN-title.md`, with Context / Decision / Consequences.

| # | Decision |
|---|----------|
| [0001](0001-torch-wheels-and-accelerators.md) | Torch wheels and accelerators: PyPI torch by default, conflicting `cpu` / `cu126` extras, torch 2.14 pin, sync discipline |
| [0002](0002-nt-xent-denominator.md) | NT-Xent denominator: the legacy `1 + Σ_neg exp` (close to DCL) by default, the paper's formula behind `nt_xent_legacy_denominator: false` |
| [0003](0003-carrasco2021-preset.md) | The `carrasco2021` preset: ALeRCE's code over the paper text (frozen feature BN, inactive dropout, Glorot fc biases), Adam (0.9, 0.999) with torch's ε placement, the without-features assumption, the NaN approximation, the leaked stamp out of Train only |
| [0004](0004-evaluation-protocol.md) | Evaluation protocol: stored splits (the file is the truth), seed-mean stratified paired bootstrap, exact McNemar, the leaked stamp removed from Train only, CD21 on exactly 5 seeds |
| [0005](0005-paper-model-variants.md) | The paper model's 2.5 architecture variants as orthogonal flags (`paper_arch`, `group_fields`, `context`, `rot90_augment`), faithful defaults listed in the presets, Glorot fans of group layers (the readout as a dense layer), the context normalised like the crop, hand-written p4/p4m layers |
| [0006](0006-class-aware-augmentation.md) | Class-aware training augmentation (plan C5, C6): the difference computed from the science and template through a per-stamp fit, noise scaled to each crop's background sigma then the min-max again, discrete features never jittered, synthetic hot-pixel Bogus from the row's own stamp with a batch Bogus donor, spikes narrower than the measured PSF minimum (1.19 px), per-augmentation step-seeded generators |
