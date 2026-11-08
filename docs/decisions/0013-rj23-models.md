<!-- ADR 0013: the details of RJ23's classifiers that their missing models.py loses (task
     3.4), each with its source, the TF-default parity choices, and the pre-registered
     sensitivity arm. -->
# 0013: RJ23's classifiers: 'same' convolutions, TF-'same' max-pooling, flatten then average, dropout → concat → BN → FC → FC, ReLU, Keras defaults

- **Status:** accepted (2026-11-08, phase 3, task 3.4)
- **Implemented by:** `src/astro_simclr/rj23_model.py`
- **Checked by:** `tests/test_rj23_model.py`

## Context

Reyes-Jainaga et al. 2023 (RJ23, ApJL 952, L43, arXiv:2304.13080) train one classifier per
input scenario (full 63², cropped 16², low-resolution 16², multi-scale 8×8×24; the views of
ADR 0009). The paper gives Table 2 (the convolution-pooling stacks, the dropout and dense
sizes, the learning rates) and Appendix A (the 8 rotation and flip views averaged, the
position concatenated after a dropout, two dense layers, batch normalisation "after the
source position is given to the network", 8×8 feature maps at the end of every stack).
RJ23's code is public (MIT, `github.com/alercebroker/multiscale_stamps`), but every
script imports a `models.py` that is not in the repository, so the following are lost:
the convolution padding, the pooling padding, where the maps are flattened relative to the
8-view mean, the exact head order and BN placement, and the activation. Their training
script fixes `with_batchnorm = True` and `first_kernel_size = 3`. Carvajal et al. 2025
(C25, A&A 697, A119, §2.2) describe the same stamp block ("mirrors the structural design
from RJ23") in more detail. The task 3.0 research marks each of these points verified,
inferred or unknown; we reimplement, nothing is copied.

RJ23 trained with `tf.keras` defaults (Adam with `beta_1=0.9, beta_2=0.999`; the layers'
default initialisers and BN settings); torch's defaults differ on each.

## Decision

The architecture, each choice with its source:

1. **'same' convolution padding**, 3×3 kernels with stride 1 (Table 2's `conv3-*`). Source:
   RJ23 App. A, "8x8 feature maps at the end of the convolutional - pooling stage in all
   four scenarios". With valid convolutions the full stack (four convolutions, three
   poolings) would end on 4×4 or 5×5 and the multi-scale stack (four convolutions on 8×8,
   no pooling) on 0×0; only 'same' gives 8×8 everywhere. For an odd kernel TF's and torch's
   'same' paddings coincide (one pixel on each side).
2. **Ceil 'same' max-pooling**, 2×2 with stride 2, TF's padding: 63 → 32 → 16 → 8 in the
   full scenario, 16 → 8 in the cropped and low-resolution ones. Source: the same 8×8
   statement (a floor pooling gives 63 → 31 → 15 → 7); `MaxPool2D(padding='same')` is
   also what RJ23's code uses for its average poolings (ADR 0009). torch has no 'same'
   max-pooling, so `TFSameMaxPool2d` pads as TF (`total = (out − 1)·s + k − n`, half
   before, the rest after) with −∞, so a partial window takes the maximum of its valid
   pixels, as TF does.
3. **Flatten, then the mean over the 8 views.** Source: C25 §2.2, "After the convolutional
   layers, the images are flattened, averaged over the augmentation steps, and passed
   through a dropout layer"; RJ23 App. A, "the 8 representations are averaged". The 8 views
   are the 4 counter-clockwise rotations (`tf.image.rot90`) and the up-down flip of each;
   they are part of the model (train and test), not an augmentation. Flattening before or
   after the mean is the same function (the mean is linear and acts per position); the
   order matters only for the maps' layout, and torch's (c, h, w) flatten against TF's
   (h, w, c) is a fixed permutation of the dense layer's inputs, irrelevant without TF
   weights to load.
4. **Head: dropout → concat(x, y, z) → BN → FC → ReLU → FC-K.** Sources: RJ23 App. A, "a
   dropout operation is done and the position metadata is concatenated ... passes through
   two dense (fully-connected) layers" and "batch normalization after the source position
   is given to the network"; C25 §2.2's dropout before the concatenation; RJ23's
   `with_batchnorm = True`. The BN normalises the concatenated `64·c_L + 3` features; the
   position is RJ23's unit-sphere `(x, y, z)` (`rj23_input.unit_sphere`). The first FC has
   Table 2's size, the second `K` outputs: 5 for R5 (RJ23's six classes without the
   satellites), configurable. Logits, no softmax (RJ23's loss is
   `SparseCategoricalCrossentropy(from_logits=True)`).
5. **ReLU** after every convolution and after the first FC. Source: inferred from the
   lineage, RJ23 App. A "we took the Convolutional Network from Carrasco-Davis et al. and
   made small adjustments", whose network uses ReLU throughout (ADR 0003); RJ23 name no
   other activation.
6. **Table 2 as published** (`RJ23_TABLE2`, with the citation in code): filters, dense
   size, dropout and learning rate per scenario, the first kernel 3×3. No hyperparameter
   search (plan, task 3.4).

TF-default parity, each with a hand-computed golden test and no TF dependency:

7. **Keras BatchNormalization**: momentum 0.99 (torch's `momentum=0.01`), epsilon 1e-3,
   gamma 1, beta 0, moving statistics (0, 1); and the moving variance updated with the
   **biased** batch variance, which is what Keras' non-fused path (every 2-D input) does
   (`tf.nn.moments`), where torch's `BatchNorm1d` uses the unbiased one (a factor
   256/255 at RJ23's batch size). `KerasBatchNorm1d` keeps torch's forward and gradient,
   and replaces only the buffer update.
8. **Glorot-uniform kernels and zero biases** on every convolution and dense layer (Keras'
   `glorot_uniform` and `zeros`; Keras' fans and `xavier_uniform_`'s agree on 4-D
   kernels). The weights come from the global torch generator, so a seed fixes them.
9. **TF 'same' max-pooling** (point 2).

**Pre-registered sensitivity arm:** `bn_position="before_concat"`, the BN on the image
representation only, before the position is concatenated (the position then reaches the
first FC unnormalised). Run on the multi-scale scenario with 5 seeds, decided on the
decision half of the validation set (plan, task 3.4). The default, `after_concat`, is the
primary model; the arm is the only architectural alternative to be run.

## Consequences

- Parameter counts with 5 classes (`rj23_parameter_count`, formula in the module
  docstring, tested against the built models): full 99,778; cropped 1,202,808; low-res
  86,648; multi-scale 179,124. The sensitivity arm has 6 fewer.
- The averaged representation is exactly invariant to any rotation or flip *of the view*
  (they permute the 8 views). Of the *raw stamp* it is invariant only where the view
  commutes with the transform: every element for the full view, the transpose for every
  view. Elsewhere the crops shift by one pixel (centre 31.5, ADR 0009) and the pooled grids
  pad at the end, so a rotated stamp is a slightly different input; the tests pin both.
- The modules are engine-free (`nn.Module` builders, no Lightning; the phase-4 rule). Task
  3.4b wires them into a training engine and owns the training regime: Table 2's learning
  rates (recorded here), **Adam with Keras' epsilon 1e-7** (torch's default is 1e-8), the
  per-sample class-uniform sampler and the `rj23` schedule.
- If the sensitivity arm moves the decision-half macro-F1 beyond its noise, the report
  names it; the primary model does not change after the fact.
- Everything here is a documented choice, not RJ23's code: if ALeRCE shares `models.py`
  (task 3.0a), any difference becomes a deviation to record, and moving the defaults to it
  is a numerics change with its own ADR amendment.
