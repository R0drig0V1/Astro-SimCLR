<!-- ADR 0022: ConvNeXt-atto / pico at the native 63 px as paper_arch options (plan row B8), timm as a dependency, and the pinned Zoobot weights. -->
# 0022: ConvNeXt-atto / pico on the whole stamp, timm, and the pinned Zoobot weights

- **Status:** accepted (2026-11-08, phase 2, task 2.5, plan row B8)
- **Implemented by:** `src/astro_simclr/convnext_model.py`, `paper_variants.build_paper_model`
  / `paper_input_side` / `count_macs` / `VARIANTS`, `Settings.paper_arch` (`convnext_atto`,
  `convnext_pico`) and `Settings.backbone_weights` (`config.py`),
  `training.device_data` (`side=63`), `experiments._check_schedule` /
  `_load_backbone_weights`, `training.modules.PaperClassifierModule`, `presets`,
  `runs.canonical.HASH_NEUTRAL_DEFAULTS`, `pyproject.toml` / `uv.lock` (timm)
- **Checked by:** `tests/test_convnext_model.py`, `tests/test_paper_variants.py` (golden
  fixture), `tests/test_split_runs.py`, `tests/test_runs_canonical.py`,
  `scripts/check-torch-wheels.sh`

## Context

Plan row B8 (research of 2026-11-08): the ZTF groups that replaced custom CNNs deploy
ConvNeXt-pico, with Galaxy-Zoo (Zoobot) weights (Rehemtulla et al. 2025, arXiv:2512.11957;
ORACLE-2, arXiv:2607.00228). It is the modern-CNN control that any transformer row must
beat. Those groups upsample the 63 px stamp to 224 px (about 1.4 G MACs per view, 26 times
the paper model). The row asks for the native resolution instead, padded to 64, with the
true 32-16-8-4 stage maps, the paper's 4-rotation averaging and its head, from scratch and
from the Zoobot weights. ConvNeXt is not in torch or torchvision at these sizes (atto and
pico are timm's), and the Zoobot weights are timm state dicts.

## Decision

1. **timm is a runtime dependency** (`timm>=1.0.31,<2`, locked at 1.0.31). It is the
   reference ConvNeXt implementation, the format of the Zoobot weights, and the home of the
   B9 hybrids (CoAtNet, MaxViT) the plan lists next. It depends on torch and torchvision
   without pinning them, so uv resolves it against the torch of each ADR 0001 fork:
   `scripts/check-torch-wheels.sh` passes unchanged (no second torch, no nvidia wheel in
   the CPU fork). It pulls `huggingface-hub` and `safetensors` (and their httpx stack).
   timm is imported only when a ConvNeXt is built, so the other subcommands do not pay its
   import time. Writing ConvNeXt by hand was rejected: about 200 lines that would have to
   match timm's key names and initialisation exactly to load the weights, with no gain.
2. **The whole stamp, a new input side.** The ConvNeXt `paper_arch` values take
   (N, 3, 63, 63): the paper's float path at the full side (`crop_split(images, 63)`: the
   same per-channel min-max without epsilon, NaN approximation and NaN -> 0, on all the
   stamp's pixels). `image_size` must be 63 for them (21 for the others), so the recorded
   config says what the model saw. The model zero-pads to 64 at the bottom and the right,
   after rotating. The device-resident splits grow 9 times (about 2.4 GB for `red`): fine
   on the 5090, tight next to the model on the 4 GB GTX 1050.
3. **Stage maps 32-16-8-4 with an overlapping stride-2 stem.** timm's patch stem is 4x4 at
   stride 4 (16-8-4-2 at 64 px). `patch_size=2` alone does not change the maps: timm then
   adds a stride-2 downsample to stage 0 (32 after the stem, 16 after stage 0). The research
   proposed `patch_size=2` plus replacing `stages[0].downsample` with `Identity`; that gives
   a 2x2 stem kernel, which the Zoobot weights (a 4x4 stem kernel) do not fit. Instead the
   backbone keeps timm's default layout (4x4 stem kernel, no stage-0 downsample) and its
   stem convolution runs at stride 2 with padding 1: 64 -> 32, then 32, 16, 8, 4. Scratch
   and Zoobot rows are then one architecture that differs only in its initial weights; the
   stem costs about 3 M more MACs per view than a 2x2 one (under 1 %). The constructor
   refuses a timm whose stage 0 has a downsample, so a timm change cannot silently move the
   maps; a test pins the timm trap itself.
4. **The paper's frame around the backbone.** The 4 rotations go through the backbone as one
   batch; its pooled, normalised features (320 / 512) go through `fc1` to 64, ReLU and a
   dropout layer per rotation, then the mean over the rotations, then `PaperHead` (feature
   BN on the `feature_set` columns, fc2, fc3, logits). The invariance to 90-degree rotations
   is exact up to the order of the 4-term mean (float64 tests, perturbed parameters). timm's
   initialisation for the backbone, the paper's for fc1 and the head, all from the seeded
   global generator. No stochastic depth or timm head dropout: the paper's `dropout_active`
   is the only dropout. Counts (with the 23 features): atto 3,405,227 parameters and 178.6 M
   MACs per view, pico 8,575,667 and 446.3 M (the paper model: 441,075 and 52.7 M).
5. **The 21x21-crop flags are refused** with a ConvNeXt: `context: avgpool63` (the input
   already is the whole stamp), C5's `aug_noise`, C6 and C7 (their fits, spikes and PSF are
   defined on the crop). `rot90_augment` and `aug_feature_jitter` work.
6. **Pretrained weights, pinned and fetched at run time.** `backbone_weights: zoobot`
   (`convnext_pico` only: no Zoobot ConvNeXt-atto exists) loads
   `mwalmsley/zoobot-encoder-convnext_pico` at commit `ca4dc5278aa6...`, file
   `pytorch_model.bin`, 34,178,370 bytes, SHA-256 `b0311878fd13...` (full values in
   `convnext_model.ZOOBOT_CONVNEXT_PICO`). It is downloaded by `huggingface_hub` into its
   cache, never committed; a file of another size or SHA-256 is refused; it is read with
   `torch.load(weights_only=True)` and loaded strictly (every key and shape). It is loaded
   after the seeded initialisation, and recorded in `metrics.json["paper_data"]
   ["backbone_weights"]`. From scratch (`none`, the default) never imports
   `huggingface_hub` nor opens a socket (tested with sockets disabled).
7. **The licence gate.** Every weights entry records its licence; an entry with none is
   refused before any download. The Zoobot pico card's front matter declares
   `license: apache-2.0` (also the Hub tag), read at the pinned revision on 2026-11-08; the
   card has no other text. The Zoobot code (github.com/mwalmsley/zoobot) is GPL-3.0, but no
   Zoobot code is used, only the weights file. Apache-2.0 asks for attribution: the docs
   cite the weights and Walmsley et al. 2024 (arXiv:2404.02973).
8. **The faithful records do not move.** `backbone_weights` joins `HASH_NEUTRAL_DEFAULTS`
   at `none` (the only initialisation before the field), so every existing hash and run id
   is unchanged (the canonical fixture only gains the pico and pico-Zoobot cases). It is
   not a module hyperparameter: the checkpoint holds the weights. It is listed in the
   presets, so a Zoobot run is a deviation, and like the other 2.5 flags only
   `paper_stamp_classifier` accepts it.

## Consequences

- Three B8 runs are one preset plus flags: `--set paper_arch=convnext_atto --set
  image_size=63` (or `convnext_pico`, plus `--set backbone_weights=zoobot`), with the split
  runs (`--splits FILE --fold K | --screen`). `paper_variants.VARIANTS` names them.
- Open questions for the runs, not settled here: Zoobot was trained at 224 px on galaxies
  at another pixel scale, so the native-64 fine-tune may lose part of its benefit (the
  research suggests the 224-px upsampled recipe as a further arm); the stamps are fed in
  [0, 1] without ImageNet mean/std (the card's `pretrained_cfg` lists timm's ImageNet
  values, which may be a template default; Zoobot's own transforms were not checked); the
  research recommends a fine-tuning rate of about 1e-4 for Zoobot ConvNeXts, while the
  preset's Adam uses 1e-3.
- The 4-rotation pico costs about 1.8 G MACs per stamp, 8.5 times the paper model; a CPU
  training step is seconds, so the CPU tests use atto and tiny batches.
- B9 (conv-attention hybrids) can reuse the dependency, the input side and the weights
  gate.
