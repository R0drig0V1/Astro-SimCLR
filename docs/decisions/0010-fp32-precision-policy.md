<!-- ADR 0010: the float32 precision policy of convolutions and matmuls (TF32), set and recorded per run. -->
# 0010: Full float32 by default; TF32 only behind `fp32_precision: tf32`

- **Status:** accepted (2026-11-08, audit item 1, "Record and set the conv TF32 policy")
- **Implemented by:** `Settings.fp32_precision` (`config.py`),
  `training.runs.apply_fp32_precision` / `fp32_precision_record`,
  `experiments.run_command` / `_machine`, `reference.machine_signature`, the `carrasco2021`
  presets (`presets.py`, `configs/phase2/carrasco2021*.yaml`)
- **Checked by:** `tests/test_fp32_precision.py`, `tests/test_cli_training.py`
  (`test_metrics_record_the_protocol_fields`), `tests/test_presets.py`
  (`test_config_files_hold_the_preset`)

## Context

torch 2.14.1 runs cuDNN convolutions in TensorFloat-32 by default
(`torch.backends.cudnn.conv.fp32_precision == "tf32"`, `cudnn.allow_tf32 == True`;
measured), while CUDA matmuls default to full float32 (`torch.get_float32_matmul_precision()
== "highest"`). TF32 keeps float32's exponent but only a 10-bit mantissa, and exists on
Ampere and newer GPUs only (compute capability 8.0+). So far every run was made on the GTX
1050 (Pascal, 6.1), the CPU or MPS, which have no TF32 and computed in float32 whatever the
flag said. The RTX 5090 (Blackwell, 12.0) would silently run the "faithful" `carrasco2021`
convolutions at the lower precision, and `metrics.json["machine"]` recorded nothing that
would show it. A comparison of a 1050 run with a 5090 run would then mix two numerics while
looking like a comparison of two machines, and one of 5090 runs made before and after a
torch default change would mix them while looking like one machine.

## Decision

1. **One setting, `fp32_precision: ieee | tf32`, default `ieee`.** It covers convolutions
   and matmuls together: the paper model has both (conv1-conv5, fc1-fc3), and a policy that
   lowers one but not the other is not a variant anyone needs. Its values are torch's own
   names. `ieee` is the faithful choice: it is what the GTX 1050 computes, so the CD21
   replication's numerics carry over to new GPUs. TF32 is a speed option (an estimated
   1.0-1.2x on the convolutions, audit row 8), never the default.
2. **Set at the start of every run, before any module is built.**
   `experiments.run_command` calls `apply_fp32_precision(settings)` right after the
   accelerator and name checks, before the dataset is opened. The setting is
   process-global, so a second run in the same process (the preflight, the tests) gets its
   own policy, not the previous run's.
3. **The 2.14 APIs, with one exception.** Convolutions use the per-operator API,
   `torch.backends.cudnn.conv.fp32_precision`. Matmuls use
   `torch.set_float32_matmul_precision("highest" | "high")` instead of
   `torch.backends.cuda.matmul.fp32_precision`: after the latter is set to `tf32`,
   `torch.get_float32_matmul_precision()` raises (torch reports a mix of the legacy and new
   APIs; measured on 2.14.1), and Lightning's CUDA accelerator calls it on every Ampere+
   device. The legacy `allow_tf32` flags are neither set nor read, because reading
   `cudnn.allow_tf32` raises once the conv and RNN precisions differ. `highest` is torch's
   CUDA default and leaves the CPU's oneDNN matmuls in float32, so the faithful preset's CPU
   golden fixtures and outputs do not move.
4. **The effective policy is recorded, and it is part of the machine signature.**
   `metrics.json["machine"]` gains `fp32_precision` (`tf32` only on a CUDA device with
   compute capability 8.0+ whose convolutions or matmuls allow it, else `ieee`),
   `fp32_backends` (the torch settings read back after the run: `cudnn_conv`, `cuda_matmul`,
   `mkldnn_matmul`) and `compute_capability`. `reference.machine_signature` appends the
   effective value, so `compare-variants` (and `--splits`) refuses to mix policies exactly as
   it refuses mixed machines (`--allow-mixed-machines` overrides both), and
   `compare-reference` warns. The effective value, not the setting, is compared: a `tf32`
   run on the 1050 computes exactly what an `ieee` run does.
5. **Runs without the field read as `ieee`.** Every run written before this ADR ran on
   hardware without TF32 (all ten recorded CUDA runs in `results/` are the GTX 1050), so the
   phase-2 replication stays comparable with new 1050 runs.
6. **`tf32` is a preset deviation.** The `carrasco2021` presets list `fp32_precision: ieee`,
   so a TF32 run is recorded in `metrics.json["preset"]["deviations"]` and is never
   `faithful`.

## Consequences

- A run that only changes the GPU keeps its numerics; a TF32 run has to say so, and is
  recorded, reported and refused in comparisons as a different build.
- A future non-faithful speed variant (bf16 autocast, channels_last with TF32) starts from
  `tf32`, never from the faithful preset.
- Caveat: with `tf32`, `set_float32_matmul_precision("high")` also lets oneDNN use TF32 for
  CPU matmuls on CPUs that support it, while the record calls every CPU run `ieee`. No
  machine of this project has such a CPU, and `fp32_backends.mkldnn_matmul` records the
  setting.
- Caveat on rule 5: a pre-ADR run made on an Ampere+ GPU would be misread as `ieee`. None
  exists; this ADR is why one will not.
- Caveat on "keeps its numerics": `ieee` only rules out TF32. cuDNN still picks its float32
  algorithm per GPU and shape, and some are far from a float32 dot product. On the RTX 5090
  (torch 2.14.1+cu130, cuDNN 9.24) the 5x5 convolutions of the faithful model and of the
  p4/p4m variants run as non-fused Winograd F(5x5) on 9x9 tiles (`winogradForwardData9x9_5x5`)
  whatever the policy, with 4e-3 relative error on a 32 -> 32 channel layer against 6e-7 on
  the CPU; the logits of the faithful model are 1.3e-5 from float64 there against 2e-7 on
  the CPU (measured in the fix/cuda-tests PR). `cudnn.deterministic` does not change the
  choice; only disabling cuDNN does. So a CUDA run is about 1e-5 from float64 in the logits
  under either policy, and two GPUs (or cuDNN versions) need not pick the same algorithm:
  "keeps its numerics" means no TF32, not the same bits.
- Revisit when torch removes `set_float32_matmul_precision` or makes the new matmul API
  readable by Lightning: then rule 3's exception goes.
