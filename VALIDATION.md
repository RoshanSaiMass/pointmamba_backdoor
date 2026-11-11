# Validation record

Executed in this CPU-only workspace with Python 3.12.14 and torch 2.15.0+cpu.

- `python -m pytest -q`: **30 passed**. Covers single/multi/degenerate hull
  containment and point counts, rotation invariance, curve radius/reduction,
  both backbone gradients, poisoning/cache semantics, OFF loading/sampling,
  defenses, checkpoint evaluation/reporting, and exact resumed CPU weights.
- `bash scripts/smoke.sh`: **completed, exit 0** using the workspace virtualenv.
  Ran a clean baseline plus spiral/rose/Lissajous two-epoch synthetic training,
  all input defenses, eight total one-epoch feature-defense refits, reports,
  30 paired figures, manifest, and a summary grid.
- DGCNN two-epoch synthetic training with `after_augmentation` insertion:
  **completed**.
- One paired PNG was visually inspected; image decoding is also tested.
- Shell syntax checks, Python compilation, and `git diff --check`: **passed**.

Generated artifacts are ignored by Git under `runs/smoke_core` and
`figures/smoke_core`. The tiny synthetic data and short schedules demonstrate
execution, not useful classification or attack performance. In particular,
high synthetic ASR can be caused by target-class collapse at chance CA.

Not executed: real ModelNet experiments, CUDA Mamba installation/kernels,
A100 memory/performance checks, or the full long-running ablation sweep.
Run `scripts/setup_a100.sh` and its CUDA checker on the target machine before
starting research training. See README for exact commands and method choices.

Downloader update: five additional fixture tests cover streamed downloads,
checksum failures, safe extraction, traversal rejection, count checks, and reuse.
Live Princeton download was blocked by the cloud outbound policy (HTTP 403
CONNECT); it was not verified here.

Official-backbone update: **33 tests passed**. Added upstream source integrity /
AST equality, FPS/KNN, and CPU upstream classifier execution with explicit test
doubles (small dimensions and a substituted mixer). No official CUDA validation
was possible here. The required on-cluster check is scripts/check_official.py.
See OFFICIAL_POINTMAMBA.md for the exact architecture/kernel/protocol distinction.
