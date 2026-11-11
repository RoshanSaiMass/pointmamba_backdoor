# Official backbone update

Use [OFFICIAL_POINTMAMBA.md](OFFICIAL_POINTMAMBA.md) and `configs/official.yaml` for the pinned official classifier architecture with an explicit PyTorch grouping backend. The default config below is retained for legacy custom-model checkpoints.

# Flat-curve point-cloud backdoor research

A reproducible PyTorch pipeline for studying geometric backdoors on ModelNet10
(default) or ModelNet40. Includes cached dirty/clean-label poisoning, three local
curves, training, evaluation, five defenses, ablation plans, and paired figures.

**Validation status:** CPU synthetic integration checks have been run. There is
no ModelNet dataset, CUDA device, or CUDA toolkit in the development environment.
No ModelNet accuracy, attack efficacy, A100 performance, or CUDA build success is
claimed. Synthetic metrics are pipeline diagnostics, not research results.

```text
attack/
  curves.py           # spiral / clipped rose / Lissajous; budget adaptation
  flat_detection.py   # batched KNN, PCA, mutual graph, connected components
  flat_spiral.py      # patch selection, budgets, hull-safe snapping
data/modelnet.py     # OFF surface sampling, NPZ loading, poison caches
models/classifier.py  # minimal PointMamba-style Mamba and DGCNN
configs/              # default A100 research and small CPU smoke configs
train.py              # training, resumable checkpoints, epoch/class CSV logs
eval.py              # CA, non-target ASR, distances, defenses and refits
defenses.py          # SOR, SRS, jitter, activation clustering, spectral filter
visualize.py          # independent static renderer / geometry metrics
generate_figures.py  # class × curve pairs, summary grid, manifest
ablation.py           # explicit, resumable experiment plans
report.py             # CSV and Markdown tables from measured artifacts
scripts/              # A100 installation, CUDA checks, CPU smoke workflow
tests/                # geometry, models, poisoning, defenses, integration/resume
```

## A100 environment

Use Linux, Python **3.11**, a **CUDA 12.4 development toolkit** (`nvcc`), and a
CUDA-12.4-compatible NVIDIA driver. An A100 with 40 GB is sufficient for the
default batch size; reduce `train.batch_size` if other jobs consume memory.
The setup script pins torch 2.6.0/cu124, causal-conv1d 1.5.0.post8, and
mamba-ssm 2.2.4. These GPU installation instructions must be validated on the
target machine; they have not been executed on this CPU-only machine.

```bash
cd pointmamba_backdoor
bash scripts/setup_a100.sh
source .venv/bin/activate
python scripts/check_cuda.py
```

`setup_a100.sh` verifies CPU tests and CUDA forward/backward for both backbones
and CUDA trigger insertion. It stops on failure. Extension builds use at most
four jobs by default (`MAX_JOBS=2 bash scripts/setup_a100.sh` for limited RAM).
Keep `a100-installed.txt` with experiment artifacts to record resolved versions.
The CUDA checker must pass before starting a long experiment. Python 3.12 and
torch 2.15.0+cpu were used for the current CPU validation; that is a separate
backend from the pinned A100 environment.

The official [PointMamba source](https://github.com/LMD0311/PointMamba) was
inspected at `80858e096e33ff7470640fe1f2132e5c1e667074`. Its upstream model
requires KNN-CUDA, pointnet2, Chamfer CUDA extensions, and a different dependency
stack unavailable here. This project therefore uses the minimal implementation
path: FPS patches → point MLP/max pooling → Morton (Z-order) serialization →
alternating forward/reverse Mamba blocks → mean/max classifier. It is **not** an
official PointMamba reproduction and cannot load its pretrained checkpoints.
`model.mamba_impl=mamba_ssm` uses real Mamba on CUDA. The explicitly selected
`reference` backend implements a readable selective SSM scan for CPU development;
it is slower and has a different parameterization, with no checkpoint conversion
or numerical-equivalence claim. Backbones are selected through `build_model`;
`model.backbone=dgcnn` uses two dynamic EdgeConv stages instead.

## Data

Download ModelNet10 automatically (standard library only; can run before GPU setup):

```bash
python3 scripts/download_modelnet10.py
```

This downloads the official HTTPS archive to `datasets/ModelNet10.zip`, safely
extracts meshes into `datasets/ModelNet10`, and checks all ten classes plus the
expected 3,991 training / 908 test objects. Repeated runs verify the existing
dataset. TLS, ZIP CRC, and optional `--sha256` verification remain enabled; a
recorded SHA256 identifies the downloaded bytes but is not an independently
trusted checksum. Failed downloads/extractions do not replace existing data.
Use `--url https://...` only for an explicitly trusted mirror if the original
server is unavailable. No automatic fallback to an unverified source is used.
The source could not be contacted from this cloud instance because its outbound
policy denied the domain; local archive extraction and download handling were
tested with fixtures. The live download must be verified on your machine.

For ModelNet40 or manual downloads, acquire meshes from the
[official ModelNet page](https://modelnet.cs.princeton.edu/), then extract them
locally. Training itself does not make network requests or require credentials.

```text
/path/to/ModelNet10/
  bathtub/train/*.off
  bathtub/test/*.off
  bed/train/*.off
  ...
```

The root must contain the class directories, not the parent directory of the
extracted dataset. OFF faces are triangulated and sampled proportional to surface
area. Each sample is centered and normalized to unit maximum radius. Sampling
and class order are deterministic for a fixed seed. The default class mapping is
`bathtub, bed, chair, desk, dresser, monitor, night_stand, sofa, table, toilet`.
ModelNet40 uses the standard alphabetical 40-class mapping in `data/modelnet.py`.

Alternatively provide `train.npz` and `test.npz`, each with `points [S,N,3]`,
`labels [S]` (integer class IDs), and optional `class_names [C]`. Labels must use
the same mapping; `class_names`, when supplied, is checked. Clouds are resampled
to the requested point count and normalized. Native HDF5 is not assumed: convert
its arrays to this documented NPZ schema if using a preprocessed release.

## Main A100 commands

Train a clean baseline and each attack under an identical schedule, evaluate
all five defenses, and produce the comparison tables:

```bash
source .venv/bin/activate
python ablation.py --config configs/default.yaml --suite core \
  --output runs/modelnet10_core \
  --set data.root=/path/to/ModelNet10 geometry_device=cuda
```

This runs four main training jobs sequentially: baseline and spiral/rose/Lissajous.
Each evaluation performs two additional fresh-model defense refits (default 30
epochs). Outputs include `results.csv`, `results.md`, `per_class.csv/.md`, and
`train_patch_modes.csv/.md`. Run names include seeds, e.g.
`runs/modelnet10_core/curve_spiral_seed42/last.pt`.

For one experiment:

```bash
python train.py --config configs/default.yaml \
  --set data.root=/path/to/ModelNet10 geometry_device=cuda output=runs/spiral
python eval.py --checkpoint runs/spiral/last.pt --feature-defenses
python generate_figures.py --config configs/default.yaml \
  --checkpoint runs/spiral/last.pt --output figures/modelnet10 --per-class 2 \
  --set data.root=/path/to/ModelNet10 geometry_device=cuda
```

The figure command generates all three curve types; predictions for each are
made with the **supplied checkpoint**, even if that checkpoint was trained on
only one curve. To assess a matching trained model for each curve, run the figure
command separately with each checkpoint and a distinct output directory. Omit
`--checkpoint` for geometry-only figures: success then says “unavailable (no
checkpoint)”, never a fabricated prediction. Each sample gets a combined PNG,
and the manifest stores full requested config, effective curve parameters,
metrics, and output paths. `summary.png` contains all classes × three curves.

Useful variants:

```bash
# Clean training baseline only (choose its own output)
python train.py --baseline --set data.root=/path/to/ModelNet10 output=runs/baseline
# Clean-label poisoning
python train.py --clean_label --set data.root=/path/to/ModelNet10 output=runs/clean_label
# Rose or Lissajous
python train.py --set data.root=/path/to/ModelNet10 attack.curve.kind=rose output=runs/rose
python train.py --set data.root=/path/to/ModelNet10 attack.curve.kind=lissajous output=runs/lissajous
# Second backbone
python train.py --set data.root=/path/to/ModelNet10 model.backbone=dgcnn output=runs/dgcnn
# ModelNet40 (use a new output)
python train.py --set data.root=/path/to/ModelNet40 data.num_classes=40 output=runs/modelnet40
# Resume at an epoch boundary with exactly the original config/schedule
python train.py --resume --set data.root=/path/to/ModelNet10 geometry_device=cuda output=runs/spiral
```

`--set` accepts YAML scalars and dotted keys, and should come after other options.
Existing runs are protected from accidental overwrite. `--overwrite` explicitly
restarts training. Resume restores optimizer, scheduler, AMP scaler, Python/NumPy/
torch/CUDA RNGs, loader RNG, and convergence history. Atomic checkpoint writes
avoid partially written `last.pt` files. Only load checkpoints/cache files you
trust: these locally generated PyTorch artifacts include Python metadata.

## Ablations and reporting

Review the full plan before allocating a long GPU job:

```bash
python ablation.py --suite full --plan-only --seeds 42 43 44 \
  --output runs/full --set data.root=/path/to/ModelNet10 geometry_device=cuda
# Execute the same plan (remove --plan-only):
python ablation.py --suite full --seeds 42 43 44 \
  --output runs/full --set data.root=/path/to/ModelNet10 geometry_device=cuda
python report.py --root runs/full
```

The full suite varies one setting at a time for each curve: own complexity,
KNN size, planarity threshold, normal angle, minimum patch size, curve density,
M, dirty/clean labels, random placement, and poison rates
`{0.001,0.005,0.01,0.03,0.05,0.10}`. Duplicate configs are collapsed. Add
`--transfer` for DGCNN training with the same geometric construction; these are
cross-backbone efficacy experiments. To assess a fixed model under a different
curve use `eval.py --curve rose ...` (saved separately under `evaluation_rose`; convergence is undefined for an unmonitored curve). Geometric triggers are model-independent;
this is not a gradient-based source-to-target transfer attack.

Plans are saved to `plan.json` before execution. Repeated invocations reuse
completed runs and resume interrupted main training jobs. Configuration mismatches
are rejected. Two feature-defense refits per run make the full sweep expensive.
`--skip-feature-defenses` is an explicit cheaper option; tables then label those
ASRs **not run**, rather than filling in zeros. A later run without that flag
fills in missing refits. Refit jobs restart if interrupted; they do not resume.
ModelNet40 point arrays and caches can consume gigabytes per distinct attack
configuration; plan disk space and archive/remove unneeded caches between suites.

Tables retain individual seeds (no undocumented averaging), source-class metrics,
class CA, non-target ASR per defense, geometry distances, global/per-class
convergence, and patch/reduction frequencies. Target-class ASR is undefined and
left blank, because target samples are excluded. Training patch statistics use
original source classes, including clean-label target-class poisons. A missing
convergence epoch means the threshold was not reached for the required streak;
it never means epoch zero.

## Algorithm and measurement choices

- Detection computes all per-point KNNs and covariance/eigendecompositions with
  tensor operations. Self neighbors are excluded. Normal compatibility uses the
  absolute dot product, so eigenvector sign and world orientation do not matter.
  Components use SciPy on CPU after graph construction. Distance memory is
  O(B N²); default insertion processes one sample at a time. Relative distance
  quantization stabilizes nearest-neighbor ties on rotated regular grids.
- A single patch is preferred when it reaches `min_patch_size` **and** has capacity
  for M. Otherwise use qualifying components, largest first, and allocate integer
  budgets approximately proportional to sizes, with minimum allocations and
  capacity bounds. If fewer than M points fit, use a logged global fallback to
  preserve M. There is no silent sample skip. Patches may be reduced in number if
  M cannot supply the configured minimum to all of them.
- Every patch has its own PCA frame and convex hull. Curves are radially projected
  into hull halfplanes before nearest-curve snapping. Rank-one hulls use their
  actual line segment; rank-zero hulls collapse to the centroid. Exactly M unique
  input indices are selected; N never changes. Multiple indices can snap to the
  same position. Ranking is literal `planarity / density`, with density defined
  as inverse squared mean neighbor distance.
- Spiral pitch is derived from radius and effective turns. Rose radii are clipped
  by omitting negative-radius intervals over **the full 2π**; using only π for an
  odd clipped rose would omit petals. This clipped convention intentionally
  differs from the signed standard rose. Lissajous amplitude is radius/√2;
  frequencies must be coprime. `a+b−1` is a conservative feature-budget proxy,
  not an exact count of topological loops.
- Complexity is reduced locally when budget/feature is too small. If even one
  feature cannot meet the minimum (e.g. budget 3, minimum 4), use the simplest
  curve and record `feature_budget_satisfied=false`. Sampling density controls
  curve discretization separately from M and is samples per unit arc length.
  Effective reductions are logged; requested and effective complexity are never
  conflated. Nearest snapping does not guarantee uniform coverage of the curve.
- `poison.rate` is a fraction of the **entire training set**: floor(rate × size),
  capped by eligible samples. Dirty-label eligibility excludes the target class;
  clean-label eligibility includes only that class and preserves labels.
  Reports include the actual count/rate, so low-rate rounding is visible.
- Cache keys include input tensor contents, seed, complete attack/poison settings,
  algorithm source hashes, library versions, geometry device, and schema version. Cache source geometry is always normalized before insertion.
  The default order is cached insertion → random anisotropic scale → random Y
  rotation → translation. Thus the local geometry transforms with the cloud.
  `train.insertion_order=after_augmentation` instead augments clean clouds first,
  then recomputes detection/insertion online for selected training samples each
  epoch. This option is more expensive and only the base cache is reusable.
- CA uses the untouched test split. ASR means target predictions on poisoned
  **non-target ground-truth** test samples, regardless of the clean prediction.
  Figure “attack succeeded” is stricter: an actual non-target → target prediction
  flip. The clean baseline is essential to detect target-collapse artifacts.
- Chamfer is the **sum** of two directed mean **squared** nearest distances.
  Hausdorff is the maximum directed **Euclidean** nearest distance. Geometry is
  measured before augmentation, without post-insertion renormalization.
- SOR removes high mean-KNN-distance points and deterministically repeats retained
  points back to N. SRS retains a random 75% subset and evaluates at that smaller
  size. Jitter is clipped Gaussian noise. Both CA and ASR use each input defense.
- Activation clustering uses per-class k=2 clustering of PCA-projected penultimate
  features and flags a smaller cluster only below a configurable fraction.
  Spectral filtering removes a configured fraction of largest squared projections
  on each class's leading singular vector. Classes with fewer than four points
  are left intact. These are explicitly specified baseline variants, not claims
  of exact reproduction of every published defense protocol.
- Feature filters inspect training activations without poison ground truth, then
  refit a **fresh model** on retained training data before measuring defended CA
  and ASR. Poison masks are used only for removal precision/recall diagnostics.
  Refit epochs are fixed/configurable; for final research use a matched full
  training budget and compare clean-data refits as well.
- Convergence is the first epoch starting `convergence_patience` consecutive ASRs
  ≥ `asr_threshold` (defaults 3 epochs, 0.95). Test metrics are monitored each
  epoch as requested; checkpoint selection is always the fixed final epoch.
  Tune hyperparameters on a separate validation split for a publication protocol;
  the code does not claim an untouched test set after test-driven ablations.
- Seeds and deterministic flags are set, with warnings for unsupported CUDA
  deterministic kernels. Exact reproducibility across GPU/library versions is
  not guaranteed. Save configs, versions, and seeds with reported results.

## CPU development and tests

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install torch==2.15.0+cpu --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
OMP_NUM_THREADS=2 bash scripts/smoke.sh
```

`requirements-lock.txt` records the actual CPU validation environment, including
CPU torch; install torch from its CPU index first, then this lock to reproduce it.
It is **not** the CUDA dependency lock. The smoke workflow uses 64-point synthetic
clouds and short schedules. It covers all curves, baseline training, checkpoint
loading, all five defenses, reports, and figures. Unit tests additionally check
hull containment in single/multi modes, exact N/M, degenerate geometries,
rotation invariance, radius/feature adaptation, both backbone gradients, label
semantics/cache reuse, OFF surface sampling, and epoch-boundary resume equality.

Generated experiments/caches/figures are ignored by Git. No training is silently
launched on import, and no experiment results are hardcoded.
