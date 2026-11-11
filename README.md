# Point-cloud backdoor robustness research — staged implementation

## Current status

Stage 1 is in progress. Implemented `attack/curves.py`,
`attack/flat_detection.py`, and the standalone `visualize.py` renderer.
22 geometry unit tests pass on CPU. No ModelNet experiments, training,
attack insertion, ASR measurements, or finished sample figures exist yet.
`train.py` and `eval.py` are intentionally deferred until attack and figure review.

## Environment and verification

Validated with Python 3.12 and torch 2.15.0+cpu. From this directory:

```bash
python -m venv .venv
.venv/bin/pip install torch==2.15.0+cpu --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q
```

`requirements-lock.txt` records the complete installed CPU environment.
CUDA code paths have not been tested; this session has no CUDA device.

## Module interfaces

```python
from attack.curves import CurveConfig, generate_curve, spiral, rose, lissajous
result = generate_curve(radius=0.2, budget=32, config=CurveConfig())
# result.points: [S, 2]; result.metadata: requested/effective settings, reductions

from attack.flat_detection import DetectionConfig, detect_flat_patches
# result = detect_flat_patches(points, DetectionConfig())
# points: floating torch [B,N,3] or [N,3], on CPU or CUDA
# result: normals, planarity, density, patches, neighbors; retains batch dimension

from visualize import save_comparison
# save_comparison(original, poisoned, patches, trigger_indices, metadata, output_path,
#                 prediction_before=None, prediction_after=None, target_class=None)
```

### Geometry conventions

- Detector: K neighbors excluding self, centered neighborhood covariance,
  ascending symmetric eigendecomposition, sign-invariant normal comparisons,
  mutual KNN graph, SciPy connected components. No world-axis comparisons.
- Degenerate rank-one neighborhoods receive zero planarity. Graph construction
  and PCA are tensorized; only connected-component processing loops over batches.
- Distance ordering uses relative 1e-12 quantization and stable index tie-breaking
  to prevent regular-grid neighbor ties from changing after rotation.
  Exact covariance computations retain the original coordinates.
- Detector distance memory is O(B N²); caller controls batch size.
- Curve coordinates are always clamped to the supplied circular radius.
  Polygonal convex-hull clamping belongs to the forthcoming insertion module.
- Spiral starts at radius zero; pitch is radius/(2*pi*effective_turns).
- Clipped rose and signed rose are both implemented as explicit options.
  Clipped rose samples the full 2pi and omits negative-r intervals. Signed rose
  uses pi for odd k and 2pi for even k. The experiment default awaits confirmation.
- Lissajous uses amplitude radius/sqrt(2). Frequencies must be coprime.
  Feature budget proxy is a+b-1; this conservative crossing/loop proxy is logged
  via effective frequencies and is not a claim of exact topological loop count.
- Density controls dense sample resolution; snapped-point budget controls
  complexity separately. Reduction preserves coprimality for Lissajous curves.
- Impossible feature budgets support explicit `raise` or `simplest` policy.
  The default currently raises; no silent failure of the feature minimum.
- Chamfer is the sum of two directed mean **squared** nearest-neighbor distances.
  Hausdorff is the maximum of the directed **Euclidean** nearest-neighbor distances.
- Visualization uses fixed elevation/azimuth and common pair bounds. Without
  model predictions, the success field explicitly says checkpoint unavailable.
  When predictions are supplied, success means an actual flip from non-target
  to target, rather than simply an after-poisoning target prediction.

## Pending decisions before insertion/sample generation

1. Synthetic samples first, or a local ModelNet path.
2. Clipped full-interval versus standard signed rose convention.
3. Impossible per-patch feature budget: simplify and log, or require more points.
4. Eligible multi-patch capacity below M: global fallback versus reduced actual M.
5. Ranking: planarity then density, or literal planarity divided by density.

After these choices: implement hull-safe insertion, insertion tests, figure script
and manifest, then present sample figures for review before training work.
