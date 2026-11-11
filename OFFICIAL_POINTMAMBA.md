# Official PointMamba integration — scope and use

This update adds a separate `official_pointmamba` backbone. It executes the
actual upstream `PointMambaScan` classifier definitions at commit
80858e096e33ff7470640fe1f2132e5c1e667074, from a bundled, hash-checked snapshot.
Upstream source, configuration and Apache-2.0 license are included unchanged in
`third_party/pointmamba_official/`. SOURCE.json records all source hashes.

## Precisely what is official and what is adapted

The upstream Encoder, Group, two Hilbert serializations (hilbert/hilbert-trans),
OrderScale, MixerModel, Block, and classification head execute from upstream
source. The ModelNet classifier config uses dim=384, depth=12, 64 groups,
32 neighbors, average pooling; only cls_dim is changed from 40 to 10.
The loader selects classifier definitions without importing unused pretraining
classes, Chamfer extensions or upstream's conflicting top-level utils package.
It does not rewrite any classifier method. CPU tests compare ASTs against the
upstream definitions, aside from removing registry-only class decorators.

The default explicitly named `grouping_backend: torch` uses PyTorch CUDA FPS
and KNN instead of pointnet2/KNN-CUDA extension kernels. FPS starts at point
zero as in pointnet2; KNN uses torch.cdist. This preserves the grouping algorithm
and trainable architecture, but tie-breaking and floating-point differences can
produce different neighborhoods. It is not bitwise equivalent to native kernels.
This choice avoids the old KNN_CUDA dependency, whose source URL could not be
resolved here. `native` is optional only if pointnet2_ops and knn_cuda are already
installed; there is no silent fallback.

The state-space primitive uses the installed mamba-ssm 2.2.4, with timm 0.9.16
for compatible DropPath/trunc_normal, rather than upstream's older bundled Mamba
and timm 0.4.5 stack. This is the official classifier architecture adapted to the
working modern environment, NOT a reproduction of paper accuracy or pretraining.
It trains from scratch. Paper results use a different pretraining/finetuning
protocol. No official pretrained checkpoint is downloaded or implied.

## Deployment without connecting Git on the cluster

Copy the supplied updated codebase ZIP to your Sarvesh folder and extract it
there, replacing project source files. Keep your existing datasets, cache,
Conda environment and old runs. Do NOT copy Markdown fences or shell file-creation
commands into batch files. The included scripts/run_official.sbatch is ready
as a file, with #!/bin/bash on its first line and the confirmed 12-hour limit.

Submit from the project directory:

    sbatch scripts/run_official.sbatch

It loads the existing pointmamba-a100 Conda environment, installs only additional
compatible torchvision/timm/addict/transformers dependencies, runs a mandatory
GPU check, then runs the four official-model experiments, defenses and figures.
The script stops if dependency setup or GPU checks fail. The archive does not
include your Conda environment, datasets, caches or generated experiments.

New output root: runs/official_modelnet10_core. Old runs remain untouched.
Checkpoints from the old custom model are incompatible and are not reused.
If the previous job is still running it may block the new job under your
one-running-job QoS; inspect squeue -u achyut before deciding whether to cancel it.

Logs: official-pointmamba-JOBID.out and official-pointmamba-JOBID.err.
This model is substantially larger, so earlier seconds/epoch estimates do not
apply. No guarantee is made that the suite completes within one 12-hour job.
Resubmit the same batch file to resume main training at the last complete epoch;
interrupted defense refits restart. Do not change the config between resumes.

The existing default/smoke configs still select the legacy custom model so old
checkpoints remain interpretable. Always use configs/official.yaml for this work.
The official config uses drop_last=true for training to protect the upstream
BatchNorm head from singleton batches after feature filtering. Partial final
training batches are excluded. Evaluation retains every test example.

## Validation limits

CPU tests check source integrity, exact definition preservation, FPS/KNN,
upstream classifier forward/backward and feature extraction using explicitly
marked lightweight test doubles for unavailable GPU dependencies. Those tests
are NOT evidence of CUDA Mamba correctness. scripts/check_official.py must pass
on the A100 before research training. No live official GPU run or ModelNet run
was possible in this cloud environment. Results cannot be claimed before that.
