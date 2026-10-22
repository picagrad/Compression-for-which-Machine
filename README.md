# Compression for Which Machine? On the Effects of Downstream Task-Models on Learned Compression

Expanded training and testing details, results and reproduction for "Compression for Which Machine? On the Effects
of Downstream Task-Models on Learned Compression" (submitted for peer review at the Data Compression Conference,
DCC 2027).

One feature codec trained and evaluated on a range of object-detection and semantic-segmentation models. This repository (package `cfwm_repro`) holds the codec (ELFC-GQVRF - a QVRF-inspired variable rate feature codec based on ELIC), its train and test scripts, the published results and the guide to reproducing them; the split task models come from the [CFM-Model-Zoo](https://github.com/picagrad/CFM-Model-Zoo) library (Based on MMDetection and MMSegmentation).

Authors: Alon Harell. Code: <https://github.com/picagrad/Compression-for-which-Machine>.

## Install

You need a CUDA GPU: training and testing run on `cuda:0`, in 32-bit precision.

This repository always uses the exact package versions our results were produced with. You can get them from our
container image, or install them yourself.

### The container image

The image `ghcr.io/picagrad/cfwm-repro:0.1.1-cu124` holds this repository, CFM-Model-Zoo and every dependency, at
those versions. The repository is at `/opt/Compression-for-which-Machine`, where every command in this README runs as
written. Mount your data, the published weights and a folder for the results:

```bash
# A shell in the repository; in the commands below, use --data_root /data and --checkpoint /weights/<name>.ckpt
docker run -it --rm --gpus all -v $DATA:/data -v $WEIGHTS:/weights -v $RESULTS:/results \
    -v ~/.cache/mim:/root/.cache/mim ghcr.io/picagrad/cfwm-repro:0.1.1-cu124 bash

# A Python session
docker run -it --rm --gpus all -v ~/.cache/mim:/root/.cache/mim ghcr.io/picagrad/cfwm-repro:0.1.1-cu124

# On a cluster with Apptainer, a shell with your data and weights mounted
apptainer shell --nv --bind $DATA:/data,$WEIGHTS:/weights docker://ghcr.io/picagrad/cfwm-repro:0.1.1-cu124
```

The image is built from [`docker/Dockerfile`](docker/Dockerfile). It contains NVIDIA software; see [License](#license).

### Installing it yourself

1. Install [CFM-Model-Zoo](https://github.com/picagrad/CFM-Model-Zoo#install) as its README describes, choosing its
   pinned versions (`-c constraints-pinned.txt`).
2. In the same environment, from a clone of this repository, install this package:

   ```bash
   pip install --no-deps -r constraints-pinned.txt -e .
   ```

   [`constraints-pinned.txt`](constraints-pinned.txt) lists every package and version, including CFM-Model-Zoo at
   the pinned commit and MLflow for tracking. [Library version](#library-version) explains `--no-deps`.

## Data

`--data_root` is the dataset root, and the scripts find the rest.

The datasets are not distributed here; download them from their publishers, whose terms apply:
[COCO 2017](https://cocodataset.org/#download) (`train2017`, `val2017` and their annotations) and
[ADE20K's scene-parsing release](http://sceneparsing.csail.mit.edu/) (`ADEChallengeData2016`).

- **COCO** (detection): our own training/validation split of the 2017 training set's annotations,
  `coco2017_trn_split_anns.json` and `coco2017_val_split_anns.json` in `<root>/annotations/`, in addition to the
  standard COCO 2017 layout. Our split files can be built with `python scripts/make_coco_split.py --coco <root>` (from
  the image-id lists in `splits/`), or downloaded with `python scripts/download_weights.py --coco-splits <root>`.
- **ADE20K** (segmentation): the `ADEChallengeData2016` directory, unchanged.

The task models' configurations and weights are downloaded by the library, from their publishers, on
first use (cache: `--mim_cache_dir`, else the library's default).

## Train

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.train_variable_rate --task_model FasterRCNN --data_root $DATA \
  --lambda_ref 5 --epochs 100 --checkpoint_dir $CKPT
```

Trains one variable-rate codec for one task model. The main inputs (all of them: `--help`):
- `--task_model`: the task model, by its CFM-Model-Zoo short name (`FasterRCNN`, `SegUperNetSWIN`, ...; the list is
  in the CFM-Model-Zoo README, and the published ones in [`REPRODUCE.md`](REPRODUCE.md#task-models)).
- `--data_root`: the dataset root (see Data).
- `--lambda_ref`, `--lambda_min`, `--lambda_max`: the training lambda is sampled from
  `[lambda_min, lambda_max] x lambda_ref`.
- `--epochs`, `--lr_scheduler` and the schedule flags, `--train_batch_size` (default `auto`: the largest batch that
  fits the GPU).
- `--checkpoint_dir`: where the run saves `<checkpoint_name>.ckpt` (default name `last`) every epoch.

The training recipes of the published codecs are in [`REPRODUCE.md`](REPRODUCE.md#4-training-parameters), and
every run's training command in [`REPRODUCE_COMMANDS.md`](REPRODUCE_COMMANDS.md#the-training-commands). Training uses
512x512 crops of the training images (details: `make_safe_pipeline` in `cfwm_repro/tools/utils.py`, and REPRODUCE.md).

**Where a run starts.** At launch it takes the first case that applies, and prints which:
1. **`<checkpoint_dir>/<checkpoint_name>.ckpt` exists:** it resumes from it fully (optimiser, scheduler and training
   stage included), so relaunching an interrupted run with the same command continues it; `--init_checkpoint` and
   `--restart` are then ignored.
2. **`--init_checkpoint FILE`:** it continues that checkpoint's training state, or, with `--restart True`, starts a
   new run (fresh optimiser and scheduler, epoch 0) from its weights only.
3. **Neither:** it trains from scratch.

## Test

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.test_rate_guided --task_model FasterRCNN --data_root $DATA \
  --checkpoint $WEIGHTS/FasterRCNN.ckpt --results_dir results/mine
```

A rate-guided sweep of a trained codec: it measures the task model's uncompressed score, then chooses each next
lambda from the points measured so far ([how](REPRODUCE.md#how-the-rate-guided-sweep-chooses-its-points)). The
inputs:
- `--task_model`, `--data_root`: as for training.
- `--checkpoint`: the codec checkpoint, loaded strictly (a missing file or another codec's checkpoint is an error).
- `--results_dir`: the results go to `<results_dir>/<task_model>__<checkpoint stem>.json` (default directory
  `data/logs/rd_sweeps/`), one point per lambda, saved as each completes; the same command resumes.
- The sweep flags (`--floor_frac`, `--max_points`, `--min_points`, `--plateau_tol`, `--q_step`, `--rate_stall`,
  `--hi_frac`): where the sweep places its points and when it stops, explained in
  [REPRODUCE.md](REPRODUCE.md#how-the-rate-guided-sweep-chooses-its-points); their defaults are the values every
  published sweep used.

## Cross-model

<!-- check: parse -->
```bash
python -m cfwm_repro.tools.test_cross_model --task_model FasterRCNN --checkpoint $WEIGHTS/FasterRCNN.ckpt \
  --target_task_models RetinaNet,FCOS,ATSSResNet --data_root $DATA --results_dir results/mine
```

Tests one codec against other task models that emit the same cut-point shape, in one resumable JSON file. The
inputs:
- `--task_model`, `--checkpoint`: the model the codec was trained on, and the codec.
- `--target_task_models`: the models to test it on (comma-separated short names).
- `--sweep rate_guided` (default) with the sweep flags of Test, or `--sweep grid` with `--lambda_min`, `--lambda_max`,
  `--lambda_space`, `--lambda_space_steps` (a fixed lambda grid).
- `--calibration_stats train|test|none` (default `train`): corrects the feature-scale difference between the two
  models with per-channel statistics from the first 2000 training images (`train`), the test set (`test`), or not at
  all (`none`). ConvNet backbones only.
- `--results_dir`: as for Test; the file is `<results_dir>/<task_model>__<checkpoint stem>.json`.

What it does with them:
- A target whose cut-point shape differs from the codec's, or whose checkpoint load fails, is recorded as
  `skipped_incompatible`.
- It is safe to run in parallel: several jobs can split one codec's targets between them with the same
  `--results_dir`.

## Tracking

`--tracking none|mlflow` is on every entry point; the default is `none`.

- `none` never imports MLflow. Metrics go to `<output>/cfwm/<random id>/metrics.csv`, whose path is
  printed at start. One CSV per launch: a resumed run writes a new file, and the history is their
  concatenation.
- `mlflow` logs the same metrics as the published runs (parameter names may differ), and tags each run
  `cfwm_repro`. It needs the `[mlflow]` extra; the tracking server comes from MLflow's own environment
  configuration. Without the extra, the script stops at once and says to install it or pass
  `--tracking none`.

The results files (the RD points that REPRODUCE.md compares) are written in both modes.

## Published weights

The trained codecs behind every sweep in the paper (32 checkpoints, about 2.7 GB) are archived on Zenodo,
<https://doi.org/10.5281/zenodo.23408855>, with a faster mirror hosted by the Digital Research Alliance of Canada.
They are licensed under CC BY-NC 4.0 (see License). Download and verify them with:

```bash
python scripts/download_weights.py --to $WEIGHTS              # all 32; --only NAME ... for some
```

The script reads `weights/manifest.json` (each file's name, size, sha256 and the run it reproduces), downloads
from the mirror or, failing that, from Zenodo, and checks every file's size and sha256. The files hold the codec
weights only (the optimiser state is not included): they test and reproduce the published sweeps, and
`--init_checkpoint FILE --restart True` starts a new training run from them, but a training run cannot resume
from them. The same script fetches the COCO split annotation files (`--coco-splits`, see Data).

## Reproducing the paper

[`REPRODUCE.md`](REPRODUCE.md) explains how every result file was made, with one example command per method,
and gives the training parameters; [`REPRODUCE_COMMANDS.md`](REPRODUCE_COMMANDS.md) holds every test and training
command behind the published results.

The results are in `results/`. From them, `scripts/run_paper_results.sh` draws, without re-running any test, into
`results/paper/` (committed):
- each model's RD curve;
- summary curves;
- cross-model testing statistics;
- BD-rate summaries, overall and per codec.

Exact reproduction is not guaranteed. Training and testing both use GPU operations that have no deterministic
implementation, so results vary slightly with the GPU type and from run to run; and training picks its batch size
automatically by default (`--train_batch_size auto`, the largest that fits the GPU), so a codec trained on other
hardware can differ more. REPRODUCE.md gives the tolerances that the published runs were checked against.

## Library version

This repository is pinned to one commit of CFM-Model-Zoo and to CompressAI 1.2.8, the versions the published results
were produced with. Both pins are in `pyproject.toml` and in `constraints-pinned.txt`. You can point them at another
library version by hand, but results are only guaranteed for the pinned one.

**Why `--no-deps`.** CompressAI 1.2.8 says it needs a NumPy older than 2, but it works with NumPy 2, and our published
results were produced with NumPy 2.3.5. Left to resolve dependencies itself, pip would follow CompressAI's claim and
downgrade NumPy to 1.26. `constraints-pinned.txt` already lists the complete set of packages, so `--no-deps` tells pip
to install exactly that list and nothing else. `pip check` will still report CompressAI's NumPy requirement; you can
ignore it. The container image is installed the same way.

## AI assistance

Parts of this codebase, including its tests and documentation, were written with the assistance of an AI coding tool
(Anthropic's Claude, via Claude Code). The research design, the experiments, the release decisions, and the
verification of every reported result were carried out by the author, who takes full responsibility for the code and its
correctness.

## License

The whole package (its code and configuration files) is for academic use only: see [`LICENSE`](LICENSE). The
published codec weights are licensed separately, under CC BY-NC 4.0. Third-party components keep their own licences; the CompressAI licence text is in
`third_party_licenses/`, and [`NOTICE`](NOTICE) lists what comes from where. The task models, and the
split-model library they run through, are covered by the CFM-Model-Zoo repository's own licence and
notices.

The container image is built on NVIDIA's CUDA base image and contains NVIDIA software, under the
[NVIDIA Deep Learning Container License](https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license)
(its text is in the image at `/NGC-DL-CONTAINER-LICENSE`) and the [CUDA EULA](https://docs.nvidia.com/cuda/eula/index.html).
This software contains source code provided by NVIDIA Corporation.

## Citation

Cite this software with [`CITATION.cff`](CITATION.cff) (GitHub's "Cite this repository"; it also gives the
weights' Zenodo DOI). If you use the codec or
any of the task models, please also cite the papers listed in [`ATTRIBUTION.md`](ATTRIBUTION.md), with the BibTeX
entries provided there: the codec's (ELIC, Harell et al., QVRF, CompressAI), OpenMMLab's, and each task model you
use.

## About this repository

This repository is a public snapshot of a private research repository, updated whenever that research is ready for
publication. It contains only the parts released here, and it starts without that repository's commit history.
