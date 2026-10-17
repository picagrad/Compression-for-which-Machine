"""Shared argument groups and setup helpers for all test_*.py scripts.

The test-side counterpart to train_common.py. Used by:
  - test_rate_guided.py     (VariableRateCFM, one task model, rate-guided RD sweep)
  - test_cross_model.py     (one source codec, many target task models)

The sweep helper here is *incremental*: it yields one RD point at a time as each lambda
finishes, so a caller can persist partial progress and resume from the first missing
lambda after a timeout/preemption. That is what makes cross-model testing survive
exceeding a single job's time limit.
"""

import copy
import hashlib
import json
from cfm_model_zoo.split_utils import task_model_mode
import os
import random
import socket
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch
from mmengine.runner import Runner

from .train_common import _make_probe_batch, _point_at_data
from .rate_guided_sweep import lambda_key  # noqa: F401 -- re-exported; test scripts import it from here

# Task metric to report, in priority order: (short name, the key the mmengine evaluator emits).
TASK_METRIC_KEYS = (
    ('mAP', 'coco/bbox_mAP'),
    ('mIoU', 'mIoU'),
)

CROSS_MODEL_RESULTS_DIR = Path('data/logs/cross_model_tests')

# Results files: one per swept checkpoint (test_rate_guided) or per source checkpoint (test_cross_model), written as
# the sweep goes; a relaunch resumes from them, and they are the sweep's output. Relative to the working directory;
# see results_file().
RD_SWEEP_RESULTS_DIR = Path('data/logs/rd_sweeps')

RESULTS_DIR_DISABLED_WARNING = (
    'WARNING: Missing results directory -- results will only appear in stdout, and any '
    'repeated run (including an automatic requeue after preemption) will repeat the whole '
    'lambda sweep from the beginning. Pass --results_dir to enable resume.'
)


# ---------------------------------------------------------------------------
# Dataloader / evaluator construction
# ---------------------------------------------------------------------------

def build_test_data(cfm_module, args):
    """Build the test dataloader + evaluator for the task model held by cfm_module.

    Reads ``data_root``/``test_ann``/``test_prefix``/``output`` off args. Returns (test_dl, test_eval, cfg_tr).
    """
    cfg_tr = cfm_module.task_model.cfg.copy()
    cfg_tr['work_dir'] = args.output
    # One test dataset (every published task model has one); see train_common._point_at_data.
    use_coco = _point_at_data(cfg_tr['test_dataloader']['dataset'], args.data_root, args.test_prefix, args.test_ann)
    assert cfg_tr['test_dataloader']['batch_size'] == 1, 'Test batch size must be 1'

    print('Building test dataloader and evaluator...')
    test_dl = Runner.build_dataloader(cfg_tr['test_dataloader'])
    if use_coco:
        cfg_tr['test_evaluator']['ann_file'] = os.path.join(args.data_root, cfg_tr['test_dataloader']['dataset']['ann_file'])
    test_eval = Runner.build_evaluator(None, cfg_tr['test_evaluator'])
    setattr(test_eval, 'dataset_meta', test_dl.dataset.metainfo)
    return test_dl, test_eval, cfg_tr



def build_calibration_data(cfm_module, args):
    """The dataloader the cross-model calibration stats are computed on (--calibration_stats).

    'test': the test set (test_prefix / test_ann). 'train': the first calibration_train_images training images through
    the test pipeline, so no test image is used to fit the calibration. Repeatable: COCO keeps the order of the
    annotation file's image list (calibration_train_ann; a test-mode dataset neither filters nor reorders), ADE20K
    mmseg's path-sorted listing of images/training.
    """
    if args.calibration_stats == 'test':
        return build_test_data(cfm_module, args)[0]
    cfg = copy.deepcopy(cfm_module.task_model.cfg.test_dataloader)
    ds = cfg['dataset']
    if not _point_at_data(ds, args.data_root, args.calibration_train_prefix, args.calibration_train_ann):
        if 'ADE20K' not in ds['type']:
            raise ValueError(f"--calibration_stats train: no training split is known for a {ds['type']}")
        ds['data_prefix'] = {'img_path': 'images/training', 'seg_map_path': 'annotations/training'}
    ds['indices'] = args.calibration_train_images
    return Runner.build_dataloader(cfg)


# ---------------------------------------------------------------------------
# Lambda sweep
# ---------------------------------------------------------------------------

def resolve_lambda_ref(cfm_module, fallback: float) -> float:
    """The lambda_ref the sweep is expressed relative to.

    Prefers the value carried by the loaded checkpoint (a codec parameter); falls back
    to the CLI value for fixed-rate / no-checkpoint runs where it isn't a parameter.
    """
    codec_params = dict(cfm_module.codec.named_parameters())
    if 'lambda_ref' in codec_params:
        value = float(codec_params['lambda_ref'].item())
        print(f'Using lambda_ref={value:.6g} from model for RD sweep')
        return value
    print(f'lambda_ref not a model parameter — using lambda_ref={fallback} for RD sweep')
    return fallback


def pick_task_metric(eval_result: Optional[dict]) -> tuple[Optional[str], Optional[float]]:
    """Reduce an mmengine evaluator result dict to (metric_name, value).

    Tries each entry of TASK_METRIC_KEYS in order; returns (None, None) if the dict holds
    no key this project knows how to report (a new task type gets added there).
    """
    if not eval_result:
        return None, None
    for name, key in TASK_METRIC_KEYS:
        if key in eval_result:
            return name, float(eval_result[key])
    return None, None


def sweep_lambdas(trainer, cfm_module, test_dl, lambda_range) -> Iterator[dict]:
    """Run one full test pass per lambda, yielding an RD point as each one finishes.

    Yields ``{'lambda': float, 'bpp': float|None, 'value': float|None, 'metric': str|None}``.
    Yielding (rather than returning a completed list) is what makes lambda-level resume
    possible: the caller persists each point before the next lambda starts.

    Stops iterating if the trainer is interrupted, so a caller can tell "swept every
    lambda" from "was cut short" by comparing counts against ``len(lambda_range)``.
    """
    for lmbda in lambda_range:
        lmbda = float(lmbda)
        print(f'Testing with lambda={lmbda:.4f}')
        cfm_module.lmbda = torch.tensor(lmbda, dtype=torch.float32)
        cfm_module.last_test_eval = None
        trainer.test(model=cfm_module, dataloaders=test_dl, ckpt_path=None)
        if trainer.interrupted:
            return

        metric, value = pick_task_metric(getattr(cfm_module, 'last_test_eval', None))
        bpp_key = f'full_test_bpp_{lmbda:.3f}'
        bpp = trainer.logged_metrics.get(bpp_key)
        if bpp is None:
            print(f'    Warning: no {bpp_key} in logged metrics — recording bpp=None')
        yield {
            'lambda': lmbda,
            'bpp': float(bpp) if bpp is not None else None,
            'value': value,
            'metric': metric,
        }


def run_baseline(trainer, cfm_module, test_dl) -> tuple[Optional[str], Optional[float]]:
    """Uncompressed reference score for the task model (rank 0 only; None elsewhere).

    Runs trainer.test() with the module at stage -1: the task model's own frontend and backend, no codec (see
    FixedRateCFM.test_step). The baseline does not depend on the codec, so callers measure it only when neither the
    results file nor the baseline store has it. `trainer`'s ModelEvaluator must hold the test evaluator.
    """
    print('Running uncompressed baseline...')
    prior_stage = cfm_module.stage
    cfm_module.stage = -1
    cfm_module.last_test_eval = None
    try:
        trainer.test(model=cfm_module, dataloaders=test_dl, ckpt_path=None)
    finally:
        cfm_module.stage = prior_stage
    if trainer.interrupted:
        print('Uncompressed baseline: interrupted before completing')
        return None, None
    metric, value = pick_task_metric(getattr(cfm_module, 'last_test_eval', None))
    print(f'Uncompressed results: {metric}={value}')
    return metric, value


# ---------------------------------------------------------------------------
# Feature calibration stats (cross-model scale-mismatch correction)
# ---------------------------------------------------------------------------

CALIBRATION_EPS = 1e-6  # avoids divide-by-zero in FeatureCalibration.forward/invert


class FeatureCalibration:
    """Test-time-only per-channel affine correction for cross-model feature-scale
    mismatch. Not an nn.Module -- nothing
    here is trained or persisted; a caller (test_cross_model.py) builds one per target from
    compute_feature_stats() output and assigns it directly to cfm_module.calibration.

    Duck-typed: anything with forward(x)/invert(x_hat) works as cfm_module.calibration
    (see trainers.FixedRateCFM._apply_calibration).
    """

    def __init__(self, mu_t, sigma_t, mu_s, sigma_s):
        """Each arg: a length-C list/array (from compute_feature_stats' 'mean'/'std').
        _t = the target's own stats, _s = the source codec's own stats."""
        def _as_tensor(x):
            return torch.as_tensor(x, dtype=torch.float32).view(1, -1, 1, 1)
        self.mu_t, self.sigma_t = _as_tensor(mu_t), _as_tensor(sigma_t)
        self.mu_s, self.sigma_s = _as_tensor(mu_s), _as_tensor(sigma_s)

    def _to(self, x):
        return self.mu_t.to(x.device), self.sigma_t.to(x.device), self.mu_s.to(x.device), self.sigma_s.to(x.device)

    def forward(self, x):
        """Target's raw scale -> source's calibrated scale. z-score using the TARGET's own
        per-channel stats, then re-express in the SOURCE codec's own per-channel scale --
        the codec itself runs completely unmodified on the result."""
        mu_t, sigma_t, mu_s, sigma_s = self._to(x)
        return (x - mu_t) / (sigma_t + CALIBRATION_EPS) * sigma_s + mu_s

    def invert(self, x_hat):
        """Inverse of forward, applied to the codec's reconstructed output before it
        reaches the target's own downstream layers."""
        mu_t, sigma_t, mu_s, sigma_s = self._to(x_hat)
        return (x_hat - mu_s) / (sigma_s + CALIBRATION_EPS) * sigma_t + mu_t


def compute_feature_stats(cfm_module, test_dl, max_batches: Optional[int] = None) -> dict:
    """Per-channel mean/std of the raw cut-point tensor, over every pixel of every image
    in test_dl. Never touches the codec -- just frontend_preprocessor + feature_frontend
    (cfm_module._prepare_batch, the exact extraction the RD sweep itself uses) -- so this
    is cheap relative to the lambda sweep that follows it: no g_a/g_s, no entropy coding.

    Streaming (sum/sumsq accumulated in float64 for numerical stability, never holding the
    full dataset's activations in memory at once) rather than a single big tensor, since
    this is meant to run over the full validation set, not a small sample.

    max_batches: stop after this many batches (None: the whole of test_dl). This function
    iterates test_dl directly rather than through trainer.test(), so no trainer-side batch
    limit applies here.

    Caller is responsible for cfm_module already being on the right device (this calls
    cfm_module.transfer_batch_to_device per batch, matching what pl.Trainer would do, but
    doesn't move the module itself -- this function is used outside of trainer.test()).
    """
    device = next(cfm_module.parameters()).device
    count = 0
    n_images = 0
    sum_ = sumsq = None
    with torch.no_grad():
        for i, batch in enumerate(test_dl):
            if max_batches is not None and i >= max_batches:
                break
            batch = cfm_module.transfer_batch_to_device(batch, device, 0)
            _, _, feat_tensor, _ = cfm_module._prepare_batch(batch, 'test')
            f = feat_tensor.double()
            b, c, h, w = f.shape
            if sum_ is None:
                sum_ = torch.zeros(c, dtype=torch.float64, device=device)
                sumsq = torch.zeros(c, dtype=torch.float64, device=device)
            sum_ += f.sum(dim=(0, 2, 3))
            sumsq += (f ** 2).sum(dim=(0, 2, 3))
            count += b * h * w
            n_images += b
    mean = sum_ / count
    var = (sumsq / count - mean ** 2).clamp(min=0)
    std = var.sqrt()
    return {'mean': mean.float().cpu().tolist(), 'std': std.float().cpu().tolist(),
           'n_images': n_images}


# ---------------------------------------------------------------------------
# Cross-model compatibility checks
# ---------------------------------------------------------------------------

def codec_input_signature(uses_swin: bool, codec_args: dict) -> tuple[int, int]:
    """(ndim, channels) the *codec's own weights* expect at their input.

    Token-domain codecs (any SWIN backbone) take (B, N, C) with C = embed_dims; spatial
    codecs take (B, C, H, W) with C = feature_chs. Derived from the source's codec args
    rather than from a model, so the source model never needs building to know this.
    """
    if uses_swin:
        return 3, int(codec_args['embed_dims'])
    return 4, int(codec_args['feature_chs'])


def probe_cut_point_signature(task_model, device, image_h: int = 512, image_w: int = 512
                              ) -> tuple[int, int]:
    """(ndim, channels) of the real tensor a task model emits at its cut point.

    Runs the exact calls the test loop makes — frontend_preprocessor then
    feature_frontend — on a synthetic single-image batch. Authoritative, unlike the
    name-substring heuristic in ``get_embed_dims``.
    """
    batch = _make_probe_batch(1, image_h, image_w, device)
    task_model = task_model.to(device)
    # This forward is on synthetic images, so it must not touch the task model's norm statistics: eval mode, restored
    # on exit through the split model's patched train(), which keeps the norms frozen.
    with task_model_mode(task_model, training=False):
        with torch.no_grad():
            data = task_model.frontend_preprocessor(batch, False)
            feats = task_model.feature_frontend(data, preprocessed=True)
            tensor = feats['inputs'].outs[0]
    if tensor.ndim == 3:
        return 3, int(tensor.shape[-1])
    if tensor.ndim == 4:
        return 4, int(tensor.shape[1])
    raise ValueError(f'Unexpected cut-point tensor rank {tensor.ndim} (shape {tuple(tensor.shape)})')


def name_heuristic_compatible(source_task_model: str, target_task_model: str,
                              cut_point: int) -> tuple[bool, str]:
    """Tier-1 pre-filter — rejects obvious mismatches without building any model.

    Three cheap checks: same task family (a detector's COCO dataloader/evaluator can't
    score a segmentor, and one --data_root can't point at both datasets), same backbone
    style, and — for SWIN — the same embed_dims the name heuristic reports. Passing this
    proves nothing; the tensor probe is the real gate.
    """
    from cfm_model_zoo.directory import backbone_style, detectors, segmentors
    from .utils import get_embed_dims

    if target_task_model not in backbone_style:
        return False, f'unknown task_model {target_task_model!r}'

    def family(name):
        return 'segmentation' if name in segmentors else ('detection' if name in detectors else None)

    src_family, tgt_family = family(source_task_model), family(target_task_model)
    if src_family != tgt_family:
        return False, (f'task family mismatch: source={src_family}, target={tgt_family} '
                       f'(different dataset and evaluator — run those as separate jobs)')

    src_style = backbone_style.get(source_task_model)
    tgt_style = backbone_style[target_task_model]
    if src_style != tgt_style:
        return False, f'backbone_style mismatch: source={src_style}, target={tgt_style}'

    if tgt_style == 'SWIN':
        src_dims = get_embed_dims(source_task_model, cut_point)
        tgt_dims = get_embed_dims(target_task_model, cut_point)
        if src_dims != tgt_dims:
            return False, f'embed_dims mismatch: source={src_dims}, target={tgt_dims}'

    return True, ''


# ---------------------------------------------------------------------------
# Results file
# ---------------------------------------------------------------------------
def results_file(results_dir: Optional[str], file_id: str, default_dir: Path) -> Optional[Path]:
    """The results file '<results_dir>/<file_id>.json', or None when writing results is disabled.

    Inputs:
        results_dir: --results_dir. None: default_dir (relative to the working directory). '', 'none', 'null',
            'false' or '0': disabled, results only printed (the caller warns). Anything else: used as given.
        file_id: '<task_model>__<checkpoint stem>', fixed per checkpoint (never per launch), so a relaunch finds
            its file again.
        default_dir: RD_SWEEP_RESULTS_DIR or CROSS_MODEL_RESULTS_DIR.
    """
    if results_dir is None:
        return Path(default_dir) / f'{file_id}.json'
    text = str(results_dir).strip()
    if not text or text.lower() in ('none', 'null', 'false', '0'):
        return None
    return Path(text) / f'{file_id}.json'


# A save holds the results lock for milliseconds, so a lock this old was left by a job that died holding it.
RESULTS_LOCK_STALE_SECONDS = 60


@contextmanager
def results_lock(path: Path, poll: float = 0.2):
    """Exclusive lock on a results file that several jobs save into: ``<file>.lock``, created with
    O_CREAT|O_EXCL, which only one process can win, also across the nodes of a cluster's shared
    filesystem (flock/fcntl locks are not reliable there). The lock records host, pid and time, for
    whoever finds one left behind. Waiters retry every ~``poll`` seconds; a lock older
    than RESULTS_LOCK_STALE_SECONDS is removed as a dead job's. Another node's process cannot be
    checked, so age is the only test."""
    lock = path.with_name(path.name + '.lock')
    note = json.dumps({'host': socket.gethostname(), 'pid': os.getpid(), 'time': time.time()})
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            _break_if_stale(lock)
            time.sleep(poll * (0.5 + random.random()))
            continue
        with os.fdopen(fd, 'w') as f:
            f.write(note)
        mine = os.stat(lock).st_ino
        break
    try:
        yield
    finally:
        try:   # remove it only if it is still ours (it is not if we held it so long it was broken as stale)
            if os.stat(lock).st_ino == mine:
                os.unlink(lock)
        except FileNotFoundError:
            pass


def _break_if_stale(lock: Path) -> None:
    """Remove ``lock`` if it is older than RESULTS_LOCK_STALE_SECONDS. It is renamed aside first, so of
    several waiters only one removes it; if a fresh lock replaced it in the meantime, that one is put back."""
    try:
        st = os.stat(lock)
    except FileNotFoundError:
        return
    age = time.time() - st.st_mtime
    if age <= RESULTS_LOCK_STALE_SECONDS:
        return
    aside = lock.with_name(f'{lock.name}.stale.{socket.gethostname()}.{os.getpid()}')
    try:
        os.rename(lock, aside)
    except FileNotFoundError:
        return
    try:
        if os.stat(aside).st_ino != st.st_ino:   # not the stale lock: a job took a fresh one meanwhile
            try:
                os.link(aside, lock)
            except FileExistsError:
                pass
            return
        print(f'Removed a results lock left by a dead job ({age:.0f}s old): {aside.read_text()!r}')
    finally:
        aside.unlink(missing_ok=True)


def save_results(path: Path, doc: dict, merge=None) -> None:
    """Atomic write — a preemption mid-save must not corrupt the resume file.

    With ``merge`` (results files that several jobs save into), the write happens under results_lock and
    writes ``merge(on_disk, doc)``: on_disk is the file as it is now (None if there is none yet), so
    what other jobs saved since this job read the file is kept."""
    if merge is None:
        _write_json_atomically(path, doc)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with results_lock(path):
        on_disk = None
        if path.exists():
            with open(path) as f:
                on_disk = json.load(f)
        _write_json_atomically(path, merge(on_disk, doc))


def _write_json_atomically(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix='.tmp')
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(doc, f, indent=2)
            f.flush()
            os.fsync(f.fileno())   # data stored before the rename: a node crash leaves the old or the new file, never an empty one
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def checkpoint_identity(checkpoint_path: str) -> dict:
    """A checkpoint's identity for results-file resume: size in bytes and sha256 (what is compared), plus its real
    path (recorded only to name the checkpoint in a refusal; the same bytes staged elsewhere still resume)."""
    real = os.path.realpath(checkpoint_path)
    digest = hashlib.sha256()
    with open(real, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            digest.update(block)
    return {'realpath': real, 'size': os.path.getsize(real), 'sha256': digest.hexdigest()}


def load_results_for_checkpoint(path: Path, identity: dict) -> Optional[dict]:
    """The results doc at ``path`` to resume, or None when there is none yet.

    The file name (<task_model>__<checkpoint stem>) does not identify the weights: two training
    runs that both saved last.ckpt share it. So a doc is resumed only when the identity it
    recorded (checkpoint_identity, stored under 'checkpoint_identity' when the doc was created)
    has this checkpoint's size and sha256; the path is not compared, so a copy of the same file
    elsewhere (e.g. a per-job scratch copy) resumes. A doc recorded for other bytes, or one that
    recorded no identity, stops the job: resuming it would mix RD points measured on other weights into this
    sweep, and starting afresh would silently overwrite them.
    """
    if not path.exists():
        return None
    try:
        with open(path) as f:
            doc = json.load(f)
    except json.JSONDecodeError:
        print(f'Warning: {path} is not valid JSON (truncated write?) -- starting fresh')
        return None

    def describe(identity):
        return f"{identity.get('realpath')} ({identity.get('size')} bytes, sha256 {identity.get('sha256')})"

    recorded = doc.get('checkpoint_identity')
    if recorded is not None and all(recorded.get(k) == identity[k] for k in ('size', 'sha256')):
        return doc
    if recorded is None:
        old = doc.get('checkpoint') or doc.get('source_checkpoint') or 'an unrecorded checkpoint'
        what = f'records no checkpoint identity (it was written for {old})'
    else:
        what = f'was recorded for checkpoint {describe(recorded)}'
    raise SystemExit(
        f'Results file {path} {what}, but this sweep is of {describe(identity)}. Its RD points do not '
        f'belong to these weights, so it is not resumed. Pass a different --results_dir, or delete or move '
        f'{path}.')


# ---------------------------------------------------------------------------
# Uncompressed-baseline store
# ---------------------------------------------------------------------------
#
# A task model's uncompressed score doesn't depend on any codec, so one number per task model
# serves every job. The store is a JSON file keyed by task_model ({metric, value, source}),
# relative to the working directory. Jobs only READ it.

BASELINE_STORE = Path('data/logs/baselines.json')


def expected_metric(task_model: str) -> str:
    """The short metric name (TASK_METRIC_KEYS) this task model's evaluator reports."""
    from cfm_model_zoo.directory import segmentors
    return 'mIoU' if task_model in segmentors else 'mAP'


def lookup_baseline(task_model: str, metric: str, path: Optional[Path] = None) -> Optional[dict]:
    """{'metric', 'value', 'source'} from the baseline store, or None (with the reason printed).

    A stored value is used only if it is for this metric and positive.
    """
    path = Path(path) if path is not None else BASELINE_STORE
    entry, why = None, None
    if not path.exists():
        why = 'no store file'
    else:
        try:
            with open(path) as f:
                entry = json.load(f).get(task_model)
        except (OSError, json.JSONDecodeError) as e:
            why = f'unreadable ({e!r})'
        if why is None and not entry:
            why = 'no entry'
    if why is None:
        value = entry.get('value')
        if entry.get('metric') != metric or value is None or not float(value) > 0:
            why = f'unusable entry {entry} (expected a positive {metric})'
    if why:
        print(f'No stored baseline for {task_model} in {path}: {why}')
        return None
    return {'metric': metric, 'value': float(value),
            'source': f"store {path}: {entry.get('source', '')}".rstrip(': ')}


# ---------------------------------------------------------------------------
# Grid sweep (test_cross_model --sweep grid)
# ---------------------------------------------------------------------------

def compute_lambda_range(lambda_ref: float, lambda_min: float, lambda_max: float,
                         lambda_space: str, steps: int) -> np.ndarray:
    """lambda_min/max are ratios relative to lambda_ref (same convention as training).

    Returned ascending; callers sweep it reversed (largest lambda first, cheapest to
    interrupt early).
    """
    lambda_lo = lambda_ref * lambda_min
    lambda_hi = lambda_ref * lambda_max
    if lambda_space == 'linear':
        return np.linspace(lambda_lo, lambda_hi, steps, dtype=np.float32)
    return np.logspace(np.log10(lambda_lo), np.log10(lambda_hi), steps, dtype=np.float32)


def completed_lambda_keys(entry: Optional[dict]) -> set[str]:
    if not entry:
        return set()
    return {lambda_key(pt['lambda']) for pt in entry.get('rd_points') or []}
