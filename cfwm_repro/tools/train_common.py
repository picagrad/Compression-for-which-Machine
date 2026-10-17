"""Shared argument groups and setup helpers for the training entry point, train_variable_rate.py
(VariableRateCFM); the test scripts import some of them too.
"""

import argparse
import logging
import os
import warnings

# mmdet's SwinBlock calls torch.utils.checkpoint.checkpoint() without use_reentrant, and PyTorch warns about it on
# every call (still only a warning; tracked upstream as mmengine issue #1539). Silence that one warning.
warnings.filterwarnings(
    'ignore',
    message='.*use_reentrant.*',
    category=UserWarning,
    module='torch',
)

import numpy as np
import pytorch_lightning as pl
import torch
from mmengine.logging import MMLogger
from mmengine.runner import Runner
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies import SingleDeviceStrategy
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau

from cfm_model_zoo.directory import LEGACY_NAMES, backbone_style, detectors, resolve_download_path, segmentors
from .callbacks import ChangeModelStage, ModelEvaluator, LambdaRangeWarmup
from .checkpoint_io import checkpoint_file, require_checkpoint
from .utils import MlFLowBoolWorkaround, get_codec_type, get_embed_dims, make_safe_pipeline

# At import, so it runs for the test scripts too (they import this module).
torch.set_float32_matmul_precision('high')

# Used to keep the channel-group ratios identical to ELIC's (these are its groups for M=320) when M changes:
# codec_class_and_args scales them to codec_bottleneck_chs (M=120 -> [6, 6, 12, 24, 72]), drops empty groups and gives the
# rounding remainder to the last group.
GROUPS_RATIO_FOR_ELIC = np.array([16, 16, 32, 64, 192])


# ---------------------------------------------------------------------------
# The command-line surface: the frozen publication settings and the argument groups
# ---------------------------------------------------------------------------

# The publication runs' value for every setting they all used with ONE value. The rule: a setting the publication
# runs used with only one value is not a CLI flag. parser.set_defaults(**FROZEN_*) puts each into the args
# namespace, so vars(args), the MLflow params and every args.x read keep working, while passing it on the
# command line is an "unrecognized arguments" error. A setting becomes a flag again when we choose to expose it.
FROZEN_CODEC = dict(
    codec_family='elfc',  # The only family
    var_rate='gqvrf', cut_point=1, codec_down_strides='2', codec_bottleneck_chs=120, codec_mid_chs=192,
    feature_chs=256,
    embed_dims=96,        # logged only: the codec takes get_embed_dims(task_model), 192 for SWIN-L
    shuffle_ratio=1)
FROZEN_RUN = dict(
    precision='32-true',  # Train AND test (bf16 roughly halved an iterative decode head's mIoU)
    strategy='single', interactive=False, mmdet_log_level='WARNING', deterministic_algorithms_warn_only=True)
FROZEN_TRAIN = dict(
    **FROZEN_CODEC, **FROZEN_RUN,
    startup=5, lr=1e-4, lr_warmup_epochs=-1, lr_patience=5, lr_factor=0.25, lambda_range_ramp_epochs=-1,
    grad_clip=0.1, seed=1987, val_batch_size=1, log_interval=200,
    val_inference=True, val_inference_freq=5, full_test=True,
    trn_prefix='train2017', val_prefix='train2017', test_prefix='val2017',
    trn_ann='annotations/coco2017_trn_split_anns.json', val_ann='annotations/coco2017_val_split_anns.json',
    test_ann='annotations/instances_val2017.json', image_h=512, image_w=512, crop_h=512, crop_w=512)
FROZEN_TEST = dict(
    **FROZEN_CODEC, **FROZEN_RUN,
    # The training band the published sweeps logged (0.01, 2.0). Inert at test time: it only builds the codec's
    # lambda_ratio, and the test path always passes an explicit lambda (trainers.VariableRateCFM._compute_codec_output),
    # so it is never a flag (it would be a silent no-op).
    lambda_min=0.01, lambda_max=2.0,
    test_prefix='val2017', test_ann='annotations/instances_val2017.json')
FROZEN_CROSS = dict(
    **FROZEN_CODEC, **FROZEN_RUN, image_h=512, image_w=512,
    retry_skipped=False, test_prefix='val2017', test_ann='annotations/instances_val2017.json',
    # --calibration_stats train: the first 2000 training images (COCO: in instances_train2017.json's order)
    calibration_train_prefix='train2017', calibration_train_ann='annotations/instances_train2017.json',
    calibration_train_images=2000)


def _int_or_auto(v):
    """Argument type that accepts a positive integer or the string 'auto'."""
    if isinstance(v, str) and v.lower() == 'auto':
        return 'auto'
    try:
        return int(v)
    except (ValueError, TypeError):
        raise argparse.ArgumentTypeError(f"Expected a positive integer or 'auto', got: {v!r}")


# The task models are named by their CFM-Model-Zoo short names only (the library's LEGACY_NAMES): a full registered
# name would build the wrong codec for a Swin-L model, which get_embed_dims recognises by its short name.
TASK_MODELS = tuple(name for name in LEGACY_NAMES if name in segmentors or name in detectors)


def add_task_model_arg(parser, *, required: bool = True, help: str = 'Task model (a CFM-Model-Zoo short name)'):
    parser.add_argument('--task_model', choices=TASK_MODELS, required=required, default=None, help=help)


def task_model_list(text: str) -> str:
    """An argparse type: comma-separated task-model short names, returned unchanged; any other name is an error."""
    unknown = [t.strip() for t in text.split(',') if t.strip() and t.strip() not in TASK_MODELS]
    if unknown:
        raise argparse.ArgumentTypeError(f"not a task-model short name: {', '.join(unknown)} (e.g. FasterRCNN, "
                                         'SegUperNetSWIN; the list is in the CFM-Model-Zoo README)')
    return text


def add_schedule_args(parser):
    g = parser.add_argument_group('Schedule')
    g.add_argument('-e', '--epochs', type=int, default=10,
                   help='Number of training epochs (basic recipe: 100 (detection), 200 (segmentation))')
    g.add_argument('--lr_scheduler', choices=['wsd', 'plateau'], default='plateau',
                   help="'wsd' = warmup-stable-decay: linear warmup over the warmup stage, a long stable phase, "
                        'then a linear decay to --lr_min_factor * lr over the last --lr_decay_frac of the run. '
                        "'plateau' = ReduceLROnPlateau on val_loss, active from stage 2 (basic recipe: wsd)")
    g.add_argument('--lr_decay_frac', type=float, default=0.2,
                   help='wsd: fraction of the FULL run spent decaying at the end (forced to start inside stage 2) '
                        '(basic recipe: 0.3)')
    g.add_argument('--lr_min_factor', type=float, default=0.1,
                   help='wsd: final LR as a fraction of the initial LR (basic recipe: 0.05)')
    g.add_argument('--lambda_warmup_epochs', type=int, default=0,
                   help='Length of stage 1 in epochs, over which the lambda-sampling range widens to its target; '
                        '0 skips stage 1 (basic recipe: 5)')
    g.add_argument('--train_batch_size', type=_int_or_auto, default='auto', metavar='N|auto',
                   help='Training batch size, or "auto" (the largest that fits in ~75%% of GPU memory)')


def add_path_args(parser, *, kind: str, output_default=None):
    """Training: where THIS run saves (--checkpoint_dir, --checkpoint_name) and where it starts from (--init_checkpoint,
    --restart; order in train_common.resolve_training_start). Testing: the checkpoint to test (--checkpoint)."""
    g = parser.add_argument_group('Paths')
    g.add_argument('--data_root', type=str, required=True,
                   help='COCO root (annotations/, train2017/, val2017/) or the ADEChallengeData2016 directory')
    g.add_argument('--output', type=str, default=output_default, help='Working directory')
    g.add_argument('--mim_cache_dir', type=str, default=None,
                   help='Cache for the task models\' configs and weights (default: CFM_MODEL_ZOO_CACHE_DIR, '
                        'else mim\'s own cache)')
    if kind == 'train':
        g.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                       help='Where this run saves its checkpoint, <checkpoint_dir>/<checkpoint_name>.ckpt, every epoch; '
                            'if that file exists at startup the run fully resumes from it')
        g.add_argument('--checkpoint_name', type=str, default='last', help='The checkpoint file name without .ckpt')
        g.add_argument('--init_checkpoint', type=str, default='',
                       help='Start from this checkpoint (used only when <checkpoint_dir>/<checkpoint_name>.ckpt does '
                            'not exist): its full training state, or with --restart its weights only')
        g.add_argument('--restart', action=MlFLowBoolWorkaround, default=False, metavar='True|False',
                       help='True or False. With --init_checkpoint: load its weights only (fresh optimiser and '
                            'scheduler, epoch 0)')
    else:
        g.add_argument('--checkpoint', type=str, required=True, help='The codec checkpoint (.ckpt) to test')


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def configure_mmdet_logging(args):
    """Suppress mmengine/mmdet/transformers verbose INFO output."""
    _level = getattr(logging, args.mmdet_log_level)
    mm_logger = MMLogger.get_instance('mmengine', log_level=args.mmdet_log_level)
    mm_logger.setLevel(_level)
    for _h in mm_logger.handlers:
        _h.setLevel(_level)

    import mmengine.logging.logger as _mml
    _orig_print_log = _mml.print_log
    def _quiet_print_log(msg, logger=None, level=logging.INFO):
        _orig_print_log(msg, logger='current' if logger is None else logger, level=level)
    _mml.print_log = _quiet_print_log
    import mmengine.runner.checkpoint as _chkpt
    _chkpt.print_log = _quiet_print_log
    del _mml, _chkpt

    try:
        import transformers as _tr
        _tr_level = (_tr.logging.ERROR if _level >= logging.WARNING
                     else getattr(_tr.logging, args.mmdet_log_level, _tr.logging.WARNING))
        _tr.logging.set_verbosity(_tr_level)
        if _level >= logging.WARNING:
            _tr.logging.disable_progress_bar()
        del _tr, _tr_level
    except ImportError:
        pass
    del _level


def apply_seed(args):
    if args.seed >= 0:
        pl.seed_everything(args.seed, workers=True)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        logging.info(f'Setting random seed to {args.seed}')


def resolve_codec_strides(codec_down_strides) -> list:
    """Parse --codec_down_strides into a full stride list.

    A plain int N (or numeric string, for backward compatibility with existing int-typed
    call sites/launch yamls) generates the original [1] + [2]*N pattern. A comma-separated
    string (e.g. '1,1,2') is used verbatim as a custom stride list, for patterns the auto-
    generated form can't express (e.g. downsampling less aggressively at one stage).
    """
    if isinstance(codec_down_strides, str) and ',' in codec_down_strides:
        return [int(s) for s in codec_down_strides.split(',') if s.strip()]
    return [1] + [2] * int(codec_down_strides)


def codec_class_and_args(args):
    """The codec class and its arguments (without the lambda arguments): the GQVRF ELFC codec on a (B, C, H, W)
    feature map for every backbone; for a SWIN backbone the codec reshapes the tokens through UnProject and also
    takes the backbone's embed_dims (192 for SWIN-L, whatever the logged embed_dims). Returns (codec_class, codec_args).
    """
    uses_swin = backbone_style[args.task_model] == 'SWIN'
    codec_cls = get_codec_type(uses_swin, args.codec_family, args.var_rate)
    down_strides = resolve_codec_strides(args.codec_down_strides)
    codec_args = {
        'analysis_strides': down_strides,
        'synthesis_strides': down_strides,
        'M': args.codec_bottleneck_chs, 'N': args.codec_mid_chs, 'feature_chs': args.feature_chs,
    }
    groups = GROUPS_RATIO_FOR_ELIC * args.codec_bottleneck_chs // GROUPS_RATIO_FOR_ELIC.sum()
    groups = [gr for gr in groups if gr > 0]
    groups[-1] = args.codec_bottleneck_chs - sum(groups[:-1])
    codec_args['groups'] = groups
    if uses_swin:
        codec_args['embed_dims'] = get_embed_dims(args.task_model, args.cut_point)
    codec_args['scale_per_group'] = True   # GQVRF: one rate scale per channel group
    return codec_cls, codec_args


def task_model_name_and_args(args):
    """(task_model_name, task_model_args) for cfm_model_zoo.directory.build_split_model (keyword-only after cut_point).
    Every model resolves with source='mim': OpenMMLab's files through mim, and the three models mim cannot serve
    (RepPointsV2, DDQDETRResNet, DDQDETRSWIN-L) from the library's shared-checkpoint entries, which download their
    publishers' files and check size and sha256."""
    return args.task_model, {'cut_point': args.cut_point, 'source': 'mim',
                             'download_path': resolve_download_path(args.mim_cache_dir)}


def build_sched_dict(args):
    if args.lr_scheduler == 'plateau':
        return {'scheduler': ReduceLROnPlateau, 'factor': args.lr_factor,
                'patience': args.lr_patience, 'mode': 'min'}
    # wsd: _active_stage=0, the schedule spans the WHOLE run (warmup during stage 0)
    return {'scheduler': LambdaLR, 'lr_lambda': make_wsd_lambda(args), '_active_stage': 0}


def make_wsd_lambda(args):
    """Warmup-Stable-Decay (WSD; Hu et al., 2024, "MiniCPM: Unveiling the Potential of Small Language Models with
    Scalable Training Strategies", arXiv:2404.06395) aligned to the STAGE MACHINE, over the full epoch range.

        stage 0            stage 1                 stage 2
        |<-- warmup -->|<------- stable ------------------>|<-- decay -->|
        0           startup                            decay_start    epochs

    Warmup spans stage 0 (the fixed-lambda codec warmup). Stable holds through all of
    stage 1 (where LambdaRangeWarmup is widening the lambda range -- the LR must not be
    moving at the same time) and well into stage 2. Decay only begins deep in stage 2.

    This schedule is NOT wrapped in DelayedLRScheduler (see _active_stage=0 above): that
    wrapper keeps a scheduler dormant until stage 2, which would make a stage-0 warmup
    impossible and would also reset the epoch counter mid-run.
    """
    # Everything below is an EPOCH INDEX, not a count. Lightning runs epochs 0..last_epoch
    # inclusive, so last_epoch -- not `total` -- is the right end of the schedule (ending at
    # `total` would leave the decay one step short of min_factor).
    total = max(int(args.epochs), 1)             # a COUNT (what the user passes as --epochs)
    last_epoch = total - 1                       # the last epoch INDEX that actually runs
    startup = int(getattr(args, 'startup', 0) or 0)
    intermediate = int(getattr(args, 'lambda_warmup_epochs', 0) or 0)
    warmup = int(args.lr_warmup_epochs) if int(args.lr_warmup_epochs) >= 0 else startup
    warmup = max(min(warmup, last_epoch), 1)     # warmup occupies indices 0..warmup-1
    stage2_start = startup + intermediate        # first index of stage 2
    # decay must land inside stage 2, and must leave at least one stable epoch after warmup
    decay_start = int(round((1.0 - float(args.lr_decay_frac)) * total))
    decay_start = max(decay_start, stage2_start + 1, warmup + 1)
    decay_start = min(decay_start, last_epoch)
    min_factor = float(args.lr_min_factor)

    def wsd(epoch: int) -> float:
        if epoch < warmup:
            # stage 0: linear warmup over indices 0..warmup-1, reaching 1.0 ON index
            # warmup-1 (the last warmup epoch), so stage 1 opens already at full LR.
            return float(epoch + 1) / float(warmup)
        if epoch < decay_start:                  # stage 1 + early stage 2: stable
            return 1.0
        # late stage 2: linear decay over the decay_epochs indices
        # decay_start..last_epoch. Same "+1" convention as the warmup above: the FIRST
        # epoch of the phase has already made one step through it, so epoch decay_start
        # is already below 1.0 (the last stable epoch, decay_start-1, is the 1.0 anchor)
        # and last_epoch lands exactly on min_factor. Anchoring 1.0 on decay_start
        # instead would waste that epoch as a second flat one and force the remaining
        # steps to be steeper -- --lr_decay_frac asks for decay_epochs epochs of decay,
        # so all decay_epochs of them must actually decay.
        decay_epochs = last_epoch - decay_start + 1
        frac = min(float(epoch - decay_start + 1) / float(decay_epochs), 1.0)
        return 1.0 + frac * (min_factor - 1.0)

    wsd.total, wsd.warmup, wsd.decay_start = total, warmup, decay_start
    wsd.last_epoch = last_epoch
    wsd.stage2_start, wsd.min_factor = stage2_start, min_factor
    return wsd


# The batch-size probe below is kept identical to the code the published runs used (it uses CUDA randomness, so
# editing it can change a run); do not edit it.
def _make_probe_batch(batch_size, image_h, image_w, device):
    """Create a minimal dummy batch for _probe_batch_size.

    Tries mmdet DetDataSample first, then mmseg SegDataSample, then a plain
    namespace fallback used by tests that swap in _FakeCFMForProbe.
    """
    try:
        import numpy as np
        from mmdet.structures import DetDataSample
        from mmdet.structures.mask import BitmapMasks
        from mmengine.structures import InstanceData
        data_samples = []
        for _ in range(batch_size):
            ds = DetDataSample()
            ds.set_metainfo({'img_shape': (image_h, image_w),
                             'ori_shape': (image_h, image_w),
                             'scale_factor': (1.0, 1.0)})
            gt = InstanceData()
            gt.bboxes = torch.zeros((0, 4))
            gt.labels = torch.zeros(0, dtype=torch.long)
            gt.masks = BitmapMasks(np.zeros((0, image_h, image_w), dtype=np.uint8),
                                   image_h, image_w)
            ds.gt_instances = gt
            data_samples.append(ds)
    except ImportError:
        try:
            from mmseg.structures import SegDataSample
            data_samples = []
            for _ in range(batch_size):
                ds = SegDataSample()
                ds.set_metainfo({'img_shape': (image_h, image_w),
                                 'ori_shape': (image_h, image_w),
                                 'scale_factor': (1.0, 1.0)})
                data_samples.append(ds)
        except ImportError:
            from types import SimpleNamespace as _NS
            data_samples = [_NS(metainfo={'scale_factor': (1.0, 1.0)}) for _ in range(batch_size)]
    return {
        'inputs': [torch.zeros(3, image_h, image_w, device=device) for _ in range(batch_size)],
        'data_samples': data_samples,
    }


def _is_out_of_memory(exc: BaseException) -> bool:
    """A genuine allocator out-of-memory: torch's OutOfMemoryError, or a RuntimeError carrying the
    allocator's 'out of memory' wording (e.g. 'CUDA error: out of memory')."""
    return isinstance(exc, torch.cuda.OutOfMemoryError) or (
        isinstance(exc, RuntimeError) and 'out of memory' in str(exc))


@pl.utilities.rank_zero_only
def _probe_batch_size(cfm_module, args, cfg_tr=None, target_fraction=0.75):
    """Run the GPU memory probe on rank 0 only. Returns the found batch size int.

    Runs the exact same code path as a real training step (_prepare_batch +
    _compute_codec_output + _compute_losses + backward) so that distillation,
    gradient checkpointing, and all other training-time memory consumers are
    measured correctly.
    """
    if not torch.cuda.is_available():
        print('[auto batch size] No CUDA device available, defaulting to 4')
        return 4

    device = torch.device('cuda:0')
    total_mem = torch.cuda.get_device_properties(device).total_memory
    target_mem = int(total_mem * target_fraction)
    print(f'[auto batch size] Profiling on {torch.cuda.get_device_name(device)} '
          f'({total_mem / 1e9:.1f} GB total, target {target_fraction * 100:.0f}% = '
          f'{target_mem / 1e9:.2f} GB)')

    cfm_module.to(device)

    try:
        image_h = getattr(args, 'image_h', None) or getattr(args, 'crop_h', None) or 512
        image_w = getattr(args, 'image_w', None) or getattr(args, 'crop_w', None) or 512

        def probe(batch_size):
            """Returns peak GPU memory in bytes, or None on out-of-memory. Any other error
            propagates: treating it as "does not fit" hid real bugs behind batch size 1."""
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.empty_cache()
            cfm_module.codec.zero_grad(set_to_none=True)
            try:
                batch = _make_probe_batch(batch_size, image_h, image_w, device)
                data, feats, feat_tensor, N = cfm_module._prepare_batch(batch, 'train')
                comp_out = cfm_module._compute_codec_output(feat_tensor, mode='train')
                loss, *_ = cfm_module._compute_losses(comp_out, data, feats, 'train')
                if loss.requires_grad:
                    loss.backward()
                return torch.cuda.max_memory_allocated(device)
            except RuntimeError as e:          # torch.cuda.OutOfMemoryError is a RuntimeError
                if not _is_out_of_memory(e):
                    raise
                return None
            finally:
                torch.cuda.empty_cache()
                cfm_module.codec.zero_grad(set_to_none=True)

        # --- Extrapolate from BS=2 and BS=4 ---
        mem2 = probe(2)
        if mem2 is None:
            print('[auto batch size] OOM at batch size 2. Using 1.')
            return 1

        mem4 = probe(4)
        if mem4 is not None and mem4 > mem2:
            per_sample = (mem4 - mem2) / 2
            fixed = mem2 - 2 * per_sample
            estimated = max(1, int((target_mem - fixed) / per_sample))
        else:
            estimated = 2

        estimated = min(estimated, 256)  # sanity cap

        # --- Verify and back off via binary search if needed ---
        if estimated > 4 and probe(estimated) is None:
            lo, hi = 4, estimated - 1
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if probe(mid) is not None:
                    lo = mid
                else:
                    hi = mid - 1
            estimated = lo

        mem4_str = f'{mem4 / 1e9:.2f} GB' if mem4 is not None else 'OOM'
        print(f'[auto batch size] Probe: BS=2 → {mem2 / 1e9:.2f} GB | BS=4 → {mem4_str} '
              f'→ selected batch size: {estimated}')
        return estimated

    finally:
        cfm_module.cpu()
        torch.cuda.empty_cache()


def auto_find_batch_size(cfm_module, args, target_fraction=0.75):
    """Find the per-GPU training batch size that fills ~target_fraction of GPU memory.

    The probe runs on rank 0 only (@rank_zero_only). If torch.distributed is already
    initialised (e.g. when launched via torchrun), the result is broadcast to all
    other ranks so every process ends up with the same batch size. In our typical
    single-process-per-job setup (Lightning spawns DDP workers internally after
    trainer.fit()), this function is called exactly once and rank_zero_only is a no-op.

    Returns an int.
    """
    # NB: keyword, not positional -- the signature is (cfm_module, args, cfg_tr=None,
    # target_fraction=0.75), so a positional third argument would land in cfg_tr and leave
    # target_fraction at its default. The probe profiles on all-zero dummy images (the task model's norms are
    # already frozen, so they cannot move its running statistics).
    result = _probe_batch_size(cfm_module, args, target_fraction=target_fraction)  # None on non-rank-0
    # If distributed is already up (torchrun / multi-process launch), broadcast so
    # every rank gets the same value instead of each probing independently.
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        t = torch.tensor([result if result is not None else 0], dtype=torch.long)
        torch.distributed.broadcast(t, src=0)
        return int(t.item())
    return result if result is not None else 4


def _point_at_data(ds_cfg, data_root, prefix, ann) -> bool:
    """Point one dataset config at data_root, with this split's image prefix and annotation file.

    Inputs:
        ds_cfg: one mm dataset config (a dict, edited in place).
        data_root: the --data_root directory.
        prefix, ann: this split's image directory and annotation file under data_root (empty: keep the config's).
    Outputs:
        True for a COCO-style dataset (its evaluator then needs the annotation file); ODVG-format datasets keep their
        own paths, and ADE20K keeps the config's layout under data_root.
    """
    ds_cfg['data_root'] = data_root
    if ds_cfg['type'] == 'ODVGDataset' or 'coco' not in ds_cfg['type'].lower():
        return False
    ds_cfg['data_prefix'] = {'img': prefix} if prefix else ds_cfg['data_prefix']
    ds_cfg['ann_file'] = ann if ann else ds_cfg['ann_file']
    return True


def build_dataloaders(cfm_module, args):
    """Configure dataloaders from the task model cfg.

    Returns (train_dl, val_dl, test_dl, cfg_tr, use_coco); use_coco: whether the evaluators need COCO annotation files.
    """
    from mmdet.models.data_preprocessors.data_preprocessor import BatchFixedSizePad

    # Training pads every batch to a fixed size (BatchFixedSizePad, in training only). The training pipeline's last
    # size-changing step is the crop (make_safe_pipeline), so pad to the crop size; do it before the memory probe so
    # the probe sees the real training resolution, not the config's (often much larger) padded size.
    if 'batch_augments' in dir(cfm_module.task_model.frontend_preprocessor):
        if cfm_module.task_model.frontend_preprocessor.batch_augments is not None:
            for m in cfm_module.task_model.frontend_preprocessor.batch_augments:
                if isinstance(m, BatchFixedSizePad):
                    print(f'Changed preprocessing module {m} to use size '
                          f'{(args.crop_h, args.crop_w)} instead of {m.size}')
                    m.size = (args.crop_h, args.crop_w)

    if args.train_batch_size == 'auto':
        final_train_batch_size = auto_find_batch_size(cfm_module, args)
    else:
        final_train_batch_size = args.train_batch_size

    cfg_tr = cfm_module.task_model.cfg.copy()
    cfg_tr['work_dir'] = args.output
    # Point each split at --data_root with its image prefix and annotation file. Every published task model has one
    # dataset per split; train may be wrapped (RepeatDataset in FasterRCNN, ATSSSWIN-L and GLIP): train on the
    # dataset it wraps.
    train = cfg_tr['train_dataloader']
    if 'dataset' in train['dataset']:
        train['dataset'] = train['dataset']['dataset']
    use_coco = _point_at_data(train['dataset'], args.data_root, args.trn_prefix, args.trn_ann)
    use_coco = _point_at_data(cfg_tr['val_dataloader']['dataset'], args.data_root, args.val_prefix, args.val_ann) or use_coco
    use_coco = _point_at_data(cfg_tr['test_dataloader']['dataset'], args.data_root, args.test_prefix, args.test_ann) or use_coco

    cfg_tr['train_dataloader']['batch_size'] = final_train_batch_size
    cfg_tr['val_dataloader']['batch_size'] = args.val_batch_size
    assert cfg_tr['test_dataloader']['batch_size'] == 1, 'Test batch size must be 1 for now'

    print('Ensuring pipeline is safe in terms of output size:\n')
    cfg_tr['train_dataloader']['dataset']['pipeline'] = make_safe_pipeline(
        cfg_tr['train_dataloader']['dataset']['pipeline'], args, is_seg=args.task_model in segmentors)

    print('Preparing Data Loaders and Task Evaluators: \n')
    if 'sampler' in cfg_tr['train_dataloader']:
        cfg_tr['train_dataloader']['sampler']['type'] = (
            'DefaultSampler' if 'Infinite' in cfg_tr['train_dataloader']['sampler']['type']
            else cfg_tr['train_dataloader']['sampler']['type'])

    train_dataloader = Runner.build_dataloader(cfg_tr['train_dataloader'])
    val_dataloader = Runner.build_dataloader(cfg_tr['val_dataloader'])
    test_dataloader = Runner.build_dataloader(cfg_tr['test_dataloader'])

    return train_dataloader, val_dataloader, test_dataloader, cfg_tr, use_coco


def build_evaluators(cfg_tr, train_dl, test_dl, use_coco, data_root):
    """Build val and test evaluators. Returns (val_eval, test_eval)."""
    if use_coco:
        cfg_tr['val_evaluator']['ann_file'] = os.path.join(data_root, cfg_tr['val_dataloader']['dataset']['ann_file'])
        cfg_tr['test_evaluator']['ann_file'] = os.path.join(data_root, cfg_tr['test_dataloader']['dataset']['ann_file'])

    val_eval = Runner.build_evaluator(None, cfg_tr['val_evaluator'])
    setattr(val_eval, 'dataset_meta', train_dl.dataset.metainfo)
    test_eval = Runner.build_evaluator(None, cfg_tr['test_evaluator'])
    setattr(test_eval, 'dataset_meta', test_dl.dataset.metainfo)
    return val_eval, test_eval


def build_checkpoint_callback(args):
    """Training keeps ONE checkpoint, <checkpoint_dir>/<checkpoint_name>.ckpt (default name `last`), rewritten in place
    every epoch (the same settings as the publication runs' `last` instance). No `best` checkpoint: no published sweep
    tested one, and for segmentation it would be selected on the test images."""
    # enable_version_counter=False: always overwrite DIR/NAME.ckpt. With the counter, a run resumed from a moved or
    # copied directory saved NAME-v1.ckpt beside it, and every later relaunch resumed the stale NAME.ckpt.
    return ModelCheckpoint(dirpath=args.checkpoint_dir, filename=args.checkpoint_name, monitor=None, save_top_k=-1,
                           verbose=False, enable_version_counter=False)


def resolve_training_start(args):
    """How this training run starts: (resume_path, weights_only_path, description), in this order:
    (1) DIR/NAME.ckpt exists -> full resume from it; --init_checkpoint and --restart are ignored (a relaunch of a
        preempted run never goes back to its starting point);
    (2) else --init_checkpoint -> continue its full training state, or with --restart load its WEIGHTS ONLY (fresh
        optimiser and scheduler, epoch 0); this run saves to DIR/NAME.ckpt from then on;
    (3) else from scratch."""
    if args.restart and not args.init_checkpoint:
        raise SystemExit('--restart needs --init_checkpoint: it says to take only the weights of that checkpoint')
    own = checkpoint_file(args)
    if os.path.isfile(own):
        return own, None, f'resume: {own} exists, full training state (--init_checkpoint/--restart ignored)'
    if args.init_checkpoint:
        init = require_checkpoint(args.init_checkpoint)
        if args.restart:
            return None, init, f'restart: weights only from {init}; fresh optimiser and scheduler, epoch 0'
        return init, None, f'continue: full training state of {init}; saving to {own}'
    return None, None, f'scratch: no checkpoint; saving to {own}'


def build_base_callbacks(args, checkpoint_callback, intermediate_stage_epochs: int = 0,
                         lambda_range_ramp_epochs: int = None, val_evaluator=None, test_evaluator=None):
    """Build callbacks common to all scripts.

    Always includes ModelEvaluator and ChangeModelStage (the task model's norms are kept frozen by the split model
    itself). Appends checkpoint_callback if not None.

    intermediate_stage_epochs: duration of stage 1 passed to ChangeModelStage.
    0 skips stage 1 (transitions directly 0→2 at startup_epochs).
    Use a positive value for variable-rate training: it is a settling-in period
    after lambda sampling begins, before LR/inference/saving activate at stage 2. Whenever
    positive, a LambdaRangeWarmup callback is also added, widening the codec's
    lambda-sampling range toward the target ratio from args.lambda_min/lambda_max (see
    LambdaRangeWarmup's docstring).

    lambda_range_ramp_epochs: length of that widening RAMP, which need not equal
    stage 1. None (or -1) ramps across the whole of stage 1. A shorter ramp reaches the
    target range early and leaves the rest of stage 1 to settle. Clamped to
    intermediate_stage_epochs; a longer ramp would never finish.

    val_evaluator, test_evaluator: the task-metric evaluators, held by ModelEvaluator.
    """
    # LearningRateMonitor logs the optimizer's actual LR every epoch as 'lr-<OptimName>': without it, whether (and
    # when) ReduceLROnPlateau fired cannot be read from the logs. Cheap, and the only thing that makes the schedule
    # auditable.
    callbacks = [ModelEvaluator(val_evaluator=val_evaluator, test_evaluator=test_evaluator),
                 ChangeModelStage(args.startup, intermediate_stage_epochs),
                 LearningRateMonitor(logging_interval='epoch')]
    if lambda_range_ramp_epochs is None:
        # NB: no `or -1` here -- 0 is falsy but meaningful (stage 1 with no widening at all).
        v = getattr(args, 'lambda_range_ramp_epochs', -1)
        lambda_range_ramp_epochs = -1 if v is None else int(v)
    ramp = (intermediate_stage_epochs if lambda_range_ramp_epochs < 0
            else min(lambda_range_ramp_epochs, intermediate_stage_epochs))
    if intermediate_stage_epochs > 0 and ramp > 0:
        callbacks.append(LambdaRangeWarmup(
            stage1_start_epoch=args.startup,
            duration_epochs=ramp,
            target_ratio=(args.lambda_min, args.lambda_max),
        ))
    if checkpoint_callback is not None:
        callbacks.append(checkpoint_callback)
    return callbacks


def build_trainer(args, callbacks, logger):
    return pl.Trainer(
        max_epochs=args.epochs, callbacks=callbacks, logger=logger, precision=args.precision,
        strategy=SingleDeviceStrategy(device='cuda:0'), num_sanity_val_steps=0,
        log_every_n_steps=args.log_interval, enable_progress_bar=args.interactive,
        gradient_clip_val=args.grad_clip)


def run_full_test(trainer, cfm_module, test_dl, args):
    torch.use_deterministic_algorithms(True, warn_only=args.deterministic_algorithms_warn_only)
    # This runs after fit's teardown moved the module to the CPU, so the end-of-training test's tables are built on the
    # CPU, as for every published run. Keep Lightning's per-call placement and teardown.
    cfm_module.update_codec_bottlenecks()   # no quantile refit -- see update_codec_bottlenecks
    cfm_module.stage = 3
    trainer.test(model=cfm_module, dataloaders=test_dl)
