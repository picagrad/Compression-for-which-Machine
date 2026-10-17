"""Cross-model codec testing: one codec checkpoint, many target task models, one JSON file.

Given a codec trained against task model A, evaluate it against a list of *other* task
models that share A's cut-point tensor shape — without one script invocation per (A, B)
pair, and with every result landing in a single resumable summary file.

Why this works at all: checkpoints only ever contain ``codec.*`` weights (FixedRateCFM
filters out everything under ``task_model`` in ``state_dict()``), and the task model is
always rebuilt from its own pretrained source. So loading A's checkpoint while
instantiating task model B is mechanically fine — provided the codec's expected input
shape matches what B emits at the cut point. That last condition is what this script
checks, per target, before loading.

The codec is built for the *source* task model (--task_model), not from the target's name:
the checkpoint's weights were built for the source's codec, and the target only supplies its
own task model.

Everything is resumable at the lambda level. Re-running the same command after a timeout
or preemption picks up at the first lambda with no recorded RD point.

Two ways to choose the lambdas, per target (--sweep):
  grid         a fixed log grid from --lambda_min/--lambda_max/--lambda_space(_steps),
               the same for every target.
  rate_guided  (default) the rate-guided logic of test_rate_guided (rate_guided_sweep.py): start at the
               source's lambda_ref x 1, 2, 1/2, climb the x2 ladder to the plateau, descend to the
               quality floor, refine below the knee. Each target's next lambda is decided from its
               OWN points relative to its OWN uncompressed baseline, so a target whose rates sit far
               from the source's still gets placed points. The baseline is therefore resolved
               BEFORE the sweep (results file, else the baseline store, else measured). No early
               stop for collapsed targets: they run the same rules, backstop included.

Usage:
    python -m cfwm_repro.tools.test_cross_model \
        --task_model FasterRCNN --checkpoint <path/to/checkpoint.ckpt> \
        --target_task_models RetinaNet,FCOS,ATSSResNet \
        --data_root data/coco --output ./
"""

import argparse
import os
import traceback
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.strategies import SingleDeviceStrategy

from cfm_model_zoo.directory import backbone_style
from .callbacks import ModelEvaluator
from .checkpoint_io import load_checkpoint_state, require_checkpoint
from .rate_guided_sweep import SweepConfig, run_sweep
from .test_common import (
    FeatureCalibration, build_calibration_data, build_test_data, codec_input_signature, completed_lambda_keys,
    checkpoint_identity, compute_feature_stats, compute_lambda_range, expected_metric, lambda_key,
    load_results_for_checkpoint,
    lookup_baseline, name_heuristic_compatible, results_lock, probe_cut_point_signature, resolve_lambda_ref,
    CROSS_MODEL_RESULTS_DIR, results_file, run_baseline, save_results, sweep_lambdas,
)
from .train_common import (
    FROZEN_CROSS, add_path_args, add_task_model_arg, task_model_list,
    codec_class_and_args, task_model_name_and_args, configure_mmdet_logging,
)
from .tracking import add_tracking_arg, build_logger, nested_run, parent_run, require_mlflow
from .trainers import VariableRateCFM
from .utils import MlFLowBoolWorkaround


def build_parser():
    parser = argparse.ArgumentParser(description='Test one codec checkpoint against many target task models',
                                     allow_abbrev=False)
    add_task_model_arg(parser, help='Task model the source codec was trained on')

    g = parser.add_argument_group('Cross-Model Arguments')
    g.add_argument('--target_task_models', type=task_model_list, required=True,
                   help='Comma-separated task models (short names) to test the source codec against')
    g.add_argument('--results_dir', type=str, default='',
                   help='Directory for the JSON summary file (default: data/logs/cross_model_tests)')
    g.add_argument('--lambda_ref', type=float, default=1.0,
                   help='Reference lambda used to build the codec; replaced by the checkpoint\'s own value')
    g.add_argument('--sweep', choices=['grid', 'rate_guided'], default='rate_guided',
                   help="How each target's lambdas are chosen. 'rate_guided': test_rate_guided's logic, per "
                        "target, relative to that target's own uncompressed baseline (resolved first). 'grid': "
                        'the fixed log grid from --lambda_min/--lambda_max/--lambda_space(_steps). The '
                        'rate-guided flags act only with rate_guided, the grid flags only with grid')
    g.add_argument('--calibration_stats', choices=['train', 'test', 'none'], default='train',
                   help="Cross-model feature-scale calibration, and where its per-channel stats come from: 'train' "
                        "(default) the first 2000 training images, 'test' the test set, 'none' no calibration. "
                        "Calibration z-scores each target's raw cut-point tensor with the target's own stats, rescales "
                        "it into the source codec's, and inverts that on x_hat. ConvNet backbones only (the SWIN "
                        "cross-model results did not need it)")

    g = parser.add_argument_group("Rate-guided Sweep Arguments (--sweep rate_guided only)")
    g.add_argument('--floor_frac', type=float, default=0.6,
                   help='Walk down the ladder until a point scores below this fraction of the '
                        "target's uncompressed baseline")
    g.add_argument('--max_points', type=int, default=14, help='Hard cap on points per target')
    g.add_argument('--min_points', type=int, default=8,
                   help='A target whose sweep would end with fewer points steps down past the floor '
                        'until it has them, or until the rate stops responding')
    g.add_argument('--plateau_tol', type=float, default=0.015,
                   help='Plateau = every point in the top 2x of bpp within this fraction of '
                        'baseline of the best score')
    g.add_argument('--q_step', type=float, default=0.03,
                   help='A gap is clean when its score step is within this fraction of baseline '
                        '(and its rate step within sqrt(2)x)')
    g.add_argument('--rate_stall', type=float, default=0.10,
                   help='A ladder direction ends when one x2 step changes bpp by less than this')
    g.add_argument('--hi_frac', type=float, default=0.85,
                   help='Refinement fills only the gaps whose upper point scores at or above this '
                        'fraction of baseline')

    g = parser.add_argument_group('Grid Sweep Arguments (--sweep grid only)')
    g.add_argument('--lambda_min', type=float, default=0.1,
                   help='Minimum lambda for RD curve sweep, as a RATIO of the checkpoint\'s lambda_ref '
                        '(--sweep grid only)')
    g.add_argument('--lambda_max', type=float, default=10.0,
                   help='Maximum lambda for RD curve sweep, as a RATIO of the checkpoint\'s lambda_ref '
                        '(--sweep grid only)')
    g.add_argument('--lambda_space', choices=['linear', 'log'], default='log',
                   help='Interpolation space for lambda sweep (--sweep grid only)')
    g.add_argument('--lambda_space_steps', type=int, default=6,
                   help='Number of lambda values in the sweep (--sweep grid only)')

    add_path_args(parser, kind='test', output_default='./')
    add_tracking_arg(parser)
    parser.set_defaults(**FROZEN_CROSS)
    return parser


def resolve_checkpoint_path(args):
    """The source checkpoint, checked once up front -- it's reloaded per target."""
    return require_checkpoint(args.checkpoint)


def build_target_module(args, target, codec_cls, codec_args):
    """A VariableRateCFM holding the *target's* task model and the *source's* codec."""
    target_args = argparse.Namespace(**vars(args))
    target_args.task_model = target
    task_model_name, task_model_args = task_model_name_and_args(target_args)
    return VariableRateCFM(
        task_model_name, task_model_args, codec_cls, codec_args,
        stage=3,
        lmbda=args.lambda_ref,
    )


def build_trainer(args, logger, test_evaluator):
    return pl.Trainer(max_epochs=1, callbacks=[ModelEvaluator(test_evaluator=test_evaluator)], logger=logger,
                      precision=args.precision, strategy=SingleDeviceStrategy(device='cuda:0'),
                      num_sanity_val_steps=0, enable_progress_bar=args.interactive)


def merge_cross_model_doc(on_disk, mine, owned, path):
    """The doc to write when several jobs save into one results file, each sweeping its own targets of
    the same source: the file as it is on disk, with this job's top-level fields and only the entries of
    its own targets (``owned``) replaced or added. Other targets' entries come from disk, never from this
    job's copy, which may be older. A file recorded for other weights stops the job."""
    if on_disk is None:
        return mine
    theirs, ours = on_disk.get('checkpoint_identity') or {}, mine.get('checkpoint_identity') or {}
    if any(theirs.get(k) != ours.get(k) for k in ('size', 'sha256')):
        raise SystemExit(f"Results file {path} was rewritten for other weights ({theirs.get('realpath')}) while "
                         f"this job sweeps {ours.get('realpath')}; not merging the two. Pass a different "
                         f"--results_dir, or delete or move {path}.")
    if on_disk.get('calibration_stats') and mine.get('calibration_stats') and \
            on_disk['calibration_stats'] != mine['calibration_stats']:
        check_calibration_source(on_disk, mine['calibration_stats'], path)
    merged = {**on_disk, **{k: v for k, v in mine.items() if k != 'results'}}
    own = {e['tested_on_task_model']: e for e in mine['results'] if e['tested_on_task_model'] in owned}
    merged['results'] = [own.pop(e.get('tested_on_task_model'), e) for e in on_disk.get('results', [])]
    merged['results'] += list(own.values())
    return merged


def check_calibration_source(doc, setting, path):
    """Stop when the results file was swept with another calibration than this run's ('train', 'test' or 'none'):
    resuming it would mix two calibrations in one file. A file with stats but no record predates --calibration_stats and
    used the test set; a file with neither has nothing to conflict with."""
    held = doc.get('calibration_stats') or ('test' if 'source_stats' in doc else None)
    if held is not None and held != setting:
        raise SystemExit(f'Results file {path} was swept with --calibration_stats {held}, but this run asks for '
                         f'--calibration_stats {setting}. Pass a different --results_dir, or delete or move {path}.')


def load_shared_results(path, identity):
    """load_results_for_checkpoint under results_lock. Other jobs may be saving into the same file, and on
    NFS a read while another node renames over the file fails (ESTALE); saves rename only under the lock."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with results_lock(path):
        return load_results_for_checkpoint(path, identity)


def find_entry(doc, target):
    for entry in doc['results']:
        if entry.get('tested_on_task_model') == target:
            return entry
    return None


def sweep_config(args, lambda_ref: float) -> SweepConfig:
    return SweepConfig(lambda_ref=lambda_ref, floor_frac=args.floor_frac,
                       max_points=args.max_points, min_points=args.min_points,
                       plateau_tol=args.plateau_tol, q_step=args.q_step,
                       rate_stall=args.rate_stall, hi_frac=args.hi_frac)


def resolve_target_baseline(entry, target, lookup, measure_baseline, persist, log=print, required=True):
    """The target's uncompressed score, resolved before either sweep (a rate-guided sweep decides relative to
    it): the entry's own, else the baseline store, else measured. Stored on the entry
    as a float (this file's schema) with its source. Returns the value, or None if measuring was
    interrupted. A measurement that returns no value raises when `required` (the rate-guided sweep cannot
    run without it); otherwise (the grid sweep) nothing is stored and None is returned.

    lookup(target) -> {'metric', 'value', 'source'} | None
    measure_baseline() -> (metric, value, interrupted)
    """
    if entry.get('baseline') is not None:
        log(f"  Baseline from the results file: {entry['baseline']}")
        return entry['baseline']
    found = lookup(target)
    if found is not None:
        entry['baseline'] = found['value']
        entry['baseline_source'] = found.get('source', 'baseline store')
        entry.setdefault('metric', found['metric'])
    else:
        metric, value, interrupted = measure_baseline()
        if interrupted:
            return None
        if value is None:
            if not required:
                log(f'  {target}: the baseline measurement gave no value; continuing without one')
                return None
            raise RuntimeError(f'No usable uncompressed baseline for {target} -- the rate-guided '
                               f'sweep decides relative to it and cannot run without one')
        entry['baseline'] = value
        entry['baseline_source'] = 'measured'
        entry.setdefault('metric', metric)
    persist()
    log(f"  Baseline: {entry.get('metric')}={entry['baseline']} ({entry['baseline_source']})")
    return entry['baseline']


def rate_guided_target(entry, target, cfg, lookup, measure_baseline, measure, persist,
                       log=print) -> bool:
    """One target's rate-guided sweep. True if it ran to completion, False if interrupted.
    The next lambda is a pure function of entry['rd_points'], so a resumed job carries on."""
    entry.setdefault('rd_points', [])
    baseline = resolve_target_baseline(entry, target, lookup, measure_baseline, persist, log)
    if baseline is None:
        log(f'  {target}: interrupted while measuring the baseline')
        return False
    if entry['rd_points']:
        log(f"  Resuming: {len(entry['rd_points'])} point(s) already recorded")
    interrupted = run_sweep(entry, baseline, cfg, measure, persist, log)
    return not interrupted


def run_target(args, target, entry, codec_cls, codec_args, source_signature,
               checkpoint_path, doc, persist, use_calibration):
    """Test one target, persisting after every lambda. Mutates `entry` in place.

    Returns True if the target's sweep ran to completion, False if it was skipped
    (incompatible) or cut short (interrupted).
    """
    already_done = completed_lambda_keys(entry)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    cfm_module = build_target_module(args, target, codec_cls, codec_args)

    # Tier 2 -- authoritative tensor probe. Skipped on resume: a target with a persisted
    # RD point already proved its compatibility the expensive way.
    if not already_done:
        try:
            probe = probe_cut_point_signature(cfm_module.task_model, device,
                                              args.image_h, args.image_w)
        except Exception:
            entry.update({'status': 'skipped_incompatible',
                          'reason': f'cut-point probe failed: {traceback.format_exc(limit=1).strip()}'})
            print(f'  SKIP {target}: probe raised\n{traceback.format_exc()}')
            persist()
            return False
        if probe != source_signature:
            reason = (f'cut-point signature mismatch: source expects '
                      f'ndim={source_signature[0]}, channels={source_signature[1]}; '
                      f'target emits ndim={probe[0]}, channels={probe[1]}')
            entry.update({'status': 'skipped_incompatible', 'reason': reason})
            print(f'  SKIP {target}: {reason}')
            persist()
            return False
        print(f'  Probe OK: ndim={probe[0]}, channels={probe[1]}')

    # Final safety net beneath both tiers: anything that still slips through and blows up
    # at load time skips this one target, exactly like a probe failure would.
    try:
        load_checkpoint_state(checkpoint_path, cfm_module)   # strict: a mismatch skips the target
    except Exception:
        entry.update({'status': 'skipped_incompatible',
                      'reason': f'load_state_dict failed: {traceback.format_exc(limit=1).strip()}'})
        print(f'  SKIP {target}: checkpoint load failed\n{traceback.format_exc()}')
        persist()
        return False

    test_dl, test_eval, cfg_tr = build_test_data(cfm_module, args)

    # Cross-model feature-scale calibration: this target's own stats, computed once and cached on its entry so a
    # resumed job doesn't redo it. doc['source_stats'] was already computed (or this whole thing skipped) back in
    # main(). cfm_module.calibration stays None -- a no-op -- if disabled.
    if use_calibration:
        if 'target_stats' not in entry:
            print(f'  Computing per-channel calibration stats for target {target} ({args.calibration_stats} images)...')
            # .eval(): nothing has put this freshly built module in eval mode yet (Lightning does so only inside
            # trainer.test()), and in train mode BatchNorm would use per-batch statistics instead of the trained
            # running ones (it shifted SegKNetUPerResNet's target_stats by up to 0.47 on some channels). Same as
            # main()'s source module.
            cfm_module = cfm_module.to(device).eval()
            calib_dl = test_dl if args.calibration_stats == 'test' else build_calibration_data(cfm_module, args)
            entry['target_stats'] = compute_feature_stats(cfm_module, calib_dl, max_batches=None)
            persist()
        cfm_module.calibration = FeatureCalibration(
            mu_t=entry['target_stats']['mean'], sigma_t=entry['target_stats']['std'],
            mu_s=doc['source_stats']['mean'], sigma_s=doc['source_stats']['std'],
        )

    lambda_ref_sweep = resolve_lambda_ref(cfm_module, args.lambda_ref)
    if args.sweep == 'rate_guided':
        return _run_rate_guided(args, target, entry, cfm_module, test_dl, cfg_tr,
                                lambda_ref_sweep, doc, persist, test_eval)
    lambda_range = compute_lambda_range(lambda_ref_sweep, args.lambda_min, args.lambda_max,
                                        args.lambda_space, args.lambda_space_steps)
    # Reversed: largest lambda first (cheapest to interrupt early).
    remaining = [lm for lm in lambda_range[::-1] if lambda_key(lm) not in already_done]
    if already_done:
        print(f'  Resuming: {len(already_done)} lambda(s) already recorded, '
              f'{len(remaining)} remaining')

    entry.pop('status', None)
    entry.pop('reason', None)
    entry['lambda_ref'] = lambda_ref_sweep

    # The baseline is resolved the same way for both sweeps (resolve_target_baseline: this results file, else the
    # baseline store, else measured), but the grid sweep does not need it: a measurement with no value is not an
    # error here. The trainer is built only when something has to run: the baseline or a lambda.
    trainer = None

    def get_trainer():
        nonlocal trainer
        if trainer is None:
            logger = build_logger(cfm_module, cfg_tr, args)
            entry['mlflow_run_id'] = logger.run_id
            trainer = build_trainer(args, logger, test_eval)
        return trainer

    def measure_baseline():
        t = get_trainer()
        metric, value = run_baseline(t, cfm_module, test_dl)
        return metric, value, bool(t.interrupted)

    resolve_target_baseline(entry, target, lambda name: lookup_baseline(name, expected_metric(name)),
                            measure_baseline, persist, required=False)
    if trainer is not None and trainer.interrupted:
        print(f'  {target}: interrupted while measuring the baseline')
        return False
    if remaining:
        get_trainer()

    completed = 0
    for point in sweep_lambdas(trainer, cfm_module, test_dl, remaining):
        print(f"  lambda={point['lambda']:.4f}  bpp={point['bpp']}  "
              f"{point['metric']}={point['value']}")
        entry.setdefault('rd_points', []).append(point)
        # Ascending bpp; a point whose bpp went missing sorts last rather than breaking the comparison.
        entry['rd_points'].sort(key=lambda p: (p['bpp'] is None, p['bpp'] or 0.0))
        if point['metric']:
            entry['metric'] = point['metric']
        persist()   # persist per lambda, not per target
        completed += 1

    if completed < len(remaining):
        print(f'  {target}: interrupted after {completed}/{len(remaining)} lambda(s)')
        return False
    return True


def _run_rate_guided(args, target, entry, cfm_module, test_dl, cfg_tr, lambda_ref_sweep,
                     doc, persist, test_eval) -> bool:
    """run_target's --sweep rate_guided branch: wires the trainer into rate_guided_target."""
    entry.pop('status', None)
    entry.pop('reason', None)
    entry['lambda_ref'] = lambda_ref_sweep
    entry['sweep'] = 'rate_guided'
    cfg = sweep_config(args, lambda_ref_sweep)
    print(f'  Rate-guided sweep: {cfg}')
    logger = build_logger(cfm_module, cfg_tr, args)
    entry['mlflow_run_id'] = logger.run_id
    trainer = build_trainer(args, logger, test_eval)

    def sort_and_persist():
        # Ascending bpp (next_lambda sorts for itself).
        entry['rd_points'].sort(key=lambda p: (p.get('bpp') is None, p.get('bpp') or 0.0))
        for p in entry['rd_points']:
            if p.get('metric'):
                entry.setdefault('metric', p['metric'])
        persist()

    def lookup(name):
        return lookup_baseline(name, expected_metric(name))

    def measure_baseline():
        metric, value = run_baseline(trainer, cfm_module, test_dl)
        return metric, value, bool(trainer.interrupted)

    def measure(lam):
        points = list(sweep_lambdas(trainer, cfm_module, test_dl, [lam]))
        if trainer.interrupted or not points:
            return None
        return points[0]

    finished = rate_guided_target(entry, target, cfg, lookup, measure_baseline, measure, sort_and_persist)
    return finished and not trainer.interrupted


def main(args):
    require_mlflow(args)
    torch.use_deterministic_algorithms(True, warn_only=args.deterministic_algorithms_warn_only)
    configure_mmdet_logging(args)

    targets = [t.strip() for t in args.target_task_models.split(',') if t.strip()]
    if not targets:
        raise SystemExit('--target_task_models is empty')

    # Codec class + args come from the source, once, for every target.
    codec_cls, codec_args = codec_class_and_args(args)
    if args.var_rate in ('qvrf', 'gqvrf'):
        codec_args.update({'lambda_ref': args.lambda_ref,
                           'lambda_ratio': [args.lambda_min, args.lambda_max]})
    uses_swin = backbone_style[args.task_model] == 'SWIN'
    source_signature = codec_input_signature(uses_swin, codec_args)
    print(f'Source codec {codec_cls.__name__} expects ndim={source_signature[0]}, '
          f'channels={source_signature[1]}')

    checkpoint_path = resolve_checkpoint_path(args)
    print(f'Source checkpoint: {checkpoint_path}')

    file_id = f'{args.task_model}__{Path(args.checkpoint).stem}'   # the SOURCE task model
    out_path = results_file(args.results_dir or None, file_id, CROSS_MODEL_RESULTS_DIR)   # '' -> the default
    identity = checkpoint_identity(checkpoint_path)
    doc = load_shared_results(out_path, identity) or {   # refuses another checkpoint's file
        'source_checkpoint': os.path.basename(args.checkpoint),
        'checkpoint_identity': identity,
        'trained_on_task_model': args.task_model,
        'results': [],
    }
    doc['trained_on_task_model'] = args.task_model
    # The calibration this file is swept with; 'none' for a SWIN source, which is never calibrated.
    calibration = args.calibration_stats if not uses_swin else 'none'
    check_calibration_source(doc, calibration, out_path)
    doc['calibration_stats'] = calibration
    print(f'Results file: {out_path}')
    # Jobs that split one source's targets between them save into this same file; each save keeps what the others
    # saved and replaces only this job's own targets (merge_cross_model_doc, under results_lock).
    owned = set(targets)

    def persist():
        save_results(out_path, doc, merge=lambda on_disk, mine: merge_cross_model_doc(on_disk, mine, owned, out_path))

    # One parent run for the whole cross-model job -- everything a "sweep" produces (one
    # source codec x N targets) shows up grouped under it in MLflow. Each target below
    # gets its own nested child run (same logger/trainer construction as test_rate_guided),
    # so per-target metrics keep clean, uncollided key namespaces. A launcher that
    # pre-creates the MLflow run (e.g. `mlflow run --run-id`) is resumed by start_run; a bare
    # invocation opens a new run.
    with parent_run(args, f'cross_model_{args.task_model}') as parent_run_obj:
        doc['mlflow_parent_run_id'] = parent_run_obj.info.run_id if parent_run_obj is not None else None
        if args.tracking == 'mlflow':
            import mlflow
            mlflow.log_params(vars(args))

        # Cross-model feature-scale calibration: source's own stats computed once here, cached in doc so a
        # resumed job doesn't redo it; each target computes its own further down in run_target().
        # ConvNet only -- real cross-model data shows SWIN doesn't need this.
        use_calibration = args.calibration_stats != 'none' and not uses_swin
        if args.calibration_stats != 'none' and uses_swin:
            print('Calibration skipped: the source is a SWIN backbone (the SWIN cross-model results did not need it).')
        if use_calibration and 'source_stats' not in doc:
            print(f'Computing per-channel calibration stats for source {args.task_model} '
                  f'({args.calibration_stats} images)...')
            calib_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            source_cfm = build_target_module(args, args.task_model, codec_cls, codec_args)
            source_cfm = source_cfm.to(calib_device).eval()
            source_dl = build_calibration_data(source_cfm, args)
            doc['source_stats'] = compute_feature_stats(source_cfm, source_dl, max_batches=None)
            persist()
            del source_cfm

        interrupted = False
        for target in targets:
            print(f'\n=== {args.task_model} codec -> {target} ===')
            entry = find_entry(doc, target)
            if entry is None:
                entry = {'tested_on_task_model': target}
                doc['results'].append(entry)

            if entry.get('status') == 'skipped_incompatible' and not args.retry_skipped:
                print(f'  Already recorded as incompatible ({entry.get("reason")}) — skipping')
                continue

            # Tier 1 -- name heuristic, no model built. Cheap enough to re-check on resume.
            ok, reason = name_heuristic_compatible(args.task_model, target, args.cut_point)
            if not ok:
                entry.update({'status': 'skipped_incompatible', 'reason': reason})
                print(f'  SKIP {target}: {reason}')
                persist()
                continue

            with nested_run(args, f'{args.task_model}->{target}'):
                finished = run_target(args, target, entry, codec_cls, codec_args,
                                      source_signature, checkpoint_path, doc, persist,
                                      use_calibration)
            if not finished and entry.get('status') != 'skipped_incompatible':
                interrupted = True
                break

    persist()
    done = sum(1 for e in doc['results'] if e.get('rd_points'))
    skipped = sum(1 for e in doc['results'] if e.get('status') == 'skipped_incompatible')
    print(f'\nWrote {out_path}: {done} target(s) with RD points, {skipped} skipped as '
          f'incompatible, {len(targets)} requested.')
    # Exit status reflects whether the target list was worked through, not whether every
    # target turned out compatible -- an incompatible target is a recorded result, not a
    # job failure.
    return interrupted


if __name__ == '__main__':
    main(build_parser().parse_args())
