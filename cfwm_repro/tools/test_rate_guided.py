"""Rate-guided RD sweep for VariableRateCFM checkpoints.

Instead of a fixed lambda grid, each next lambda is picked from the points measured so far
(rate_guided_sweep.next_lambda): lambda_ref x {1, 2, 1/2}, a x2 ladder up to the plateau and
down to floor_frac of uncompressed, then refinement below the knee.

The uncompressed baseline is REQUIRED -- every decision is relative to it -- so it is resolved
FIRST: this checkpoint's results file, else the baseline store (test_common.lookup_baseline),
else measured before any compressed point.

The lambda_ref point is measured by the sweep itself, like any other point.

Usage:
    python -m cfwm_repro.tools.test_rate_guided \
        --task_model SegUperNetSWIN --data_root <data_root> \
        --checkpoint <path/to/checkpoint.ckpt> --output ./
"""

import argparse
import os
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.strategies import SingleDeviceStrategy

from .callbacks import ModelEvaluator
from .checkpoint_io import load_checkpoint_state, require_checkpoint
from .rate_guided_sweep import SweepConfig, run_sweep
from .test_common import (
    RESULTS_DIR_DISABLED_WARNING, build_test_data, expected_metric,
    RD_SWEEP_RESULTS_DIR, checkpoint_identity, load_results_for_checkpoint, lookup_baseline,
    resolve_lambda_ref, results_file, run_baseline, save_results,
    sweep_lambdas,
)
from .train_common import (
    FROZEN_TEST, add_path_args, add_task_model_arg,
    codec_class_and_args, task_model_name_and_args,
    configure_mmdet_logging,
)
from .tracking import add_tracking_arg, build_logger, require_mlflow
from .trainers import VariableRateCFM


def build_parser():
    parser = argparse.ArgumentParser(description='Rate-guided RD sweep of a trained variable-rate ELFC codec',
                                     allow_abbrev=False)
    add_task_model_arg(parser)
    parser.add_argument('--lambda_ref', type=float, default=1.0,
                        help='Reference lambda used to build the codec; replaced by the checkpoint\'s own value')
    g = parser.add_argument_group('Rate-guided sweep')
    g.add_argument('--floor_frac', type=float, default=0.6,
                   help='Walk down the ladder until a point scores below this fraction of '
                        'the uncompressed baseline')
    g.add_argument('--max_points', type=int, default=14,
                   help='Hard cap on points for this checkpoint. Counts the points this script '
                        'placed plus earlier-sweep points inside [floor, plateau] '
                        '(rate_guided_sweep.select_points)')
    g.add_argument('--min_points', type=int, default=8,
                   help='A sweep that would end with fewer points than this steps down past the '
                        'floor (x1/2 below its lowest point) until it has them, or until the rate '
                        'stops responding. Only fires when the climb ends early (a collapsed or '
                        'weak-plateau run); lower it for a run known to be collapsed')
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
    g.add_argument('--results_dir', type=str, default=None,
                   help='Directory for the per-checkpoint results JSON (default: data/logs/rd_sweeps under the '
                        'working directory); each point is saved as it completes, so a relaunch carries on. '
                        'An empty string disables it')
    add_path_args(parser, kind='test')
    add_tracking_arg(parser)
    parser.set_defaults(**FROZEN_TEST)
    return parser


def main(args):
    require_mlflow(args)
    torch.use_deterministic_algorithms(True, warn_only=args.deterministic_algorithms_warn_only)
    configure_mmdet_logging(args)

    codec_cls, codec_args = codec_class_and_args(args)
    codec_args.update({'lambda_ref': args.lambda_ref,
                       'lambda_ratio': [args.lambda_min, args.lambda_max]})

    task_model_name, task_model_args = task_model_name_and_args(args)

    cfm_module = VariableRateCFM(
        task_model_name, task_model_args, codec_cls, codec_args,
        stage=3,
        lmbda=args.lambda_ref,
    )

    # --- Build test dataloader and evaluator ---
    test_dl, test_eval, cfg_tr = build_test_data(cfm_module, args)

    # --- Load the given checkpoint ---
    checkpoint_path = require_checkpoint(args.checkpoint)
    print(f'Testing checkpoint: {checkpoint_path}')
    load_checkpoint_state(checkpoint_path, cfm_module)   # strict

    # --- Set up trainer ---

    logger = build_logger(cfm_module, cfg_tr, args)
    callbacks = [ModelEvaluator(test_evaluator=test_eval)]
    trainer = pl.Trainer(
        max_epochs=1, callbacks=callbacks, logger=logger, precision=args.precision,
        strategy=SingleDeviceStrategy(device='cuda:0'), num_sanity_val_steps=0,
        enable_progress_bar=args.interactive,
    )

    # --- Sweep config, read back off the built object ---
    lambda_ref_sweep = resolve_lambda_ref(cfm_module, args.lambda_ref)
    cfg = SweepConfig(lambda_ref=lambda_ref_sweep, floor_frac=args.floor_frac,
                      max_points=args.max_points, min_points=args.min_points,
                      plateau_tol=args.plateau_tol, q_step=args.q_step,
                      rate_stall=args.rate_stall, hi_frac=args.hi_frac)
    print(f'Rate-guided sweep: {cfg}')

    # --- Results file: the state the sweep decides from ---
    identity = checkpoint_identity(checkpoint_path)
    doc = {'checkpoint': os.path.basename(checkpoint_path), 'checkpoint_identity': identity,
           'task_model': args.task_model, 'rd_points': []}
    out_path = results_file(args.results_dir, f'{args.task_model}__{Path(args.checkpoint).stem}', RD_SWEEP_RESULTS_DIR)
    if out_path is None:
        print(RESULTS_DIR_DISABLED_WARNING)
    else:
        doc = load_results_for_checkpoint(out_path, identity) or doc   # refuses another checkpoint's file
        doc.setdefault('rd_points', [])
        print(f"Results file: {out_path} ({len(doc['rd_points'])} point(s) already recorded)")

    def persist():
        if out_path is not None:
            save_results(out_path, doc)

    # --- Uncompressed baseline: required, so resolved first ---
    baseline = doc.get('baseline')
    if baseline:
        print(f'Baseline from the results file: {baseline}')
    else:
        baseline = lookup_baseline(args.task_model, expected_metric(args.task_model))
        if baseline is None:
            metric, value = run_baseline(trainer, cfm_module, test_dl)
            if trainer.interrupted:
                return trainer.interrupted
            baseline = ({'metric': metric, 'value': value, 'source': 'measured'}
                        if value is not None else None)
        if not baseline or not baseline.get('value'):
            raise RuntimeError(f'No usable uncompressed baseline for {args.task_model} -- the '
                               f'rate-guided sweep decides relative to it and cannot run without one')
        doc['baseline'] = baseline
        persist()
    print(f"Baseline: {baseline['metric']}={baseline['value']} ({baseline.get('source', 'results file')})")

    # No training-run seed: the lambda_ref point is measured by the sweep itself, like any other point.

    # --- The sweep ---
    def measure(lam):
        points = list(sweep_lambdas(trainer, cfm_module, test_dl, [lam]))
        if trainer.interrupted or not points:
            return None
        return points[0]

    interrupted = run_sweep(doc, baseline['value'], cfg, measure, persist)
    return interrupted or trainer.interrupted


if __name__ == '__main__':
    main(build_parser().parse_args())
