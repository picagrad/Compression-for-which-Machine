"""Train the variable-rate (GQVRF) ELFC codec for one task model (VariableRateCFM).

The settings every publication run shared are frozen (train_common.FROZEN_TRAIN); the command line takes only what
varied between them. Example:

    python -m cfwm_repro.tools.train_variable_rate \
        --task_model FasterRCNN --data_root data/coco \
        --lambda_ref 5 --lambda_min 0.01 --lambda_max 2 \
        --epochs 100 --lr_scheduler wsd --lr_decay_frac 0.3 --lr_min_factor 0.05 --lambda_warmup_epochs 5 \
        --checkpoint_dir checkpoints/FasterRCNN --output ./

Relaunching the same command resumes from <checkpoint_dir>/<checkpoint_name>.ckpt.
"""

import argparse

import torch

from .train_common import (
    FROZEN_TRAIN,
    add_path_args, add_schedule_args, add_task_model_arg,
    apply_seed, build_base_callbacks, build_checkpoint_callback, codec_class_and_args,
    build_dataloaders, build_evaluators, build_sched_dict,
    task_model_name_and_args, build_trainer, configure_mmdet_logging, resolve_training_start, run_full_test,
)
from .callbacks import PeriodicInferenceValidation
from .tracking import add_tracking_arg, build_logger, require_mlflow
from .trainers import VariableRateCFM


def build_parser():
    parser = argparse.ArgumentParser(description='Train the variable-rate (GQVRF) ELFC codec for one task model',
                                     allow_abbrev=False)
    add_task_model_arg(parser)
    g = parser.add_argument_group('Rate')
    g.add_argument('--lambda_ref', type=float, default=1,
                   help='Reference lambda (the GQVRF scaler reference) (basic recipe: 5)')
    g.add_argument('--lambda_min', type=float, default=0.01,
                   help='Smallest training lambda, as a RATIO of --lambda_ref')
    g.add_argument('--lambda_max', type=float, default=1,
                   help='Largest training lambda, as a RATIO of --lambda_ref (basic recipe: 2)')
    add_schedule_args(parser)
    add_path_args(parser, kind='train')
    add_tracking_arg(parser)
    parser.set_defaults(**FROZEN_TRAIN)
    return parser


def main(args: argparse.Namespace):
    require_mlflow(args)
    resume_path, weights_path, branch = resolve_training_start(args)
    print(f'Training start: {branch}')
    configure_mmdet_logging(args)
    apply_seed(args)

    codec_cls, codec_args = codec_class_and_args(args)
    codec_args.update({'lambda_ref': args.lambda_ref,
                       'lambda_ratio': [args.lambda_min, args.lambda_max]})

    task_model_name, task_model_args = task_model_name_and_args(args)
    sched_dict = build_sched_dict(args)

    cfm_module = VariableRateCFM(
        task_model_name, task_model_args, codec_cls, codec_args,
        stage=0,
        lmbda=args.lambda_ref,
        val_inference=args.val_inference,
        sched_dict=sched_dict,
        lr=args.lr,
    )

    train_dl, val_dl, test_dl, cfg_tr, use_coco = build_dataloaders(cfm_module, args)
    val_eval, test_eval = build_evaluators(cfg_tr, train_dl, test_dl, use_coco, args.data_root)
    if weights_path is not None:
        # --restart: the WEIGHTS of --init_checkpoint only (fresh optimiser and scheduler, epoch 0), loaded where the
        # weights-only restart always loaded them: after build_dataloaders, so after the batch-size probe. Not strict.
        with open(weights_path, 'rb') as ckpt:
            cfm_module.load_state_dict(torch.load(ckpt, cfm_module.device)['state_dict'], strict=False)

    # Validation inference (task metrics) runs periodically, armed by PeriodicInferenceValidation and
    # ChangeModelStage; the normal validation loop always cycles lambda (no inference).
    callbacks = build_base_callbacks(
        args, build_checkpoint_callback(args),
        intermediate_stage_epochs=args.lambda_warmup_epochs,
        val_evaluator=val_eval if args.val_inference else None, test_evaluator=test_eval,
    )
    if args.val_inference_freq > 0:
        callbacks.append(PeriodicInferenceValidation(every_n_epochs=args.val_inference_freq))

    logger = build_logger(cfm_module, cfg_tr, args)
    trainer = build_trainer(args, callbacks, logger)

    cfm_module.stage = 0
    trainer.fit(
        model=cfm_module,
        train_dataloaders=train_dl,
        val_dataloaders=val_dl,
        ckpt_path=resume_path,
    )
    if trainer.interrupted:
        return trainer.interrupted

    cfm_module._reset_lmbda()
    trainer.test(model=cfm_module, dataloaders=test_dl)
    if trainer.interrupted:
        return trainer.interrupted

    if args.full_test:
        run_full_test(trainer, cfm_module, test_dl, args)

    return trainer.interrupted


if __name__ == '__main__':
    main(build_parser().parse_args())
