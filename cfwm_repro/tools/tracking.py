"""The tracking switch: --tracking none (default; MLflow is never imported; metrics go to a CSV under --output) or
mlflow (MLflow exactly as the paper's runs used it; needs the optional [mlflow] extra)."""
import contextlib
import os
import sys
import uuid

from pytorch_lightning.loggers import CSVLogger

from .. import __version__

TRACKING_CHOICES = ('mlflow', 'none')


def add_tracking_arg(parser):
    parser.add_argument(
        '--tracking', choices=TRACKING_CHOICES, default='none',
        help="'none': no MLflow; metrics go to <output>/cfwm/<random id>/metrics.csv (the path is printed at start). "
             "'mlflow': log to MLflow as the paper's runs did (needs the [mlflow] extra; the server comes from "
             "MLflow's own environment variables). Default: none")


def require_mlflow(args) -> None:
    """Stop with one line, before any work, when --tracking mlflow is asked for and MLflow is not installed."""
    if args.tracking != 'mlflow':
        return
    try:
        import mlflow  # noqa: F401
    except ImportError:
        raise SystemExit("--tracking mlflow needs MLflow: pip install 'cfwm-repro[mlflow]', or pass --tracking none") \
            from None


def tracking_errors() -> tuple:
    """The MLflow exception class when MLflow is loaded (--tracking mlflow), else nothing, so `except
    tracking_errors():` catches exactly what the MLflow-only code caught before and never imports MLflow."""
    module = sys.modules.get('mlflow.exceptions')
    return (module.MlflowException,) if module is not None else ()


class OfflineLogger(CSVLogger):
    """Lightning's CSV logger with the two things the MLflow path relies on: a run_id and log_params."""

    def __init__(self, save_dir: str, run_id: str):
        super().__init__(save_dir=save_dir, name='cfwm', version=run_id)
        self.run_id = run_id
        print(f'Metrics: {self.log_dir}/metrics.csv')

    def log_params(self, params: dict) -> None:
        self.log_hyperparams(params)


def build_logger(cfm_module, cfg_tr, args):
    """The run's logger (never None: ModelEvaluator and sweep_lambdas need one)."""
    if args.tracking == 'none':
        logger = OfflineLogger(save_dir=args.output or '.', run_id=uuid.uuid4().hex)
    else:
        from .loggers import MLFlowLogger   # the only importer of loggers: --tracking none never loads it
        logger = MLFlowLogger(tags={
            'task_model': cfm_module.task_model.__class__.__name__,
            'codec': cfm_module.codec.__class__.__name__,
            'dataset': cfg_tr['train_dataloader']['dataset']['type'],
            'job-id': os.environ.get('JOB_ID'),
            'cfwm_repro': __version__,   # marks this repo's runs
        })
    logger.log_params(vars(args))
    return logger


def parent_run(args, run_name: str):
    """A top-level MLflow run (the context yields it), or a no-op yielding None under --tracking none."""
    if args.tracking != 'mlflow':
        return contextlib.nullcontext(None)
    import mlflow
    return mlflow.start_run(run_name=run_name)


def nested_run(args, run_name: str):
    """A child of the active MLflow run, or a no-op yielding None under --tracking none."""
    if args.tracking != 'mlflow':
        return contextlib.nullcontext(None)
    import mlflow
    return mlflow.start_run(nested=True, run_name=run_name)
