import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only

rank_zero_print = rank_zero_only(print)
from pytorch_lightning import LightningModule, Trainer
import traceback
from .tracking import tracking_errors


class LambdaRangeWarmup(Callback):
    """Widens the codec's lambda-sampling range from [1, 1] to target_ratio linearly over stage 1, one step per epoch,
    reaching it on stage 1's last epoch: a narrow range first lets the codec's Scaler settle before it sees the full
    range. Added only when stage 1 is non-empty (build_base_callbacks); assumes the codec reads lambda_ratio on every
    forward (the QVRF codecs do).

    Inputs:
        stage1_start_epoch: the first epoch of stage 1 (= the startup epochs).
        duration_epochs: the epochs over which the range widens (> 0).
        target_ratio: (lo, hi), the final range as ratios of lambda_ref; lo < 1 < hi.
    """
    def __init__(self, stage1_start_epoch: int, duration_epochs: int,
                 target_ratio: 'tuple[float, float]'):
        assert duration_epochs > 0, 'LambdaRangeWarmup should only be constructed when duration_epochs > 0'
        assert target_ratio[0] < 1.0 < target_ratio[1], f'target_ratio must straddle 1, got {target_ratio}'
        self.stage1_start_epoch = stage1_start_epoch
        self.duration_epochs = duration_epochs
        self.target_ratio = target_ratio
        super().__init__()

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if pl_module.stage != 1:
            return
        k = trainer.current_epoch - self.stage1_start_epoch
        # Evenly divide the expansion into duration_epochs steps; step k uses progress
        # (k+1)/N so the FIRST epoch already takes one full step away from 1.0 (not a
        # zero-width start), and the LAST epoch (k=N-1) reaches the target exactly.
        progress = min(1.0, (k + 1) / self.duration_epochs)
        lo_t, hi_t = self.target_ratio
        lo = 1.0 + progress * (lo_t - 1.0)
        hi = 1.0 + progress * (hi_t - 1.0)
        pl_module.codec.lambda_ratio = [lo, hi]


class ChangeModelStage(Callback):
    """Drives the 4-stage training progression:
        0  warmup         — fixed lambda, no val-inference, no LR tuning
        1  intermediate   — variable-rate sampling, no inference/LR/saving:
                            a settling-in period after lambda sampling begins
                            (skipped when intermediate_stage_epochs is 0)
        2  bulk training  — LR active, inference enabled, best-model saving
        3  testing        — test_step routes to compress→decompress (set manually)

    Transitions 0→1 and 1→2 are driven by epoch counts.
    Stage 3 is set externally (e.g. when launching a test run).
    With intermediate_stage_epochs=0 both transitions fire at the same epoch,
    effectively skipping stage 1.

    If the run's ModelEvaluator holds a validation evaluator and trainer.val_dataloaders is available,
    pl_module.val_inference is set True in on_train_epoch_end so that the scheduled
    validation immediately following the transition epoch runs a second codec pass
    with the reference lambda for inference metrics — before any training under the
    new stage begins.
    """
    def __init__(self, startup_epochs: int, intermediate_stage_epochs: int = 0):
        self.startup_epochs = startup_epochs
        self.intermediate_stage_epochs = intermediate_stage_epochs
        self._need_inference_reset: bool = False
        self._warned_no_delayed_sched: bool = False
        super().__init__()

    def _sync_scheduler_stage(self, pl_module: pl.LightningModule) -> None:
        """Push pl_module.stage to the DelayedLRScheduler so it activates at the right time.

        A failure here is silent in its effect but severe: the scheduler never reaches
        its activation stage, so the LR stays at its initial value for the whole run and
        nothing in the logs says so. Warn instead of swallowing -- but never raise, since
        a plain scheduler is a legitimate configuration (wsd runs from epoch 0, unwrapped).
        """
        try:
            sched = pl_module.lr_schedulers()
        except RuntimeError as e:   # raised only when the module is not attached to a Trainer
            rank_zero_print(f'WARNING: could not read lr_schedulers() to sync stage: {e!r}. '
                            f'A DelayedLRScheduler will never activate.')
            return
        if sched is None:
            return
        if 'stage' in sched.__dict__:
            sched.stage = pl_module.stage
        elif not self._warned_no_delayed_sched:
            # Not an error on its own -- only a DelayedLRScheduler carries .stage.
            self._warned_no_delayed_sched = True
            rank_zero_print(f'Note: active LR scheduler {type(sched).__name__} has no .stage '
                            f'attribute; stage-gated activation does not apply to it.')

    def _can_do_inference(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> bool:
        return (
            getattr(pl_module, 'variable_rate', False)
            and _val_evaluator(trainer) is not None
            and getattr(trainer, 'val_dataloaders', None) is not None
        )

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._sync_scheduler_stage(pl_module)
        # Immediate transitions when there is no warmup period (startup_epochs == 0).
        # The normal path fires in on_train_epoch_end; with no warmup there is no epoch
        # end to hook into before the first training step.
        if pl_module.stage == 0 and self.startup_epochs == 0:
            pl_module.stage = 1
            self._sync_scheduler_stage(pl_module)
            rank_zero_print('Changing model stage to 1 (variable-rate warm-up: the lambda range widens). '
                             'No val-inference, LR tuning, or model saving until stage 2.')
            if self.intermediate_stage_epochs == 0:
                pl_module.stage = 2
                self._sync_scheduler_stage(pl_module)
                rank_zero_print('Changing model stage to 2 (bulk training). '
                                 'LR tuning, val-inference, and best-model saving are now active.')

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        # Resume guard: if training resumes at an epoch where stage should already have
        # advanced (e.g. checkpoint was saved mid-stage), catch up without re-arming
        # val_inference (that already happened at the transition epoch).
        if pl_module.stage == 0 and self.startup_epochs > 0 and trainer.current_epoch >= self.startup_epochs:
            pl_module.stage = 1
            self._sync_scheduler_stage(pl_module)
        if pl_module.stage == 1 and trainer.current_epoch >= self.startup_epochs + self.intermediate_stage_epochs:
            pl_module.stage = 2
            self._sync_scheduler_stage(pl_module)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # 0 → 1: fire at the END of the last warmup epoch so the immediately following
        # scheduled validation is the transition snapshot (before any variable-rate training).
        if (pl_module.stage == 0 and self.startup_epochs > 0
                and trainer.current_epoch + 1 >= self.startup_epochs):
            if self._can_do_inference(trainer, pl_module):
                pl_module.val_inference = True
                self._need_inference_reset = True
            rank_zero_print('Changing model stage to 1 (variable-rate warm-up: the lambda range widens). '
                             'No val-inference, LR tuning, or model saving until stage 2.')
            pl_module.stage = 1
            self._sync_scheduler_stage(pl_module) 
        # 1 → 2: fire at the END of the last intermediate epoch.
        # When intermediate_stage_epochs == 0, this fires at the same epoch as 0→1
        # (both thresholds equal startup_epochs - 1), skipping stage 1 entirely.
        if (pl_module.stage == 1
                and trainer.current_epoch + 1 >= self.startup_epochs + self.intermediate_stage_epochs):
            if self.intermediate_stage_epochs > 0 and self._can_do_inference(trainer, pl_module):
                pl_module.val_inference = True
                self._need_inference_reset = True
            rank_zero_print('Changing model stage to 2 (bulk training). '
                             'LR tuning, val-inference, and best-model saving are now active.')
            pl_module.stage = 2
            self._sync_scheduler_stage(pl_module)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self._need_inference_reset:
            pl_module.val_inference = False
            self._need_inference_reset = False


def _epoch_metric(trainer, name: str, default=None):
    """A logged epoch value under either spelling: bare for metrics logged epoch-only (validation
    and test), '<name>_epoch' for metrics also logged per step (training)."""
    for key in (name, f'{name}_epoch'):
        if key in trainer.logged_metrics:
            return trainer.logged_metrics[key]
    return default


class ModelEvaluator(Callback):
    """Owns the task-metric evaluators: passes them the task model's predictions after every validation-inference
    and test batch, and computes and logs the task metrics at the end of the epoch.

    Inputs:
        val_evaluator: the mmengine Evaluator for the validation-inference pass, or None (no validation metrics).
        test_evaluator: the mmengine Evaluator for test, or None (no test metrics).
    """
    def __init__(self, val_evaluator=None, test_evaluator=None):
        super().__init__()
        self.val_evaluator = val_evaluator
        self.test_evaluator = test_evaluator
        self._val_inference_did_run = False

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0) -> None:
        # A validation step returns a bare loss, or a dict with the predictions when it ran inference.
        if self.val_evaluator is not None and isinstance(outputs, dict) and 'out' in outputs:
            self.val_evaluator.process(outputs['out'], outputs['data_samples'])
            self._val_inference_did_run = True

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0) -> None:
        if self.test_evaluator is not None and outputs is not None:
            _, result = outputs                    # test_step returns (loss, {'out', 'data_samples', ...})
            self.test_evaluator.process(result['out'], result['data_samples'])

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.val_evaluator is not None and self._val_inference_did_run:
            self._val_inference_did_run = False
            rank_zero_print(f'Inference validation complete (lambda={float(pl_module.lmbda):.3f}). Running evaluator ...')
            ln = len(trainer.val_dataloaders.dataset)
            final_acc = self.val_evaluator.evaluate(ln)
            final_acc = {f'val_{k}_lmbda_{float(pl_module.lmbda):.3f}':v for k,v in final_acc.items()}
            rank_zero_print(final_acc)
            # Prefer the reference-lambda inference bpp over the cycled-lambda average.
            bpp = _epoch_metric(trainer, 'val_bpp_loss_inference')
            bpp = _epoch_metric(trainer, 'val_bpp_loss') if bpp is None else bpp
            rank_zero_print(f'Approximate bpp for lambda value {float(pl_module.lmbda):.3f} is {bpp}')
            try:
                pl_module.logger.log_metrics(final_acc, step=trainer.current_epoch)
            except tracking_errors():
                print(f'Could not log to mlflow, continuing without logging. For record purposes this is the error trace: {traceback.format_exc()}')
        return super().on_validation_epoch_end(trainer, pl_module)

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.test_evaluator is not None:
            ln = len(trainer.test_dataloaders.dataset)
            final_acc = self.test_evaluator.evaluate(ln)
            # Keep the raw (un-prefixed) scores on the module before the keys get lambda-tagged:
            # evaluate() resets the evaluator, and test_common.sweep_lambdas records them to JSON.
            pl_module.last_test_eval = dict(final_acc)
            # Stage -1 is the uncompressed baseline (test_common.run_baseline). It runs with lmbda
            # reset to lambda_ref, so tagging it by lambda would overwrite the sweep point at
            # lambda_ref in the logged metrics: tag it 'baseline' instead.
            tag = 'baseline' if pl_module.stage == -1 else f'{float(pl_module.lmbda):.3f}'
            final_acc = {f'test_{k}_{tag}':v for k,v in final_acc.items()}
            rank_zero_print(final_acc)
            try:
                pl_module.logger.log_metrics(final_acc)
            except tracking_errors():
                print(f'Could not log to mlflow, continuing without logging. For record purposes this is the error trace: {traceback.format_exc()}')
        return super().on_test_epoch_end(trainer, pl_module)


def _val_evaluator(trainer):
    """The validation evaluator held by the run's ModelEvaluator callback, or None."""
    for callback in getattr(trainer, 'callbacks', None) or []:
        if isinstance(callback, ModelEvaluator):
            return callback.val_evaluator
    return None


class PeriodicInferenceValidation(Callback):
    """Arms a second codec pass (reference lambda, inference metrics) every k stage-2 epochs.

    Sets pl_module.val_inference=True in on_train_epoch_end so the following
    scheduled validation runs the dual-pass path in VariableRateCFM.validation_step.
    No-ops without a validation evaluator (held by ModelEvaluator) or validation dataloader.
    """

    def __init__(self, every_n_epochs: int = 5):
        self.every_n_epochs = every_n_epochs
        self._epochs_in_stage2 = 0
        self._need_inference_reset: bool = False
        super().__init__()

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if pl_module.stage != 2:
            return
        self._epochs_in_stage2 += 1
        if self._epochs_in_stage2 % self.every_n_epochs != 0:
            return
        if _val_evaluator(trainer) is None:
            return
        if getattr(trainer, 'val_dataloaders', None) is None:
            return
        pl_module.val_inference = True
        self._need_inference_reset = True

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self._need_inference_reset:
            pl_module.val_inference = False
            self._need_inference_reset = False
