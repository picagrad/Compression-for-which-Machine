import logging
import os
from typing import Any,  Dict,  Optional, Mapping
from warnings import warn
# import lightning_fabric.loggers.logger as logger_lib

from pytorch_lightning.loggers import MLFlowLogger as MLFlowLoggerBase
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers.logger import rank_zero_experiment

# MLflow is the optional [mlflow] extra and is imported inside the methods below, never at module level.
RUN_NAME_TAG = 'mlflow.runName'  # equal to MLflow's run-name tag constant (pinned by a check)


class MLFlowLogger(MLFlowLoggerBase):
    def __init__(
        self,
        run_name: Optional[str] = None,
        tracking_uri: Optional[str] = os.getenv('MLFLOW_TRACKING_URI'),
        tags: Optional[Dict[str, Any]] = None,
        save_dir: Optional[str] = './mlruns',
        prefix: str = '',
        artifact_location: Optional[str] = None,
    ) -> None:
        super().__init__(
            run_name=run_name,
            tracking_uri=tracking_uri,
            tags=tags,
            save_dir=save_dir,
            prefix=prefix,
            artifact_location=artifact_location,
        )

        self.logger = logging.getLogger(__name__)

    @property  # type: ignore[misc]
    @rank_zero_experiment
    def experiment(self):
        import mlflow
        from mlflow.entities import RunTag

        if self._run_id and self._experiment_id:
            return self._mlflow_client

        self.tags = self.tags or {}

        if self._run_name is not None:
            if RUN_NAME_TAG in self.tags:
                self.logger.warning(
                    f'The tag {RUN_NAME_TAG} is found in tags. ' f'The value will be overridden by {self._run_name}.'
                )

            self.tags[RUN_NAME_TAG] = self._run_name

        run = mlflow.active_run() or mlflow.start_run()

        self._mlflow_client.log_batch(
            run_id=run.info.run_id,
            tags=[RunTag(key, str(value)) for key, value in self.tags.items()],
        )

        self._run_id = run.info.run_id
        self._experiment_id = run.info.experiment_id
        self._experiment_name = mlflow.get_experiment(run.info.experiment_id).name

        return self._mlflow_client

    @rank_zero_only
    def finalize(self, status: str = "success") -> None:
        if not self._initialized:
            return
        if status == "success":
            status = "FINISHED"
        elif status == "failed":
            status = "FAILED"
        elif status == "finished":
            status = "FINISHED"


        if self.experiment.get_run(self.run_id):
            self.experiment.set_terminated(self.run_id, status)
    
    @rank_zero_only
    def log_params(self, params: Dict[str, Any]) -> None:
        print('Logging params')
        for key, value in params.items():
            self.experiment.log_param(self._run_id, key, value)

    @rank_zero_only
    def log_metrics(self, metrics: Mapping[str, float], step: Optional[int] = None) -> None:
        from mlflow.exceptions import MlflowException
        try:
            super().log_metrics(metrics, step)
        except MlflowException as e:
            warn(
                f"Failed to log metrics: {e}. "
                "This may be due to an issue with the MLflow server or the metrics format."
                "Resuming training without logging metrics.")
            