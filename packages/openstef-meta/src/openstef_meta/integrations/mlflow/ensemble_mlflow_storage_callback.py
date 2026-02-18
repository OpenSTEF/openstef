# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""MLflow storage callback for ensemble forecasting models.

Extends the base MLFlowStorageCallback with ensemble-specific behavior:
- Logs hyperparameters for each base forecaster and the combiner
- Stores feature importance plots for each explainable forecaster component
"""

import logging
from pathlib import Path
from typing import override

from pydantic import PrivateAttr

from openstef_core.mixins.predictor import HyperParams
from openstef_meta.models.ensemble_forecasting_model import EnsembleForecastingModel
from openstef_models.explainability import ExplainableForecaster
from openstef_models.integrations.mlflow.mlflow_storage_callback import MLFlowStorageCallback
from openstef_models.models.base_forecasting_model import BaseForecastingModel


class EnsembleMLFlowStorageCallback(MLFlowStorageCallback):
    """MLFlow callback with ensemble-specific logging for multi-model forecasting.

    Extends the base MLFlowStorageCallback to handle EnsembleForecastingModel
    instances by:
    - Logging combiner hyperparameters as the primary model hyperparams
    - Logging per-forecaster hyperparameters with name-prefixed keys
    - Storing feature importance plots for each explainable base forecaster

    For non-ensemble models, falls back to the base class behavior.
    """

    _logger: logging.Logger = PrivateAttr(default=logging.getLogger(__name__))

    @override
    def _get_hyperparams(self, model: BaseForecastingModel) -> HyperParams | None:
        """Extract hyperparameters from the ensemble combiner.

        For ensemble models, the combiner's hyperparams are treated as the
        primary hyperparameters. Per-forecaster hyperparams are logged
        separately via _log_additional_hyperparams.

        Falls back to base class behavior for non-ensemble models.

        Returns:
            The combiner hyperparams for ensemble models, or base class result otherwise.
        """
        if isinstance(model, EnsembleForecastingModel):
            return model.combiner.config.hyperparams
        return super()._get_hyperparams(model)

    @override
    def _log_additional_hyperparams(self, model: BaseForecastingModel, run_id: str) -> None:
        """Log per-forecaster hyperparameters to the MLflow run.

        Each base forecaster's hyperparameters are logged with a prefix
        of its name (e.g., 'lgbm.n_estimators', 'xgboost.max_depth').

        Args:
            model: The ensemble forecasting model.
            run_id: MLflow run ID to log parameters to.
        """
        if not isinstance(model, EnsembleForecastingModel):
            return

        for name, forecaster in model.forecasters.items():
            hyperparams = forecaster.hyperparams
            prefixed_params = {f"{name}.{k}": str(v) for k, v in hyperparams.model_dump().items()}
            self.storage.log_hyperparams(run_id=run_id, params=prefixed_params)
            self._logger.debug("Logged hyperparams for forecaster '%s' in run %s", name, run_id)

    @staticmethod
    @override
    def _store_feature_importance(
        model: BaseForecastingModel,
        data_path: Path,
    ) -> None:
        """Store feature importance plots for each explainable forecaster in the ensemble.

        For ensemble models, generates separate feature importance HTML plots for
        each base forecaster that implements ExplainableForecaster. Files are named
        'feature_importances_{forecaster_name}.html'.

        For non-ensemble models, falls back to the base class behavior.

        Args:
            model: The forecasting model (ensemble or single).
            data_path: Directory path where HTML plots will be saved.
        """
        if not isinstance(model, EnsembleForecastingModel):
            MLFlowStorageCallback._store_feature_importance(model=model, data_path=data_path)  # noqa: SLF001
            return

        for name, forecaster in model.forecasters.items():
            if isinstance(forecaster, ExplainableForecaster):
                fig = forecaster.plot_feature_importances()
                fig.write_html(data_path / f"feature_importances_{name}.html")  # pyright: ignore[reportUnknownMemberType]


__all__ = ["EnsembleMLFlowStorageCallback"]
