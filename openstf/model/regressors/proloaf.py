import numpy as np
import pandas as pd
import torch
import utils.datahandler as dh
from openstf.model.regressors.regressor import OpenstfRegressor
from utils.modelhandler import ModelWrapper
from typing import List, Dict, Any


class OpenstfProloafRegressor(OpenstfRegressor, ModelWrapper):
    def __init__(
        self,
        name: str = "model",
        core_net: str = "torch.nn.LSTM",
        relu_leak: float = 0.1,
        encoder_features: List[str] = None,
        decoder_features: List[str] = None,
        core_layers: int = 1,
        rel_linear_hidden_size: float = 1.0,
        rel_core_hidden_size: float = 1.0,
        dropout_fc: float = 0.4,
        dropout_core: float = 0.3,
        training_metric: str = "nllgauss",
        metric_options: Dict[str, Any] = {},
        optimizer_name: str = "adam",
        early_stopping_patience: int = 7,
        early_stopping_margin: float = 0.0,
        learning_rate: float = 1e-4,
        max_epochs: int = 1,
        device: str = "cpu",
        batch_size: int = 10,
        # split_percent: float = 0.85,  # XXX now unsued
        history_horizon: int = 24,
        horizon_minutes: int = None,
    ):
        self.device = device
        self.batch_size = batch_size
        # self.split_percent = split_percent  # XXX now unsued
        self.history_horizon = history_horizon
        self.forecast_horizon = int(horizon_minutes / 60)
        ModelWrapper.__init__(
            self,
            name=name,
            core_net=core_net,
            relu_leak=relu_leak,
            encoder_features=encoder_features,
            decoder_features=decoder_features,
            core_layers=core_layers,
            rel_linear_hidden_size=rel_linear_hidden_size,
            rel_core_hidden_size=rel_core_hidden_size,
            dropout_fc=dropout_fc,
            dropout_core=dropout_core,
            training_metric=training_metric,
            metric_options=metric_options,
            optimizer_name=optimizer_name,
            early_stopping_patience=early_stopping_patience,
            early_stopping_margin=early_stopping_margin,
            learning_rate=learning_rate,
            max_epochs=max_epochs,
        )

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        # x = dh.fill_if_missing(x, periodicity=24)
        # This would be needed if no previous scaling is done
        # selected_features, scalers = dh.scale_all(x, **config)
        inputs_enc = torch.tensor(x[self.encoder_features]).unsqueeze(dim=0)
        inputs_dec = torch.tensor(x[self.decoder_features]).unsqueeze(dim=0)
        return super().predict(inputs_enc, inputs_dec)[:, :, 0].squeeze().numpy()

    def fit(
        self,
        x: pd.DataFrame,
        y: pd.DataFrame,
        eval_set: tuple = None,
        early_stopping_rounds: int = None,
        verbose: bool = False,
    ) -> ModelWrapper:
        y = y.to_frame()
        self.target_id = [y.columns[0]]
        df_train = pd.concat([x, y], axis="columns", verify_integrity=True)
        print(f"{self.encoder_features = }")
        print(f"{self.decoder_features = }")
        train_dl, _, _ = dh.transform(
            df=df_train,  # TODO this needs to be x and y cacatinated
            encoder_features=self.encoder_features,
            decoder_features=self.decoder_features,
            batch_size=self.batch_size,
            history_horizon=self.history_horizon,
            forecast_horizon=self.forecast_horizon,
            target_id=self.target_id,
            train_split=1.0,
            validation_split=1.0,
            device=self.device,
        )
        df_val = pd.concat(eval_set[1],
                           axis="columns", verify_integrity=True)
        _, validation_dl, _ = dh.transform(
            df=df_val,  # TODO this needs to be x and y cacatinated
            encoder_features=self.encoder_features,
            decoder_features=self.decoder_features,
            batch_size=self.batch_size,
            history_horizon=self.history_horizon,
            forecast_horizon=self.forecast_horizon,
            target_id=self.target_id,
            train_split=0.0,
            validation_split=1.0,
            device=self.device,
        )
        self.to(self.device)
        self.init_model()
        return self.run_training(train_dl, validation_dl)

    def get_params(self, deep=True):
        model_params = self.get_model_config()
        training_params = self.get_training_config()
        return {**model_params, **training_params}

    def set_params(self, **params):
        self.update(**params)
        return self


if __name__ == "__main__":
    test = OpenstfProloafRegressor()
    print(test)
