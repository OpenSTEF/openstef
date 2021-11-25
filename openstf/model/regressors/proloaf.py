# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pandas as pd
import torch
import proloaf.datahandler as dh
from openstf.model.regressors.regressor import OpenstfRegressor
from proloaf.modelhandler import ModelWrapper
from typing import List, Dict, Any, Tuple
from torch.utils.tensorboard import SummaryWriter

# TODO: implement the hyperparameter optimalisation via optuna
# TODO: set the default for hyperparameters in the init of OpenstfProloafRegressor
# TODO: implement function for defining encoder and decoder features

def divide_scaling_groups(x: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    """Divides the column names over different type of scaling groups

        Args:
            x (pd.DataFrame): Dataframe from which columns have to be divided

        Returns:
            List of all the grouped features for scaling (three groups)

    """
    minmax_scale_features = []
    oh_scale_features = []
    no_scale_features = []

    for column in x.columns:
        if (x[column].min() <= -1) or (x[column].max() >= 1):
            minmax_scale_features.append(column)
        elif x[column].dtype == 'bool':
            oh_scale_features.append(column)
        else:
            no_scale_features.append(column)

    print(minmax_scale_features, oh_scale_features, no_scale_features)
    return minmax_scale_features, oh_scale_features, no_scale_features


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
        early_stopping_margin: float = 0,
        learning_rate: float = 1e-3,
        max_epochs: int = 50,
        device: str = "cpu",
        batch_size: int = 6,
        history_horizon: int = 24,
        horizon_minutes: int = 2880,  # 2 days in minutes,
    ):
        self.device = device
        self.batch_size = batch_size
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
        # Apply scaling and interpolation for NaN values
        x = dh.fill_if_missing(x, periodicity=24)
        selected_features, scalers = dh.scale_all(
            x,
            scalers=self.scalers,
            feature_groups=[
                {
                    "name": "main",
                    "scaler": ["minmax", -1.0, 1.0],
                    "features": self.minmax_scale_features,
                },
                {"name": "aux", "scaler": None, "features": self.no_scale_features},
            ],
        )

        # One hot encoding certain features
        onehot_feature_groups = [
            {
                "name": "main",
                "scaler": [
                    "onehot",
                ],
                "features": self.oh_scale_features,
            }
        ]
        for group in onehot_feature_groups:
            df_onehot = x.filter(group["features"])
            result_oh_scale = np.transpose(np.array(df_onehot.iloc[:, :], dtype=np.int))
            df_onehot.iloc[:, :] = result_oh_scale.T

        selected_features = pd.concat([selected_features, df_onehot], axis=1)

        x = selected_features.iloc[:, :].replace(np.nan, 0)
        inputs_enc = torch.tensor(
            x[self.encoder_features].to_numpy(), dtype=torch.float
        ).unsqueeze(dim=0)
        inputs_dec = torch.tensor(
            x[self.decoder_features].to_numpy(), dtype=torch.float
        ).unsqueeze(dim=0)
        prediction = (
            ModelWrapper.predict(self, inputs_enc, inputs_dec)[:, :, 0]
            .squeeze()
            .detach()
            .numpy()
        )
        return prediction

    def fit(
        self,
        x: pd.DataFrame,
        y: pd.DataFrame,
        eval_set: tuple = None,
        early_stopping_rounds: int = None,
        verbose: bool = False,
        **kwargs,
    ) -> ModelWrapper:
        # Apply scaling and interpolation for NaN values
        x = dh.fill_if_missing(x, periodicity=24)
        (self.minmax_scale_features, self.oh_scale_features, self.no_scale_features) = divide_scaling_groups(x)
        selected_features, self.scalers = dh.scale_all(
            x,
            scalers=self.scalers,
            feature_groups=[
                {
                    "name": "main",
                    "scaler": ["minmax", -1.0, 1.0],
                    "features": self.minmax_scale_features,
                },
                {"name": "aux", "scaler": None, "features": self.no_scale_features},
            ],
        )

        # One hot encoding certain features
        onehot_feature_groups = [
            {
                "name": "main",
                "scaler": [
                    "onehot",
                ],
                "features": self.oh_scale_features,
            }
        ]
        for group in onehot_feature_groups:
            df_onehot = x.filter(group["features"])
            result_oh_scale = np.transpose(np.array(df_onehot.iloc[:, :], dtype=np.int))
            df_onehot.iloc[:, :] = result_oh_scale.T

        selected_features = pd.concat([selected_features, df_onehot], axis=1)

        x = selected_features.iloc[:, :].replace(np.nan, 0)
        y = y.to_frame()
        self.target_id = [y.columns[0]]
        df_train = pd.concat([x, y], axis="columns", verify_integrity=True)
        print(f"{self.encoder_features = }")
        print(f"{self.decoder_features = }")
        train_dl, _, _ = dh.transform(
            df=df_train,
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
        df_val = pd.concat(eval_set[1], axis="columns", verify_integrity=True)
        _, validation_dl, _ = dh.transform(
            df=df_val,
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

        writer_tb = SummaryWriter()

        return self.run_training(train_dl, validation_dl, log_tb=writer_tb)

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
