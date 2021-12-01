# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
from typing import List, Dict, Any, Tuple, Union

import numpy as np
import pandas as pd

from openstef.model.regressors.regressor import OpenstfRegressor

# These imports will require the proloaf optional dependencies to be installed
import proloaf.datahandler as dh
from proloaf.modelhandler import ModelWrapper
import torch


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
        elif x[column].dtype == "bool":
            oh_scale_features.append(column)
        else:
            no_scale_features.append(column)

    return minmax_scale_features, oh_scale_features, no_scale_features


def apply_scaling(
    scaler_features: Tuple[List[str], List[str], List[str]],
    data: pd.DataFrame,
    scalers=None,
):
    """Applies different scaling methods to a certain dataframe (minmax, one hot, or no scaling)

    Args:
        scaler_features (Tuple[List[str], List[str], List[str]]): Three different lists with features for each scaling
        x (pd.DataFrame): Dataframe from which columns have to be divided
        scalers: scalers resulting from the previous scaling

    Returns:
        Dataframe with all the scaled features

    """
    selected_features, scalers = dh.scale_all(
        data,
        scalers=scalers,
        feature_groups=[
            {
                "name": "main",
                "scaler": ["minmax", -1.0, 1.0],
                "features": scaler_features[0],
            },
            {"name": "aux", "scaler": None, "features": scaler_features[2]},
        ],
    )

    # One hot encoding certain features
    onehot_feature_groups = [
        {
            "name": "main",
            "scaler": [
                "onehot",
            ],
            "features": scaler_features[1],
        }
    ]
    for group in onehot_feature_groups:
        df_onehot = data.filter(group["features"])
        result_oh_scale = np.transpose(np.array(df_onehot.iloc[:, :], dtype=np.int))
        df_onehot.iloc[:, :] = result_oh_scale.T

    if not df_onehot.columns.empty:
        selected_features = pd.concat([selected_features, df_onehot], axis=1)
    data = selected_features.iloc[:, :].replace(np.nan, 0)

    return data, scalers


class OpenstfProloafRegressor(OpenstfRegressor, ModelWrapper):
    def __init__(
        self,
        name: str = "model",
        core_net: str = "torch.nn.LSTM",
        relu_leak: float = 0.1,
        encoder_features: List[str] = [
            "historic_load",
        ],  # make sure historic load is present, TODO: implement so you can use None
        decoder_features: List[str] = [
            "air_density"
        ],  # TODO: implement so you can use None
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
        max_epochs: int = 100,
        device: Union[str, int] = "cpu",  # "cuda" or "cpu"
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
        self.to(device)

    @property
    def feature_names(self):
        return (
            ["load"] + self.encoder_features + self.decoder_features
        )  # TODO: gehele range, of een enkele feature

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        x = x[list(self.feature_names)[1:]]
        # Apply scaling and interpolation for NaN values
        x = dh.fill_if_missing(x, periodicity=24)
        x, _ = apply_scaling(
            [
                self.minmax_scale_features,
                self.oh_scale_features,
                self.no_scale_features,
            ],
            x,
            self.scalers,
        )

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
        x = x[list(self.feature_names)[1:]]
        # Apply scaling and interpolation for NaN values
        x = dh.fill_if_missing(x, periodicity=24)
        (
            self.minmax_scale_features,
            self.oh_scale_features,
            self.no_scale_features,
        ) = divide_scaling_groups(x)
        x, self.scalers = apply_scaling(
            [
                self.minmax_scale_features,
                self.oh_scale_features,
                self.no_scale_features,
            ],
            x,
            self.scalers,
        )
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

        val_x, val_y = eval_set[1][0], eval_set[1][1]
        val_x = dh.fill_if_missing(val_x, periodicity=24)
        val_x, _ = apply_scaling(
            [
                self.minmax_scale_features,
                self.oh_scale_features,
                self.no_scale_features,
            ],
            val_x,
            self.scalers,
        )

        df_val = pd.concat([val_x, val_y], axis="columns", verify_integrity=True)
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
        self.is_fitted_ = True

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
