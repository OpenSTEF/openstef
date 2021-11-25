import numpy as np
import pandas as pd
import torch
import proloaf.datahandler as dh
from openstf.model.regressors.regressor import OpenstfRegressor
from proloaf.modelhandler import ModelWrapper
from typing import List, Dict, Any, Union
from torch.utils.tensorboard import SummaryWriter

NO_SCALE_FEATURES = ['humidity',
                        'sjv_E1A',
                        'sjv_E1B',
                        'sjv_E1C',
                        'sjv_E2A',
                        'sjv_E2B',
                        'sjv_E3A',
                        'sjv_E3B',
                        'sjv_E3C',
                        'sjv_E3D',
                        'sjv_E4A',]

MINMAX_SCALE_FEATURES = ['APX',
                         'clearSky_dlf',
                         'clearSky_ulf',
                         'clouds',
                         'mxlD',
                         'pressure',
                         'radiation',
                         'snowDepth',
                         'temp',
                         'winddeg',
                         'windspeed',
                         'windspeed_100m',
                         'rain',
                         'Month',
                         'Quarter',
                         'air_density',
                         'dewpoint',
                         'saturation_pressure',
                         'historic_load',]

OH_SCALE_FEATURES = ['IsSunday',
                     'IsWeekDay',
                     'IsWeekendDay',]

ENCODER_FEATURES = ['APX',
                     'sjv_E1A',
                     'sjv_E1B',
                     'sjv_E1C',
                     'sjv_E2A',
                     'sjv_E2B',
                     'sjv_E3A',
                     'sjv_E3B',
                     'sjv_E3C',
                     'sjv_E3D',
                     'historic_load']

DECODER_FEATURES = ['clouds',
                     'radiation',
                     'temp',
                     'winddeg',
                     'windspeed',
                     'windspeed_100m',
                     'pressure',
                     'humidity',
                     'rain',
                     'mxlD',
                     'snowDepth',
                     'clearSky_ulf',
                     'clearSky_dlf',
                     'air_density',
                     'dewpoint',
                     'saturation_pressure']

class OpenstfProloafRegressor(OpenstfRegressor, ModelWrapper):
    def __init__(
        self,
        name: str = "model",
        core_net: str = "torch.nn.LSTM",
        relu_leak: float = 0.1,
        encoder_features: List[str] = ENCODER_FEATURES,#None,
        decoder_features: List[str] = DECODER_FEATURES,#None,
        core_layers: int = 1,
        rel_linear_hidden_size: float = 1.0,
        rel_core_hidden_size: float = 1.0,
        dropout_fc: float = 0.4,
        dropout_core: float = 0.3,
        training_metric: str = "nllgauss", #"PinnballLoss", #"nllgauss",
        metric_options: Dict[str, Any] = {},#{'quantiles': [0.05, 0.95]}, #{},
        optimizer_name: str = "adam",
        early_stopping_patience: int = 7,
        early_stopping_margin: float = 0,
        learning_rate: float = 1e-3,
        max_epochs: int = 100,
        device: Union[str,int] = "cuda",
        batch_size: int = 6,
        # split_percent: float = 0.85,  # XXX now unsued
        history_horizon: int = 24,
        horizon_minutes: int = 2880,  # 2 days in minutes,
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
        self.to(device)

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        x = dh.fill_if_missing(x, periodicity=24) #for scaling and NAN
        selected_features, scalers = dh.scale_all(x, scalers=self.scalers, feature_groups = [
                {
                    "name": "main",
                    "scaler": [
                        "minmax",
                        -1.0,
                        1.0
                    ],
                    "features": MINMAX_SCALE_FEATURES
                },
                {
                    "name": "aux",
                    "scaler": None,
                    "features": NO_SCALE_FEATURES
                }
        ]) #for scaling and NAN

        # One hot encoding certain features
        onehot_feature_groups = [
                {
                    "name": "main",
                    "scaler": [
                        "onehot",
                    ],
                    "features": OH_SCALE_FEATURES
                }
        ]
        for group in onehot_feature_groups:
            df_onehot = x.filter(group["features"])
            result_oh_scale = np.transpose(np.array(df_onehot.iloc[:, :], dtype=np.int))
            df_onehot.iloc[:, :] = result_oh_scale.T

        selected_features = pd.concat([selected_features, df_onehot], axis=1)

        selected_features=selected_features.iloc[:, :].replace(np.nan, 0)

        x = selected_features # added, remove if not necessary
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
        x = dh.fill_if_missing(x, periodicity=24) #for scaling and NAN
        selected_features, self.scalers = dh.scale_all(x, scalers=self.scalers, feature_groups=[
                {
                    "name": "main",
                    "scaler": [
                        "minmax",
                        -1.0,
                        1.0
                    ],
                    "features": MINMAX_SCALE_FEATURES
                },
                {
                    "name": "aux",
                    "scaler": None,
                    "features": NO_SCALE_FEATURES
                }
        ]) #**config) #for scaling and NAN

        # One hot encoding certain features
        onehot_feature_groups = [
                {
                    "name": "main",
                    "scaler": [
                        "onehot",
                    ],
                    "features": OH_SCALE_FEATURES,
                }
        ]
        for group in onehot_feature_groups:
            df_onehot = x.filter(group["features"])
            result_oh_scale = np.transpose(np.array(df_onehot.iloc[:, :], dtype=np.int))
            df_onehot.iloc[:, :] = result_oh_scale.T

        selected_features = pd.concat([selected_features, df_onehot], axis=1)

        selected_features = selected_features.iloc[:, :].replace(np.nan, 0)

        x = selected_features # added, remove if not necessary
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
        df_val = pd.concat(eval_set[1], axis="columns", verify_integrity=True)
        _, validation_dl, _ = dh.transform(
            df=df_val,  # TODO this needs to be x and y concatinated
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

        return self.run_training(train_dl, validation_dl, log_tb = writer_tb)

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
