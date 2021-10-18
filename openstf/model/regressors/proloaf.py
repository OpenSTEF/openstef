import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import utils.datahandler as dh
from openstf.model.regressors.regressor import OpenstfRegressor
from utils.modelhandler import ModelWrapper


class OpenstfProloafRegressor(OpenstfRegressor, ModelWrapper):
    def __init__(
            self,
            json_path: Path = "model/regressors/parameters/proloaf_parameters.json"
    ):
        with open(json_path) as json_file:
            parameters = json.load(json_file)

        ModelWrapper.__init__(
            self,
            **parameters
        )

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        # x = dh.fill_if_missing(x, periodicity=24)
        # This would be needed if no previous scaling is done
        # selected_features, scalers = dh.scale_all(x, **config)
        inputs_enc = torch.tensor(x[self.encoder_features]).unsqueeze(dim=0)
        inputs_dec = torch.tensor(x[self.decoder_features]).unsqueeze(dim=0)
        return super().predict(inputs_enc, inputs_dec)[:,:,0].squeeze().numpy()

    def fit(
            self,
            x: pd.DataFrame,
            y: pd.DataFrame,
            device: str = "cpu",
            batch_size: int = 10,
            split_percent: float = 0.85,
            history_horizon: int = 24,
            forecast_horizon: int = 24,
            eval_set: tuple = None,
            early_stopping_rounds: int = None,
            verbose: bool = False
    ) -> ModelWrapper:
        y = y.to_frame()
        self.target_id = [y.columns[0]]
        df = pd.concat([x, y], axis="columns", verify_integrity=True)
        train_dl, validation_dl, _ = dh.transform(
            df=df,  # TODO this needs to be x and y cacatinated
            encoder_features=self.encoder_features,
            decoder_features=self.decoder_features,
            batch_size=batch_size,
            history_horizon=history_horizon,
            forecast_horizon=forecast_horizon,
            target_id=self.target_id,
            train_split=split_percent,
            validation_split=1.0,
            device=device,
        )
        self.to(device)
        self.init_model()
        return self.run_training(train_dl, validation_dl)

    def get_params(self,  deep=True):
        model_params = self.get_model_config()
        training_params = self.get_training_config()
        return {**model_params, **training_params}

    def set_params(self, encoder_features = ['feature_a', 'feature_b'], decoder_features = ['feature_a', 'feature_b'], **params):
        self.update(**params)
        self.update(encoder_features=encoder_features)
        self.update(decoder_features=decoder_features)

        return self



if __name__ == "__main__":
    test = OpenstfProloafRegressor()
    print(test)
