# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

import dill
import structlog


class XGBQuantileModel:
    def __init__(self, quantile_models=None, model_file=None):

        self.logger = structlog.get_logger(self.__class__.__name__)

        if quantile_models is not None and model_file is None:
            self.quantile_models = quantile_models
            self.best_ntree_limit = self.quantile_models[0.5].best_ntree_limit
            self.feature_names = self.quantile_models[0.5].feature_names
            self.feature_types = self.quantile_models[0.5].feature_types
        elif model_file is not None and quantile_models is None:
            self._load_from_disk(model_file)
        else:
            # TODO describe reason and cases (if en elif case)
            print("Could not initialize!")

        # Check if 0.5 quantile is present, raise exception otherwise
        if 0.5 not in self.quantile_models.keys():
            raise ValueError("Quantile model does not have a model for the median")

    def predict(self, *args, quantile=0.5, **kwargs):
        return self.quantile_models[quantile].predict(*args, **kwargs)

    def get_score(self, *args, quantile=0.5, **kwargs):
        return self.quantile_models[quantile].get_score(*args, **kwargs)

    def save_model(self, save_location):

        # Save dict with quantile models
        save_location = save_location.parent / "model_quantile.bin"
        dill.dump(self.quantile_models, file=open(save_location, "wb"))

        # Save best n_trees
        for quantile in self.quantile_models.keys():
            try:
                save_location = save_location.parent / f"best_iteration_{quantile}.pkl"
                dill.dump(
                    self.quantile_models[quantile].best_ntree_limit,
                    file=open(save_location, "wb"),
                )
            except Exception as e:
                print(
                    "No best iteration found in the output, best iteration not saved for quantile "
                    + str(quantile)
                    + ".",
                    e,
                )

    def _load_from_disk(self, model_file):
        # Save dict with quantile models
        self.quantile_models = dill.load(open(model_file, "rb"))
        # Save best n_trees
        for quantile in self.quantile_models.keys():
            try:
                filename = model_file.parent / f"best_iteration_{quantile}.pkl"
                self.quantile_models[quantile].best_ntree_limit = dill.load(
                    open(filename, "rb")
                )
            except Exception as e:
                print(
                    "No best iteration found for quantile " + str(quantile) + ".",
                    e,
                )
                self.quantile_models[quantile].best_ntree_limit = 15
