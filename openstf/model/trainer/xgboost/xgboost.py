# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import timedelta

import numpy as np
import pandas as pd
import pytz
import xgboost as xgb
import structlog

from openstf.metrics import metrics
from openstf.validation import validation
from openstf.feature_engineering.general import remove_extra_feature_columns
from openstf.model_selection.model_selection import split_data_train_validation_test
from openstf.model.trainer.trainer import AbstractModelTrainer

# Available trainings period durations for optimization
# After preprocessing, the data consists of 75 days (of the original 90 days).
# To prevent overfitting on a very short period of time, the test and validation
# sets are always a constant fraction of the total data period (75 days).
# We take an optimized fraction of the remaining data as our train-data.
# Given the current fractions for test and validation, the longest train-block
# is 60 days.


class XGBModelTrainer(AbstractModelTrainer):
    def __init__(self, pj):
        super().__init__(pj)
        self.logger = structlog.get_logger(self.__class__.__name__)
        # See XGBoost Parameters
        # https://xgboost.readthedocs.io/en/latest/parameter.html
        # look at the parameters for Tree booster
        self.hyper_parameters = {
            # general model specific hyper paramater (NOT optimized)
            # "silent": 1,
            "disable_default_eval_metric": 1,
            # model specific hyper parameters (optimized)
            "subsample": 0.9,
            "min_child_weight": 4,
            "max_depth": 8,
            "gamma": 0.5,
            "colsample_bytree": 0.85,
            "eta": 0.1,
            # model specific hyper parameters (NOT optimized)
            "objective": "reg:squarederror",
            # generic hyper parameters
            "featureset_name": "G",
            "training_period_days": 90,
        }
        # private key lists for filtering
        self._xgb_hyper_parameter_keys = [
            # "silent",
            "disable_default_eval_metric",
            "subsample",
            "min_child_weight",
            "max_depth",
            "gamma",
            "colsample_bytree",
            "eta",
            "objective",
        ]

    @property
    def feature_importance(self):
        """Return feature importances and weights of trained model.

        Returns:
            pandas.DataFrame: A DataFrame describing the feature importances and
            weights of the trained model.

        """
        if self.trained_model is None:
            return None

        feature_gain = pd.DataFrame(
            self.trained_model.get_score(importance_type="gain"), index=["gain"]
        ).T
        feature_gain /= feature_gain.sum()

        feature_weight = pd.DataFrame(
            self.trained_model.get_score(importance_type="weight"), index=["weight"]
        ).T
        feature_weight /= feature_weight.sum()

        feature_importance = pd.merge(
            feature_gain, feature_weight, left_index=True, right_index=True
        )
        feature_importance.sort_values(by="gain", ascending=False, inplace=True)

        return feature_importance

    def train(
        self,
        train_data,
        validation_data,
        callbacks=None,
        early_stopping_rounds=10,
        num_boost_round=500,
    ):
        """Method that trains XGBoost model based on train and validation data.

        Args:
            train_data (pandas.DataFrame): The train data. Assumed is that the
                first column is the label.
            validation_data (pandas.DataFrame): The validation data.
            callbacks (list of callable): List of callback functions that can be
                called at the end of each training iteration
            early_stopping_rounds (int): early stop training of new estimators
                after this many rounds of no improvement
            num_boost_round (int): can be large since we use early stopping.

        Returns:
            xgboost.Booster: this object is also stored in the self.trained_model
            attribute.

        """
        # Convert train and validation sets to Dmatrix format for computational
        #  efficiency. Drop Horizon column
        dtrain = xgb.DMatrix(
            train_data.iloc[:, 1:].drop("Horizon", axis=1, errors="ignore"),
            label=train_data.iloc[:, 0],
        )  # [:,1:-1] excludes label and 'horizon' column
        dvalidation = xgb.DMatrix(
            validation_data.iloc[:, 1:].drop("Horizon", axis=1, errors="ignore"),
            label=validation_data.iloc[:, 0],
        )

        # Define data set to be monitored during training, the last(validation)
        #  will be used for early stopping
        watchlist = [(dtrain, "train"), (dvalidation, "validation")]

        # Define callback function that is used to calculate metrics while training
        def custom_eval(preds, dset):
            return "MAE", metrics.mae(dset.get_label(), preds)

        # get the xgb (hyper) parameters
        params = {k: self.hyper_parameters[k] for k in self._xgb_hyper_parameter_keys}

        # Train and validate model
        self.trained_model = xgb.train(
            dtrain=dtrain,
            params=params,
            evals=watchlist,
            num_boost_round=num_boost_round,  # Can be large because we are early stopping anyway
            feval=custom_eval,
            verbose_eval=False,
            early_stopping_rounds=early_stopping_rounds,
            callbacks=callbacks,
        )

        # Add confidence interval data
        self.calculate_confidence_interval(validation_data)

        # Return model object
        return self.trained_model

    def better_than_old_model(self, test_data, penalty_factor=1.2):
        """Method that checks if newly trained model is better than the old model.

        Args:
            test_data (pandas.DataFrame): Data to test the models on.
            penalty_factor (float): Factor used to penalize the old model.


        Returns:
            bool: True if the new model performs better, False otherwise.

        """
        # Check if the old model is not None and try to make a prediction with the old model
        if self.old_model is None:
            self.logger.warning("No old model available, using new model")
            return True

        # Check if trained model is not None
        elif self.trained_model is None:
            self.logger.warning(
                "New model is not yet trained, could not compare performance!"
            )
            return False

        # Check if old model has same feature names. If not, use new model
        # TODO instead of model.predict, 'predictionmodel'.make_forecast would be better
        # However, this would require significant restructuring on model design.
        elif self.trained_model.feature_names != self.old_model.feature_names:
            self.logger.warning("Old model had different features. Using new model")
            return True

        else:
            try:
                # Ask old model for prediction
                prediction_old_model = self.old_model.predict(
                    xgb.DMatrix(
                        test_data.iloc[:, 1:].drop("Horizon", axis=1, errors="ignore")
                    )
                )
            except Exception as e:
                self.logger.error("Could not compare to old model!", exc_info=e)
                return True

        try:
            # Ask new model for prediction
            prediction_new_model = self.trained_model.predict(
                xgb.DMatrix(
                    test_data.iloc[:, 1:].drop("Horizon", axis=1, errors="ignore")
                ),
                ntree_limit=self.trained_model.best_ntree_limit,
            )
        except Exception as e:
            self.logger.error("Could not get prediction from new model!", exc_info=e)
            return False

        # Calculate scores
        old_mae = metrics.mae(prediction_old_model, test_data.iloc[:, 0])
        new_mae = metrics.mae(prediction_new_model, test_data.iloc[:, 0])

        # Compare and return True if new model is better, False otherwise
        if (new_mae < (old_mae * penalty_factor)) or (prediction_old_model is np.nan):
            return True

        return False

    def store_model(self):
        """Stores the model."""
        self._store_model()

    def calculate_confidence_interval(self, validation_data):
        """Gets confidence interval data.

        This data is required to add confidence intervals to the predictions.

        Args:
            validation_data (pandas.DataFrame): The data to get the confidence
            intervals over.

        Returns:
            pandas.DataFrame: The confidence interval data. This is also stored
            in the self.confidence_interval attribute.

        """

        # Define some variables
        predicted = None
        self.confidence_interval = pd.DataFrame()

        # Loop over Horizons and ask prediction for each specific horizon
        for horizon in validation_data.Horizon.unique():
            # Make subset for this specific horizon
            sub_val = validation_data[validation_data.Horizon == horizon]
            # Check if returned object is not None and try to make a prediction with the new model
            if self.trained_model is None:
                self.logger.info(
                    "New model is not yet trained, could not compute corrections!"
                )
            else:
                try:
                    predicted = self.trained_model.predict(
                        xgb.DMatrix(
                            sub_val.iloc[:, 1:].drop("Horizon", axis=1, errors="ignore")
                        ),
                        ntree_limit=self.trained_model.best_ntree_limit,
                    )
                except Exception as e:
                    self.logger.error(
                        "Could not get prediction from new model!", exc_info=e
                    )

            # Calculate confidence interval for this horizon
            confidence_interval_horizon = self._calculate_confidence_interval(
                sub_val.iloc[:, 0], predicted
            )
            confidence_interval_horizon[
                "horizon"
            ] = horizon  # Label with respective horizon
            self.confidence_interval = self.confidence_interval.append(
                confidence_interval_horizon
            )

        return self.confidence_interval

    def training_period_objective(
        self,
        trial,
        error_function,
        unprocessed_data,
        training_durations_days,
        optimized_parameters,
        featuresets,
    ):
        """Objective function that picks the optimal training_duration value.

        This function should be used after the hyper_params_objective has been completed
        by optuna.

        Args:
            trial: optuna trial object that is passed on during hyper parameter
                optimalisation.
            error_function (callable): Function to calculate the error metric to be
                optimized, preferably one from openstf.metrics.metrics.
            unprocessed_data (pandas.DataFrame): Data and features that have not yet
                been pre-processed.
            training_durations_days (list of int): Candidate training durations.
            parameter_space (dict): Parameters previously optimized by optuna using the
                hyper_parameters_objective function.
            featuresets (dict): All feature sets: with keys the featureset_name and
                values a list of feature names.

        Returns:
            float: Error metric value that is either minimized or maximized by the
            optuna study.

        """

        # Make selection of training-duration
        training_period = trial.suggest_categorical(
            "training_period_days", training_durations_days
        )
        featureset_names = list(featuresets.keys())
        featureset_name = trial.suggest_categorical("featureset_name", featureset_names)
        self.logger.debug(
            "Current iteration of model trainer",
            featureset_name=featureset_name,
        )

        # Update hyper parameters
        self.hyper_parameters.update(optimized_parameters)

        # Set training duration and shrink incoming data accordingly
        datetime_start = (
            unprocessed_data.index.max() - timedelta(days=training_period)
        ).replace(tzinfo=pytz.UTC)
        shortened_data = unprocessed_data.loc[unprocessed_data.index > datetime_start]

        # Validate input data
        validated_data = validation.validate(shortened_data)

        # Select feature set
        featureset = featuresets[featureset_name]

        validated_data_data_with_features = remove_extra_feature_columns(
            validated_data, featurelist=featureset
        )

        # Clean up data
        total_data = validation.clean(validated_data_data_with_features)

        # Split data in train, test and validation sets, note we are using the
        # backtest option here because we assume hyperparameters do not change
        # much over time
        train_data, validation_data, test_data = split_data_train_validation_test(
            total_data,
            test_fraction=0.1,
            validation_fraction=0.1,
            back_test=True,
        )

        # Train model
        model = self.train(train_data, validation_data)

        # Make prediction on test data and prepare dataframes for comparison
        prediction = pd.DataFrame(
            model.predict(
                xgb.DMatrix(
                    test_data.iloc[:, 1:].drop("Horizon", axis=1, errors="ignore")
                ),
                ntree_limit=model.best_ntree_limit,
            )
        )
        realised = pd.DataFrame(test_data.iloc[:, 0])
        prediction.index = realised.index  # Set correct DateTime index

        # Calculate error metric
        error = error_function(realised.iloc[:, 0], prediction.iloc[:, 0])
        return error

    def hyper_params_objective(
        self, trial, error_function, clean_data_with_all_features, featuresets
    ):
        """Objective function used during hyperparameter optimalization.

        This objective function picks hyperparameter values in the pre-defined
        (in this method) parameter space and returns a resulting error. This
        error metric is either maximized or minimized by optuna.
        The metric type can be chosen by passing an error_function.

        Args:
            trial: optuna trial object that is passed on during hyper parameter
                optimalisation.
            error_function (callable): Function to calculate the error metric to be
                optimized, preferably one from openstf.metrics.metrics.
            clean_data_with_all_features (pandas.DataFrame): Data and features ready for
                model training
            featuresets (dict): All feature sets: with keys the featureset_name and
                values a list of feature names.

        Returns:
            float: Error metric value that is either minimized or maximized by the
            optuna study.

        """

        # This is an example on how to implement training period optimalisation,
        # if this gets to large a separate objective function should be build
        # Configure parameter space
        # Here the six parameters that give the most benfefit to prediction quality are optimized:
        # 1. eta [xgb_default=0.3],
        #   - Analogous to learning rate in GBM,
        #   - Makes the model more robust by shrinking the weights on each step
        #   - Typical final values to be used: 0.01-0.2
        # 2. subsample [xgb_default=1]
        #   - Same as the subsample of GBM. Denotes the fraction of observations to be randomly sampled for each tree.
        #   - Lower values make the algorithm more conservative and prevents overfitting but too small values might lead to under-fitting.
        #   - Typical values: 0.5-1
        # 3. min_child_weight [xgb_default=1]
        #   - Defines the minimum sum of weights of all observations required in a child.
        #   - This is similar to min_child_leaf in GBM but not exactly. This refers to min “sum of weights” of observations while GBM has min “number of observations”.
        #   - Used to control over-fitting. Higher values prevent a model from learning relations which might be highly specific to the particular sample selected for a tree.
        #   - Too high values can lead to under-fitting hence.
        # 4. max_depth [xgb_default=6]
        #   - The maximum depth of a tree, same as GBM.
        #   - Used to control over-fitting as higher depth will allow model to learn relations very specific to a particular sample.
        #   - Typical values: 3-10
        # 5. gamma [xgb_default=0]
        #   - A node is split only when the resulting split gives a positive reduction in the loss function. Gamma specifies the minimum loss reduction required to make a split.
        #   - Makes the algorithm conservative. The values can vary depending on the loss function and should be tuned.
        # 6. colsample_bytree [xgb_default=1]
        #   - Similar to max_features in GBM. Denotes the fraction of columns to be randomly samples for each tree.
        #   - Typical values: 0.5-1
        #
        # More information about these and other parameters and their limits can be found at
        # https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
        parameter_space = {
            "eta": trial.suggest_float("eta", 0.01, 0.2),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 6),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        }

        # TODO featureset and trial period optimization is probably general
        # for all model types. When we start using other models we might want to
        # move this to the base class
        featureset_names = list(featuresets.keys())
        featureset_name = trial.suggest_categorical("featureset_name", featureset_names)
        self.logger.debug(
            "Current iteration of model trainer",
            featureset_name=featureset_name,
            parameter_space=parameter_space,
        )

        # Update hyper parameters (used by self.train())
        self.hyper_parameters.update(parameter_space)
        self.hyper_parameters["featureset_name"] = featureset_name

        # Split data in train, test and validation sets, note we are using the
        # backtest option here because we assume hyperparameters do not change
        # much over time
        train_data, validation_data, test_data = split_data_train_validation_test(
            clean_data_with_all_features,
            test_fraction=0.1,
            validation_fraction=0.1,
            back_test=True,
        )
        # Train model
        model = self.train(train_data, validation_data)

        # Make prediction on test data and prepare dataframes for comparison
        prediction = pd.DataFrame(
            model.predict(
                xgb.DMatrix(
                    test_data.iloc[:, 1:].drop("Horizon", axis=1, errors="ignore")
                ),
                ntree_limit=model.best_ntree_limit,
            )
        )
        realised = pd.DataFrame(test_data.iloc[:, 0])
        prediction.index = realised.index  # Set correct DateTime index

        # Calculate error metric
        error = error_function(realised.iloc[:, 0], prediction.iloc[:, 0])
        return error
