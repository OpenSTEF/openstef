    forecast_input_data = input_data_with_features.loc[
        forecast_start:forecast_end, prediction_model.feature_names
    ]

    # Dit zijn eigenlijk twee transformers misschien?
    # 1. een range transformer?
    # 2. een feature selection transformer?

########################################################################################

from sklearn.pipeline import Pipeline

pipe = Pipeline(

)

pipe.fit()      # train the model

pipe.predict()  # make a prediction with a trained model