class Calibrator[I, O](ABC):

    def fit(data: I) -> None:
        pass

    def calibrate(predictions: O) -> O:
        pass
