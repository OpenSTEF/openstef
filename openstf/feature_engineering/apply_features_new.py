from abc import ABC, abstractmethod

class AbstractFeatureAplicator(ABC):

    @abstractmethod
    def add_features(self, df):
        pass


class TrainFeatureAplicator(AbstractFeatureAplicator)

    def add_features(self, df):

        df =
