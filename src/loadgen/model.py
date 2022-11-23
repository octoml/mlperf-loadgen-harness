import abc
import typing

import numpy as np

ModelInput = typing.Dict[str, np.array]


class Model(abc.ABC):
    @abc.abstractmethod
    def predict(self, input: ModelInput):
        pass


class ModelInputSampler(abc.ABC):
    @abc.abstractmethod
    def sample(self, id: int) -> ModelInput:
        pass
