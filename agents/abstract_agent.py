from abc import ABC, abstractmethod

import numpy as np


class AbstractAgent(ABC):
    def __init__(self, name):
        self.name = name
        self.done = False

    @abstractmethod
    def predict(self, state):
        raise NotImplementedError

    @abstractmethod
    def update(
        self,
        state: np.ndarray,
        action: int,
        next_state: np.ndarray,
        reward: float,
        info: str,
        done: bool,
        truncated: bool,
    ):
        raise NotImplementedError
