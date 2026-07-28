from abc import ABC, abstractmethod
from typing import Any, Optional


class Optimizer(ABC):
    def __init__(self, rf, parameters: dict, function_space: Optional[Any] = None):
        self.rf = rf
        self.parameters = parameters
        self.function_space = function_space

    @abstractmethod
    def optimize(self):
        """
        Optimize the control variables for the given PDE solver.
        """
        pass

    @abstractmethod
    def get_optimal_control(self):
        """
        Get the optimal control variables after optimization.
        """
        pass
