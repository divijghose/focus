from abc import ABC, abstractmethod


class Optimizer(ABC):
    def __init__(self, rf, **parameters: dict):
        self.rf = rf
        self.parameters = parameters

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
