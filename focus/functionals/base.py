from abc import ABC, abstractmethod


class BaseLoss(ABC):
    def __init__(self, u_desired, pde_solver, weighting: dict):
        self.weighting = weighting
        self.u_desired = u_desired
        self.pde_solver = pde_solver

    @abstractmethod
    def __call__(self, control, t_current, t_window):
        pass

    @abstractmethod
    def misfit_loss(self, t_current, t_window):
        """
        Returns the misfit loss at time t.
        """
        pass

    @abstractmethod
    def control_cost(self, control):
        pass

    @abstractmethod
    def get_desired_solution(self, t):
        """
        Returns the desired solution at time t.
        """
        return self.u_desired(t)
