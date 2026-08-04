from abc import ABC, abstractmethod


class Solver(ABC):

    def __init__(self, mesh, function_space):
        self.mesh = mesh
        self.V = function_space

    @abstractmethod
    def set_forcing_function(self, f_expr, t=0.0):
        pass

    @abstractmethod
    def set_initial_condition(self, u0):
        pass

    @abstractmethod
    def set_bcs(self, bcs):
        pass

    @abstractmethod
    def build_solver(self):
        pass

    @abstractmethod
    def solve(self):
        pass

    @abstractmethod
    def allocate_control_variables(self):
        pass

    @abstractmethod
    def set_control(self):
        pass

    @abstractmethod
    def allocate_parameters(self):
        pass

    @abstractmethod
    def set_parameters(self):
        pass

    @abstractmethod
    def set_desired_solution(self, expression):
        pass
