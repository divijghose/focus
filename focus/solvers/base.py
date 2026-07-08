from abc import ABC, abstractmethod

class Solver(ABC):

    def __init__(self, mesh, function_space):
        self.mesh = mesh
        self.V = function_space
    
    @abstractmethod
    def set_forcing_function(self):
        pass
    
    @abstractmethod
    def set_initial_condition(self):
        pass

    @abstractmethod
    def assign_initial_condition(self):
        pass
    
    @abstractmethod
    def set_bcs(self):
        pass

    @abstractmethod
    def build_solver(self):
        pass
    
    @abstractmethod
    def solve(self):
        pass

    @abstractmethod
    def assign_solution(self):
        pass

    @abstractmethod
    def allocate_control_variables(self):
        pass

    @abstractmethod
    def set_control_variables(self):
        pass

    
    
