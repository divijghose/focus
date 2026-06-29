from firedrake import *
from abc import ABC, abstractmethod

class PDESolver1D(ABC):

    def __init__(self, mesh, function_space):
        self.mesh = mesh
        self.V = function_space
    
    @abstractmethod
    def build_solver(self):
        pass
    
    @abstractmethod
    def solve(self):
        pass

    
