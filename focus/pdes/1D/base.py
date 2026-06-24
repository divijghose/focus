from firedrake import *
from abc import ABC, abstractmethod

class PDESolver1D(ABC):

    def __init__(self, mesh):
        self.mesh = mesh
    
    @abstractmethod
    def solve(self):
        pass

    
