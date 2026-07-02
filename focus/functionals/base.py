from abc import ABC, abstractmethod
class LossFunctional(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, u_desired, u_sol):
        pass
    
    @abstractmethod
    def cost_control(self, control):
        pass

    