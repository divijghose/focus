from abc import ABC, abstractmethod
from firedrake import inner, dx, exp, assemble
class LossFunctional(ABC):
    def __init__(self, weighting:dict):
        self.lambda_t = weighting["lambda_t"]
        self.misfit_weight = weighting["misfit_weight"]
        pass

    @abstractmethod
    def __call__(self, u_desired, u_sol, t_window, control):
        assemble(self.control_cost(control) + self.misfit_loss(u_desired, u_sol, t_window))
    
    @abstractmethod
    def misfit_loss(self, u_desired, u_sol, t_window):
        return (exp(-self.lambda_t*t_window)*self.misfit_weight*inner(u_desired - u_sol, u_desired - u_sol)) 
        
    
    @abstractmethod
    def control_cost(self, control):
        return inner(control, control)*dx
