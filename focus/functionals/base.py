from abc import ABC, abstractmethod
from firedrake import inner, dx, exp
class LossFunctional(ABC):
    def __init__(self, weighting):
        pass

    @abstractmethod
    def __call__(self, u_desired, u_sol):
        assemble(self)
        pass
    
    @abstractmethod
    def misfit_loss(self, u_desired, u_sol):
        return (exp(-lambda_t*t_window)*misfit_weight*inner(u_desired - u_sol, u_desired - u_sol)) 
        
    
    @abstractmethod
    def control_cost(self, control):
        return inner(control, control)*dx

    def loss_functional(t_current, t_window):
        u_desired.interpolate(u_desired_expr(t_current))
        return assemble(((exp(-lambda_t*t_window)*misfit_weight*inner(u_desired - u_new, u_desired - u_new)) + inner(m, m))*dx
    )