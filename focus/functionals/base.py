from abc import ABC, abstractmethod
from firedrake import inner, dx, exp, assemble
class LossFunctional(ABC):
    def __init__(self, u_desired, pde_solver, control, weighting: dict):
        self.lambda_t = weighting["lambda_t"]
        self.misfit_weight = weighting["misfit_weight"]
        self.u_desired = u_desired
        self.pde_solver = pde_solver
        self.control = control
        self.t_window = 0.0
        pass

    @abstractmethod
    def __call__(self):
        assemble((self.control_cost() + self.misfit_loss()))*dx
    
    @abstractmethod
    def misfit_loss(self):
        return (exp(-self.lambda_t*self.t_window)*self.misfit_weight*inner(self.u_desired - self.pde_solver.u_new, self.u_desired - self.pde_solver.u_new)) 
        
    
    @abstractmethod
    def control_cost(self):
        return inner(self.control, self.control)*dx
