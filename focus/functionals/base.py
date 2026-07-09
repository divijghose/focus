from abc import ABC, abstractmethod
from firedrake import inner, dx, exp, assemble
class LossFunctional(ABC):
    def __init__(self, u_desired, pde_solver, windowing, weighting: dict):
        self.lambda_t = weighting["lambda_t"]
        self.control_weight = weighting["control_weight"]
        self.u_desired = u_desired
        self.pde_solver = pde_solver
        self.windowing = windowing
        self.window_size = windowing.window_size
        self.window_stride = windowing.window_stride
   
    # FIXME: There should be a better way to pass the time information
    @abstractmethod
    def __call__(self, control, t_current, t_window):
        assemble((self.control_cost(control) + self.misfit_loss(t_current, t_window)))*dx
    
    @abstractmethod
    def misfit_loss(self, t_current, t_window):
        """
        Returns the misfit loss at time t.
        """
        u_desired = self.u_desired(t_current)
        return (exp(-self.lambda_t*t_window)*inner(u_desired - self.pde_solver.u_new, u_desired - self.pde_solver.u_new)) 
        
    
    @abstractmethod
    def control_cost(self, control):
        return self.control_weight*inner(control, control)*dx
    
    @abstractmethod
    def get_desired_solution(self, t):
        """
        Returns the desired solution at time t.
        """
        return self.u_desired(t)

#TODO: Rewrite the LossFunctional class to align with the updated solver and windowing classes
