from .base import BaseLoss
from firedrake import inner, dx, exp, assemble
class LossFunctional(BaseLoss):
    def __init__(self, u_desired, pde_solver,  weighting: dict):
        super().__init__(u_desired, pde_solver, weighting)
        self.lambda_t = self.weighting["lambda_t"]
        self.control_weight =self.weighting["control_weight"]
      
   
    # FIXME: There should be a better way to pass the time information
    def __call__(self, control, t_current, t_window):
        return assemble((self.control_cost(control) + self.misfit_loss(t_current, t_window)))
    
    def misfit_loss(self, t_current, t_window):
        """
        Returns the misfit loss at time t.
        """
        u_desired = self.u_desired(t_current)
        return (exp(-self.lambda_t*t_window)*inner(u_desired - self.pde_solver.u_new, u_desired - self.pde_solver.u_new))*dx
        
    
    def control_cost(self, control):
        return self.control_weight*inner(control, control)*dx
    
    def get_desired_solution(self, t):
        """
        Returns the desired solution at time t.
        """
        return self.u_desired(t)

#TODO: Rewrite the LossFunctional class to align with the updated solver and windowing classes
