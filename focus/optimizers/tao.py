from .base import Optimizer
from pyadjoint.optimization.tao_solver import TAOSolver, MinimizationProblem

class TAOOptimizer(Optimizer):
    def __init__(self, rf, **parameters: dict):
        super().__init__(rf, parameters)
        self.problem = MinimizationProblem(self.rf)
        self.tao_solver = TAOSolver(self.problem, **self.parameters)

    def optimize(self):
        """
        Optimize the control variables for the given PDE solver using TAO.
        """
        controls = self.tao_solver.solve()
        return controls
    
    def check_optimal_controls(self, controls):
        """
        Check if the controls are valid.
        """
        if len(controls) != len(self.rf.controls):
            raise ValueError("Number of controls from TAO Solver does not match the number of controls in the Reduced Functional.")
        if not all(isinstance(c, type(self.rf.controls[0])) for c in controls):
            raise TypeError("Controls from TAO Solver are not of the same type as the controls in the Reduced Functional.")
        
    
    def get_optimal_control(self):
        """
        Get the optimal control variables after optimization.
        """
        controls = self.optimize()
        self.check_optimal_controls(controls)
        return controls

    #TODO: Rewrite the optimizer class to align with the updated solver and windowing classes
    #TODO: Add second order optimization
    