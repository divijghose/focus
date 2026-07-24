import pytest
import numpy as np
from focus.windowing.fixed_window import FixedWindow
from firedrake import *
from focus.optimizers.tao import TAOOptimizer
from firedrake.adjoint import *
from pyadjoint import *

# ============================================================================
#       Helper classes to build a dummy pde solver  and loss functional to pass to windowing classes
# ============================================================================
class DummyPDESolver:
    def __init__(self, root1=0.0, root2=0.0):
        
        self.num_controls = 1
        self.mesh = UnitIntervalMesh(1)
        self.V = FunctionSpace(self.mesh, "CG", 1)

        self.dt = 0.1
        self.u_new = Function(self.V)
  
        self.control = Function(self.V)
        self.control.interpolate(Constant(2.0))
        self.p = Function(self.V)
        self.p.interpolate(Constant(1.0))
        self.root1 = root1
        self.root2 = root2

    def solve(self):
        self.u_new = (self.control - self.root1)*(self.control - self.root2)*self.p

    def set_parameters(self):
        self.p = self.p


class DummyLossFunctional:
    def __init__(self,pde_solver):

        self.pde_solver = pde_solver

    def __call__(self, control, t_current, t_window):
        self.lambda_t = 10.0
        self.control_weight = 0.001
        self.control_cost = self.control_weight * inner(control, control) * dx
        self.misfit_loss = self.lambda_t * inner(self.pde_solver.u_new, self.pde_solver.u_new) * dx
        self.total_loss = assemble(self.misfit_loss + self.control_cost)
        return self.total_loss
@pytest.mark.parametrize("root1, root2", [(1.0, 10.0), (2.1, 49.5), (0.0, 56.4)])
def test_optimizer_with_fixed_window(root1, root2):
    # Create a dummy PDE solver and loss functional
    pde_solver = DummyPDESolver(root1=root1, root2=root2)
    loss_functional = DummyLossFunctional(pde_solver)

    # Create a FixedWindow instance
    window_size = 1
    window_stride = 1
    fixed_window = FixedWindow(window_size, window_stride, pde_solver)

    # Initialize controls for the fixed window
    fixed_window.initialize_controls(initial_expression=Constant(2.0))

    # Run the first window to set up the Reduced Functional and the Optimizer
    fixed_window.run_first_window(loss_functional)
    parameters_tao = {
        "method": "lbfgs",
        "max_it": 20,
        "fatol": 0.0,
        "frtol": 0.0,
        "gatol": 1e-9,
        "grtol": 0.0,
    }
    optimizer = TAOOptimizer(fixed_window.Jhat, parameters=parameters_tao)
    optimal_controls = optimizer.get_optimal_control()
    assert optimal_controls is not None
    assert np.allclose(optimal_controls[0].dat.data, root1), "Optimal control is not as expected."
    pde_solver.p.interpolate(Constant(1.0))
    optimal_controls = optimizer.get_optimal_control()
    assert optimal_controls is not None
    assert np.allclose(optimal_controls[0].dat.data, root1), "Optimal control is not as expected after re-interpolating pde_solver.p."