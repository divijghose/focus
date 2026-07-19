import pytest
import numpy as np
from focus.windowing.fixed_window import FixedWindow
from firedrake import *
# from firedrake.adjoint import *
from pyadjoint import *
from pyadjoint.optimization.tao_solver import TAOSolver, MinimizationProblem
# from firedrake.adjoint import *
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
        self.p.interpolate(Constant(0.0))
        self.root1 = root1
        self.root2 = root2

    def solve(self):
        self.u_new = (self.control - self.root1)*(self.control - self.root2) + self.p

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
        



# ============================================================================
#       Tests for FixedWindow class
# ============================================================================
def test_fixed_window_initialization():
    solver = DummyPDESolver()
    window_size = 5
    window_stride = 2
    fixed_window = FixedWindow(window_size=window_size, window_stride=window_stride, pde_solver=solver)

    assert fixed_window.window_size == window_size
    assert fixed_window.window_stride == window_stride
    assert fixed_window.current_window_start == 0
    assert fixed_window.current_window_end == window_size

    window_size = 2
    window_stride = 5
    with pytest.raises(AssertionError):
        FixedWindow(window_size=window_size, window_stride=window_stride, pde_solver=solver)


def test_fixed_window_advance():
    solver = DummyPDESolver()
    window_size = 5
    window_stride = 2

    fixed_window = FixedWindow(window_size=window_size, window_stride=window_stride, pde_solver=solver)

    fixed_window.advance_window()

    assert fixed_window.current_window_start == window_stride
    assert fixed_window.current_window_end == window_size + window_stride


def test_initialize_controls():

    solver = DummyPDESolver()

    window_size = 3
    window_stride = 1
    fixed_window = FixedWindow(window_size=window_size, window_stride=window_stride, pde_solver=solver)

    controls = fixed_window.initialize_controls()

    assert len(controls) == window_size
    for control in controls:
        assert isinstance(control, Function)

def test_run_first_window():
    continue_annotation()
    solver = DummyPDESolver()
    window_size = 3
    window_stride = 1
    fixed_window = FixedWindow(window_size=window_size, window_stride=window_stride, pde_solver=solver)

    controls = fixed_window.initialize_controls()

    loss_functional = DummyLossFunctional(solver)

    fixed_window.run_first_window(loss_functional)
    pause_annotation()

    assert fixed_window.Jhat is not None
    assert fixed_window.Jhat.derivative() is not None

@pytest.mark.parametrize("root1, root2", [(1.0, 10.0), (2.1, 49.5), (0.0, 56.4)])
def test_optimization_after_first_window(root1, root2):
    root1 = root1
    root2 = root2
    continue_annotation()
    solver = DummyPDESolver(root1=root1, root2=root2)
    window_size = 1
    window_stride = 1
    fixed_window = FixedWindow(window_size=window_size, window_stride=window_stride, pde_solver=solver)

    controls = fixed_window.initialize_controls()

    loss_functional = DummyLossFunctional( solver)
    

    fixed_window.run_first_window(loss_functional)
    pause_annotation()
    assert fixed_window.Jhat is not None
    assert fixed_window.Jhat.derivative() is not None
    tape = get_working_tape()
    tape.visualise("tape_test.pdf")

    


    problem = MinimizationProblem(fixed_window.Jhat)
    tao_solver = TAOSolver(problem, {
    "method": "lbfgs",
    "max_it": 4,
    "fatol": 0.0,
    "frtol": 0.0,
    "gatol": 1e-9,
    "grtol": 0.0,
        
    })
    assert fixed_window.Jhat is not None
    assert fixed_window.Jhat.derivative() is not None
    optimal_controls = tao_solver.solve()
    assert optimal_controls is not None
    assert np.allclose(optimal_controls[0].dat.data, root1), "Optimal control is not as expected."
    



