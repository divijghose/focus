import pytest
import numpy as np
from focus.windowing.fixed_window import FixedWindow
from firedrake import *
from firedrake.adjoint import *
continue_annotation()

# ============================================================================
#       Helper classes to build a dummy pde solver  and loss functional to pass to windowing classes
# ============================================================================
class DummyPDESolver:
    def __init__(self):

        self.num_controls = 1
        self.mesh = UnitIntervalMesh(10)
        self.V = FunctionSpace(self.mesh, "CG", 1)
        self.u = TrialFunction(self.V)
        self.v = TestFunction(self.V)
        self.kappa = Constant(1.0)
        self.dt = Constant(0.1)
        self.u_new = Function(self.V)
        self.u_old = Function(self.V)
        self.f = Function(self.V)
        self.a = self.u * self.v * dx + self.dt * self.kappa * dot(grad(self.u), grad(self.v)) * dx
        self.L = (self.u_old + self.dt * self.f) * self.v * dx
        self.solver = LinearVariationalSolver(LinearVariationalProblem(self.a, self.L, self.u_new))
        self.u_desired = Function(self.V)

class DummyLossFunctional:
    def __init__(self, control, pde_solver, ):
        self.lambda_t = 1.0
        self.control_weight = 1.0
        self.control_cost = self.control_weight * inner(control, control) * dx
        self.misfit_loss = inner(pde_solver.u_desired - pde_solver.u_new, pde_solver.u_desired - pde_solver.u_new) * dx
        self.total_loss = assemble(self.control_cost + self.misfit_loss)
        



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


