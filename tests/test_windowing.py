import pytest
import numpy as np
from focus.windowing.fixed_window import FixedWindow
from firedrake import *


# ============================================================================
#       Helper functions to build a dummy pde solver to pass to windowing classes
# ============================================================================
def build_heat_equation_solver():
    mesh = UnitIntervalMesh(10)
    V = FunctionSpace(mesh, "CG", 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    kappa = Constant(1.0)
    dt = Constant(0.1)
    u_new = Function(V)
    u_old = Function(V)
    f = Function(V)
    a = u * v * dx + dt * kappa * dot(grad(u), grad(v)) * dx
    L = (u_old + dt * f) * v * dx
    solver = LinearVariationalSolver(LinearVariationalProblem(a, L, u_new))
    return solver


# ============================================================================
#       Tests for FixedWindow class
# ============================================================================
def test_fixed_window_initialization():
    window_size = 5
    window_stride = 2
    fixed_window = FixedWindow(window_size, window_stride)

    assert fixed_window.window_size == window_size
    assert fixed_window.window_stride == window_stride
    assert fixed_window.current_window_start == 0
    assert fixed_window.current_window_end == window_size

    window_size = 2
    window_stride = 5
    with pytest.raises(AssertionError):
        FixedWindow(window_size, window_stride)


def test_fixed_window_advance():
    window_size = 5
    window_stride = 2

    fixed_window = FixedWindow(window_size, window_stride)

    fixed_window.advance_window()

    assert fixed_window.current_window_start == window_stride
    assert fixed_window.current_window_end == window_size + window_stride


def test_initialize_controls():
    from firedrake import UnitIntervalMesh, FunctionSpace, Function

    mesh = UnitIntervalMesh(10)
    V = FunctionSpace(mesh, "CG", 2)

    window_size = 3
    window_stride = 1
    fixed_window = FixedWindow(window_size, window_stride)

    controls = fixed_window.initialize_controls(V)

    assert len(controls) == window_size
    for control in controls:
        assert isinstance(control, Function)
