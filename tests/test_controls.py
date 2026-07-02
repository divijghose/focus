import pytest
import numpy as np
from focus.equations.pde1D.heat_equation import HeatEquationSolver1D
from focus.controls import additive_control
from firedrake import *

def test_additive_control_assignment():
    mesh = UnitIntervalMesh(10)
    V = FunctionSpace(mesh, "CG", 2)
    heat_solver = HeatEquationSolver1D(mesh, V, kappa=0.1, dt=0.01)
    heat_solver.set_forcing_function(Constant(1.0))
    control = additive_control(heat_solver)
    assert control is not None
    assert np.allclose(control.dat.data, 1.0), "Control variable was not initialized correctly."
    control.assign(2.0)
    assert np.allclose(heat_solver.f.dat.data, 2.0), "Control variable was not assigned correctly."