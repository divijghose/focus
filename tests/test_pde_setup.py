import numpy as np
import pytest
from firedrake import *
from focus.equations.pde1D.heat_equation import HeatEquationSolver1D

def test_heat_equation_solver_initialization():
    mesh = UnitIntervalMesh(10)
    V = FunctionSpace(mesh, "CG", 2)
    kappa = 1.0
    dt = 0.1
    heat_solver = HeatEquationSolver1D(mesh, V, kappa=kappa, dt=dt)
    
    assert isinstance(heat_solver, HeatEquationSolver1D), "HeatEquationSolver1D instance not created."
    assert heat_solver.kappa == kappa, "Incorrect thermal diffusivity."
    assert heat_solver.dt == dt, "Incorrect time step size."
    assert isinstance(heat_solver.f, Function), "Forcing function not initialized."
    assert isinstance(heat_solver.u0, Function), "Initial condition not initialized."

def test_heat_equation_solver_set_forcing_function():
    mesh = UnitIntervalMesh(10)
    V = FunctionSpace(mesh, "CG", 2)
    heat_solver = HeatEquationSolver1D(mesh, V)
    
    f_new = Constant(1.0)
    heat_solver.set_forcing_function(f_new)
    
    assert np.allclose(heat_solver.f.dat.data, f_new.dat.data), "Forcing function not set correctly."

    x = SpatialCoordinate(mesh)
    f_expr = sin(pi * x[0])
    f_expr_func = Function(V).interpolate(f_expr)
    heat_solver.set_forcing_function(f_expr_func)
    assert np.allclose(heat_solver.f.dat.data, f_expr_func.dat.data), "Forcing function not set correctly with expression."

def test_heat_equation_solver_set_initial_condition():
    mesh = UnitIntervalMesh(10)
    V = FunctionSpace(mesh, "CG", 2)
    heat_solver = HeatEquationSolver1D(mesh, V)
    
    u0_new = Constant(2.0)
    heat_solver.set_initial_condition(u0_new)
    
    assert np.allclose(heat_solver.u0.dat.data, u0_new.dat.data), "Initial condition not set correctly."

    x = SpatialCoordinate(mesh)
    u0_expr = cos(pi * x[0])
    u0_expr_func = Function(V).interpolate(u0_expr)
    heat_solver.set_initial_condition(u0_expr_func)
    assert np.allclose(heat_solver.u0.dat.data, u0_expr_func.dat.data), "Initial condition not set correctly with expression."

def test_heat_equation_set_bcs():
    mesh = UnitIntervalMesh(10)
    V = FunctionSpace(mesh, "CG", 2)
    heat_solver = HeatEquationSolver1D(mesh, V)
    
    bc = [0.0, 0.0]
    heat_solver.set_bcs(bc)
    
    assert len(heat_solver.bcs) == 2, "Boundary conditions not set correctly."
    assert all(isinstance(bc, DirichletBC) for bc in heat_solver.bcs), "Boundary conditions are not DirichletBC instances."


def test_heat_equation_solver_build_and_solve():
    mesh = UnitIntervalMesh(10)
    V = FunctionSpace(mesh, "CG", 2)
    heat_solver = HeatEquationSolver1D(mesh, V)
    bc = [0.0, 0.0]
    heat_solver.set_bcs(bc)
    heat_solver.build_solver()
    assert heat_solver.solver is not None, "Solver not built correctly."
    
    heat_solver.solve()
    assert isinstance(heat_solver.u_new, Function), "Solution not a Function instance after solving."
