from focus.equations.pde1D.heat_equation import HeatEquationSolver1D
from focus.windowing.windowing import FixedWindow
from focus.utils.input_utils import get_user_config, pretty_print_config
from firedrake import *

config = get_user_config()
pretty_print_config(config)

mesh = UnitIntervalMesh(10)
V = FunctionSpace(mesh, "CG", 2)
f = Function(V)
function_space = FunctionSpace(mesh, "CG", 2)
heat_solver = HeatEquationSolver1D(mesh, function_space)
print("HeatEquationSolver1D instance created successfully.")

