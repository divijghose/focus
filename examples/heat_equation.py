from focus.solvers.heat_equation import HeatEquationSolver
from focus.windowing.fixed_window import FixedWindow
from focus.optimizers.tao import TAOOptimizer
from focus.controls import additive_control
from focus.utils.input_utils import get_user_config, pretty_print_config
from focus.functionals.base import LossFunctional
from firedrake import *

# Read configuration
config = get_user_config()
pretty_print_config(config)
weighting = {"lambda_t": config["decay_constant"], "misfit_weight": config["misfit_weight"]}

# Define mesh and function space
mesh = UnitIntervalMesh(10)
V = FunctionSpace(mesh, "CG", 2)

x = SpatialCoordinate(mesh)

def initial_condition_expression(x):
    """Define the initial condition expression."""
    return exp(-100 * ((x[0] - 0.5) ** 2))

def forcing_function_expression(x, t):
    """Define the forcing function expression."""
    return exp(-100 * ((x[0] - 0.5) ** 2))

def desired_solution_expression(x, t):
    """Define the desired solution expression."""
    return exp(-100 * ((x[0] - 0.5) ** 2)) * exp(t)

# Initialize the PDE solver
heat_solver = HeatEquationSolver(mesh, V, kappa = 0.1, dt = 0.01)
# Set initial condition and boundary conditions
heat_solver.set_initial_condition(initial_condition_expression(x))
heat_solver.set_bcs([Constant(0.0), Constant(0.0)])
heat_solver.build_solver()


# Initialize the forcing function and set it as the control variable
heat_solver.set_forcing_function(Constant(0.0))
windowing = FixedWindow(window_size=config["window_size"], window_stride=config["window_step"], pde_solver=heat_solver)
windowing.initialize_controls(initial_expression=Constant(1.0))
windowing.run_first_window

exit()
t = 0.0
while t < config["T"]:
    # Run the PDE solver for one time step
    heat_solver.solve()
    
    # Update time
    t += heat_solver.dt
    
    # Print the current time and the maximum value of the solution
    print(f"Time: {t:.4f}, Max u: {heat_solver.u_new.dat.data.max():.4f}")



# Build a windowing object


# Initialize the loss functional
loss_functional = LossFunctional(weighting)




t = 0.0
while t < config["T"]:
    # Run the PDE solver for one time step
    heat_solver.solve()
    
    # Update time
    t += heat_solver.dt
    
    # Print the current time and the maximum value of the solution
    print(f"Time: {t:.4f}, Max u: {heat_solver.u_new.dat.data.max():.4f}")


exit(0)
