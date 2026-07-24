from focus.solvers.heat_equation import HeatEquationSolver
from focus.windowing.fixed_window import FixedWindow
from focus.optimizers.tao import TAOOptimizer
from focus.controls import additive_control
from focus.utils.input_utils import get_user_config, pretty_print_config
from focus.utils.output_utils import OutputUtils1D
from focus.functionals.loss import LossFunctional
from firedrake import *
from firedrake.adjoint import *

# continue_annotation()

# Read configuration
config = get_user_config()
pretty_print_config(config)
weighting = {
    "lambda_t": config["decay_constant"],
    "control_weight": config["misfit_weight"],
}  # FIXME: Change misfit_weight to control_weight in config file


# Define mesh and function space
mesh = UnitIntervalMesh(80)
V = FunctionSpace(mesh, "CG", 2)

x = SpatialCoordinate(mesh)


def initial_condition_expression(x):
    """Define the initial condition expression."""
    return exp(-100 * ((x[0] - 0.5) ** 2))


def forcing_function_expression(x, t):
    """Define the forcing function expression."""
    return exp(-100 * ((x[0] - 0.5) ** 2))


def desired_solution_expression(t):
    """Define the desired solution at time t."""
    return exp(-100 * ((x[0] - 0.5) ** 2)) * exp(-t)


# Initialize the PDE solver
heat_solver = HeatEquationSolver(mesh, V, kappa=0.1, dt=0.01)
# Set initial condition and boundary conditions
heat_solver.set_initial_condition(initial_condition_expression(x))
heat_solver.set_bcs([Constant(0.0), Constant(0.0)])
heat_solver.build_solver()


# Initialize the forcing function and set it as the control variable
heat_solver.set_forcing_function(Constant(0.0))
desired_solution = heat_solver.set_desired_solution(desired_solution_expression)
windowing = FixedWindow(window_size=1, window_stride=1, pde_solver=heat_solver)
windowing.initialize_controls(initial_expression=Constant(1.0))

loss_functional = LossFunctional(desired_solution, heat_solver, weighting)
windowing.run_first_window(
    loss_functional
)  # Dummy run of first window to set up the Reduced Functional and the Optimizer
parameters_tao = {
    "method": "lbfgs",
    "max_it": 20,
    "fatol": 0.0,
    "frtol": 0.0,
    "gatol": 1e-9,
    "grtol": 0.0,
}
optimizer = TAOOptimizer(windowing.Jhat, parameters=parameters_tao)
output_manager = OutputUtils1D({"Solution" : heat_solver.u_new, "Control" : heat_solver.control}, "./results", vtk_filename="test_vtk")
# save_to_vtk([heat_solver.u0], "./results", "initial_solution")
t = 0.0
while t < config["T"]:
    print(windowing.get_window_start_time())
    windowing.time_hop_loop(loss_functional)
    optimal_controls = optimizer.get_optimal_control()
    windowing.time_step_loop()
    t =windowing.global_step_time
    windowing.reinitialize_window_controls(optimal_controls)
    output_manager.save_to_vtk()
    print("Comes here")

    # optimizer.get_optimal_control()
    t = config["T"]

