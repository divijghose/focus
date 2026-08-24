from focus.utils.input_utils import get_user_config, pretty_print_config
from focus.solvers.heat_equation import HeatEquationSolver
from focus.windowing.fixed_window import FixedWindow
from focus.optimizers.tao import TAOOptimizer
# from focus.optimizers.covariance import NewtonOptimizer
from focus.controls import additive_control
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
    "control_weight": config["control_weight"],
}  


# Define mesh and function space
mesh = UnitIntervalMesh(80)
V = FunctionSpace(mesh, "CG", 2)

x = SpatialCoordinate(mesh)


def initial_condition_expression(x):
    """Define the initial condition expression."""
    return exp(-100 * ((x[0] - 0.5) ** 2))

def forcing_function_expression(t):
    """Define the forcing function expression."""
    return exp(-100 * ((x[0] - 0.5) ** 2)) * exp(-t)  # Example time-dependent forcing function


def desired_solution_expression(t):
    """Define the desired solution at time t."""
    # return exp(-100 * ((x[0] - 0.5) ** 2)) * exp(-t)
    return exp(-100 * ((x[0] - 0.5) ** 2)) + 0.0*t


# Initialize the PDE solver
heat_solver = HeatEquationSolver(mesh, V, kappa=0.01, dt=0.01)
# Set initial condition and boundary conditions
heat_solver.set_initial_condition(initial_condition_expression(x))
heat_solver.set_bcs([Constant(0.0), Constant(0.0)])
heat_solver.build_solver()

point_wise_error, l2_err, linf_err = heat_solver.errors()


# Initialize the forcing function and set it as the control variable
heat_solver.set_forcing_function(forcing_function_expression)
heat_solver.update_forcing_function(0.0)  # Update the forcing function at the initial time
desired_solution = heat_solver.set_desired_solution(desired_solution_expression)
windowing = FixedWindow(window_size=config["window_size"], window_stride=config["window_stride"], pde_solver=heat_solver)
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
# optimizer = NewtonOptimizer(windowing.Jhat, function_space=V)

output_manager = OutputUtils1D({"Solution" : heat_solver.u_new, "Control" : heat_solver.control, "Error" : heat_solver.point_wise_error, "Desired" : heat_solver.u_desired}, "./results", vtk_filename="test_vtk")
t = 0.0
while t < config["T"]:
    print(windowing.get_window_start_time())
    windowing.time_hop_loop(loss_functional)
    optimal_controls = optimizer.get_optimal_control()
    windowing.time_step_loop()
    _, l2_err, linf_err = heat_solver.errors()
    print(f"L2 error: {l2_err}, Linf error: {linf_err}")
    t =windowing.global_step_time
    windowing.reinitialize_window_controls(optimal_controls)
    output_manager.save_to_vtk()
output_manager.plot_results()


