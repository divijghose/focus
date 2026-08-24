from focus.utils.input_utils import get_user_config, pretty_print_config, get_ensemble_config
from focus.solvers.heat_equation import HeatEquationSolver
from focus.windowing.fixed_window import FixedWindow
from focus.optimizers.tao import TAOOptimizer
# from focus.optimizers.covariance import NewtonOptimizer
from focus.controls import additive_control
from focus.utils.output_utils import OutputUtilsEnsemble2D
from focus.functionals.loss import LossFunctional
from firedrake import *
from firedrake.adjoint import *
import numpy as np
from focus.utils.diagnostics import diagnose_ensemble
rng =np.random.default_rng(42) 

# continue_annotation()

# Read configuration
config = get_user_config()
pretty_print_config(config)
weighting = {
    "lambda_t": config["decay_constant"],
    "control_weight": config["control_weight"],
} 
ensemble_config = get_ensemble_config()
M = ensemble_config["processes_per_member"]
ensemble_size = ensemble_config["ensemble_size"]
my_ensemble = Ensemble(comm=COMM_WORLD, M=M)
assert my_ensemble.ensemble_size == ensemble_size, f"Ensemble size mismatch: expected {my_ensemble.ensemble_size}, got {ensemble_size}"
ensemble_member = my_ensemble.ensemble_rank


# Define mesh and function space
mesh = UnitSquareMesh(50, 50, comm=my_ensemble.comm)
V = FunctionSpace(mesh, "CG", 2)

x, y = SpatialCoordinate(mesh)
alpha = 100.0


def initial_condition_expression(x, y, ensemble_member=ensemble_member):
    """Define the initial condition expression."""
    rng = np.random.default_rng(seed=ensemble_member)  # Seed the RNG with the ensemble member rank for reproducibility
    x_offset = rng.normal(loc=0.0, scale=0.2)
    y_offset = rng.normal(loc=0.0, scale=0.2)
    print(f"Ensemble Member {ensemble_member}: Initial condition offsets: x_offset={x_offset}, y_offset={y_offset}")
    return exp(-alpha * (((x - 0.5 + x_offset) ** 2) + ((y - 0.5 + y_offset) ** 2)))




def forcing_function_expression(x, y, t):
    """Define the forcing function expression."""
    return Constant(0.0)


def desired_solution_expression(t):
    """Define the desired solution at time t."""
    # return exp(-100 * ((x[0] - 0.5) ** 2)) * exp(-t)
    return exp(-alpha * (((x - 0.5) ** 2) + ((y - 0.5) ** 2))) * exp(0.1*t)



# Initialize the PDE solver
heat_solver = HeatEquationSolver(mesh, V, kappa=0.01, dt=0.01)
# Set initial condition and boundary conditions
heat_solver.set_initial_condition(initial_condition_expression(x, y, ensemble_member=ensemble_member))
heat_solver.set_bcs([Constant(0.0), Constant(0.0), Constant(0.0), Constant(0.0)])
heat_solver.build_solver()
point_wise_error, l2_err, linf_err = heat_solver.errors()


# Initialize the forcing function and set it as the control variable
heat_solver.set_forcing_function()
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

output_manager = OutputUtils2D({"Solution" : heat_solver.u_new, "Control" : heat_solver.control, "Error" : heat_solver.point_wise_error, "Desired" : heat_solver.u_desired}, "./results", vtk_filename="test_vtk")
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


