# examples/heat_equation_mpc.py

"""
heat_equation_mpc.py

Receding horizon MPC for the 2D heat equation with a distributed control.

Problem setup:
    - Domain:    Omega = [0,1]^2
    - Initial:   u0 = exp(-alpha * ((x-0.5)^2 + (y-0.5)^2))
    - Desired:   u_d = exp(-alpha * ((x-0.5)^2 + (y-0.5)^2))  (steady)
    - Forcing:   f = 0
    - Control:   distributed, defined on CG1
    - BCs:       homogeneous Dirichlet on all boundaries
"""

from firedrake import (
    Constant,
    FunctionSpace,
    SpatialCoordinate,
    UnitSquareMesh,
    VTKFile,
    exp,
    Function
)

from focus.controls.distributed import DistributedControl
from focus.functionals.loss import LossFunctional
from focus.optimizers.tao import TAOOptimizer
from focus.solvers.heat import HeatEquationSolver
from focus.utils.input_utils import get_user_config, pretty_print_config
from focus.utils.output_utils import get_logger, setup_logger
from focus.windowing.windowing import FixedWindow

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

config = get_user_config()
setup_logger(verbose=config.verbose)
pretty_print_config(config)


# ---------------------------------------------------------------------------
# Mesh and function spaces
# ---------------------------------------------------------------------------

mesh = UnitSquareMesh(80, 80)
V = FunctionSpace(mesh, "CG", 2)
V_control = FunctionSpace(mesh, "CG", 1)
x, y = SpatialCoordinate(mesh)

alpha = 100.0


# ---------------------------------------------------------------------------
# Problem expressions
# ---------------------------------------------------------------------------

def initial_condition_expression():
    """Gaussian bump centred at (0.5, 0.5)."""
    return exp(-alpha * ((x - 0.5) ** 2 + (y - 0.5) ** 2))


def forcing_function_expression(t):
    """Zero forcing."""
    return Constant(0.0)


def desired_solution_expression(t):
    """Steady Gaussian bump. Accepts a float or a Firedrake Constant for t
    so it can be called both inside the tape (with t_hop Constant) and
    outside (with a plain float for error evaluation).
    """
    return exp(-alpha * ((x - 0.5) ** 2 + (y - 0.5) ** 2))


# ---------------------------------------------------------------------------
# Solver setup
# ---------------------------------------------------------------------------

heat_solver = HeatEquationSolver(mesh, V, kappa=0.01, dt=config.t_max / 100)

heat_solver.set_forcing_function(forcing_function_expression)
heat_solver.set_initial_condition(initial_condition_expression())
heat_solver.set_bcs([
    Constant(0.0),
    Constant(0.0),
    Constant(0.0),
    Constant(0.0),
])

distributed_control = DistributedControl(V_control, name="distributed_control")
heat_solver.attach_control(distributed_control)
heat_solver.build_solver()


# ---------------------------------------------------------------------------
# Loss functional
# ---------------------------------------------------------------------------

weighting = {
    "lambda_t": config.decay_constant,
    "control_weight": config.control_weight,
}

loss_functional = LossFunctional(
    u_desired=desired_solution_expression,
    pde_solver=heat_solver,
    weighting=weighting,
)


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------

windowing = FixedWindow(
    window_size=config.window_size,
    window_stride=config.window_stride,
    pde_solver=heat_solver,
)
windowing.initialize_controls(initial_expression=Constant(0.0))

# Record the tape once. After this call, Jhat is ready.
windowing.run_first_window(loss_functional)


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

parameters_tao = {
    "method": "lbfgs",
    "max_it": 20,
    "fatol": 0.0,
    "frtol": 0.0,
    "gatol": 1e-9,
    "grtol": 0.0,
}
optimizer = TAOOptimizer(windowing.Jhat, parameters=parameters_tao)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

vtk_file = VTKFile("./results/heat_equation_mpc.pvd")

desired_plot = Function(V, name="Desired solution")
def save_output(t: float) -> None:
    heat_solver.u_new.rename("Solution")
    windowing.window_controls[0][0].rename("Control")
    heat_solver.point_wise_error.rename("Pointwise error")
    desired_plot.interpolate(desired_solution_expression(t))

    vtk_file.write(
        heat_solver.u_new,
        windowing.window_controls[0][0],
        heat_solver.point_wise_error,
        desired_plot,
        t=t,
    )


# ---------------------------------------------------------------------------
# MPC loop
# ---------------------------------------------------------------------------

t = 0.0

while t < config.t_max:
    logger.info(
        f"t = {t:.4f} | "
        f"Window [{windowing.get_window_start_time():.4f}, "
        f"{windowing.get_window_end_time():.4f}]"
    )

    windowing.Jhat(windowing.window_controls[0])
    windowing.window_controls[0] = optimal_controls = optimizer.get_optimal_control()
    windowing.time_step_loop()

    # Evaluate errors against the desired state at the current time.
    u_desired_now = desired_solution_expression(t)
    _, l2_err, linf_err = heat_solver.errors(u_desired_now)
    logger.info(f"L2 error: {l2_err:.6e} | Linf error: {linf_err:.6e}")

    t = windowing.global_step_time

    optimal_controls = [[windowing.window_controls[0][i] for i in range(windowing.window_size)]]
    windowing.reinitialize_window_controls(optimal_controls)

    save_output(t)

