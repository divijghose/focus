# examples/cahn_hilliard_mpc.py

"""
cahn_hilliard_mpc.py

Receding horizon MPC for the Cahn-Hilliard equation with a distributed
control source term on a 2D unit square domain.

Problem setup:
    - Domain:    Omega = [0,1]^2
    - Initial:   c0 = 0.5 + 0.01 * (0.5 - xi),  xi ~ U(0,1)
    - Desired:   u_d = <user defined>
    - Forcing:   f = 0
    - Control:   distributed source term in concentration equation
    - BCs:       none (natural boundary conditions)
"""

from firedrake import (
    Constant,
    Function,
    FunctionSpace,
    SpatialCoordinate,
    UnitSquareMesh,
    VTKFile,
    pi,
    sin,
)
from firedrake.adjoint import continue_annotation

from focus.controls.distributed_nonlinear import DistributedControl
from focus.functionals.loss import LossFunctional
from focus.optimizers.tao import TAOOptimizer
from focus.solvers.cahn_hilliard import CahnHilliardSolver
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

N = 128
mesh = UnitSquareMesh(N, N)
x, y = SpatialCoordinate(mesh)
V = FunctionSpace(mesh, "CG", 1)
V_control = FunctionSpace(mesh, "CG", 1)


# ---------------------------------------------------------------------------
# Problem expressions
# ---------------------------------------------------------------------------

def forcing_function_expression(t: float):
    """Zero external forcing."""
    return Constant(0.0)


def desired_solution_expression(t):
    """Desired concentration field. """
    return 1.0*sin(8*pi*x)*sin(8*pi*y)


# ---------------------------------------------------------------------------
# Solver setup
# ---------------------------------------------------------------------------

ch_solver = CahnHilliardSolver(
    mesh,
    V,
    lmbda=1e-2,
    theta=0.5,
)

ch_solver.set_forcing_function(forcing_function_expression)
ch_solver.set_initial_condition()  
ch_solver.set_bcs([])             

distributed_control = DistributedControl(V_control, name="distributed_control")
ch_solver.attach_control(distributed_control)

ch_solver.build_solver()


# ---------------------------------------------------------------------------
# Loss functional
# ---------------------------------------------------------------------------

weighting = {
    "lambda_t": config.decay_constant,
    "control_weight": config.control_weight,
}

loss_functional = LossFunctional(
    u_desired=desired_solution_expression,
    pde_solver=ch_solver,
    weighting=weighting,
)


# ---------------------------------------------------------------------------
# Windowing and optimizer
# ---------------------------------------------------------------------------

windowing = FixedWindow(
    window_size=config.window_size,
    window_stride=config.window_stride,
    pde_solver=ch_solver,
)
windowing.initialize_controls(initial_expression=sin(8*pi*x)*sin(8*pi*y))
windowing.run_first_window(loss_functional)

parameters_tao = {
    "method": "lbfgs",
    "max_it": 20,
    "fatol": 0.0,
    "frtol": 0.0,
    "gatol": 1e-9,
    "grtol": 0.0,
}
optimizer = TAOOptimizer(windowing.tape_manager.Jhat, parameters=parameters_tao)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

vtk_file = VTKFile("./results/cahn_hilliard_mpc.pvd")

desired_plot = Function(V, name="Desired solution")
def save_output(t: float) -> None:
    ch_solver.u_new.rename("Concentration")
    windowing.window_controls[0][0].rename("Control")
    ch_solver.point_wise_error.rename("Pointwise error")
    desired_plot.interpolate(desired_solution_expression(t))
    vtk_file.write(  # ty: ignore[unresolved-attribute]
        ch_solver.u_new,
        windowing.window_controls[0][0],
        ch_solver.point_wise_error,
        desired_plot,
        t=t,
    )


# ---------------------------------------------------------------------------
# MPC loop
# ---------------------------------------------------------------------------

t = 0.0
save_output(t)
while t < config.t_max:
    logger.info(
        f"t = {t:.6f} | "
        f"Window [{windowing.get_window_start_time():.6f}, "
        f"{windowing.get_window_end_time():.6f}]"
    )

    # Explicit evaluation before optimise
    windowing.Jhat(
        windowing.window_controls[0]
    )

    windowing.window_controls[0] = optimal_controls = optimizer.get_optimal_control()
    windowing.time_step_loop()
    print(f"solved for {t}")

    u_desired_now = desired_solution_expression(t)
    _, l2_err, linf_err = ch_solver.errors(u_desired_now)
    logger.info(f"L2 error: {l2_err:.6e} | Linf error: {linf_err:.6e}")

    t = windowing.global_step_time

    windowing.reinitialize_window_controls(
        [[windowing.window_controls[0][i] for i in range(windowing.window_size)]]
    )

    save_output(t)
