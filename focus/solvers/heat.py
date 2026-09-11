# solvers/heat_equation.py

"""
heat_equation.py

Solver class for the heat equation for optimal control problems using Firedrake.
"""

from firedrake import (
    Constant,
    LinearVariationalProblem,
    LinearVariationalSolver,
    dx,
    grad,
    inner,
)
from firedrake.bcs import DirichletBC
from firedrake.function import Function
from firedrake.functionspaceimpl import WithGeometry
from firedrake.mesh import MeshGeometry
from firedrake.ufl_expr import TestFunction, TrialFunction

from ..utils.output_utils import get_logger
from .controlled import ControlledSolver

logger = get_logger(__name__)


class HeatEquationSolver(ControlledSolver):
    """Solve a transient heat equation with Firedrake.

    Implements the unsteady heat equation with Dirichlet boundary conditions
    and optional optimal controls:

    .. math::

        \\begin{eqnarray}
        u_t - \\kappa \\, \\Delta u & = & f + m,
            \\qquad \\text{in} \\quad \\Omega \\times (0, T] \\\\
        u & = & g + m_\\Gamma,
            \\qquad \\text{on} \\quad \\partial\\Omega \\times (0, T] \\\\
        u(\\cdot, 0) & = & u_0,
            \\qquad \\text{in} \\quad \\Omega
        \\end{eqnarray}

    where :math:`m` is an optional distributed control and
    :math:`m_\\Gamma` is an optional Dirichlet boundary control.

    """

    def __init__(
        self,
        mesh: MeshGeometry,
        function_space: WithGeometry,
        kappa: float = 1.0,
        dt: float = 0.1,
    ):
        """
        :param mesh: Firedrake mesh defining the spatial domain.
        :type mesh: MeshGeometry
        :param function_space: Function space used for the state variable.
        :type function_space: WithGeometry
        :param kappa: Thermal diffusivity coefficient. Must be positive.
        :type kappa: float
        :param dt: Time step size. Must be positive.
        :type dt: float
        :raises ValueError: If kappa is not positive.
        """
        if kappa <= 0:
            raise ValueError(
                f"Thermal diffusivity kappa must be positive, got {kappa}."
            )
        super().__init__(mesh, function_space, dt)
        self.kappa: float = kappa
        self.f: Function = Function(self.V, name="Forcing function")
        self._f_expr = None
        self.p.interpolate(Constant(0.0))
        logger.debug(
            f"Initialized HeatEquationSolver with kappa={kappa}, dt={dt}."
        )


    def set_forcing_function(self, f_expr, t: float = 0.0) -> None:
        """Set the forcing term for the heat equation.

        :param f_expr: A callable taking time as a float and returning a
            Firedrake expression or Function.
        :param t: Initial time at which to evaluate the forcing function.
        :type t: float
        :raises TypeError: If f_expr is not callable.
        """
        if not callable(f_expr):
            raise TypeError(
                "f_expr must be a callable that takes time and returns "
                "a Firedrake expression."
            )
        self._f_expr = f_expr
        self.f.interpolate(self._f_expr(t))
        logger.debug("Forcing function set.")

    def update_forcing_function(self, t: float) -> None:
        """Update the forcing function to the given time.

        :param t: Current time.
        :type t: float
        :raises RuntimeError: If the forcing function has not been set.
        """
        if self._f_expr is None:
            raise RuntimeError(
                "Forcing function has not been set. "
                "Call set_forcing_function() before update_forcing_function()."
            )
        self.f.interpolate(self._f_expr(t))

    def set_initial_condition(self, u0=Constant(0.0)) -> None:
        """Set the initial condition for the state variable.

        Initialises :attr:`u_old`, :attr:`u_new`, and the parameter field
        :attr:`p` from the given expression. All three are set consistently
        so the first window starts from a well-defined state.

        :param u0: A Firedrake expression or Constant for the initial state.
        """
        self.u_old.interpolate(u0)
        self.u_new.interpolate(u0)
        self.p.interpolate(u0)
        logger.debug("Initial condition set on u_old, u_new, and p.")

    def set_bcs(self, bcs: list | None = None) -> None:
        """Set the static Dirichlet boundary conditions.

        Each entry in ``bcs`` is applied to the corresponding subdomain,
        indexed from 1. Any boundary subsequently controlled by a
        :class:`DirichletControl` will have its static BC replaced at
        build time.

        :param bcs: A list of Firedrake expressions or Constants, one per
            subdomain boundary. Defaults to no BCs if None.
        :type bcs: list or None
        """
        if bcs is None:
            logger.warning(
                "No boundary conditions provided. "
                "Defaulting to homogeneous Dirichlet on all boundaries."
            )
            bcs = []
        self.bcs = [
            DirichletBC(self.V, bc_value, bc_subdomain + 1)
            for bc_subdomain, bc_value in enumerate(bcs)
        ]
        logger.debug(f"Set {len(self.bcs)} static boundary condition(s).")

 

    def _build_forms(self) -> None:
        """Assemble the bilinear and linear forms for the heat equation.

        Uses :attr:`u_old` in the linear form to carry the solution forward
        within each window. :attr:`u_old` is assigned from :attr:`p` at the
        start of each window in :meth:`run_first_window`, anchoring the tape
        to the window initial condition.

        Controls append to :attr:`L` after this method returns.
        """
        self.u = TrialFunction(self.V)
        self.v = TestFunction(self.V)

        self.a = (
            inner(self.u, self.v)
            + self.dt * self.kappa * inner(grad(self.u), grad(self.v))
        ) * dx

        # u_old carries the solution forward within the window.
        # p anchors the tape as the window initial condition via
        # Jhat.update_parameters between windows.
        self.L = (
            inner(self.u_old, self.v)
            + self.dt * inner(self.f, self.v)
        ) * dx

        logger.debug("Variational forms assembled.")

    def _build_linear_solver(self) -> None:
        """Construct the Firedrake LinearVariationalSolver.

        Called after forms are assembled and all controls have been applied.
        Stores the solver in :attr:`solver`.
        """
        self.solver = LinearVariationalSolver(
            LinearVariationalProblem(self.a, self.L, self.u_new, bcs=self.bcs)
        )
        logger.debug("LinearVariationalSolver constructed.")

    # ------------------------------------------------------------------
    # Time stepping
    # ------------------------------------------------------------------

    def solve(self) -> None:
        """Advance the solution by one time step.

        Copies :attr:`u_new` into :attr:`u_old` and calls the assembled
        variational solver to update :attr:`u_new`.

        :raises RuntimeError: If the solver has not been built yet.
        """
        if not self._solver_built:
            raise RuntimeError(
                "solve() cannot be called before build_solver()."
            )
        self.u_old.assign(self.u_new)
        self.solver.solve()

    def set_parameters(self) -> None:
        """Update the parameter field from the current solution.

        Assigns :attr:`u_new` into :attr:`p` after the stride has been
        executed. This makes :attr:`p` the correct initial condition for
        the next window, ready to be passed to ``Jhat.update_parameters``.
        """
        self.p.assign(self.u_new)
        logger.debug("Parameter field p updated from u_new.")
