# solvers/base.py

from abc import ABC, abstractmethod

from firedrake import Constant
from firedrake.bcs import DirichletBC
from firedrake.function import Function
from firedrake.functionspaceimpl import WithGeometry
from firedrake.mesh import MeshGeometry

from ..controls.base import ControlBase
from ..utils.error_utils import l2_error, linf_error, point_wise_error
from ..utils.output_utils import get_logger

logger = get_logger(__name__)


class Solver(ABC):
    """Abstract base class for all PDE solvers.

    Users subclass this to define a new PDE solver. The class enforces a
    consistent interface for setting up the variational problem, boundary
    conditions, forcing function, and initial condition.

    Subclasses must implement:
        - :meth:`_build_forms`
        - :meth:`_build_linear_solver`
        - :meth:`set_forcing_function`
        - :meth:`set_initial_condition`
        - :meth:`set_bcs`
        - :meth:`solve`
    """

    def __init__(self, mesh: MeshGeometry, function_space: WithGeometry):
        """
        :param mesh: Firedrake mesh defining the spatial domain.
        :type mesh: MeshGeometry
        :param function_space: Function space used for the state variable.
        :type function_space: WithGeometry
        :raises TypeError: If mesh or function_space are not the expected types.
        """
        if not isinstance(mesh, MeshGeometry):
            raise TypeError("mesh must be an instance of firedrake.mesh.MeshGeometry.")
        if not isinstance(function_space, WithGeometry):
            raise TypeError("function_space must be an instance of WithGeometry.")

        self.mesh = mesh
        self.V = function_space
        self.bcs: list[DirichletBC] = []
        self.u_old: Function = Function(self.V, name="Solution at previous time step")
        self.u_new: Function = Function(self.V, name="Solution at current time step")
        self.point_wise_error: Function = Function(self.V, name="Pointwise error")

    @abstractmethod
    def set_forcing_function(self, f_expr, t: float = 0.0) -> None:
        """Set the forcing term for the PDE.

        :param f_expr: A callable taking time and returning a Firedrake expression.
        :param t: Initial time at which to evaluate the forcing function.
        :type t: float
        """

    @abstractmethod
    def set_initial_condition(self, u0=Constant(0.0)) -> None:
        """Set the initial condition for the state variable.

        :param u0: A Firedrake expression or Constant for the initial state.
        """

    @abstractmethod
    def set_bcs(self, bcs: list) -> None:
        """Set the static Dirichlet boundary conditions.

        :param bcs: A list of boundary values indexed by subdomain.
        :type bcs: list
        """

    @abstractmethod
    def solve(self) -> None:
        """Advance the solution by one time step."""


class ControlledSolver(Solver, ABC):
    """Library-internal base class for PDE solvers with optimal control support.

    Extends :class:`Solver` with control attachment, a staged build process,
    parameter management, and error computation. Users should subclass this
    when implementing a new controlled PDE solver, but are not expected to
    interact with the control mechanism directly.

    Subclasses must implement:
        - :meth:`_build_forms`
        - :meth:`_build_linear_solver`
        - :meth:`set_forcing_function`
        - :meth:`set_initial_condition`
        - :meth:`set_bcs`
        - :meth:`solve`
        - :meth:`update_forcing_function`
    """

    def __init__(self, mesh: MeshGeometry, function_space: WithGeometry, dt: float):
        """
        :param mesh: Firedrake mesh defining the spatial domain.
        :type mesh: MeshGeometry
        :param function_space: Function space used for the state variable.
        :type function_space: WithGeometry
        :param dt: Time step size.
        :type dt: float
        :raises ValueError: If dt is not positive.
        """
        if dt <= 0:
            raise ValueError(f"Time step dt must be positive, got {dt}.")
        super().__init__(mesh, function_space)
        self.dt: float = dt
        self.p: Function = Function(self.V, name="Parameters")
        self.controls: list[ControlBase] = []
        self._solver_built: bool = False

    # ------------------------------------------------------------------
    # Control attachment
    # ------------------------------------------------------------------

    def attach_control(self, control: ControlBase) -> None:
        """Attach a control to the solver.

        Must be called before :meth:`build_solver`. Controls are applied
        in the order they are attached.

        :param control: A control instance to attach.
        :type control: ControlBase
        :raises TypeError: If control is not a ControlBase instance.
        :raises RuntimeError: If the solver has already been built.
        """
        if not isinstance(control, ControlBase):
            raise TypeError("control must be an instance of ControlBase.")
        if self._solver_built:
            raise RuntimeError(
                "Cannot attach controls after build_solver() has been called."
            )
        self.controls.append(control)
        logger.debug(
            f"Attached control '{control.name}' to solver "
            f"'{self.__class__.__name__}'. "
            f"Total controls: {len(self.controls)}."
        )

    # ------------------------------------------------------------------
    # Staged build
    # ------------------------------------------------------------------

    @abstractmethod
    def _build_forms(self) -> None:
        """Assemble the bilinear and linear forms for the variational problem.

        Subclasses define ``self.a``, ``self.L``, and trial/test functions
        here. Controls are applied after this method returns.
        """

    @abstractmethod
    def _build_linear_solver(self) -> None:
        """Construct the Firedrake LinearVariationalSolver.

        Called after forms are assembled and controls are applied.
        Subclasses store the solver in ``self.solver``.
        """

    def build_solver(self) -> None:
        """Assemble the variational problem and build the linear solver.

        Orchestrates the full build sequence:
            1. Assemble variational forms via :meth:`_build_forms`.
            2. Apply all attached controls via :meth:`ControlBase.apply`.
            3. Construct the linear solver via :meth:`_build_linear_solver`.

        :raises RuntimeError: If called more than once.
        """
        if self._solver_built:
            raise RuntimeError("build_solver() has already been called.")

        logger.debug(f"Building solver '{self.__class__.__name__}'.")
        self._build_forms()

        for control in self.controls:
            control.apply(self)
            logger.debug(f"Applied control '{control.name}'.")

        self._build_linear_solver()
        self._solver_built = True
        logger.debug(
            f"Solver '{self.__class__.__name__}' built with "
            f"{len(self.controls)} control(s)."
        )

    # ------------------------------------------------------------------
    # Parameter management
    # ------------------------------------------------------------------

    def set_parameters(self) -> None:
        """Update the parameter field from the current solution.

        Assigns the latest state :attr:`u_new` into :attr:`p`. Called by
        the windowing class after each time step to advance the initial
        condition for the next window.
        """
        self.p.assign(self.u_new)
        logger.debug(f"Updated parameters for solver '{self.__class__.__name__}'.")

    # ------------------------------------------------------------------
    # Forcing function update
    # ------------------------------------------------------------------

    @abstractmethod
    def update_forcing_function(self, t: float) -> None:
        """Update the forcing function to the given time.

        :param t: Current time.
        :type t: float
        """

    # ------------------------------------------------------------------
    # Error computation
    # ------------------------------------------------------------------

    def errors(self, u_desired: Function) -> tuple:
        """Compute pointwise, L2, and Linf errors against the desired state.

        :param u_desired: The desired state at the current time.
        :type u_desired: Function
        :returns: The pointwise error field, the L2 error, and the Linf error.
        :rtype: tuple[Function, float, float]
        :raises RuntimeError: If the solver has not been built yet.
        """
        if not self._solver_built:
            raise RuntimeError(
                "errors() cannot be called before build_solver()."
            )
        point_wise_error(self.point_wise_error, self.u_new, u_desired)
        l2_err = l2_error(self.u_new, u_desired)
        linf_err = linf_error(self.point_wise_error)
        return self.point_wise_error, l2_err, linf_err