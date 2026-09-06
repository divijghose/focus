# solvers/controlled.py

from abc import ABC, abstractmethod

from firedrake import Constant
from firedrake.bcs import DirichletBC
from firedrake.function import Function
from firedrake.functionspaceimpl import WithGeometry
from firedrake.mesh import MeshGeometry

from ..controls.base import ControlBase
from ..utils.error_utils import l2_error, linf_error, point_wise_error
from ..utils.output_utils import get_logger
from .base import Solver

logger = get_logger(__name__)


class ControlledSolver(Solver, ABC):
    """Library-internal base class for PDE solvers with optimal control support.

    Extends :class:`Solver` with control attachment, a staged build process,
    parameter management, and error computation. Users subclass this when
    implementing a new controlled PDE solver but are not expected to interact
    with the control mechanism directly.


    """

    def __init__(
        self,
        mesh: MeshGeometry,
        function_space: WithGeometry,
        dt: float,
    ):
        """
        :param mesh: Firedrake mesh defining the spatial domain.
        :type mesh: MeshGeometry
        :param function_space: Function space used for the state variable.
        :type function_space: WithGeometry
        :param dt: Time step size. Must be positive.
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
        to the variational problem in the order they are attached.

        :param control: A control instance to attach.
        :type control: ControlBase
        :raises TypeError: If control is not a :class:`ControlBase` instance.
        :raises RuntimeError: If called after :meth:`build_solver`.
        """
        if not isinstance(control, ControlBase):
            raise TypeError(
                f"control must be an instance of ControlBase, "
                f"got {type(control).__name__}."
            )
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

    @property
    def num_controls(self) -> int:
        """The number of controls attached to the solver."""
        return len(self.controls)

    # ------------------------------------------------------------------
    # Staged build
    # ------------------------------------------------------------------

    @abstractmethod
    def _build_forms(self) -> None:
        """Assemble the bilinear and linear forms for the variational problem.

        Subclasses define ``self.a``, ``self.L``, and trial/test functions
        here. Controls are applied to ``self.L`` and ``self.bcs`` after
        this method returns.
        """

    @abstractmethod
    def _build_linear_solver(self) -> None:
        """Construct the Firedrake LinearVariationalSolver.

        Called after forms are assembled and all controls have been applied.
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
            raise RuntimeError(
                "build_solver() has already been called and cannot be called again."
            )
        logger.debug(f"Building solver '{self.__class__.__name__}'.")

        self._build_forms()
        logger.debug("Variational forms assembled.")

        for control in self.controls:
            control.apply(self)

        self._build_linear_solver()
        self._solver_built = True
        logger.debug(
            f"Solver '{self.__class__.__name__}' built successfully with "
            f"{self.num_controls} control(s)."
        )

    # ------------------------------------------------------------------
    # Parameter management
    # ------------------------------------------------------------------

    def set_parameters(self) -> None:
        """Update the parameter field from the current solution.

        Assigns the latest state :attr:`u_new` into :attr:`p`. Called by
        the windowing class after each stride to advance the initial
        condition for the next window.
        """
        self.p.assign(self.u_new)
        logger.debug(
            f"Parameters updated for solver '{self.__class__.__name__}'."
        )

    # ------------------------------------------------------------------
    # Abstract interface extensions
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

    def errors(self, u_desired: Function) -> tuple[Function, float, float]:
        """Compute pointwise, L2, and Linf errors against the desired state.

        :param u_desired: The desired state at the current time.
        :type u_desired: Function
        :returns: The pointwise error field, the L2 error, and the Linf error.
        :rtype: tuple[Function, float, float]
        :raises RuntimeError: If called before :meth:`build_solver`.
        """
        if not self._solver_built:
            raise RuntimeError(
                "errors() cannot be called before build_solver()."
            )
        point_wise_error(self.point_wise_error, self.u_new, u_desired)
        l2_err = l2_error(self.u_new, u_desired)
        linf_err = linf_error(self.point_wise_error)
        return self.point_wise_error, l2_err, linf_err