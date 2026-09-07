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
