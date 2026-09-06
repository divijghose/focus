# controls/base.py

from abc import ABC, abstractmethod

from firedrake.function import Function
from firedrake.functionspaceimpl import WithGeometry

from ..utils.output_utils import get_logger

logger = get_logger(__name__)


class ControlBase(ABC):
    """Abstract base class for all control types.

    A control encapsulates a field defined on a dedicated function space
    and knows how to apply itself to a variational problem. Controls are
    attached to a solver before :meth:`build_solver` is called and may
    not be modified afterwards.
    """

    def __init__(self, function_space: WithGeometry, name: str = "control"):
        """
        :param function_space: The function space on which the control is defined.
        :type function_space: WithGeometry
        :param name: A human-readable name for the control, used in logging and output.
        :type name: str
        """
        if not isinstance(function_space, WithGeometry):
            raise TypeError("function_space must be an instance of WithGeometry.")
        self._V = function_space
        self._name = name
        self._function: Function = Function(self._V, name=name)
        self._is_applied: bool = False
        logger.debug(f"Initialized control '{self._name}' on function space {self._V}.")

    @property
    def function_space(self) -> WithGeometry:
        """The function space on which the control is defined."""
        return self._V

    @property
    def name(self) -> str:
        """The name of the control."""
        return self._name

    @property
    def function(self) -> Function:
        """The underlying Firedrake Function holding the control values."""
        return self._function

    @property
    def is_applied(self) -> bool:
        """Whether the control has been applied to a solver."""
        return self._is_applied

    def assign(self, value) -> None:
        """Assign a new value to the control function.

        :param value: A Firedrake expression, Function, or Constant to assign.
        """
        self._function.interpolate(value)

    @abstractmethod
    def apply(self, solver) -> None:
        """Apply the control to the variational problem of the given solver.

        This method is called once during :meth:`build_solver`. It modifies
        the weak form or boundary conditions of the solver in place.
        Subclasses must set ``self._is_applied = True`` at the end.

        :param solver: The controlled solver to apply this control to.
        :type solver: ControlledSolver
        """

    def _check_not_applied(self) -> None:
        """Raise an error if the control has already been applied.

        :raises RuntimeError: If :meth:`apply` has already been called.
        """
        if self._is_applied:
            raise RuntimeError(
                f"Control '{self._name}' has already been applied to a solver "
                "and cannot be modified or re-applied."
            )