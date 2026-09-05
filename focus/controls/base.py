from abc import ABC, abstractmethod
from firedrake.function import Function
from firedrake.functionspaceimpl import WithGeometry
from ..utils.output_utils import get_logger

logger = get_logger(__name__)

class ControlBase(ABC):

    def __init__(self, function_space: WithGeometry, name: str = "control"):
        self._control_function_space = function_space
        self._name = name
        self._control_function: Function = Function(self._control_function_space, name=self._name)
        self._is_control_applied: bool = False
        logger.debug(f"Initialized control {self._name} on function space {self._control_function_space}")

    @property
    def control_function_space(self) -> WithGeometry:
        """The function space on which the control is defined.

        :return: The function space of the control.
        :rtype: WithGeometry
        """
        return self._control_function_space

    @property
    def name(self) -> str:
        """The name of the control.

        :return: The name of the control.
        :rtype: str
        """
        return self._name

    @property
    def control_function(self) -> Function:
        """The Firedrake function representing the control.

        :return: The control function.
        :rtype: Function
        """
        return self._control_function

    @property
    def is_control_applied(self) -> bool:
        """Indicates whether the control has been applied.

        :return: True if the control has been applied, False otherwise.
        :rtype: bool
        """
        return self._is_control_applied

    def assign_control(self, value) -> None:
        """Assigns a new value to the control function.

        :param value: The new value to assign to the control function.
        :type value: Any
        """
        self._control_function.interpolate(value)

    @abstractmethod
    def apply_control(self, solver) -> None:
        pass

    def _check_control_applied(self) -> None:
        """Checks if the control has been applied and raises an error control has already been applied."""
        if self._is_control_applied:
            raise RuntimeError(f"Control '{self._name}' has already been applied to a solver and cannot be modified or re-applied.")
        