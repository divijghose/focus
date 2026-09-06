# controls/distributed.py

from firedrake.functionspaceimpl import WithGeometry
from firedrake import inner, dx
from .base import ControlBase
from ..utils.logger import get_logger

log = get_logger(__name__)


class DistributedControl(ControlBase):
    """A distributed control defined on a function space over the domain.

    This control enters the PDE as an additive forcing term in the weak form:

    .. math::

        L \\mathrel{+}= \\Delta t \\int_{\\Omega} m \\, v \\, \\mathrm{d}x

    where :math:`m` is the control field and :math:`v` is the test function.
    The control is defined on a dedicated function space :math:`V_c`, which
    may differ from the state space :math:`V`, and is interpolated onto
    :math:`V` before entering the weak form.
    """

    def __init__(self, function_space: WithGeometry, name: str = "distributed_control"):
        """
        :param function_space: The function space on which the control is defined.
        :type function_space: WithGeometry
        :param name: A human-readable name for the control.
        :type name: str
        """
        super().__init__(function_space, name)

    def apply(self, solver) -> None:
        """Interpolate the control onto the state space and add it to the weak form.

        Modifies ``solver.L`` in place by appending the control forcing term.
        Must be called before the :class:`LinearVariationalSolver` is assembled.

        :param solver: The controlled solver to apply this control to.
        :type solver: ControlledSolver
        :raises RuntimeError: If the control has already been applied.
        """
        self._check_not_applied()

        self._control_on_V = self._function
        self._control_on_V.interpolate(self._function)

        solver.L += (solver.dt * inner(self._control_on_V, solver.v)) * dx

        self._is_applied = True
        log.debug(f"Applied distributed control '{self._name}' to solver '{solver.__class__.__name__}'.")