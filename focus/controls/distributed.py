# controls/distributed.py

from firedrake import dx, inner
from firedrake.function import Function
from firedrake.functionspaceimpl import WithGeometry

from ..utils.output_utils import get_logger
from .base import ControlBase

logger = get_logger(__name__)


class DistributedControl(ControlBase):
    """A distributed control defined on a function space over the domain.

    This control enters the PDE as an additive forcing term in the weak form:

    .. math::

        L \\mathrel{+}= \\Delta t \\int_{\\Omega} m \\, v \\, \\mathrm{d}x

    where :math:`m` is the control field interpolated onto the state space,
    and :math:`v` is the test function. The control is defined on a dedicated
    function space :math:`V_c`, which may differ from the state space
    :math:`V` in polynomial degree, and is interpolated onto :math:`V` before
    entering the weak form.
    """

    def __init__(self, function_space: WithGeometry, name: str = "distributed_control"):
        """
        :param function_space: The function space :math:`V_c` on which the
            control is defined.
        :type function_space: WithGeometry
        :param name: A human-readable name for the control.
        :type name: str
        """
        super().__init__(function_space, name)
        self._control_on_V: Function | None = None

    @property
    def control_on_V(self) -> Function:
        """The control interpolated onto the state function space.

        :raises RuntimeError: If the control has not yet been applied to a solver.
        """
        if self._control_on_V is None:
            raise RuntimeError(
                f"Control '{self._name}' has not been applied to a solver yet. "
                "Call solver.build_solver() first."
            )
        return self._control_on_V

    def apply(self, solver) -> None:
        """Interpolate the control onto the state space and add it to the weak form.

        Allocates a :class:`Function` on the solver state space ``solver.V``,
        interpolates the control onto it, and appends the forcing term to
        ``solver.L`` in place. The interpolated function is stored in
        :attr:`control_on_V` and must be kept in sync with :attr:`function`
        by calling :meth:`sync` before each solve.

        :param solver: The controlled solver to apply this control to.
        :type solver: ControlledSolver
        :raises RuntimeError: If the control has already been applied.
        """
        self._check_not_applied()

        self._control_on_V = Function(solver.V, name=f"{self._name}_on_V")
        self._control_on_V.interpolate(self._function)

        solver.L += (solver.dt * inner(self._control_on_V, solver.v)) * dx

        self._is_applied = True
        logger.debug(
            f"Applied distributed control '{self._name}' to solver "
            f"'{solver.__class__.__name__}'. Control interpolated from "
            f"{self._V} onto {solver.V}."
        )

    def sync(self) -> None:
        """Synchronise the state-space representation with the current control values.

        Must be called after :meth:`assign` and before each solve to ensure
        the weak form sees the latest control values.

        :raises RuntimeError: If the control has not yet been applied to a solver.
        """
        _ = self.control_on_V  # triggers the RuntimeError if not applied
        self._control_on_V.interpolate(self._function)  # ty: ignore[unresolved-attribute]
        logger.debug(f"Synced distributed control '{self._name}' onto state space.")