# controls/dirichlet.py

from firedrake.bcs import DirichletBC
from firedrake.function import Function
from firedrake.functionspaceimpl import WithGeometry

from ..utils.output_utils import get_logger
from .base import ControlBase

logger = get_logger(__name__)


class DirichletControl(ControlBase):
    """A boundary control applied as a Dirichlet condition on a prescribed boundary.

    This control replaces the static Dirichlet boundary condition on the
    prescribed boundary with a time-varying control field. The
    :class:`DirichletBC` object holds a live reference to the control
    :class:`Function`, so updating the control via :meth:`assign` is
    immediately reflected in the boundary condition without reconstruction.

    .. math::

        u = m \\qquad \\text{on} \\quad \\Gamma_c \\times (0, T]

    where :math:`m` is the scalar control field and :math:`\\Gamma_c` is
    the prescribed boundary.
    """

    def __init__(
        self,
        function_space: WithGeometry,
        boundary_id: int | list[int],
        name: str = "dirichlet_control",
    ):
        """
        :param function_space: The function space :math:`V_c` on which the
            control is defined. Must be a scalar space defined on the same
            mesh as the state space.
        :type function_space: WithGeometry
        :param boundary_id: The boundary subdomain ID or list of IDs on which
            the control is applied. Must match the mesh boundary markers.
        :type boundary_id: int or list[int]
        :param name: A human-readable name for the control.
        :type name: str
        """
        super().__init__(function_space, name)

        if isinstance(boundary_id, int):
            boundary_id = [boundary_id]
        if not all(isinstance(b, int) and b > 0 for b in boundary_id):
            raise ValueError(
                f"boundary_id must be a positive integer or a list of positive integers, "
                f"got {boundary_id}."
            )
        self._boundary_id: list[int] = boundary_id
        self._bc: DirichletBC | None = None

    @property
    def boundary_id(self) -> list[int]:
        """The boundary subdomain IDs on which the control is applied."""
        return self._boundary_id

    @property
    def bc(self) -> DirichletBC:
        """The Firedrake DirichletBC object holding a live reference to the control.

        :raises RuntimeError: If the control has not yet been applied to a solver.
        """
        if self._bc is None:
            raise RuntimeError(
                f"Control '{self._name}' has not been applied to a solver yet. "
                "Call solver.build_solver() first."
            )
        return self._bc

    def apply(self, solver) -> None:
        """Replace the static boundary condition on the prescribed boundary.

        Constructs a :class:`DirichletBC` holding a live reference to the
        control :class:`Function` and replaces any existing BC on
        :attr:`boundary_id` in ``solver.bcs``. The control function is
        defined on :attr:`function_space` and is used directly by the BC
        without interpolation, so no :meth:`sync` call is required.

        :param solver: The controlled solver to apply this control to.
        :type solver: ControlledSolver
        :raises RuntimeError: If the control has already been applied.
        :raises ValueError: If the solver has no BCs defined on the
            prescribed boundary.
        """
        self._check_not_applied()

        self._bc = DirichletBC(solver.V, self._function, self._boundary_id)

        # Replace any existing static BC on the prescribed boundary
        retained_bcs = [
            bc for bc in solver.bcs
            if not any(b in self._boundary_id for b in _get_bc_nodes(bc))
        ]
        if len(retained_bcs) == len(solver.bcs):
            logger.warning(
                f"DirichletControl '{self._name}' found no existing BC on "
                f"boundary {self._boundary_id} to replace. Adding as a new BC."
            )
        solver.bcs = retained_bcs + [self._bc]

        self._is_applied = True
        logger.debug(
            f"Applied DirichletControl '{self._name}' on boundary "
            f"{self._boundary_id} to solver '{solver.__class__.__name__}'."
        )


def _get_bc_nodes(bc: DirichletBC) -> list[int]:
    """Extract the subdomain IDs from a DirichletBC.

    :param bc: A Firedrake DirichletBC object.
    :type bc: DirichletBC
    :returns: The subdomain IDs associated with the BC.
    :rtype: list[int]
    """
    sub_domain = bc.sub_domain
    if isinstance(sub_domain, int):
        return [sub_domain]
    if isinstance(sub_domain, (list, tuple)):
        return list(sub_domain)
    return []