# loss/loss_functional.py

from firedrake import inner, dx, exp,assemble
from .base import BaseLoss
from ..utils.output_utils import get_logger

logger = get_logger(__name__)


class LossFunctional(BaseLoss):
    """Loss functional for optimal control of a PDE.

    Combines a misfit term measuring deviation from the desired state and a
    regularisation term penalising the control cost. The misfit is weighted
    by an exponential time decay factor.

    The total loss at each time hop is:

    .. math::

        J_i = \\int_\\Omega m^2 \\, \\mathrm{d}x
            + w \\int_\\Omega e^{-\\lambda t_{\\text{hop}}}
            (u_d - u)^2 \\, \\mathrm{d}x

    where :math:`m` is the control, :math:`u_d` is the desired state,
    :math:`u` is the current state, :math:`\\lambda` is the decay constant,
    and :math:`w` is the control weight.

    .. note::

        :meth:`__call__` returns an **unassembled UFL form**. Assembly is
        performed once on the accumulated total loss in
        :meth:`FixedWindow.run_first_window` before passing to
        :class:`ReducedFunctional`. This is required for pyadjoint to
        differentiate through the functional correctly.
    """

    def __init__(self, u_desired, pde_solver, weighting: dict):
        """
        :param u_desired: A callable taking a time argument (float or
            Firedrake Constant) and returning a UFL expression for the
            desired state.
        :param pde_solver: The controlled PDE solver instance.
        :type pde_solver: ControlledSolver
        :param weighting: A dictionary with keys ``lambda_t`` and
            ``control_weight``.
        :type weighting: dict
        :raises KeyError: If ``lambda_t`` or ``control_weight`` are missing
            from ``weighting``.
        """
        super().__init__(u_desired, pde_solver, weighting)
        if "lambda_t" not in weighting:
            raise KeyError("weighting must contain 'lambda_t'.")
        if "control_weight" not in weighting:
            raise KeyError("weighting must contain 'control_weight'.")
        self.lambda_t = self.weighting["lambda_t"]
        self.control_weight = self.weighting["control_weight"]

    def __call__(self, control, t_hop):
        """Return the unassembled UFL form of the total loss at one time hop.

        :param control: The control function or list of control functions at
            the current time hop.
        :param t_hop: A Firedrake Constant holding the current window-local
            time. Updated in place during the recorded forward pass so the
            tape sees a single symbolic node.
        :type t_hop: firedrake.Constant
        :returns: The unassembled UFL form of the total loss.
        :rtype: ufl.Form
        """
        return assemble(self.control_weight*self.control_cost(control) +  self.misfit_loss(t_hop))
        

    def misfit_loss(self, t_hop):
        """Return the unassembled misfit UFL form at the current time hop.

        :param t_hop: A Firedrake Constant holding the current window-local
            time.
        :type t_hop: firedrake.Constant
        :returns: The unassembled misfit UFL form.
        :rtype: ufl.Form
        """
        u_desired = self.u_desired(t_hop)
        residual = u_desired - self.pde_solver.u_new
        return exp(-self.lambda_t * t_hop) * inner(residual, residual) * dx

    def control_cost(self, control):
        """Return the unassembled control cost UFL form.

        :param control: The control function at the current time hop.
        :returns: The unassembled control cost UFL form.
        :rtype: ufl.Form
        """
        return inner(control, control) * dx

    def get_desired_solution(self, t):
        """Return the desired solution at time t.

        :param t: Time value, either a float or a Firedrake Constant.
        :returns: A UFL expression for the desired state.
        """
        return self.u_desired(t)
