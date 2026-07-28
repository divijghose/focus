"""
heat_equation.py

Solver class for the heat equation for optimal control problems using Firedrake.

Author: Divij Ghose
"""
from .base import Solver
from firedrake.mesh import MeshGeometry
from firedrake.function import Function
from firedrake.functionspaceimpl import WithGeometry
from firedrake import Constant
from firedrake.ufl_expr import TrialFunction, TestFunction
from firedrake import inner, grad, dx
from firedrake import LinearVariationalProblem, LinearVariationalSolver
from firedrake.bcs import DirichletBC
from ..controls import *
from ..utils.error_utils import *


class HeatEquationSolver(Solver):
    """Solve a transient heat equation with Firedrake.

    This solver assembles a linear variational problem for the unsteady heat
    equation with Dirichlet boundary conditions,

    .. math::
        :nowrap:

        \\begin{eqnarray}

        u_t - \kappa * \Delta u & = &  f,     \\qquad \\text{in} \\qquad \Omega \\times (0, T]
        \\\\
        
        u                           & = &  g, \\qquad \\text{on} \\qquad \partial \Omega \\times (0, T] \\\\
        
        u(\cdot, 0)               & = &  u_0,  \\qquad \\text{in} \\qquad \Omega

        \\end{eqnarray}
    """



    def __init__(
        self,
        mesh: MeshGeometry,
        function_space: WithGeometry,
        kappa: float = 1.0,
        dt: float = 0.1,
    ):
        """Initialize the solver with a mesh, function space, diffusivity, and time step.

        :param mesh: Firedrake mesh defining the spatial domain.
        :type mesh: MeshGeometry
        :param function_space: Function space used for the state.
        :type function_space: WithGeometry
        :param kappa: Thermal diffusivity coefficient.
        :type kappa: float
        :param dt: Time step size.
        :type dt: float
        """
        if not isinstance(mesh, MeshGeometry):
            raise TypeError("mesh must be an instance of firedrake.mesh.MeshGeometry")
        if not isinstance(function_space, WithGeometry):
            raise TypeError(
                "function_space must be an instance of firedrake.functionspaceimpl.WithGeometry"
            )
        super().__init__(mesh, function_space)
        self.kappa: float = kappa
        self.dt: float = dt
        self.f = Function(self.V, name="Forcing function")
        self.u0 = Function(self.V, name="Initial condition")
        self.u_desired = Function(self.V, name="Desired solution")
        self.point_wise_error = Function(self.V, name="Pointwise error")



    def set_forcing_function(self, f=Constant(0.0)):
        """Set the forcing term for the heat equation.

        :param f: A Firedrake expression or function assigned to :attr:`f`.
        :type f: firedrake.function.Function
        """
        self.f.interpolate(f)

    def set_initial_condition(self, u0=Constant(0.0)):
        """Set the initial condition for the state variable.

        :param u0: A Firedrake expression or function assigned to :attr:`u0`.
        :type u0: firedrake.function.Function
        """
        self.u0.interpolate(u0)

    def set_bcs(self, bcs: list = [Constant(0.0), Constant(0.0)]):
        """Set the Dirichlet boundary conditions for the solver.

        :param bcs: Boundary values indexed by subdomain, typically for the
            left and right boundaries.
        :type bcs: list
        """
        # TODO: This only accounts for Dirichlet BCs, need to add support for Neumann BCs
        self.bcs = [
            DirichletBC(self.V, bc_value, bc_subdomain + 1)
            for bc_subdomain, bc_value in enumerate(bcs)
        ]

    def build_solver(self):
        """Assemble the variational problem and build the Firedrake solver.

        This creates the trial and test functions, the bilinear form, the load
        form, allocates control and parameter variables, and stores the resulting
        solver in :attr:`solver`.
        """
        self.u = TrialFunction(self.V)
        self.v = TestFunction(self.V)
        self.u_old = Function(self.V, name="Solution at previous time step")
        self.u_new = Function(self.V, name="Solution at new time step")

        self.a = (
            self.dt * inner(grad(self.u), grad(self.v)) * self.kappa
            + inner(self.u, self.v)
        ) * dx
        self.L = (self.dt * inner(self.f, self.v)) * dx + inner(self.u_old, self.v) * dx

        self.allocate_control_variables()
        self.set_control()
        self.allocate_parameters()
        # self.set_parameters()

        self.solver = LinearVariationalSolver(
            LinearVariationalProblem(self.a, self.L, self.u_new, bcs=self.bcs)
        )


    def solve(self):
        """Advance the solution by one time step.

        The previous solution is copied into :attr:`u_old` and the assembled
        variational solver is invoked to update :attr:`u_new`.
        """
        self.u_old.assign(self.u_new)
        self.solver.solve()

    #FIXME: Simplify the interface, user should just be able to inherit HeatEquationSolver and make changes
    def allocate_control_variables(self):
        """Allocate the control variable used by the PDE system.

        This creates the control field and stores it in :attr:`control`.
        """
        self.num_controls = 1  # Add control to forcing function
        self.control = additive_control(self.V, num_controls=self.num_controls)

    def set_control(self):
        """Add the control contribution to the weak form.

        The control term is appended to :attr:`L` so the PDE includes the
        control variable in the forcing term.
        """
        self.L += (self.dt * inner(self.control, self.v)) * dx

    def allocate_parameters(self):
        """Allocate the parameter field used by the solver.

        The parameter field is stored in :attr:`p` and initialized from the
        initial condition.
        """
        self.p = Function(self.V, name="Parameters for the heat equation solver")
        self.p.interpolate(self.u0)

    def set_parameters(self):
        """Update the parameter field from the current solution.

        This assigns the latest state :attr:`u_new` into :attr:`p`.
        """
        self.p.interpolate(self.u_new)

    def set_desired_solution(self, expression):
        """Create a callable that interpolates the desired solution in time.

        :param expression: A callable taking time and returning a Firedrake
            expression or function for the target state.
        :type expression: callable
        :returns: A function that updates :attr:`u_desired` for a given time
            value.
        :rtype: callable
        """

        def desired_solution(t):
            self.u_desired.interpolate(expression(t))
            return self.u_desired

        return desired_solution

    def errors(self):
        """Compute pointwise, L2, and Linf errors against the desired solution.

        :returns: The pointwise error field, the L2 error, and the Linf error.
        :rtype: tuple
        """
        point_wise_error(self.point_wise_error, self.u_new, self.u_desired)
        l2_err = l2_error(self.u_new, self.u_desired)
        linf_err = linf_error(self.point_wise_error)
        return self.point_wise_error, l2_err, linf_err

