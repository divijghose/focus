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


class HeatEquationSolver(Solver):
    """A class to solve the heat equation using Firedrake."""

    def __init__(self, mesh: MeshGeometry, function_space: WithGeometry, kappa: float = 1.0, dt: float = 0.1, ):
        """Initialize the HeatEquationSolver1D with a mesh, function space, thermal diffusivity, and time step size."""
        if not isinstance(mesh, MeshGeometry):
            raise TypeError("mesh must be an instance of firedrake.mesh.MeshGeometry")
        if not isinstance(function_space, WithGeometry):
            raise TypeError("function_space must be an instance of firedrake.functionspaceimpl.WithGeometry")
        super().__init__(mesh, function_space)
        self.kappa: float = kappa
        self.dt: float = dt
        self.f = Function(self.V, name="Forcing function")
        self.set_forcing_function()
        self.u0 = Function(self.V, name="Initial condition")
        self.set_initial_condition()

        
    #FIXME: Forcing function can be time-dependent, doesn't make sense to initialize a Function each time
    def set_forcing_function(self, f = Constant(0.0)):
        """Set the forcing function for the heat equation."""
        self.f.interpolate(f)
    
    def set_initial_condition(self, u0 = Constant(0.0)):
        """Set the initial condition for the heat equation."""
        self.u0.interpolate(u0)

    
    def set_bcs(self, bcs: list = [Constant(0.0)]):
        """Set the boundary conditions for the heat equation."""
        #TODO: This only accounts for Dirichlet BCs, need to add support for Neumann BCs
        self.bcs = [DirichletBC(self.V, bc_value, bc_subdomain+1) for bc_subdomain, bc_value in enumerate(bcs)]

    def build_solver(self):
        """Build the linear variational solver for the heat equation."""
        self.u = TrialFunction(self.V)
        self.v = TestFunction(self.V)
        self.u_old = Function(self.V, name="Solution at previous time step")
        self.u_new = Function(self.V, name="Solution at new time step")
        self.u_new.interpolate(self.u0)
        
        self.a = (self.dt*inner(grad(self.u), grad(self.v))*self.kappa + inner(self.u, self.v))*dx 
        self.L = (self.dt*inner(self.f, self.v))*dx + inner(self.u_old, self.v)*dx 

        self.allocate_control_variables()
        self.set_control()

        self.solver = LinearVariationalSolver(LinearVariationalProblem(self.a, self.L, self.u_new, bcs=self.bcs))
    
    def solve(self):
        """Solve the heat equation for one time step."""
        self.u_old.assign(self.u_new)
        self.solver.solve()
    
    def allocate_control_variables(self):
        """Allocate the control variable(s) for the heat equation solver."""
        self.num_controls = 1  # Add control to forcing function
        self.m =   additive_control(self.V, num_controls=self.num_controls)

    def set_control(self):
        """Set the control variable for the heat equation solver."""
        self.L += (self.dt*inner(self.m, self.v))*dx

    def allocate_parameters(self):
        """Allocate which variables are parameters for the heat equation solver."""
        self.p = self.u0
    
    def set_parameters(self):
        """Set the parameter values for the heat equation solver."""
        self.p = self.u_new
    
