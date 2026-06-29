from .base import PDESolver1D
from firedrake.mesh import MeshGeometry
from firedrake.function import Function
from firedrake.functionspaceimpl import WithGeometry
from firedrake import Constant
from firedrake.ufl_expr import TrialFunction, TestFunction
from firedrake import inner, grad, dx
from firedrake import LinearVariationalProblem, LinearVariationalSolver
from firedrake.bcs import DirichletBC


class HeatEquationSolver1D(PDESolver1D):
    """A class to solve the 1D heat equation using the Firedrake."""

    def __init__(self, mesh: MeshGeometry, function_space: WithGeometry, kappa: float = 1.0, dt: float = 0.1, ):
        """Initialize the HeatEquationSolver1D with a mesh, function space, thermal diffusivity, and time step size."""
        if not isinstance(mesh, MeshGeometry):
            raise TypeError("mesh must be an instance of firedrake.mesh.MeshGeometry")
        if not isinstance(function_space, WithGeometry):
            raise TypeError("function_space must be an instance of firedrake.functionspaceimpl.WithGeometry")
        super().__init__(mesh, function_space)
        self.kappa: float = kappa
        self.solver = None
        self.dt: float = dt
        self.set_forcing_function()
        self.set_initial_condition()
    
    def set_forcing_function(self, f = Constant(0.0)):
        """Set the forcing function for the heat equation."""
        self.f = Function(self.V, name="Forcing function")
        self.f.interpolate(f)
    
    def set_initial_condition(self, u0 = Constant(0.0)):
        """Set the initial condition for the heat equation."""
        self.u0 = Function(self.V, name="Initial condition")
        self.u0.interpolate(u0)
    
    def set_bcs(self, bcs: list ):
        """Set the boundary conditions for the heat equation."""
        self.bcs = [DirichletBC(self.V, bc_value, bc_subdomain+1) for bc_subdomain, bc_value in enumerate(bcs)]
        

    def build_solver(self):
        """Build the linear variational solver for the heat equation."""
        u = TrialFunction(self.V)
        v = TestFunction(self.V)
        self.u_old = Function(self.V, name="Solution at previous time step")
        self.u_new = Function(self.V, name="Solution at new time step")
        self.u_new.interpolate(self.u0)
        
        a = (self.dt*inner(grad(u), grad(v))*self.kappa + inner(u, v))*dx 
        L = (self.dt*inner(self.f, v))*dx + inner(self.u_old, v)*dx

        self.solver = LinearVariationalSolver(LinearVariationalProblem(a, L, self.u_new, bcs=self.bcs))
    
    def solve(self):
        """Solve the heat equation for one time step."""
        if self.solver is None:
            raise RuntimeError("Solver has not been built. Call build_solver() before solve().")
        self.solver.solve()
        self.u_old.assign(self.u_new)


