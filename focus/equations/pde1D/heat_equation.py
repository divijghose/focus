from .base import PDESolver1D
from firedrake.mesh import MeshGeometry
from firedrake.function import Function
from firedrake.functionspaceimpl import WithGeometry
from firedrake import Constant
from firedrake.ufl_expr import TrialFunction, TestFunction


class HeatEquationSolver1D(PDESolver1D):
    def __init__(self, mesh: MeshGeometry, function_space: WithGeometry, kappa: float = 1.0, dt: float = 0.1, ):
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
        self.f = Function(self.V, name="Forcing function")
        self.f.interpolate(f)
    
    def set_initial_condition(self, u0 = Constant(0.0)):
        self.u0 = Function(self.V, name="Initial condition")
        self.u0.interpolate(u0)
    
    def build_solver(self):
        u = TrialFunction(self.V)
        v = TestFunction(self.V)
        u_old = Function(self.V, name="Solution at previous time step")
        self.u_new = Function(self.V, name="Solution at new time step")
        self.u_new.interpolate(self.u0)
        
        a = (self.dt*inner(grad(u), grad(v))*self.kappa + inner(u, v))*dx 
        L = (self.dt*inner(self.f, v))*dx + inner(u_old, v)*dx
        
        bc = DirichletBC(self.V, 0.0, "on_boundary")
        self.solver = LinearVariationalSolver(LinearVariationalProblem(a, L, self.u_new, bcs=bc))
    
    def solve(self):
        V = FunctionSpace(self.mesh, self.fefamily, self.feorder)
        u = TrialFunction(V)
        u_new = Function(V, name="Solution at new time step")
        u_init = Function(V, name="Initial condition")
        u_desired = Function(V, name="Desired state")
        m = Function(V, name="Control")
        m_list = [Function(V, name=f"Control at time hop {i}") for i in range(window_size)]
        v = TestFunction(V)
        u_point_wise_error = Function(V, name="Pointwise error")
        a = (dt*inner(grad(u), grad(v))*k + inner(u, v))*dx 
        L = (dt*inner(m, v))*dx + inner(u_old, v)*dx
        bc = DirichletBC(V, 0.0, "on_boundary")

