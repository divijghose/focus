from .base import PDESolver1D
from firedrake.mesh import MeshGeometry
from firedrake.function import Function, FunctionSpace
from firedrake.ufl_expr import TrialFunction, TestFunction
from ufl.measure.measure import dx

class HeatEquationSolver1D(PDESolver1D):
    def __init__(self, mesh: MeshGeometry, feargs: dict, kappa: float, dt: float, f):
        if not isinstance(mesh, MeshGeometry):
            raise TypeError("mesh must be an instance of firedrake.mesh.MeshGeometry")
        super().__init__(mesh)
        self.kappa = kappa
        if not isinstance(f, Function):
            raise TypeError("f must be an instance of firedrake.function.Function")
        self.f = f
        self.fefamily = feargs.get("fefamily", "CG")
        self.feorder = feargs.get("feorder", 1)
        self.solver = None
        self.dt = dt
    
    def build_solver(self):
        V = FunctionSpace(self.mesh, self.fefamily, self.feorder)
        u = TrialFunction(V)
        v = TestFunction(V)
        a = (self.dt*inner(grad(u), grad(v))*self.kappa + inner(u, v))*dx 
        L = (self.dt*inner(m, v))*dx + inner(u_old, v)*dx
        bc = DirichletBC(V, 0.0, "on_boundary")
        self.solver = LinearVariationalSolver(LinearVariationalProblem(a, L, u_new, bcs=bc))
    
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

