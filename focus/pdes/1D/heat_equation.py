from ..base import PDESolver1D
from firedrake.mesh import MeshGeometry
from firedrake.function import Function, FunctionSpace
class HeatEquationSolver1D(PDESolver1D):
    def __init__(self, mesh: MeshGeometry, feargs: dict,kappa: float, f):
        if not isinstance(mesh, MeshGeometry):
            raise TypeError("mesh must be an instance of firedrake.mesh.MeshGeometry")
        super().__init__(mesh)
        self.kappa = kappa
        if not isinstance(f, Function):
            raise TypeError("f must be an instance of firedrake.function.Function")
        self.f = f
        self.fefamily = feargs.get("fefamily", "CG")
        self.feorder = feargs.get("feorder", 1)
    
    def solve(self):
        V = FunctionSpace(self.mesh, self.fefamily, self.feorder)
        

