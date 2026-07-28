from .tao import TAOOptimizer
from firedrake import *
from firedrake.adjoint import *
from pyadjoint import TAOSolver

class NewtonOptimizer(TAOOptimizer):
    def __init__(self, rf, function_space):
        self.tao_parameters = {
            'tao_view': ':tao_view.log',
            'tao_monitor': None,
            'tao_converged_reason': None,
            'tao_gttol': 3e-1,
            'tao_type': 'nls',
            'tao_ls_type': 'unit',
            'tao_nls': {
                'ksp_monitor': None,
                'ksp_converged_reason': None,
                'ksp_convergence_test': 'skip',
                'ksp_max_it': 10,
                'ksp_type': 'cg',
                'pc_type': 'python',
                'pc_python_type': 'firedrake.CovariancePC',
            },
        }
        self.V = function_space
        super().__init__(rf, parameters=parameters)
        self.B = self.covariance_operator()
        self.tao_solver = TAOSolver(
            self.problem,
            Pmat=CovarianceMat(self.B, 'inverse'),
            parameters=self.tao_parameters,
            options_prefix="",
        )


    def covariance_operator(self):
        B = AutoregressiveCovariance(self.V, L=0.2, sigma=1e-1, m=4, seed=17)
        return B


        