from .fixed_window import FixedWindow
from firedrake.adjoint import EnsembleReducedFunctional

class EnsembleFixedWindow(FixedWindow):
    def __init__(self, window_size, window_stride, pde_solver, ensemble):
        super().__init__(window_size, window_stride, pde_solver)
        self.ensemble = ensemble

    def run_first_window(self, loss_functional):
        """
        Run the first window of the ensemble simulation.

        :param loss_functional: The loss functional to be minimized.
        :type loss_functional: EnsembleReducedFunctional
        """
        if not isinstance(loss_functional, EnsembleReducedFunctional):
            raise TypeError("loss_functional must be an instance of EnsembleReducedFunctional.")
        
        # Run the PDE solver for the first window
        self.pde_solver.solve_time_window(
            start_time=self.get_window_start_time(),
            end_time=self.get_window_end_time()
        )
        
        # Compute the loss functional for the current window
        self.functional_value = loss_functional(self.pde_solver.controls)