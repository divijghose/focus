from firedrake.function import Function
from firedrake import Constant
from pyadjoint import ReducedFunctional
from firedrake.adjoint import pause_annotation

class FixedWindow():
    def __init__(self, window_size, window_stride, pde_solver):
        self.window_size = window_size
        self.window_stride = window_stride
        self.pde_solver = pde_solver
        assert self.window_size > 0, "Window size must be greater than zero."
        assert self.window_stride > 0, "Window stride must be greater than zero."
        assert self.window_stride <= self.window_size, "Window stride must be less than or equal to window size."
        self.current_window_start = 0
        self.current_window_end = window_size
        self.functional_value = 0
        self.J = 0
    
    def get_current_window(self):
        """
        Returns the start and end indices of the current window.
        """
        return self.current_window_start, self.current_window_end
    
    def advance_window(self):
        """
        Advances the window by the specified stride.
        """
        self.current_window_start += self.window_stride
        self.current_window_end += self.window_stride
    
    def initialize_controls(self, initial_expression=Constant(0.0)):
        """
        Initialize a list of controls for the fixed window.
        """
        self.window_controls = [Function(self.pde_solver.V, name=f"Control at time-hop{i}") for i in range(self.window_size)]
        for control in self.window_controls:
            control.interpolate(initial_expression)
        return self.window_controls


    def run_first_window(self, loss_functional, control):
        """"
        Run the first window to set up the Reduced Functional and the Optimizer.
        """
        self.pde_solver.u_new.interpolate(self.pde_solver.u0)
        self.pde_solver.u_old.assign(self.pde_solver.u_new)
        for m_i in self.window_controls:
            control.interpolate(m_i)
            self.pde_solver.solve()
            self.pde_solver.u_old.assign(self.pde_solver.u_new)
            # Add the "loss" functional
            self.J += loss_functional()
        self.Jhat = ReducedFunctional(self.J, controls=self.window_controls, parameters=self.pde_solver.u0)
        pause_annotation()
    

    def time_hop_loop(self, loss_functional, control):
        """
        Returns a loop over the time hops within the current window.
        """
        self.pde_solver.u_old.assign(self.pde_solver.u_new)
        for m_i in self.window_controls:
            control.interpolate(m_i)
            self.pde_solver.solve()
            self.pde_solver.u_old.assign(self.pde_solver.u_new)
            # Add the "loss" functional
            self.J += loss_functional()

    def time_step_loop(self, control):
        """
        Returns a loop over the time steps within the current window.
        """
        for i in range(self.window_stride):
            control.interpolate(self.window_controls[i])
            self.pde_solver.solve()
            self.pde_solver.u_old.assign(self.pde_solver.u_new)
            self._update_initial_condition()
            



        # t_actual = time_step_loop(m_opt[i], t_actual)
        # u_desired.interpolate(u_desired_expr(t_actual))
        # u_point_wise_error.interpolate(abs(u_desired - u_new))
        # if pvdOutput:
        #     outfile.write(u_new, m, u_desired, u_point_wise_error)
        # l2_error = norm(u_desired - u_new)
        # linf_error = max(u_point_wise_error.dat.data)
        # if verbose:
        #     PETSc.Sys.Print(f"Time {t_actual:.4f}, L2 error: {l2_error:.6f}, L-infinity error: {linf_error:.6f}")
        # l2_errors.append(l2_error)
        # linf_errors.append(linf_error)
        # u_init.interpolate(u_new)




        # t_current = t_init
        # m.interpolate(m_opt)
        # solver_heat.solve()
        # u_old.assign(u_new)
        # t_current += dt
        # return t_current
    
    def _update_initial_condition(self):
        """
        Set the new initial condition for the PDE solver.
        """
        self.pde_solver.u0.interpolate(self.pde_solver.u_new)
        self.Jhat.update_parameters(self.pde_solver.u0)
        

            
        


        

    

    