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


    def run_first_window(self, pde_solver, loss_functional, control):
        """"
        Run the first window to set up the Reduced Functional and the Optimizer.
        """
        pde_solver.u_old.assign(pde_solver.u_new)
        for m_i in self.window_controls:
            control.interpolate(m_i)
            pde_solver.solve()
            pde_solver.u_old.assign(pde_solver.u_new)
            # Add the "loss" functional
            self.J += loss_functional()
        #TODO: Get u_init into the windowing
        self.Jhat = ReducedFunctional(self.J, controls=self.window_controls, parameters=u_init)
        pause_annotation()

        # J = time_hop_loop(m_list, t_actual, J)
        # # controls.append(Control(u_init))
        # # derivative_components = tuple(range(len(m_list)))
        #     t_current = t_init # Keeps track of the time in a time-hop loop, incremented at each time-hop.
        # t_window = 0.0 # Keeps track of the time within the current window.
        # for m_i in controls:
        #     m.interpolate(m_i)
        #     solver_heat.solve()
        #     u_old.assign(u_new)
        #     t_current += dt
        #     t_window += dt
        #     # Add the "loss" functional
        #     # \int_0^T exp(-0.1*t)*0.5*||u_desired(t) - u(t)||^2 + 0.01*||m||^2 dt        
        #     J+= loss_functional(t_current, t_window)

        # # Jhat = ReducedFunctional(J, controls = controls, derivative_components=derivative_components)
        # Jhat = ReducedFunctional(J, controls = controls, parameters=u_init)
        # pause_annotation()
        pass
        

    def time_hop_loop(self, pde_solver, loss_functional, control):
        """
        Returns a loop over the time hops within the current window.
        """
        pde_solver.u_old.assign(pde_solver.u_new)
        for m_i in self.window_controls:
            control.interpolate(m_i)
            pde_solver.solve()
            pde_solver.u_old.assign(pde_solver.u_new)
            # Add the "loss" functional
            self.J += loss_functional()
        
        


        

    

    