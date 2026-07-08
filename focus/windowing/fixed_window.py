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

    def get_window_start_time(self):
        """
        Returns the start time of the current window.
        """
        return self.current_window_start * self.pde_solver.dt

    def get_window_end_time(self):
        """
        Returns the end time of the current window.
        """
        return self.current_window_end * self.pde_solver.dt

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
        if not isinstance(initial_expression, list) and self.pde_solver.num_controls > 1:
            initial_expression = [initial_expression]
   
        if self.pde_solver.num_controls > 1 and len(initial_expression) != self.pde_solver.num_controls:
            raise ValueError(f"Length of initial_expression list ({len(initial_expression)}) must match the number of controls ({self.pde_solver.num_controls}).")

        if self.pde_solver.num_controls == 1:
            self.window_controls = [Function(self.pde_solver.V, name=f"Control at time-hop{i}") for i in range(self.window_size)]
            for m_i in self.window_controls:
                m_i.interpolate(initial_expression)
        else:
            self.window_controls = [[Function(self.pde_solver.V, name=f"Control{j} at time-hop{i}") for i in range(self.window_size)] for j in range(self.pde_solver.num_controls)]

            for j, control in enumerate(self.window_controls):
                for i, m_i in enumerate(control):
                    m_i.interpolate(initial_expression[j])
        print(f"Initialized {self.pde_solver.num_controls} control variable(s) for the fixed window with size {self.window_size}.")
        return self.window_controls


    def run_first_window(self, loss_functional, control):
        """"
        Run the first window to set up the Reduced Functional and the Optimizer.
        """

        for i in range(self.window_size):
            if self.pde_solver.num_controls > 1:
                for j in range(self.pde_solver.num_controls):
                    control[j].interpolate(self.window_controls[j][i])
            else:
                control.interpolate(self.window_controls[i])
            self.pde_solver.solve()
            # Add the "loss" functional
            self.J += loss_functional()
        self.Jhat = ReducedFunctional(self.J, controls=self.window_controls, parameters=self.pde_solver.p)
        pause_annotation()
        
    #FIXME: Should run first window be called separately?
    #FIXME: The loops below don't advance the window, and don't return the actual time
    #TODO: Should time_hop_loop and time_step_loop be combined into one function? They are very similar, but the time_hop_loop also computes the loss functional.

    def _time_hop_loop(self, loss_functional, control):
        """
        Returns a loop over the time hops within the current window.
        """
        for i in range(self.window_size):
            if self.pde_solver.num_controls > 1:
                for j in range(self.pde_solver.num_controls):
                    control[j].interpolate(self.window_controls[j][i])
            else:
                control.interpolate(self.window_controls[i])
            self.pde_solver.solve()
            # Add the "loss" functional
            self.J += loss_functional()

    def _time_step_loop(self, control):
        """
        Returns a loop over the time steps within the current window.
        """
        for i in range(self.window_stride):
            if self.pde_solver.num_controls > 1:
                for j in range(self.pde_solver.num_controls):
                    control[j].interpolate(self.window_controls[j][i])
            else:
                control.interpolate(self.window_controls[i])
            self.pde_solver.solve()
    
    

    def update_parameters(self):
        """
        Update the parameters of the PDE solver for the next window.
        """
        self.pde_solver.set_parameters()
        self.Jhat.update_parameters(self.pde_solver.p)
    


        

            
        


        

    

    