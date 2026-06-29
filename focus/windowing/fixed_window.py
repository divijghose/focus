class FixedWindow():
    def __init__(self, window_size, window_stride, pdesolver, optimizer):
        self.window_size = window_size
        self.window_stride = window_stride
        assert self.window_size > 0, "Window size must be greater than zero."
        assert self.window_stride > 0, "Window stride must be greater than zero."
        assert self.window_stride <= self.window_size, "Window stride must be less than or equal to window size."
        self.pdesolver = pdesolver
        self.optimizer = optimizer
        self.current_window_start = 0
        self.current_window_end = window_size
        self.functional_value = 0
    
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
    
    def reset_functional_value(self):
        """
        Resets the functional value to zero.
        """
        self.functional_value = 0

    def run_first_window(self, dt: float, controls: list, functionals):
        """"
        Run the first window to set up the Reduced Functional and the Optimizer.
        """
        

    def time_hop_loop(self, dt: float, controls: list, functionals):
        """
        Returns a loop over the time hops within the current window.
        """
        t_current = self.current_window_start * dt
        t_window = 0
        for m_i in controls:
            self.pdesolver.f.interpolate(m_i)
            self.pdesolver.solve()
            self.pdesolver.u_old.assign(self.pdesolver.u_new)
            t_current += dt
            t_window += dt
        pass


        

    

    