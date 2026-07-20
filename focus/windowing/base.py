from abc import ABC, abstractmethod


class Windowing(ABC):
    def __init__(self, window_size: int, window_stride: int):
        self.window_size = window_size
        self.window_stride = window_stride
        assert self.window_size > 0, "Window size must be greater than zero."
        assert self.window_stride > 0, "Window stride must be greater than zero."
        assert (
            self.window_stride <= self.window_size
        ), "Window stride must be less than or equal to window size."
        self.current_window_start = 0
        self.current_window_end = window_size
        self.window_number = 0
        self.global_step_time = 0.0  # Global time across all windows, updated after each time step
        self.global_hop_time = 0.0  # Global time across all windows, updated after each window hop
        self.window_hop_time = 0.0  # Time within the current window, updated after each time hop

    def get_current_window(self):
        """Return the start and end indices of the current window"""
        return self.current_window_start, self.current_window_end

    @abstractmethod
    def advance_window(self):
        """Advance the window by the specified stride, with or without overlap."""
        pass

    @abstractmethod
    def run_first_window(self, loss_functional):
        """Run the first window to set up the Reduced Functional and the Optimizer."""
        pass
