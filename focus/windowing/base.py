from abc import ABC, abstractmethod

class Windowing(ABC):
    def __init__(self, window_size: int, window_stride: int):
        self.window_size = window_size
        self.window_stride = window_stride
        self.current_window_start = 0
        self.current_window_end = window_size

    def get_current_window(self):
        """Return the current window as a tuple (start, end)."""
        return (self.current_window_start, self.current_window_end)
    
    @abstractmethod
    def advance_window(self):
        """Advance the window by the specified stride, with or without overlap."""
        pass
    
    @abstractmethod
    def run_first_window(self):
        """Run the first window to set up the Reduced Functional and the Optimizer."""
        pass

    
