import pytest
import numpy as np
from focus.windowing.fixed_window import FixedWindow

def test_fixed_window_initialization():
    window_size = 5
    window_stride = 2
    pdesolver = None  # Replace with a mock or actual PDE solver if needed
    optimizer = None  # Replace with a mock or actual optimizer if needed

    fixed_window = FixedWindow(window_size, window_stride, pdesolver, optimizer)

    assert fixed_window.window_size == window_size
    assert fixed_window.window_stride == window_stride
    assert fixed_window.current_window_start == 0
    assert fixed_window.current_window_end == window_size

    window_size = 2
    window_stride = 5
    with pytest.raises(AssertionError):
        FixedWindow(window_size, window_stride, pdesolver, optimizer)

def test_fixed_window_advance():
    window_size = 5
    window_stride = 2
    pdesolver = None  # Replace with a mock or actual PDE solver if needed
    optimizer = None  # Replace with a mock or actual optimizer if needed

    fixed_window = FixedWindow(window_size, window_stride, pdesolver, optimizer)

    fixed_window.advance_window()

    assert fixed_window.current_window_start == window_stride
    assert fixed_window.current_window_end == window_size + window_stride

    