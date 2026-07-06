import pytest
import numpy as np
from focus.windowing.fixed_window import FixedWindow

def test_fixed_window_initialization():
    window_size = 5
    window_stride = 2
    fixed_window = FixedWindow(window_size, window_stride)

    assert fixed_window.window_size == window_size
    assert fixed_window.window_stride == window_stride
    assert fixed_window.current_window_start == 0
    assert fixed_window.current_window_end == window_size

    window_size = 2
    window_stride = 5
    with pytest.raises(AssertionError):
        FixedWindow(window_size, window_stride)

def test_fixed_window_advance():
    window_size = 5
    window_stride = 2

    fixed_window = FixedWindow(window_size, window_stride)

    fixed_window.advance_window()

    assert fixed_window.current_window_start == window_stride
    assert fixed_window.current_window_end == window_size + window_stride

def test_initialize_controls():
    from firedrake import UnitIntervalMesh, FunctionSpace, Function

    mesh = UnitIntervalMesh(10)
    V = FunctionSpace(mesh, "CG", 2)

    window_size = 3
    window_stride = 1
    fixed_window = FixedWindow(window_size, window_stride)

    controls = fixed_window.initialize_controls(V)

    assert len(controls) == window_size
    for control in controls:
        assert isinstance(control, Function)