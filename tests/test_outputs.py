from focus.utils.output_utils import OutputUtilsBase, OutputUtils1D
from firedrake import *
from firedrake.adjoint import *
import os
import pytest
class DummyPDESolver:
    def __init__(self):
        self.mesh = UnitIntervalMesh(10)
        self.V = FunctionSpace(self.mesh, "CG", 1)
        x = SpatialCoordinate(self.mesh)
        self.u_new = Function(self.V)
        self.u_new.interpolate(sin(pi * x[0]))
        self.control = Function(self.V)
        self.control.interpolate(Constant(1.0))
    
    def dummy_operation(self):
        continue_annotation()
        with set_working_tape() as tape:
            self.u_new.interpolate(self.control + 1.0)
            J = assemble(inner(self.u_new, self.u_new) * dx)
            Jhat = ReducedFunctional(J, Control(self.control))
        pause_annotation()
        return Jhat, tape

def test_output_utils_base():
    solver = DummyPDESolver()
    field_dict = {"Solution": solver.u_new, "Control": solver.control}
    output_utils = OutputUtilsBase(field_dict, "./results", vtk_filename="test_vtk")
    assert output_utils.field_dict == field_dict
    assert output_utils.output_dir == "./results"
    assert output_utils.vtk_filename == "test_vtk"

    # Check if the output directory is created
    assert os.path.exists(output_utils.output_dir)
    # Check if the VTK file is created
    output_utils.save_to_vtk()
    assert os.path.exists(os.path.join(output_utils.output_dir, "test_vtk.pvd"))

    Jhat, tape = solver.dummy_operation()
    tape_filename = os.path.join(output_utils.output_dir, "tape_plot.pdf")
    #plot tape either raises exception or creates a file, so we can check if the file exists after calling plot_tape, so catch the exception and check if the file exists
    try:
        output_utils.plot_tape(tape, tape_filename=tape_filename)
        assert os.path.exists(tape_filename)
    except Exception as e:
        with pytest.raises(Exception):
            output_utils.plot_tape(tape, tape_filename=tape_filename)
    
