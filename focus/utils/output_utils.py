import matplotlib as mpl
mpl.use('TkAgg')  # or whatever other backend that you want
import matplotlib.pyplot as plt
from firedrake import VTKFile
from pyadjoint import Tape
import numpy as np
import os
from firedrake import FunctionSpace, Function, Constant, SpatialCoordinate

class OutputUtilsBase:
    """
    Base class for output utilities.
    """

    def __init__(self, field_dict, output_dir, **kwargs):
        if not isinstance(field_dict, dict):
            raise TypeError("field_dict must be a dictionary")
        if not isinstance(output_dir, str):
            raise TypeError("output_dir must be a string")
        self.field_dict = field_dict
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        self.vtk_filename = kwargs.get("vtk_filename", "solution")
        self.plot_filename = kwargs.get("plot_filename", "solution_plot")
        
    def save_to_vtk(self):
        """
        Save the solution of the PDE solver to a VTK file.

        Parameters:
        filename: The name of the output VTK file (without extension).
        """

        if self.vtk_filename == "solution":
            print("Using default vtk_filename: 'solution'. Consider providing a custom filename.")
        elif not isinstance(self.vtk_filename, str):
            raise TypeError("vtk_filename must be a string")
        # lower case and remove spaces from filename
        filename = self.vtk_filename.lower().strip().replace(" ", "_")
        outfile_path = os.path.join(self.output_dir, filename)
        vtkfile = VTKFile(f"{outfile_path}.pvd")
        vtkfile.write(*self.field_dict.values())
        
    
    def plot_results(self):
        """
        Plot the results of the PDE solver.
        """
        pass


    def plot_tape(self, tape, tape_filename="tape_plot.pdf"):
        """
        visualize the tape
        """
        try:
            Tape.visualise(tape, tape_filename)
        except Exception as e:
            print(f"Failed to visualize tape: {e}")
            try:
                import networkx
            except ImportError:
                print("networkx is not installed. Please install it to visualize the tape.")
            try:
                import pygraphviz
            except ImportError:
                print("pygraphviz is not installed. Please install it to visualize the tape.")
            print("Ensure that the necessary dependencies (networkx and pygraphviz) for tape visualization are installed.")




class OutputUtils1D(OutputUtilsBase):
    """
    Output utilities for 1D problems.
    """

    def __init__(self, field_dict, output_dir, **kwargs):
        super().__init__(field_dict, output_dir, **kwargs)

    def check_fields(self):
        """
        Check if the fields in field_dict are suitable for 1D plotting.
        """
        given_dict = set(self.field_dict.keys())
        required_dict = {"Solution", "Desired", "Control", "Error"}
        missing = required_dict - given_dict
        if missing:
            return False, missing
        else:
            return True, None

            
    def plot_results(self):
        """
        Plot the solution of the PDE solver in 1D.
        """
        plt.rcParams["text.usetex"] = True
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Computer Modern"]
        #FIXME: This is a hack, to project to a piece-wise linear function space for plotting. This should be handled more gracefully in the future. See firedrake.pyplot
        plotting_function_space = FunctionSpace(self.field_dict["Solution"].function_space().mesh(), "CG", 1)
        solution_projected = Function(plotting_function_space)
        solution_projected.interpolate(self.field_dict["Solution"])
        self.field_dict["Solution"] = solution_projected
        error_projected = Function(plotting_function_space)
        error_projected.interpolate(self.field_dict["Error"])
        self.field_dict["Error"] = error_projected
        control_projected = Function(plotting_function_space)
        control_projected.interpolate(self.field_dict["Control"])
        self.field_dict["Control"] = control_projected
        desired_projected = Function(plotting_function_space)
        desired_projected.interpolate(self.field_dict["Desired"])
        self.field_dict["Desired"] = desired_projected
        plot_path = os.path.join(self.output_dir, self.plot_filename)
        if self.check_fields()[0]:
            plt.figure(figsize=(8, 6))
            fig, ax = plt.subplots(1, 3, figsize=(8, 6))
            x = self.field_dict["Solution"].function_space().mesh().coordinates.dat.data_ro[:]
            solution = self.field_dict["Solution"].dat.data_ro[:]
            desired = self.field_dict["Desired"].dat.data_ro[:]
            control = self.field_dict["Control"].dat.data_ro[:]
            error = self.field_dict["Error"].dat.data_ro[:]
            ax[0].plot(x, solution, label="Solution")
            ax[0].plot(x, desired, label="Desired")
            ax[0].set_xlabel("x")
            ax[0].set_ylabel("u(x)")
            ax[0].set_title("Solutions")
            ax[0].legend()
            ax[1].plot(x, control, label="Control", color='orange')
            ax[1].set_xlabel("x")
            ax[1].set_ylabel("Control")
            ax[1].set_title("Control")
            ax[2].plot(x, error, label="Error", color='green')
            ax[2].set_xlabel("x")
            ax[2].set_ylabel("Error")
            ax[2].set_title("Error")
            for a in ax:
                a.grid()
            plt.tight_layout()
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            fields = self.field_dict.keys()
            size = len(fields)
            fig, ax = plt.subplots(1, size, figsize=(8,6))
            for field in fields:
                x = self.field_dict["Solution"].function_space().mesh().coordinates.dat.data_ro[:]
                field_data = self.field_dict[field].dat.data_ro[:]
                plt.plot(x, field_data, label=field)
                plt.xlabel("x")
                plt.ylabel(field)
                plt.title(f"{field} Plot")
                plt.grid()
                plt.tight_layout()