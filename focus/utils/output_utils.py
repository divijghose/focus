import matplotlib as mpl
mpl.use('TkAgg')  # or whatever other backend that you want
import matplotlib.pyplot as plt
from firedrake import VTKFile
from pyadjoint import Tape
import numpy as np
import os

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

            
    def plot_results(self):
        """
        Plot the solution of the PDE solver in 1D.
        """
        plt.rcParams["text.usetex"] = True
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Computer Modern"]
        plt.figure(figsize=(8, 6))
        #TODO: Fill in the plotting script for 1D problems
        x = self.field_dict["Solution"].function_space().mesh().coordinates.dat.data_ro[:]
        y = np.sin(np.pi * x)  # Example solution, replace with actual solution from field_dict
        plt.figure()
        plt.plot(x, y, label="Solution")
        plt.xlabel("x")
        plt.ylabel("u(x)")
        plt.title("1D Solution Plot")
        plt.legend()
        plt.grid()
        plot_path = os.path.join(self.output_dir, f"{self.plot_filename}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        pass

