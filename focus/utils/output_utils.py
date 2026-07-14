from firedrake import VTKFile
import os


def save_to_vtk(field_list, outfile_path, filename):
    """
    Save the solution of the PDE solver to a VTK file.

    Parameters:
    pde_solver: The PDE solver object containing the solution.
    outfile_path: The path to the output directory.
    filename: The name of the output VTK file (without extension).
    """
    if not os.path.exists(outfile_path):
        os.makedirs(outfile_path, exist_ok=True)
    # lower case and remove spaces from filename
    filename = filename.lower().strip().replace(" ", "_")
    outfile_path = os.path.join(outfile_path, filename)
    vtkfile = VTKFile(f"{outfile_path}.pvd")
    for field in field_list:
        vtkfile.write(field)


# TODO: Account for ensemble writing
