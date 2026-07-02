def additive_control(pde_solver):
    """
    Additive control function for the PDE solver.
    This function returns the forcing term as the control variable.
    """
    control = pde_solver.f
    return control