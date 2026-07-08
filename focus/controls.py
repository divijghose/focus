from firedrake import Function
def additive_control(function_space, num_controls=1):
    """
    Additive control function for the PDE solver.
    This function returns the forcing term as the control variable.
    """
    control = [Function(function_space, name=f"Additive control variable {i}") for i in range(num_controls)]
    return control if num_controls > 1 else control[0]

def multiplicative_control(function_space, num_controls=1):
    """
    Multiplicative control function for the PDE solver.
    This function returns the forcing term as the control variable.
    """
    control = [Function(function_space, name=f"Multiplicative control variable {i}") for i in range(num_controls)]
    return control if num_controls > 1 else control[0]