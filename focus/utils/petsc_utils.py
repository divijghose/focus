from firedrake.petsc import PETSc

def get_user_config():
    """
    Get the user configuration from PETSc options database.
    Returns a dictionary of user-defined options.
    """
    config: dict = {}
    opts = PETSc.Options()
    config["verbose"] = opts.getBool("verbose", False)
    config["pvdOutput"] = opts.getBool("--pvd-output", default=True)
    config["T"] = opts.getReal("--final-time", default=0.01) # Final time
    config["window_size"] = opts.getInt("--window-size", default=5) # Number of time steps in each window
    config["window_step"] = opts.getInt("--window-step", default=1) # Number of time steps to step forward in each \
     # window. Must be less than or equal to window_size.
    assert config["window_step"] <= config["window_size"], \
        "The window step must be less than or equal to the window size."
    config["outfile_path"] = opts.getString("--outfile-path", default="output")
    config["summary_csv_path"] = opts.getString("--summary-csv-path", default="")
    config["decay_constant"] = opts.getReal("--decay-constant", default=0.1) # Time decay constant
    config["misfit_weight"] = opts.getReal("--misfit-weight", default=1.0) # Regularization parameter for the deviation of the state from the desired state

    return config
