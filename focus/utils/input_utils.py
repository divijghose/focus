from firedrake.petsc import PETSc


def read_petsc_inputs():
    """
    Read PETSc inputs and return a dictionary of user-defined options.
    """
    config: dict = {}
    opts = PETSc.Options()
    # Console print toggle
    config["verbose"] = opts.getBool("verbose", False)

    # Toggle .pvd output
    config["pvdOutput"] = opts.getBool("--pvd-output", default=True)

    # Final time
    config["T"] = opts.getReal("--final-time", default=0.01)

    # Number of time steps in each window
    config["window_size"] = opts.getInt("--window-size", default=5)

    #  Number of time steps by which window steps forward.
    # Must be less than or equal to window_size.
    config["window_step"] = opts.getInt("--window-step", default=1)
    assert (
        config["window_step"] <= config["window_size"]
    ), "The window step must be less than or equal to the window size."

    # Output file path
    config["outfile_path"] = opts.getString("--outfile-path", default="output")

    # Summary CSV file path
    config["summary_csv_path"] = opts.getString("--summary-csv-path", default="")

    # Time decay constant for weighting misfit in the loss functional
    config["decay_constant"] = opts.getReal("--decay-constant", default=0.1)

    # Regularization parameter for the deviation of the state from the desired state
    config["misfit_weight"] = opts.getReal("--misfit-weight", default=1.0)
    return config


def read_yaml_inputs(yaml_file_path: str):
    """
    Read inputs from a YAML file and return a dictionary of user-defined options.
    """
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required to read YAML files.")
    with open(yaml_file_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def get_user_config():
    """
    If .yaml file is passed as first argument, load config from it.
    Otherwise, read PETSc inputs.
    """
    import sys

    if len(sys.argv) > 1 and sys.argv[1].endswith(".yaml"):
        print(f"Reading configuration from YAML file: {sys.argv[1]}")
        yaml_file_path = sys.argv[1]
        config: dict = read_yaml_inputs(yaml_file_path)
    else:
        print("No .yaml file provided. Reading PETSc inputs.")
        print("Using default options where not specified.")
        config: dict = read_petsc_inputs()
    return config


def pretty_print_config(config: dict):
    """
    Pretty print the configuration dictionary in a table format.
    """
    print("Configuration:")
    print("-" * 30)
    for key, value in config.items():
        print(f"{key:<20}: {value}")
    print("-" * 30)


def print_default_config():
    """
    Print the default configuration values.
    """
    default_config = {
        "verbose": False,
        "pvdOutput": True,
        "T": 0.01,
        "window_size": 5,
        "window_step": 1,
        "outfile_path": "output",
        "summary_csv_path": "",
        "decay_constant": 0.1,
        "misfit_weight": 1.0,
    }
    pretty_print_config(default_config)


if __name__ == "__main__":
    print_default_config()
