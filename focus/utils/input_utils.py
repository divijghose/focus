import sys
from dataclasses import dataclass, field

from firedrake.petsc import PETSc

from focus.utils.output_utils import get_logger

logger = get_logger(__name__)
OPTS = PETSc.Options()

@dataclass
class UserConfig:
    """
    A dataclass to hold user-defined configuration options.
    """
    # I/O
    verbose: bool = field(default=False, metadata={"help": "Enable verbose output"})
    outfile_path: str = field(default="output", metadata={"help": "Output file path"})

    # Simulation parameters
    t_max: float = field(default=0.01, metadata={"help": "Final time"})

    # Windowing
    window_size: int = field(default=5, metadata={"help": "Number of time steps in each window"})
    window_stride: int = field(default=1, metadata={"help": "Number of time steps by which window steps forward"})

    # Cost functional
    decay_constant: float = field(default=0.1, metadata={"help": "Time decay constant for weighting misfit in the loss functional"})
    control_weight: float = field(default=1.0, metadata={"help": "Regularization parameter applied to the control constraint"})

    # Ensemble parameters (Optional)
    ensemble_size: int | None = field(default=None, metadata={"help": "Number of ensemble members"})
    processes_per_member: int | None = field(default=None, metadata={"help": "Number of processes allocated to each ensemble member"})

def _get_yaml_path() -> str | None:
    """Return the YAML file path if provided as the first command-line argument, otherwise return None."""
    if len(sys.argv) > 1 and sys.argv[1].endswith(".yaml"):
        return sys.argv[1]
    return None

def _read_yaml_inputs(yaml_file_path: str) -> UserConfig:
    """Read the input configuration YAML file

    :param yaml_file_path: Path to the YAML file
    :type yaml_file_path: str
    :raises ImportError: If PyYAML is not installed.
    :raises FileNotFoundError: If YAML file does not exist
    :raises ValueError: If YAML file cannot be read or parsed
    :raises TypeError: If the YAML file does not contain a top-level mapping
    :return: A dictionary of user-defined configurations
    :rtype: dict
    """
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required to read YAML files.")
    try:
        with open(yaml_file_path, "r") as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"YAML configuration file not found at {yaml_file_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Failed to read YAML file at {yaml_file_path} : {e}")

    if not isinstance(config, dict):
        raise TypeError(f"YAML file at {yaml_file_path} must contain the configuration dict")
    
    logger.info(f"Reading configuration from {yaml_file_path}")

    flattened_config = {}
    for value in config.values():
        if isinstance(value, dict):
            flattened_config.update(value)
        else:
            flattened_config.update(config)
            break
    defaults = UserConfig()
    return UserConfig(
        verbose=flattened_config.get("verbose", defaults.verbose),
        outfile_path=flattened_config.get("outfile_path", defaults.outfile_path),
        t_max=flattened_config.get("t_max", defaults.t_max),
        window_size=flattened_config.get("window_size", defaults.window_size),
        window_stride=flattened_config.get("window_stride", defaults.window_stride),
        decay_constant=flattened_config.get("decay_constant", defaults.decay_constant),
        control_weight=flattened_config.get("control_weight", defaults.control_weight),
        ensemble_size=flattened_config.get("ensemble_size", defaults.ensemble_size),
        processes_per_member=flattened_config.get("processes_per_member", defaults.processes_per_member)
    )

def _validate_inputs(config: UserConfig) -> None:
    """Validate the user-defined configuration options.

    :param config: Configuration object containing user-defined options
    :type config: UserConfig
    """
    if config.t_max <= 0:
        raise ValueError("Final time must be greater than zero.")
    if config.window_size < 1:
        raise ValueError("Window size must be greater than or equal to 1.")
    if config.window_stride < 1:
        raise ValueError("Window stride must be greater than or equal to 1.")
    if config.window_stride > config.window_size:
        raise ValueError("Window stride must be less than or equal to window size.")
    if config.ensemble_size is not None and config.ensemble_size < 1:
        raise ValueError("Ensemble size must be greater than or equal to 1.")
    if config.processes_per_member is not None and config.processes_per_member < 1:
        raise ValueError("Processes per member must be greater than or equal to 1.")

def _read_petsc_inputs() -> UserConfig:
    """If YAML file is not provided, get config from PETSc options where provided, switch to default otherwise

    :return: A dictionary of user-defined configurations
    :rtype: UserConfig
    """
    logger.info(msg="No YAML file provided. Reading PETSc inputs.")
    logger.info(msg="Using default options where not specified.")
    deafults = UserConfig()
    ensemble_size_raw = OPTS.getString("--ensemble-size", default="")
    ensemble_processes_raw = OPTS.getString("--processes-per-member", default="")
    ensemble_size = int(ensemble_size_raw) if ensemble_size_raw else None
    processes_per_member = int(ensemble_processes_raw) if ensemble_processes_raw else None
    config = UserConfig(
        verbose=OPTS.getBool("--verbose", default=deafults.verbose),
        outfile_path=OPTS.getString("--outfile-path", default=deafults.outfile_path),
        t_max=OPTS.getReal("--t-max", default=deafults.t_max),
        window_size=OPTS.getInt("--window-size", default=deafults.window_size),
        window_stride=OPTS.getInt("--window-stride", default=deafults.window_stride),
        decay_constant=OPTS.getReal("--decay-constant", default=deafults.decay_constant),
        control_weight=OPTS.getReal("--control-weight", default=deafults.control_weight),
        ensemble_size=ensemble_size,
        processes_per_member=processes_per_member
    )
    return config



def get_user_config():
    yaml_path = _get_yaml_path()
    config = _read_yaml_inputs(yaml_path) if yaml_path else _read_petsc_inputs()
    _validate_inputs(config)
    return config


def pretty_print_config(config: UserConfig):
    """
    Pretty print the configuration dictionary in a table format.
    """
    print("Configuration:")
    print("-" * 30)
    for key, value in vars(config).items():
        if value is not None:
            print(f"{key:<20}: {value}")
    print("-" * 30)






if __name__ == "__main__":
    pretty_print_config(UserConfig())
