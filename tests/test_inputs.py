from focus.utils.input_utils import get_user_config, get_ensemble_config

import pytest

DEFAULT_CONFIG = {
    "verbose": False,
    "pvdOutput": True,
    "T": 0.01,
    "window_size": 5,
    "window_stride": 1,
    "outfile_path": "output",
    "summary_csv_path": "",
    "decay_constant": 0.1,
    "control_weight": 1.0,
}

def test_default_user_config():
    # Test with no command line arguments (should read PETSc inputs)
    config = get_user_config()
    assert isinstance(config, dict)
    for key, value in DEFAULT_CONFIG.items():
        assert key in config
        assert config[key] == value

def test_default_ensemble_config():
    # Test with no command line arguments (should read PETSc inputs)
    ensemble_config = get_ensemble_config()
    assert isinstance(ensemble_config, dict)
    assert "ensemble_size" in ensemble_config
    assert "processes_per_member" in ensemble_config
    assert ensemble_config["ensemble_size"] == 10
    assert ensemble_config["processes_per_member"] == 2

