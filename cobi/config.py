"""
Configuration Module
====================

Provides default simulation/analysis parameters and a ``load_config`` helper
that merges a YAML / JSON / TOML file on top of those defaults so every
entry-point (notebooks, scripts, jobs) can share the same config interface.

Usage
-----
>>> from cobi.config import load_config
>>> cfg = load_config("jobs/iso/config/b0p3_goal_C3_gc60_d10s5.yaml")
>>> cfg["beta"]
0.3
"""

import json
import sys
from pathlib import Path

DEFAULTS = {
    "libdir": "/global/cfs/cdirs/sobs/cosmic_birefringence/COBIv1",
    "nside": 2048,
    "cb_model": "iso",
    "noise_model": "NC",
    "beta": 0.3,
    "alpha_lat": [0.2, 0.2],
    "alpha_lat_err": 0.05,
    "alpha_sat": 0.0,
    "alpha_sat_err": 0.01,
    "dust_model": 1,
    "sync_model": 1,
    "noise_sensitivity": "goal",
    "galcut": 90,
    "binwidth": 10,
    "aposcale": 2,
    "nsplits": 2,
    "lat_bandpass": True,
    "sat_bandpass": False,
    "verbose": True,
    "start_i": 0,
    "end_i": 100,
    "which": "EB",
}


def _read_file(path: Path) -> dict:
    suffix = path.suffix.lower()
    with path.open("rb") as handle:
        if suffix == ".json":
            return json.load(handle)
        if suffix in {".yaml", ".yml"}:
            import yaml
            return yaml.safe_load(handle) or {}
        if suffix == ".toml":
            if sys.version_info >= (3, 11):
                import tomllib
                return tomllib.load(handle)
            import toml
            return toml.load(path)
    raise ValueError(
        f"Unsupported config format '{suffix}'. Use .yaml, .yml, .json, or .toml."
    )


def load_config(path: str | Path) -> dict:
    """Load a config file and merge it on top of :data:`DEFAULTS`.

    Parameters
    ----------
    path:
        Path to a ``.yaml``, ``.yml``, ``.json``, or ``.toml`` config file.

    Returns
    -------
    dict
        A copy of :data:`DEFAULTS` updated with the values from *path*.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the file extension is not supported.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    config = DEFAULTS.copy()
    config.update(_read_file(path))
    return config
