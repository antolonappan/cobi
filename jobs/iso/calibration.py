import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.append("/global/homes/l/lonappan/workspace/cobi")


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


def load_config(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run LAT/SAT calibration maps and spectra from a config file."
    )
    parser.add_argument(
        "config",
        help="Path to a config file, for example config/beta_0p3_goal.yaml",
    )
    parser.add_argument(
        "--savemap",
        action="store_true",
        help="Save LAT and SAT observed QU maps.",
    )
    parser.add_argument(
        "--spectra",
        action="store_true",
        help="Compute cross spectra.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.savemap and not args.spectra:
        raise ValueError("Choose at least one option: --savemap and/or --spectra")

    from cobi import mpi
    from cobi.simulation import LATskyC, SATskyC
    from cobi.spectra import SpectraCross

    config = DEFAULTS.copy()
    config.update(load_config(args.config))

    lat = LATskyC(
        config["libdir"],
        config["nside"],
        config["cb_model"],
        config["beta"],
        alpha=config["alpha_lat"],
        alpha_err=config["alpha_lat_err"],
        dust_model=config["dust_model"],
        sync_model=config["sync_model"],
        bandpass=config["lat_bandpass"],
        verbose=config["verbose"],
        nsplits=config["nsplits"],
        noise_model=config["noise_model"],
        noise_sensitivity=config["noise_sensitivity"],
    )

    sat = SATskyC(
        config["libdir"],
        config["nside"],
        config["cb_model"],
        config["beta"],
        alpha=config["alpha_sat"],
        alpha_err=config["alpha_sat_err"],
        dust_model=config["dust_model"],
        sync_model=config["sync_model"],
        bandpass=config["sat_bandpass"],
        verbose=config["verbose"],
        nsplits=config["nsplits"],
        noise_model=config["noise_model"],
        noise_sensitivity=config["noise_sensitivity"],
    )

    spec = SpectraCross(
        config["libdir"],
        lat,
        sat,
        binwidth=config["binwidth"],
        galcut=config["galcut"],
        aposcale=config["aposcale"],
    )

    jobs = np.arange(config["start_i"], config["end_i"])

    if args.savemap:
        for i in jobs[mpi.rank :: mpi.size]:
            lat.SaveObsQUs(i)
            sat.SaveObsQUs(i)
        mpi.barrier()

    if args.spectra:
        for i in jobs[mpi.rank :: mpi.size]:
            spec.__spectra_matrix_core__(i, which=config["which"])
        mpi.barrier()


if __name__ == "__main__":
    main()
