import argparse
import sys

import numpy as np

sys.path.append("/global/homes/l/lonappan/workspace/cobi")

from cobi.config import load_config, DEFAULTS


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

    config = load_config(args.config)

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
