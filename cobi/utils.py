"""
Utility Functions Module
=========================

This module contains utility functions for various operations used throughout
the COBI package, including logging, coordinate transformations, map operations,
and numerical utilities.

Functions
---------
Logger
    Configurable logging class for debugging and information messages
inrad
    Convert angles from degrees to radians
cli
    Compute inverse of array elements (safe for zeros)
download_file
    Download files with progress bar
deconvolveQU
    Deconvolve beam from polarization maps
change_coord
    Transform HEALPix maps between coordinate systems
slice_alms
    Slice spherical harmonic coefficients to new lmax

The module provides essential functionality for:
- Logging with configurable verbosity levels
- Angular unit conversions
- Safe numerical operations (division, inversion)
- File downloading with progress tracking
- HEALPix map manipulations (coordinate transforms, beam operations)
- Spherical harmonic coefficient operations
"""

# This file contains utility functions that are used in the main script.

import requests
import logging
import numpy as np
from tqdm import tqdm
import healpy as hp

class Logger:
    def __init__(self, name: str, verbose: bool = False):
        """
        Initializes the logger.
        
        Parameters:
        name (str): Name of the logger, typically the class name or module name.
        verbose (bool): If True, set logging level to DEBUG, otherwise to WARNING.
        """
        self.logger = logging.getLogger(name)
        
        # Configure logging level based on verbosity
        if verbose:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.WARNING)
        
        # Prevent adding multiple handlers to the logger
        if not self.logger.hasHandlers():
            # Create console handler
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            
            # Create formatter and add it to the handler
            formatter = logging.Formatter('%(name)s : %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            
            # Add handler to the logger
            self.logger.addHandler(ch)

    def log(self, message: str, level: str = 'info'):
        """
        Logs a message at the specified logging level.
        
        Parameters:
        message (str): The message to log.
        level (str): The logging level (debug, info, warning, error, critical).
        """
        level = level.lower()
        if level == 'debug':
            self.logger.debug(message)
        elif level == 'info':
            self.logger.info(message)
        elif level == 'warning':
            self.logger.warning(message)
        elif level == 'error':
            self.logger.error(message)
        elif level == 'critical':
            self.logger.critical(message)
        else:
            self.logger.info(message)


def inrad(alpha: float) -> float:
    """
    Converts an angle from degrees to radians.

    Parameters:
    alpha (float): The angle in degrees.

    Returns:
    float: The angle in radians.
    """
    return np.deg2rad(alpha)

def cli(cl: np.ndarray) -> np.ndarray:
    """
    Computes the inverse of each element in the input array `cl`.

    Parameters:
    cl (np.ndarray): Input array for which the inverse is calculated.
                     Only positive values will be inverted; zeros and negative values will remain zero.

    Returns:
    np.ndarray: An array where each element is the inverse of the corresponding element in `cl`,
                with zeros or negative values left unchanged.
    """
    ret = np.zeros_like(cl)
    ret[np.where(cl > 0)] = 1.0 / cl[np.where(cl > 0)]
    return ret


def download_file(url, filename):
    """Download a file with a progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024 # 1 Kibibyte
    t = tqdm(total=total_size, unit='iB', unit_scale=True, desc=f'Downloading {filename}')
    with open(filename, 'wb') as file:
        for data in response.iter_content(block_size):
            t.update(len(data))
            file.write(data)
    t.close()


def deconvolveQU(QU,beam):
    """
    Deconvolves a beam from a QU map.

    Parameters:
    QU (np.ndarray): The input QU map.
    beam (np.ndarray): The beam to deconvolve.

    Returns:
    np.ndarray: The deconvolved QU map.
    """
    beam = np.radians(beam/60)
    nside = hp.npix2nside(len(QU[0]))
    elm,blm = hp.map2alm_spin(QU,2)
    lmax = hp.Alm.getlmax(len(elm))
    bl = hp.gauss_beam(beam,lmax=lmax,pol=True).T
    hp.almxfl(elm,cli(bl[1]),inplace=True)
    hp.almxfl(blm,cli(bl[2]),inplace=True)
    return hp.alm2map_spin([elm,blm],nside,2,lmax)


def change_coord(m, coord=['C', 'G']):
    npix = m.shape[-1]
    nside = hp.npix2nside(npix)
    ang = hp.pix2ang(nside, np.arange(npix))
    rot = hp.Rotator(coord=reversed(coord))
    new_ang = rot(*ang)
    new_pix = hp.ang2pix(nside, *new_ang)
    return m[..., new_pix]


def slice_alms(teb, lmax_new):
    """Returns the input teb alms sliced to the new lmax.

    teb(numpy array): input teb alms
    lmax_new(int): new lmax
    """
    nfields = len(teb)
    lmax = hp.Alm.getlmax(len(teb[0]))
    if nfields > 3:
        lmax = nfields
        nfields = 1
        dtype = teb.dtype
    else:
        dtype = teb[0].dtype
    if lmax_new > lmax:
        raise ValueError('lmax_new must be smaller or equal to lmax')
    elif lmax_new == lmax:
        return teb
    else:
        teb_new = np.zeros((nfields, hp.Alm.getsize(lmax_new)), dtype=dtype)
        indices_full = hp.Alm.getidx(lmax,*hp.Alm.getlm(lmax_new))
        indices_new = hp.Alm.getidx(lmax_new,*hp.Alm.getlm(lmax_new))
        teb_new[:,indices_new] = teb[:,indices_full]
        return teb_new
    

class Binner:
    """
    binner = Binner(n, lmin=2, lmax=..., method='linear')
    binner.b   : effective centers (mode-counting weights), never singleton at l=2
    binner.bin : bins Cl with (2l+1) or (2l+1)/vl^2 weights
    """

    def __init__(self, n, lmin=2, lmax=None, method="linear"):
        if not isinstance(n, int) or n <= 0:
            raise ValueError("n must be a positive integer")
        if not isinstance(lmin, int) or lmin < 2:
            raise ValueError("lmin must be an integer >= 2")
        if lmax is None or (not isinstance(lmax, int)) or lmax < lmin:
            raise ValueError("lmax must be an integer and >= lmin")

        self.n = n
        self.lmin = lmin
        self.lmax = lmax
        self.method = (method or "linear").lower()

        # integer slice-edges [l0:l1) covering lmin..lmax inclusive
        self.bp = self._make_edges(self.n, self.lmin, self.lmax, self.method, min_width=2)

        # precompute mode-counting effective centers (independent of cl)
        ell = np.arange(self.lmax + 1, dtype=float)
        w = 2.0 * ell + 1.0
        self.b = self._weighted_centers(ell, w, self.bp)

    def __getattribute__(self, name: str):
        if name == "bin" or name == 'bin_cell' or name == 'binner' or name == 'do_bin':
            return object.__getattribute__(self, 'bin')
        else:
            return object.__getattribute__(self, name)

    def bin(self, cl, vl=None):
        cl = np.asarray(cl)
        if cl.shape[-1] <= self.lmax:
            raise ValueError(f"cl last axis must have length >= lmax+1={self.lmax+1}")

        ell = np.arange(self.lmax + 1, dtype=float)

        if vl is None:
            w_ell = 2.0 * ell + 1.0
        else:
            vl = np.asarray(vl, dtype=float)
            if vl.shape[0] < self.lmax + 1:
                raise ValueError("vl must have length >= lmax+1")
            w_ell = (2.0 * ell + 1.0) / (vl[: self.lmax + 1] ** 2)

        cl_use = cl[..., : self.lmax + 1]
        return self._bin_core(cl_use, w_ell, self.bp)

    # ---------------- internals ----------------

    @staticmethod
    def _proposed_edges_float(n, lmin, lmax, method):
        if method == "linear":
            # IMPORTANT: use lmax+1 because edges are for half-open slices [l0:l1)
            return np.linspace(lmin, lmax + 1, n + 1)
        if method == "log":
            return lmin * np.exp(np.linspace(0.0, np.log((lmax + 1) / lmin), n + 1))
        if method == "log10":
            return lmin * 10.0 ** np.linspace(0.0, np.log10((lmax + 1) / lmin), n + 1)
        if method == "p2":
            return np.linspace(np.sqrt(lmin), np.sqrt(lmax + 1), n + 1) ** 2
        if method == "p3":
            return np.linspace(lmin ** (1 / 3), (lmax + 1) ** (1 / 3), n + 1) ** 3
        raise ValueError("Unknown method (linear/log/log10/p2/p3).")

    @classmethod
    def _make_edges(cls, n, lmin, lmax, method, min_width=2):
        """
        Build integer slice edges bp of length n+1 such that bins are [bp[i]:bp[i+1]) and
        cover lmin..lmax inclusive (so bp[-1]=lmax+1). Enforce bp[i+1]-bp[i] >= min_width.
        """
        L = (lmax - lmin + 1)  # number of multipoles included
        if n * min_width > L:
            raise ValueError(
                f"Requested n={n} is too large to keep min bin width {min_width} over "
                f"[{lmin},{lmax}] (available multipoles={L}). Reduce n."
            )

        bp_f = cls._proposed_edges_float(n, lmin, lmax, method)

        # Start from floor to avoid tiny first bin from rounding near lmin
        bp = np.floor(bp_f).astype(int)

        # enforce endpoints
        bp[0] = lmin
        bp[-1] = lmax + 1
        bp = np.clip(bp, lmin, lmax + 1)

        # enforce minimum width progressively
        for i in range(1, len(bp)):
            needed = bp[i - 1] + min_width
            if bp[i] < needed:
                bp[i] = needed

        # if we pushed the end beyond lmax+1, pull back uniformly while preserving min_width
        if bp[-1] > lmax + 1:
            bp[-1] = lmax + 1
            for i in range(len(bp) - 2, -1, -1):
                allowed = bp[i + 1] - min_width
                if bp[i] > allowed:
                    bp[i] = allowed
            bp[0] = lmin

        # final checks
        if bp[0] != lmin or bp[-1] != lmax + 1:
            raise ValueError("Failed to construct valid edges; reduce n.")
        if np.any(np.diff(bp) < min_width):
            raise ValueError("Failed to enforce minimum bin width; reduce n.")
        return bp

    @staticmethod
    def _bin_core(cl, w_ell, bp):
        out = []
        for i in range(len(bp) - 1):
            l0, l1 = bp[i], bp[i + 1]  # half-open
            w = w_ell[l0:l1]
            den = np.sum(w)
            if den == 0:
                out.append(np.zeros(cl.shape[:-1], dtype=float))
                continue

            if cl.ndim == 1:
                out.append(np.sum(w * cl[l0:l1]) / den)
            else:
                out.append(np.sum(w * cl[..., l0:l1], axis=-1) / den)

        return np.stack(out, axis=-1)

    @staticmethod
    def _weighted_centers(ell, w_ell, bp):
        b = np.zeros(len(bp) - 1, dtype=float)
        for i in range(len(bp) - 1):
            l0, l1 = bp[i], bp[i + 1]
            w = w_ell[l0:l1]
            den = np.sum(w)
            b[i] = np.sum(w * ell[l0:l1]) / den if den != 0 else 0.5 * (l0 + l1 - 1)
        return b