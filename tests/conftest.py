"""
Pytest configuration and fixtures for COBI tests.
"""
import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir)


@pytest.fixture
def nside_low():
    """Low resolution nside for fast tests."""
    return 64


@pytest.fixture
def nside_med():
    """Medium resolution nside for moderate tests."""
    return 128


@pytest.fixture
def lmax_low():
    """Low lmax for fast tests."""
    return 128


@pytest.fixture
def sample_mask(nside_low):
    """Create a simple test mask."""
    import healpy as hp
    npix = hp.nside2npix(nside_low)
    mask = np.ones(npix)
    # Mask out galactic plane
    theta, phi = hp.pix2ang(nside_low, np.arange(npix))
    lat = np.pi/2 - theta
    mask[np.abs(lat) < np.radians(20)] = 0
    return mask


@pytest.fixture
def sample_cl(lmax_low):
    """Generate sample power spectrum."""
    ell = np.arange(lmax_low + 1)
    cl_tt = 1e4 / (ell + 10)**2
    cl_ee = 1e3 / (ell + 10)**2
    cl_bb = 1e2 / (ell + 10)**2
    cl_te = 5e2 / (ell + 10)**2
    return {'tt': cl_tt, 'ee': cl_ee, 'bb': cl_bb, 'te': cl_te}
