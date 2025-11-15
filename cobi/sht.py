"""
Spherical Harmonic Transforms Module
=====================================

This module provides fast spherical harmonic transforms using the DUCC0 library
for HEALPix maps. It offers significant performance improvements over healpy's
default implementation, especially for high-resolution maps.

The module supports:
- Forward transforms (map2alm) for spin-0 and spin-2 fields
- Adjoint/inverse transforms (alm2map)
- Automatic pixel weighting for accurate transforms
- Beam smoothing operations
- Multi-threaded execution

Classes
-------
HealpixDUCC
    Main class for performing spherical harmonic transforms on HEALPix maps
    using the DUCC0 library.

Example
-------
    from cobi.sht import HealpixDUCC
    import numpy as np
    
    # Initialize for nside=512
    hp_ducc = HealpixDUCC(nside=512)
    
    # Transform Q/U polarization maps to E/B alms
    qu_maps = np.array([q_map, u_map])
    eb_alms = hp_ducc.map2alm(qu_maps, lmax=1500, nthreads=4)
    
    # Smooth and transform back
    smoothed_maps = hp_ducc.smoothing(qu_maps, lmax=1500, fwhm=5.0, nthreads=4)

Notes
-----
This module is imported from https://github.com/antolonappan/PyNILC/blob/main/pynilc/sht.py
and provides a high-performance alternative to healpy for spherical harmonic transforms.
"""

# imported from https://github.com/antolonappan/PyNILC/blob/main/pynilc/sht.py
from ducc0.sht.experimental import synthesis, adjoint_synthesis
import ducc0
import numpy as np
import healpy as hp



class HealpixDUCC:
    """
    A class to handle HEALPix geometry and perform spherical harmonic transforms.
    Attributes:
    ----------
    nside : int
        The HEALPix nside parameter.
    """
    def __init__(self, nside):
        base = ducc0.healpix.Healpix_Base(nside, "RING")
        geom = base.sht_info()
        area = (4 * np.pi) / (12 * nside ** 2)
        w = np.full((geom['theta'].size, ), area)
        phi0 = geom['phi0']
        nphi = geom['nphi']
        ringstart = geom['ringstart']
        theta = geom['theta']
        
        for arr in [phi0, nphi, ringstart, w]:
            assert arr.size == theta.size

        argsort = np.argsort(ringstart) # We sort here the rings by order in the maps
        self.theta = theta[argsort].astype(np.float64)
        self.weight = w[argsort].astype(np.float64)
        self.phi0 = phi0[argsort].astype(np.float64)
        self.nph = nphi[argsort].astype(np.uint64)
        self.ofs = ringstart[argsort].astype(np.uint64)
    
    def getsize(self, lmax, mmax):
        """
        Get size of the alm array for a given lmax and mmax.
        Parameters:
        -----------
        lmax : int
            Maximum l.
        mmax : int
            Maximum m.
        """
        return ((mmax+1) * (mmax+2)) // 2 + (mmax+1) * (lmax-mmax)

    def __tomap__(self, alm: np.ndarray, spin:int, lmax:int, mmax:int, nthreads:int, map:np.ndarray=None, **kwargs):
        """
        A helper function to perform the forward spherical harmonic transform.
        Parameters:
        -----------
        alm : np.ndarray
            The input alm array.
        spin : int
            The spin of the field.
        lmax : int
            Maximum l.
        mmax : int
            Maximum m.
        nthreads : int
            Number of threads to use.
        map : np.ndarray    
            The output map array.
        """
        alm = np.atleast_2d(alm)
        return synthesis(alm=alm, theta=self.theta, lmax=lmax, mmax=mmax, nphi=self.nph, spin=spin, phi0=self.phi0,
                         nthreads=nthreads, ringstart=self.ofs, map=map, **kwargs)

    def __toalm__(self, m: np.ndarray, spin:int, lmax:int, mmax:int, nthreads:int, alm=None, apply_weights=True, **kwargs):
        """
        A helper function to perform the adjoint spherical harmonic transform.
        Parameters:
        -----------
        m : np.ndarray
            The input map array.
        spin : int
            The spin of the field.
        lmax : int
            Maximum l.
        mmax : int
            Maximum m.
        nthreads : int
            Number of threads to use.
        alm : np.ndarray
            The output alm array.
        apply_weights : bool
            Whether to apply the weights.
        """
        m = np.atleast_2d(m)
        if apply_weights:
            for of, w, npi in zip(self.ofs, self.weight, self.nph):
                m[:, of:of + npi] *= w
        if alm is not None:
            assert alm.shape[-1] == self.getsize(lmax, mmax)
        return adjoint_synthesis(map=m, theta=self.theta, lmax=lmax, mmax=mmax, nphi=self.nph, spin=spin, phi0=self.phi0,
                                 nthreads=nthreads, ringstart=self.ofs, alm=alm,  **kwargs)

    def __alm2map_spin2__(self, alm:np.ndarray, lmax:int, nthreads:int, **kwargs):
        """
        A helper function to perform the forward spherical harmonic transform for spin-2 fields.
        Parameters:
        -----------
        alm : np.ndarray
            The input alm array.
        lmax : int
            Maximum l.
        nthreads : int
            Number of threads to use.
        """
        return self.__tomap__(alm, 2, lmax, lmax, nthreads, **kwargs)

    def __map2alm_spin2__(self, m:np.ndarray, lmax:int, nthreads:int, **kwargs):
        """
        A helper function to perform the adjoint spherical harmonic transform for spin-2 fields.
        Parameters:
        -----------
        m : np.ndarray
            The input map array.
        lmax : int
            Maximum l.
        nthreads : int
            Number of threads to use.
        """
        return self.__toalm__(m.copy(), 2, lmax, lmax, nthreads, **kwargs)

    def __alm2map_spin0__(self, alm:np.ndarray, lmax:int,  nthreads:int, **kwargs):
        """
        A helper function to perform the forward spherical harmonic transform for spin-0 fields.
        Parameters:
        -----------
        alm : np.ndarray
            The input alm array.
        lmax : int
            Maximum l.
        nthreads : int
            Number of threads to use
        """
        return self.__tomap__(alm, 0, lmax, lmax, nthreads, **kwargs).squeeze()

    def __map2alm_spin0__(self, m:np.ndarray, lmax:int, nthreads:int, **kwargs):
        """
        A helper function to perform the adjoint spherical harmonic transform for spin-0 fields.
        Parameters:
        -----------
        m : np.ndarray
            The input map array.
        lmax : int
            Maximum l.
        nthreads : int
            Number of threads to use.
        """
        return self.__toalm__(m.copy(), 0, lmax, lmax, nthreads, **kwargs).squeeze()
    
    def alm2map(self, alm:np.ndarray, lmax:int,nthreads:int, **kwargs):
        """
        Perform the forward spherical harmonic transform.
        Parameters:
        -----------
        alm : np.ndarray
            The input alm array.
        lmax : int
            Maximum l.
        nthreads : int
            Number of threads to use.
        """
        if len(alm) > 3:
            return self.__alm2map_spin0__(alm, lmax, nthreads, **kwargs)
        elif len(alm) == 2:
            return self.__alm2map_spin2__(alm, lmax, nthreads, **kwargs)
        elif len(alm) == 3:
            return np.concatenate([self.__alm2map_spin0__(alm[0], lmax, nthreads, **kwargs).reshape(1, -1),
                                   self.__alm2map_spin2__(alm[1:], lmax, nthreads, **kwargs)], axis=0)
        else:
            raise ValueError("Invalid alm shape")
    
    def map2alm(self, m:np.ndarray, lmax:int, nthreads:int, **kwargs):
        """
        Perform the adjoint spherical harmonic transform.
        Parameters:
        -----------
        m : np.ndarray
            The input map array.
        lmax : int
            Maximum l.
        nthreads : int
            Number of threads to use.
        """
        if len(m) > 3:
            return self.__map2alm_spin0__(m, lmax, nthreads, **kwargs)
        elif len(m) == 2:
            return self.__map2alm_spin2__(m, lmax, nthreads, **kwargs)
        elif len(m) == 3:
            return np.concatenate([self.__map2alm_spin0__(m[0], lmax, nthreads, **kwargs).reshape(1, -1),
                                   self.__map2alm_spin2__(m[1:], lmax, nthreads, **kwargs)], axis=0)
        else:
            raise ValueError("Invalid map shape")
        
    def smoothing_alms(self, alms:np.ndarray, lmax:int, fwhm:float, nthreads:int, **kwargs):
        """
        Smooth the alm array.
        Parameters:
        -----------
        alms : np.ndarray
            The input alm array.
        lmax : int
            Maximum l.
        fwhm : float
            The full width at half maximum of the beam.
        nthreads : int
            Number of threads to use.
        """
        bl = hp.gauss_beam(np.radians(fwhm/60.), lmax=lmax, pol=True).T
        if len(alms) > 3:
            hp.almxfl(alms, bl[0],inplace=True)
        elif len(alms) == 3:
            hp.almxfl(alms[0], bl[0],inplace=True)
            hp.almxfl(alms[1], bl[1],inplace=True)
            hp.almxfl(alms[2], bl[2],inplace=True)
        else:
            raise ValueError("Invalid alm shape")
        
    def smoothing_maps(self, map:np.ndarray, lmax:int, fwhm:float, nthreads:int, **kwargs):
        """
        Smooth the map array.
        Parameters:
        -----------
        map : np.ndarray
            The input map array.
        lmax : int
            Maximum l.
        fwhm : float
            The full width at half maximum of the beam.
        nthreads : int
            Number of threads to use.
        """
        alms = self.map2alm(map, lmax, nthreads, **kwargs)
        self.smoothing_alms(alms, lmax, fwhm, nthreads, **kwargs)
        return self.alm2map(alms, lmax, nthreads, **kwargs)
    
    def smoothing(self, data:np.ndarray, lmax:int, fwhm:float, nthreads:int, **kwargs):
        """
        Smooth the Map or alm.
        Parameters:
        -----------
        data : np.ndarray
            The input data array.
        lmax : int
            Maximum l.
        fwhm : float
            The full width at half maximum of the beam.
        nthreads : int
            Number of threads to use.
        """
        if data.dtype == np.complex64:
            return self.smoothing_alms(data, lmax, fwhm, nthreads, **kwargs)
        else:
            return self.smoothing_maps(data, lmax, fwhm, nthreads, **kwargs)