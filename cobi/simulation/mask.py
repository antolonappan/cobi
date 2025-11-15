"""
Mask Generation Module
======================

This module provides tools for creating and managing sky masks used in CMB
analysis, including galactic masks, point source masks, and survey footprints.

Features:

- Multiple mask types (LAT, SAT, CO, point sources, galactic)
- Apodization with multiple methods (C1, C2, Gaussian)
- Galactic cut options
- Combination of multiple mask types
- Automatic mask caching for efficiency

Classes
-------
Mask
    Main class for creating and managing sky masks with apodization.

Example
-------
Create a LAT survey mask with galactic cut::

    from cobi.simulation import Mask
    
    mask = Mask(
        libdir='./masks',
        nside=512,
        select='lat',
        apo_scale=1.0,  # degrees
        apo_method='C2',
        gal_cut=20  # degrees from galactic plane
    )
    
    # Get the mask
    mask_map = mask.mask

Combine multiple masks::

    mask = Mask(
        libdir='./masks',
        nside=512,
        select='lat+gal+ps',  # LAT + galactic + point sources
        apo_scale=1.0
    )

Notes
-----
Masks are cached to disk for reuse. The module supports NaMaster apodization
methods for optimal mode-coupling correction in power spectrum estimation.
"""

# General imports
import os
import numpy as np
import healpy as hp
from pymaster import mask_apodization
# Local imports
from cobi import mpi
from cobi.data import SAT_MASK, LAT_MASK, CO_MASK, PS_MASK, GAL_MASK
from cobi.utils import Logger
class Mask:
    """
    Sky mask generator and manager for CMB analysis.
    
    This class creates and handles various types of sky masks including survey
    footprints (LAT/SAT), galactic plane cuts, point source masks, and CO line
    emission masks. Supports apodization for optimal power spectrum estimation.
    
    Parameters
    ----------
    libdir : str
        Directory for caching mask files.
    nside : int
        HEALPix resolution parameter.
    select : str
        Mask type selector. Can combine multiple masks with '+':
        - 'lat': LAT survey footprint
        - 'sat': SAT survey footprint  
        - 'co': CO line emission mask
        - 'ps': Point source mask
        - 'gal': Galactic plane mask
        Example: 'lat+gal+ps' combines LAT, galactic, and point source masks.
    apo_scale : float, default=0.0
        Apodization scale in degrees. If 0, no apodization applied.
    apo_method : {'C1', 'C2', 'Gaussian'}, default='C2'
        Apodization method compatible with NaMaster.
    gal_cut : float, int, or str, default=0
        Galactic cut specification:
        - float < 1: f_sky fraction (e.g., 0.4 for 40% sky)
        - int > 1: percentage (e.g., 40 for 40% sky)
        - str: direct percentage '40', '60', '70', '80', '90'
        Only used when 'gal' or 'GAL' in select.
    verbose : bool, default=True
        Enable logging output.
    
    Attributes
    ----------
    nside : int
        HEALPix resolution.
    mask : ndarray
        The combined mask array (values 0-1).
    select : str
        Mask type identifier.
    apo_scale : float
        Apodization scale in degrees.
    fsky : float
        Sky fraction (computed from mask).
    
    Methods
    -------
    get_mask()
        Returns the mask array.
    
    Examples
    --------
    Create LAT mask with galactic cut::
    
        mask = Mask(
            libdir='./masks',
            nside=512,
            select='lat+gal',
            gal_cut=20,  # 20% sky
            apo_scale=1.0,
            apo_method='C2'
        )
        
        mask_array = mask.mask
        print(f"Sky fraction: {mask.fsky:.3f}")
    
    Simple point source mask::
    
        ps_mask = Mask(
            libdir='./masks',
            nside=512,
            select='ps',
            apo_scale=0.5
        )
    
    Combined mask for full analysis::
    
        full_mask = Mask(
            libdir='./masks',
            nside=512,
            select='lat+gal+ps+co',
            gal_cut=40,
            apo_scale=1.0
        )
    
    Notes
    -----
    - Masks are cached to disk for efficient reuse
    - Apodization reduces mode-coupling in power spectra
    - Multiple masks are combined via multiplication
    - Compatible with NaMaster (pymaster) workflows
    """
    def __init__(self, 
                 libdir: str, 
                 nside: int, 
                 select: str, 
                 apo_scale: float = 0.0,
                 apo_method: str = 'C2',
                 gal_cut: float | int | str = 0,
                 verbose: bool=True) -> None:
        """
        Initializes the Mask class for handling and generating sky masks.

        Parameters:
        nside (int): HEALPix resolution parameter.
        libdir (Optional[str], optional): Directory where the mask may be saved or loaded from. Defaults to None.
        """
        self.logger = Logger(self.__class__.__name__,verbose)
        self.libdir = libdir
        self.maskdir = os.path.join(libdir, "Masks")
        if mpi.rank == 0:
            os.makedirs(self.maskdir, exist_ok=True)
        self.nside = nside
        self.select = select
        self.apo_scale = apo_scale
        self.apo_method = apo_method

        mask_mapper = {'40':0,'60':1,'70':2,'80':3,'90':4}

        if 'GAL' in select:
            if isinstance(gal_cut, float) and gal_cut < 1 :
                self.logger.log(f"The given galactic cut value seems in fsky and it corresponds to {gal_cut*100}% of sky", level="info")
                assert str(int(gal_cut*100)) in mask_mapper.keys(), f"Invalid gal_cut value: {gal_cut}, it should be in [0.4,0.6,0.7,0.8,0.9]"
                gal_cut = mask_mapper[str(int(gal_cut*100))]
            elif isinstance(gal_cut, int) and gal_cut > 1 :
                self.logger.log(f"The given galactic cut value seems in percent of sky and it corresponds to {gal_cut}% of sky", level="info")
                assert str(gal_cut) in mask_mapper.keys(), f"Invalid gal_cut value: {gal_cut}, it should be in [40,60,70,80,90]"
                gal_cut = mask_mapper[str(gal_cut)]
            elif isinstance(gal_cut, str) :
                assert gal_cut in mask_mapper.keys(), f"Invalid gal_cut value: {gal_cut}, it should be in [40,60,70,80,90]"
                gal_cut = mask_mapper[gal_cut]
            else:
                raise ValueError(f"Invalid gal_cut value: {gal_cut}, it should be in [0,40,60,70,80,90]")

        
        self.gal_cut = gal_cut
        if apo_scale > 0:
            maskfname = os.path.join(self.maskdir, f"{select}_G{gal_cut}_N{nside}_apo{apo_scale}_{apo_method}.fits")
        else:
            maskfname = os.path.join(self.maskdir, f"{select}_G{gal_cut}_N{nside}.fits")
        if os.path.isfile(maskfname):
            self.mask = hp.read_map(maskfname,dtype=np.float64)
        else:
            self.mask = self.__load_mask__()
            if mpi.rank == 0:
                hp.write_map(maskfname, self.mask, dtype=np.float64)
        mpi.barrier()
        self.fsky = self.__calc_fsky__()

    def __mask_obj__(self, select: str):
        match select:
            case "SAT":
                mask = SAT_MASK
            case "LAT":
                mask = LAT_MASK
            case "CO":
                mask = CO_MASK
            case "PS":
                mask = PS_MASK
            case "GAL":
                mask = GAL_MASK
            case _:
                raise ValueError(f"Invalid mask selection: {self.select}")
        return mask

    def __load_mask_healper__(self) -> np.ndarray:
        """
        Loads a mask from a file.

        Returns:
        np.ndarray: The mask array.
        """
        if 'x' in self.select:
            self.logger.log("Loading composite mask", level="info")
            masks = self.select.split('x')
            final_mask = np.ones(hp.nside2npix(self.nside))
            fsky = []
            for mask in masks:
                maskobj = self.__mask_obj__(mask)
                maskobj.directory = self.libdir
                if mask == 'GAL':
                    maskobj.galcut = self.gal_cut
                smask = maskobj.data
                if hp.get_nside(smask) > self.nside:
                    self.logger.log(f"Downgrading mask {mask} resolution", level="info")
                else:
                    self.logger.log(f"Upgrading mask {mask} resolution", level="info")
                smask = hp.ud_grade(smask, self.nside)
                fsky.append(self.__calc_fsky__(smask))
                final_mask *= smask
            fskyb = sorted(set(fsky))[-2]
            fskyf = self.__calc_fsky__(final_mask)
            self.logger.log(f"Composite Mask {self.select}: fsky changed {fskyb:.2f} -> {fskyf:.2f}  ", level="info")
        else:
            mask = self.__mask_obj__(self.select)
            mask.directory = self.libdir
            final_mask = mask.data
            if hp.get_nside(final_mask) != self.nside:
                if hp.get_nside(final_mask) > self.nside:
                    self.logger.log(f"Downgrading mask {self.select} resolution", level="info")
                else:
                    self.logger.log(f"Upgrading mask {self.select} resolution", level="info")
                final_mask = hp.ud_grade(final_mask, self.nside)
        return np.array(final_mask)
    
    def __load_mask__(self) -> np.ndarray:
        """
        Loads a mask from a file.

        Returns:
        np.ndarray: The mask array.
        """
        mask = self.__load_mask_healper__()
        if self.apo_scale > 0:
            fskyb = self.__calc_fsky__(mask)
            self.logger.log(f"Apodizing mask: scale {self.apo_scale}: method: {self.apo_method}", level="info")
            mask = mask_apodization(mask, self.apo_scale, apotype=self.apo_method)
            fskya = self.__calc_fsky__(mask)
            self.logger.log(f"Apodizing changed the fsky {fskyb:.3f} -> {fskya:.3f}", level="info") 
        return mask

    def __calc_fsky__(self,mask=None) -> float:
        """
        Calculates the fraction of sky covered by the mask.

        Returns:
        float: The fraction of sky covered by the mask.
        """
        if mask is None:
            mask = self.mask
        return float(np.mean(mask ** 2) ** 2 / np.mean(mask ** 4))

    
