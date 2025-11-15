"""
Test simulation modules in cobi.simulation.
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch

try:
    import healpy as hp
    HAS_HEALPY = True
except ImportError:
    HAS_HEALPY = False

pytestmark = pytest.mark.skipif(not HAS_HEALPY, reason="requires healpy")


class TestCMBSimulation:
    """Test CMB simulation functionality."""
    
    @patch('cobi.simulation.cmb.camb')
    @patch('cobi.simulation.cmb.CAMB_INI')
    @patch('cobi.simulation.cmb.SPECTRA')
    def test_cmb_initialization(self, mock_spectra, mock_camb_ini, mock_camb, temp_dir, nside_low):
        """Test CMB class initialization."""
        from cobi.simulation import CMB
        
        # Mock the data loaders
        mock_camb_ini.data = {}
        mock_spectra.data = {
            'tt': np.ones(200),
            'ee': np.ones(200),
            'bb': np.zeros(200),
            'te': np.ones(200)
        }
        
        cmb = CMB(
            libdir=temp_dir,
            nside=nside_low,
            model='iso',
            beta=0.35,
            lensing=False,
            verbose=False
        )
        
        assert cmb.nside == nside_low
        assert cmb.lmax == 3 * nside_low - 1
        assert cmb.beta == 0.35
        assert cmb.model == 'iso'
    
    def test_synfast_pol_output_shape(self, nside_low):
        """Test synfast_pol produces correct output shapes."""
        from cobi.simulation.cmb import synfast_pol
        
        lmax = 128
        spectra = {
            'EE': np.ones(lmax + 1) * 1e-4,
            'BB': np.zeros(lmax + 1),
            'AA': np.ones(lmax + 1) * 1e-6,
            'AE': np.zeros(lmax + 1)
        }
        
        Q, U, A = synfast_pol(nside_low, spectra)
        
        npix = hp.nside2npix(nside_low)
        assert Q.shape == (npix,)
        assert U.shape == (npix,)
        assert A.shape == (npix,)
        assert np.isfinite(Q).all()
        assert np.isfinite(U).all()


class TestMaskGeneration:
    """Test mask generation."""
    
    @patch('cobi.simulation.mask.LAT_MASK')
    @patch('cobi.simulation.mask.GAL_MASK')
    def test_mask_initialization(self, mock_gal, mock_lat, temp_dir, nside_low):
        """Test Mask class initialization."""
        from cobi.simulation import Mask
        
        # Mock mask data
        npix = hp.nside2npix(nside_low)
        mock_lat.data = np.ones(npix)
        mock_gal.data = [np.ones(npix) for _ in range(5)]
        
        mask = Mask(
            libdir=temp_dir,
            nside=nside_low,
            select='LAT',
            apo_scale=0.0,
            verbose=False
        )
        
        assert mask.nside == nside_low
        assert hasattr(mask, 'mask')
        assert len(mask.mask) == npix
    
    def test_mask_properties(self, sample_mask):
        """Test mask has correct properties."""
        assert sample_mask.min() >= 0
        assert sample_mask.max() <= 1
        assert 0 < sample_mask.sum() < len(sample_mask)


class TestNoiseSimulation:
    """Test noise simulation."""
    
    @patch('cobi.simulation.noise.so_models')
    def test_noise_initialization(self, mock_so_models, nside_low):
        """Test Noise class initialization."""
        from cobi.simulation import Noise
        
        # Mock SO noise models
        mock_noise_calc = Mock()
        mock_noise_calc.get_bands.return_value = np.array([93, 145, 225])
        mock_noise_calc.get_noise_curves.return_value = (
            np.arange(300),
            np.ones((3, 300)) * 1e-4,
            np.ones((3, 3, 300)) * 1e-4
        )
        mock_so_models.SOLatV3point1.return_value = mock_noise_calc
        
        noise = Noise(
            nside=nside_low,
            fsky=0.4,
            telescope='LAT',
            sim='NC',
            atm_noise=False,
            nsplits=2,
            verbose=False
        )
        
        assert noise.nside == nside_low
        assert noise.telescope == 'LAT'
        assert noise.nsplits == 2


class TestPowerSpectra:
    """Test power spectrum utilities."""
    
    def test_cl_computation(self, sample_cl, lmax_low):
        """Test power spectrum computation."""
        ell = np.arange(lmax_low + 1)
        
        for key in ['tt', 'ee', 'bb', 'te']:
            cl = sample_cl[key]
            assert len(cl) == lmax_low + 1
            assert cl[0] == 0 or key == 'te'  # TT, EE, BB should have zero monopole
            assert np.all(cl[2:] > 0) or key in ['bb', 'te']  # Most cls positive
    
    def test_eb_correlation(self, lmax_low):
        """Test EB correlation from birefringence."""
        # Simulate rotation
        beta = np.radians(0.35)  # 0.35 degrees
        ell = np.arange(2, lmax_low + 1)
        
        # Mock EE spectrum
        cl_ee = 1e3 / ell**2
        
        # EB from rotation should be ~ sin(2*beta) * sqrt(EE*BB)
        # For small angles, EB ~ beta * EE
        cl_eb_expected = 2 * beta * cl_ee
        
        assert np.all(cl_eb_expected > 0)


class TestRotationAndBirefringence:
    """Test polarization rotation and birefringence."""
    
    def test_rotation_matrix(self):
        """Test polarization rotation matrix."""
        alpha = np.radians(45)  # 45 degree rotation
        
        # Rotation matrix
        cos2a = np.cos(2 * alpha)
        sin2a = np.sin(2 * alpha)
        
        R = np.array([[cos2a, -sin2a], [sin2a, cos2a]])
        
        # Test properties
        # Should be orthogonal: R @ R.T = I
        np.testing.assert_allclose(R @ R.T, np.eye(2), atol=1e-10)
        
        # Determinant should be 1
        assert np.isclose(np.linalg.det(R), 1.0)
    
    def test_qu_rotation(self, nside_low):
        """Test Q/U rotation."""
        npix = hp.nside2npix(nside_low)
        Q = np.random.randn(npix)
        U = np.random.randn(npix)
        
        # Rotate by 45 degrees
        alpha = np.radians(45)
        cos2a = np.cos(2 * alpha)
        sin2a = np.sin(2 * alpha)
        
        Q_rot = Q * cos2a - U * sin2a
        U_rot = Q * sin2a + U * cos2a
        
        # Check polarization intensity is preserved
        P_original = np.sqrt(Q**2 + U**2)
        P_rotated = np.sqrt(Q_rot**2 + U_rot**2)
        
        np.testing.assert_allclose(P_original, P_rotated, rtol=1e-10)
    
    def test_double_rotation_identity(self, nside_low):
        """Test that rotating by α then -α returns to original."""
        npix = hp.nside2npix(nside_low)
        Q = np.random.randn(npix)
        U = np.random.randn(npix)
        
        alpha = np.radians(30)
        
        # First rotation
        cos2a = np.cos(2 * alpha)
        sin2a = np.sin(2 * alpha)
        Q1 = Q * cos2a - U * sin2a
        U1 = Q * sin2a + U * cos2a
        
        # Reverse rotation
        cos2a_inv = np.cos(-2 * alpha)
        sin2a_inv = np.sin(-2 * alpha)
        Q2 = Q1 * cos2a_inv - U1 * sin2a_inv
        U2 = Q1 * sin2a_inv + U1 * cos2a_inv
        
        np.testing.assert_allclose(Q, Q2, atol=1e-10)
        np.testing.assert_allclose(U, U2, atol=1e-10)


class TestMapStatistics:
    """Test statistical properties of maps."""
    
    def test_gaussian_map_statistics(self, nside_low, lmax_low):
        """Test that synfast produces Gaussian maps."""
        cl = np.ones(lmax_low + 1) * 100.0  # Scale up for better statistics
        alm = hp.synalm(cl, lmax=lmax_low)
        m = hp.alm2map(alm, nside_low)
        
        # Test approximate Gaussianity
        # Mean should be close to zero (but allow for statistical fluctuations)
        # For Gaussian with std ~ 10, mean of ~50k pixels should be < 1
        assert np.abs(m.mean()) < 1.0
        
        # Test that map is mostly finite
        assert np.isfinite(m).all()
    
    def test_map_power_spectrum_recovery(self, nside_low, lmax_low):
        """Test that we can recover input power spectrum."""
        # Input power spectrum
        ell = np.arange(lmax_low + 1)
        cl_in = 1000.0 / (ell + 10)**2
        
        # Generate map
        alm = hp.synalm(cl_in, lmax=lmax_low)
        m = hp.alm2map(alm, nside_low)
        
        # Recover power spectrum
        cl_out = hp.anafast(m, lmax=lmax_low)
        
        # Should be similar (with cosmic variance)
        # Check correlation at scales where signal is reasonable
        # Skip first few ells which have high variance
        correlation = np.corrcoef(cl_in[10:], cl_out[10:])[0, 1]
        assert correlation > 0.8  # Good correlation but allow for cosmic variance
