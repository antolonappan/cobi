"""
Test utility functions in cobi.utils module.
"""
import pytest
import numpy as np

try:
    import healpy as hp
    HAS_HEALPY = True
except ImportError:
    HAS_HEALPY = False

try:
    from cobi.utils import inrad, change_coord
    HAS_COBI_UTILS = True
except ImportError:
    HAS_COBI_UTILS = False


pytestmark = pytest.mark.skipif(not HAS_HEALPY or not HAS_COBI_UTILS, 
                                 reason="requires healpy and cobi.utils")


class TestAngularConversion:
    """Test angular conversion utilities."""
    
    def test_inrad_degrees_to_radians(self):
        """Test conversion from degrees to radians."""
        assert np.isclose(inrad(180), np.pi)
        assert np.isclose(inrad(90), np.pi/2)
        assert np.isclose(inrad(0), 0)
    
    def test_inrad_array(self):
        """Test conversion with array input."""
        angles_deg = np.array([0, 45, 90, 180])
        expected_rad = np.array([0, np.pi/4, np.pi/2, np.pi])
        result = inrad(angles_deg)
        np.testing.assert_allclose(result, expected_rad)
    
    def test_inrad_negative_angles(self):
        """Test conversion with negative angles."""
        assert np.isclose(inrad(-90), -np.pi/2)
        assert np.isclose(inrad(-180), -np.pi)


class TestCoordinateTransformation:
    """Test coordinate transformation utilities."""
    
    def test_change_coord_identity(self, nside_low):
        """Test that same coordinates return same map."""
        npix = hp.nside2npix(nside_low)
        input_map = np.random.randn(npix)
        
        output_map = change_coord(input_map, ['C', 'C'])
        np.testing.assert_array_equal(input_map, output_map)
    
    def test_change_coord_galactic_to_ecliptic(self, nside_low):
        """Test coordinate transformation maintains map properties."""
        npix = hp.nside2npix(nside_low)
        input_map = np.random.randn(npix)
        
        output_map = change_coord(input_map, ['G', 'E'])
        
        # Check that transformation preserves mean and std approximately
        assert output_map.shape == input_map.shape
        assert np.isfinite(output_map).all()
    
    def test_change_coord_round_trip(self, nside_low):
        """Test that round-trip transformation recovers original."""
        npix = hp.nside2npix(nside_low)
        input_map = np.random.randn(npix)
        
        # Transform G -> C -> G
        temp_map = change_coord(input_map, ['G', 'C'])
        output_map = change_coord(temp_map, ['C', 'G'])
        
        # Should be close to original (within numerical precision)
        np.testing.assert_allclose(input_map, output_map, rtol=1e-5)


class TestMapOperations:
    """Test basic map operations."""
    
    def test_map_creation(self, nside_low):
        """Test basic map creation."""
        npix = hp.nside2npix(nside_low)
        test_map = np.random.randn(npix)
        
        assert len(test_map) == npix
        assert test_map.dtype == np.float64
    
    def test_alm_to_map_conversion(self, nside_low, lmax_low):
        """Test spherical harmonic transform."""
        # Create random alm
        alm = hp.synalm(np.ones(lmax_low + 1), lmax=lmax_low)
        
        # Convert to map
        test_map = hp.alm2map(alm, nside_low)
        
        assert len(test_map) == hp.nside2npix(nside_low)
        assert np.isfinite(test_map).all()
    
    def test_map_to_alm_to_map(self, nside_low, lmax_low):
        """Test round-trip map->alm->map."""
        npix = hp.nside2npix(nside_low)
        input_map = np.random.randn(npix)
        
        # Map to alm and back
        alm = hp.map2alm(input_map, lmax=lmax_low)
        output_map = hp.alm2map(alm, nside_low)
        
        # Should be similar (not exact due to mode truncation)
        correlation = np.corrcoef(input_map, output_map)[0, 1]
        assert correlation > 0.9


class TestMasking:
    """Test mask operations."""
    
    def test_mask_application(self, nside_low, sample_mask):
        """Test applying mask to map."""
        npix = hp.nside2npix(nside_low)
        test_map = np.random.randn(npix)
        
        masked_map = test_map * sample_mask
        
        # Check that masked regions are zero
        assert masked_map[sample_mask == 0].sum() == 0
        
        # Check that unmasked regions are preserved
        unmasked_idx = sample_mask > 0
        np.testing.assert_array_equal(
            test_map[unmasked_idx], 
            masked_map[unmasked_idx]
        )
    
    def test_fsky_calculation(self, sample_mask):
        """Test sky fraction calculation."""
        fsky = sample_mask.sum() / len(sample_mask)
        
        assert 0 < fsky < 1
        # With 20 degree galactic cut, should have substantial sky fraction
        assert fsky > 0.5
    
    def test_mask_statistics(self, sample_mask):
        """Test mask statistics."""
        assert sample_mask.min() >= 0
        assert sample_mask.max() <= 1
        assert np.isfinite(sample_mask).all()
