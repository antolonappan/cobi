"""
Test that all modules can be imported.
"""
import pytest


class TestImports:
    """Test that all main modules can be imported."""
    
    def test_import_cobi(self):
        """Test main package import."""
        import cobi
        assert hasattr(cobi, '__version__')
    
    def test_import_utils(self):
        """Test utils module import."""
        from cobi import utils
        assert hasattr(utils, 'inrad')
        assert hasattr(utils, 'change_coord')
    
    def test_import_data(self):
        """Test data module import."""
        from cobi import data
        assert hasattr(data, 'CAMB_INI')
    
    def test_import_simulation(self):
        """Test simulation subpackage import."""
        from cobi import simulation
        assert hasattr(simulation, 'CMB')
        assert hasattr(simulation, 'Foreground')
        assert hasattr(simulation, 'Mask')
        assert hasattr(simulation, 'Noise')
    
    def test_import_spectra(self):
        """Test spectra module import."""
        from cobi import spectra
        assert hasattr(spectra, 'Spectra')
    
    def test_import_calibration(self):
        """Test calibration module import."""
        from cobi import calibration
        assert hasattr(calibration, 'Sat4Lat')
    
    def test_import_mle(self):
        """Test MLE module import."""
        from cobi import mle
        assert hasattr(mle, 'MLE')
    
    def test_import_quest(self):
        """Test quest module import."""
        from cobi import quest
        assert hasattr(quest, 'FilterEB')
        assert hasattr(quest, 'QE')
    
    def test_import_sht(self):
        """Test SHT module import."""
        from cobi import sht
        # Check it has expected attributes
        assert hasattr(sht, '__name__')


class TestVersioning:
    """Test version information."""
    
    def test_version_exists(self):
        """Test that version is defined."""
        import cobi
        assert hasattr(cobi, '__version__')
        assert isinstance(cobi.__version__, str)
    
    def test_version_format(self):
        """Test version format is valid."""
        import cobi
        version = cobi.__version__
        # Should be like "1.0" or "1.0.0"
        parts = version.split('.')
        assert len(parts) >= 2
        assert all(p.isdigit() or p in ['dev', 'rc', 'a', 'b'] 
                   for p in parts[0].split('dev')[0].split('rc')[0])
