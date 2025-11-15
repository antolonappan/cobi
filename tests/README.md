# COBI Test Suite

This directory contains the test suite for the COBI package.

## Test Structure

- `conftest.py` - Pytest fixtures and configuration
- `test_imports.py` - Test that all modules can be imported
- `test_utils.py` - Test utility functions and basic operations
- `test_simulation.py` - Test simulation modules and map operations

## Running Tests

### Run all tests
```bash
pytest tests/
```

### Run with coverage
```bash
pytest tests/ --cov=cobi --cov-report=html
```

### Run specific test file
```bash
pytest tests/test_utils.py -v
```

### Run specific test class
```bash
pytest tests/test_utils.py::TestAngularConversion -v
```

### Run fast tests only (skip slow tests)
```bash
pytest tests/ -m "not slow"
```

## Test Categories

### Unit Tests
- Angular conversions
- Coordinate transformations
- Map operations
- Mask operations
- Power spectrum utilities
- Rotation matrices

### Integration Tests
- CMB simulation
- Foreground generation
- Noise simulation
- Power spectra computation

## Continuous Integration

Tests are automatically run on every push and pull request via GitHub Actions.
See `.github/workflows/tests.yml` for CI configuration.

## Requirements

Test dependencies are installed via:
```bash
pip install pytest pytest-cov pytest-xdist
```

Additional scientific dependencies:
- numpy
- scipy
- healpy
- matplotlib
- astropy
