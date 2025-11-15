# COBI Test Suite

This directory contains the test suite for the COBI package.

## Current Status

⚠️ **Currently only basic import tests are enabled.** Computational tests have been disabled until the full dependency stack is installed in CI.

## Test Structure

- `conftest.py` - Pytest configuration (fixtures removed temporarily)
- `test_imports.py` - Test that core modules can be imported (ACTIVE)
- `test_utils.py.skip` - Utility function tests (DISABLED - for future development)
- `test_simulation.py.skip` - Simulation module tests (DISABLED - for future development)

## Running Tests

### Run active tests
```bash
pytest tests/
```

## Re-enabling Computational Tests

To restore full test suite when dependencies are available:

1. Install all dependencies:
   ```bash
   pip install -e .[all]
   ```

2. Rename disabled tests:
   ```bash
   mv tests/test_simulation.py.skip tests/test_simulation.py
   mv tests/test_utils.py.skip tests/test_utils.py
   ```

3. Restore fixtures in `conftest.py`

4. Update `.github/workflows/tests.yml` to install full dependencies

## Continuous Integration

Tests are automatically run on every push via GitHub Actions.
See `.github/workflows/tests.yml` for CI configuration.
