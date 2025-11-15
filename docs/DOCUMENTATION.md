Documentation Guide
===================

This guide explains how to build and maintain the COBI documentation.

Building Documentation Locally
-------------------------------

Install documentation dependencies:

```bash
cd docs
pip install -r requirements.txt
```

Build HTML documentation:

```bash
make html
```

View the documentation by opening `_build/html/index.html` in your browser.

Clean build artifacts:

```bash
make clean
```

Automatic Documentation Updates
--------------------------------

The documentation is automatically built and deployed on:

1. **ReadTheDocs**: Builds are triggered automatically on every push to main branch
   - Visit: https://cobi.readthedocs.io/
   - Configure webhooks in your GitHub repository settings

2. **GitHub Actions**: The `.github/workflows/docs.yml` workflow:
   - Builds documentation on every push
   - Runs link checker to find broken links
   - Deploys to GitHub Pages (main branch only)

ReadTheDocs Configuration
--------------------------

The `.readthedocs.yaml` file configures the build:

- Python version: 3.10
- Sphinx configuration: `docs/conf.py`
- Requirements: `docs/requirements.txt`
- Package installation: automatic via pip

Adding New Modules
------------------

To document a new module:

1. Add module docstring at the top of the file
2. Add docstrings to all classes and functions
3. Create a new RST file in `docs/api/` directory
4. Add the new file to `docs/api.rst` toctree

Docstring Format
----------------

Use NumPy/Google style docstrings:

```python
def function_name(param1: type, param2: type) -> return_type:
    """
    Short description.
    
    Longer description with more details about the function's behavior.
    
    Parameters
    ----------
    param1 : type
        Description of param1
    param2 : type
        Description of param2
    
    Returns
    -------
    return_type
        Description of return value
    
    Examples
    --------
    >>> function_name(value1, value2)
    expected_output
    """
```

Testing Documentation
---------------------

Before committing:

1. Build locally and check for warnings:
   ```bash
   make html
   ```

2. Check for broken links:
   ```bash
   make linkcheck
   ```

3. Review the generated HTML in your browser

4. Check that all modules appear in the API documentation

Troubleshooting
---------------

**Import errors during build**:
- Add the module to `autodoc_mock_imports` in `docs/conf.py`
- Ensure the package is installed: `pip install -e .`

**Missing modules in documentation**:
- Check that the module is imported in `__init__.py`
- Verify the RST file is added to the toctree
- Ensure docstrings are present

**ReadTheDocs build failures**:
- Check the build log at readthedocs.io
- Verify `docs/requirements.txt` includes all dependencies
- Ensure `.readthedocs.yaml` configuration is valid
