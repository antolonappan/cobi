import sys
import os

# Add package to path for autodoc
sys.path.insert(0, os.path.abspath('..'))

# Project information
project = "COBI: Cosmic Birefringence Analysis Pipeline"
copyright = "2024-2025, Anto Idicherian Lonappan"
author = "Anto Idicherian Lonappan"
release = "1.0"

# Sphinx extensions
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "nbsphinx",
    "sphinx.ext.githubpages",
]

# Autodoc configuration
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}
autodoc_typehints = "description"
autodoc_mock_imports = [
    "mpi4py",
    "curvedsky", 
    "ducc0",
    "emcee",
    "getdist",
    "healpy",
    "pymaster",
    "pysm3",
    "camb",
    "scipy",
    "matplotlib",
    "astropy",
    "tqdm",
    "pandas",
    "h5py",
    "requests",
    "lenspyx",
    "pixell",
    "so_models_v3",
    "numpy",
]

# Autosummary configuration
autosummary_generate = True
autosummary_imported_members = False

# Napoleon settings for NumPy and Google style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True
napoleon_type_aliases = None

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "healpy": ("https://healpy.readthedocs.io/en/latest/", None),
    "pymaster": ("https://namaster.readthedocs.io/en/latest/", None),
}
intersphinx_disabled_domains = ["std"]

# Templates and static files
templates_path = ["_templates"]
html_static_path = ["_static"]

# Exclude patterns
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# HTML output options
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'includehidden': True,
    'titles_only': False
}

# HTML context
html_context = {
    "display_github": True,
    "github_user": "antolonappan",
    "github_repo": "cobi",
    "github_version": "main",
    "conf_py_path": "/docs/",
}

# Additional options
epub_show_urls = "footnote"
add_module_names = False