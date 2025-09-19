# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

## Make your modules available in sys.path
sys.path.insert(
    0,
    os.path.join(
        os.path.abspath(os.path.dirname(os.path.dirname(__file__))), "src"
    ),
)

from cryovit._version import __version_tuple__

project = "CryoViT"
copyright = "%Y, Vivian Li, Sanket Gupte"  # noqa: A001
author = "Vivian Li, Sanket Gupte"
version = ".".join(map(str, __version_tuple__))
release = ".".join(map(str, __version_tuple__[:2]))

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.doctest",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

## Autodoc settings
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}

## Napoleon settings
napoleon_google_docstring = True
napoleon_include_init_with_doc = True

## Autosummary settings
autosummary_generate = True
autosummary_ignore_module_all = False
add_module_names = False


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_theme_options = {"includehidden": True}
