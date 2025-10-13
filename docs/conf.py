"""
Configuration file for the Sphinx documentation builder.

This file only contains a selection of the most common options. For a full
list see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

import os
import sys
from pathlib import Path

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------

project = 'PatchMoE'
copyright = '2025, PatchMoE Team'
author = 'PatchMoE Team'

# The full version, including alpha/beta/rc tags
release = '0.1.0'
version = '0.1.0'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx.ext.napoleon',
    'myst_parser',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The suffix(es) of source filenames.
source_suffix = {
    '.rst': None,
    '.md': 'myst_parser',
}

# The master toctree document.
master_doc = 'index'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
html_sidebars = {
    '**': [
        'relations.html',  # needs 'show_related': True theme option to display
        'searchbox.html',
    ]
}

# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'PatchMoEdoc'

# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    'papersize': 'letterpaper',
    
    # The font size ('10pt', '11pt' or '12pt').
    'pointsize': '10pt',
    
    # Additional stuff for the LaTeX preamble.
    'preamble': '',
    
    # Latex figure (float) alignment
    'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'PatchMoE.tex', 'PatchMoE Documentation',
     'PatchMoE Team', 'manual'),
]

# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'patchmoe', 'PatchMoE Documentation',
     [author], 1)
]

# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'PatchMoE', 'PatchMoE Documentation',
     author, 'PatchMoE', 'A large-scale time series foundation model.',
     'Miscellaneous'),
]

# -- Options for Epub output -------------------------------------------------

# Bibliographic Dublin Core info.
epub_title = project

# The unique identifier of the text. This can be a ISBN number
# or the project homepage.
epub_identifier = 'https://github.com/patch-moe/patchmoe'

# A unique identification for the text.
epub_uid = 'PatchMoE'

# A list of files that should not be packed into the epub file.
epub_exclude_files = ['search.html']

# -- Extension configuration -------------------------------------------------

# -- Options for intersphinx extension ---------------------------------------

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'transformers': ('https://huggingface.co/docs/transformers/', None),
}

# -- Options for todo extension ----------------------------------------------

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True

# -- Options for autodoc extension -------------------------------------------

# This value selects what content will be inserted into the main body of an autoclass directive.
autoclass_content = 'both'

# This value is a list of autodoc directive flags that should be automatically applied to all autodoc directives.
autodoc_default_flags = ['members', 'undoc-members', 'show-inheritance']

# This value controls how to represent typehints.
autodoc_typehints = 'description'

# -- Options for autosummary extension ---------------------------------------

# Boolean indicating whether to scan all found documents for autosummary directives,
# and to generate stub pages for each.
autosummary_generate = True

# -- Options for Napoleon extension ------------------------------------------

# True to parse NumPy style docstrings. False to disable NumPy style docstrings.
napoleon_numpy_docstring = True

# True to parse Google style docstrings. False to disable Google style docstrings.
napoleon_google_docstring = True

# True to include private members (like _membername) with docstrings in the documentation.
napoleon_include_private_with_doc = False

# True to include special members (like __membername__) with docstrings in the documentation.
napoleon_include_special_with_doc = True

# -- Options for MyST parser -------------------------------------------------

# Enable MyST extensions
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]

# -- Custom configuration ----------------------------------------------------

# Add custom CSS
def setup(app):
    """Add custom CSS and JavaScript files."""
    app.add_css_file('custom.css')

# HTML theme options
html_theme_options = {
    'canonical_url': '',
    'analytics_id': '',
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'vcs_pageview_mode': '',
    'style_nav_header_background': '#2980B9',
    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}

# Add GitHub link
html_context = {
    "display_github": True,
    "github_user": "patch-moe",
    "github_repo": "patchmoe",
    "github_version": "main",
    "conf_py_path": "/docs/",
}
