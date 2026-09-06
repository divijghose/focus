import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'focus'
author = 'Divij Ghose'
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

autodoc_default_options = {
    'private-members': True,
}

templates_path = ['_templates']
exclude_patterns = ['_build']
html_theme = 'sphinx_rtd_theme'
