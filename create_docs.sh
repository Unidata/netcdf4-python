# Uses pdoc (https://github.com/BurntSushi/pdoc)
# to create html docs from docstrings in Cython source.
# Use hacked version at https://github.com/jswhit/pdoc
# which extracts cython method docstrings and function signatures.
pdoc --html --html-no-source --overwrite --html-dir 'docs' netCDF4
