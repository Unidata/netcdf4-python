# Uses pdoc (https://github.com/mitmproxy/pdoc)
# to create html docs from docstrings in Cython source.
/Users/jwhitaker/.local/bin/pdoc -o 'docs' netCDF4 
# use resulting docs/netCDF4/_netCDF4.html
cp docs/netCDF4/_netCDF4.html docs/index.html

