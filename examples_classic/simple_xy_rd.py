# the Scientific Python netCDF 3 interface
# http://dirac.cnrs-orleans.fr/ScientificPython/
#from Scientific.IO.NetCDF import NetCDFFile as Dataset
# the 'classic' version of the netCDF4 python interface
# http://code.google.com/p/netcdf4-python/
from netCDF4_classic import Dataset
from numpy import arange # array module from http://numpy.scipy.org
from numpy.testing import assert_array_equal, assert_array_almost_equal
"""
This is a simple example which reads a small dummy array, from a
netCDF data file created by the companion program simple_xy_wr.py.

This example demonstrates the netCDF Python API.
It will work either with the Scientific Python NetCDF version 3 interface
(http://dirac.cnrs-orleans.fr/ScientificPython/)
of the 'classic' version of the netCDF4 interface. 
(http://netcdf4-python.googlecode.com/svn/trunk/docs/netCDF4_classic-module.html)
To switch from one to another, just comment/uncomment the appropriate
import statements at the beginning of this file.

Jeff Whitaker <jeffrey.s.whitaker@noaa.gov> 20070201
"""
# open a the netCDF file for reading.
ncfile = Dataset('simple_xy.nc','r') 
# read the data in variable named 'data'.
data = ncfile.variables['data'][:]
nx,ny = data.shape
# check the data.
data_check = arange(nx*ny) # 1d array
data_check.shape = (nx,ny) # reshape to 2d array
try:
    assert_array_equal(data, data_check)
    print '*** SUCCESS reading example file simple_xy.nc'
except:
    print '*** FAILURE reading example file simple_xy.nc'
# close the file.
ncfile.close()
