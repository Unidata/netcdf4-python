# the Scientific Python netCDF 3 interface
# http://dirac.cnrs-orleans.fr/ScientificPython/
#from Scientific.IO.NetCDF import NetCDFFile as Dataset
# the 'classic' version of the netCDF4 python interface
# http://code.google.com/p/netcdf4-python/
from netCDF4_classic import Dataset
from numpy import arange, dtype # array module from http://numpy.scipy.org
from numpy.testing import assert_array_equal, assert_array_almost_equal
"""
This is an example which reads some 4D pressure and
temperatures. The data file read by this program is produced by
the companion program pres_temp_4D_wr.py.

This example demonstrates the netCDF Python API.
It will work either with the Scientific Python NetCDF version 3 interface
(http://dirac.cnrs-orleans.fr/ScientificPython/)
of the 'classic' version of the netCDF4 interface. 
(http://netcdf4-python.googlecode.com/svn/trunk/docs/netCDF4_classic-module.html)
To switch from one to another, just comment/uncomment the appropriate
import statements at the beginning of this file.

Jeff Whitaker <jeffrey.s.whitaker@noaa.gov> 20070202
"""
nrecs = 2; nlevs = 2; nlats = 6; nlons = 12
# open netCDF file for reading.
ncfile = Dataset('pres_temp_4D.nc','r') 
# latitudes and longitudes of grid
lats_check = -25.0 + 5.0*arange(nlats,dtype='float32')
lons_check = -125.0 + 5.0*arange(nlons,dtype='float32')
# output data.
press_check = 900. + arange(nlevs*nlats*nlons,dtype='float32') # 1d array
press_check.shape = (nlevs,nlats,nlons) # reshape to 2d array
temp_check = 9. + arange(nlevs*nlats*nlons,dtype='float32') # 1d array
temp_check.shape = (nlevs,nlats,nlons) # reshape to 2d array
# get latitude, longitude coordinate variable data.
# check to see it is what is expected.
lats = ncfile.variables['latitude']
lons = ncfile.variables['longitude']
try:
    assert_array_almost_equal(lats[:],lats_check)
except:
    raise ValueError('latitude data not what was expected')
try:
    assert_array_almost_equal(lons[:],lons_check)
except:
    raise ValueError('longitude data not what was expected')
# get pressure, temperature data a record at a time,
# checking to see that the data matches what we expect.
# close the file.
press = ncfile.variables['pressure']
temp = ncfile.variables['temperature']
for nrec in range(nrecs):
    try:
        assert_array_almost_equal(press[nrec],press_check)
    except:
        raise ValueError('pressure data not what was expected')
    try:
        assert_array_almost_equal(temp[nrec],temp_check)
    except:
        raise ValueError('temperature data not what was expected')
ncfile.close()
print '*** SUCCESS reading example file pres_temp_4D.nc'
