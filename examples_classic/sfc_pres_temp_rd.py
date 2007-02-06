# the Scientific Python netCDF 3 interface
# http://dirac.cnrs-orleans.fr/ScientificPython/
#from Scientific.IO.NetCDF import NetCDFFile as Dataset
# the 'classic' version of the netCDF4 python interface
# http://code.google.com/p/netcdf4-python/
from netCDF4_classic import Dataset
from numpy import arange, dtype # array module from http://numpy.scipy.org
from numpy.testing import assert_array_equal, assert_array_almost_equal
"""
This is an example which reads some surface pressure and
temperatures. The data file read by this program is produced
companion program sfc_pres_temp_wr.py.

This example demonstrates the netCDF Python API.
It will work either with the Scientific Python NetCDF version 3 interface
(http://dirac.cnrs-orleans.fr/ScientificPython/)
of the 'classic' version of the netCDF4 interface. 
(http://netcdf4-python.googlecode.com/svn/trunk/docs/netCDF4_classic-module.html)
To switch from one to another, just comment/uncomment the appropriate
import statements at the beginning of this file.

Jeff Whitaker <jeffrey.s.whitaker@noaa.gov> 20070202
"""
nlats = 6; nlons = 12
# open netCDF file for reading
ncfile = Dataset('sfc_pres_temp.nc','r') 
# expected latitudes and longitudes of grid
lats_check = -25.0 + 5.0*arange(nlats,dtype='float32')
lons_check = -125.0 + 5.0*arange(nlons,dtype='float32')
# expected data.
press_check = 900. + arange(nlats*nlons,dtype='float32') # 1d array
press_check.shape = (nlats,nlons) # reshape to 2d array
temp_check = 9. + 0.25*arange(nlats*nlons,dtype='float32') # 1d array
temp_check.shape = (nlats,nlons) # reshape to 2d array
# get pressure and temperature variables.
temp = ncfile.variables['temperature']
press = ncfile.variables['pressure']
# check units attributes.
try:
    assert(temp.units == 'celsius')
except:
    raise AttributeError('temperature units attribute not what was expected')
try:
    assert(press.units == 'hPa')
except:
    raise AttributeError('pressure units attribute not what was expected')
# check data
try:
    assert_array_almost_equal(press[:],press_check)
except:
    raise ValueError('pressure data not what was expected')
try:
    assert_array_almost_equal(temp[:],temp_check)
except:
    raise ValueError('temperature data not what was expected')
# get coordinate variables.
lats = ncfile.variables['latitude']
lons = ncfile.variables['longitude']
# check units attributes.
try:
    assert(lats.units == 'degrees_north')
except:
    raise AttributeError('latitude units attribute not what was expected')
try:
    assert(lons.units == 'degrees_east')
except:
    raise AttributeError('longitude units attribute not what was expected')
# check data
try:
    assert_array_almost_equal(lats[:],lats_check)
except:
    raise ValueError('latitude data not what was expected')
try:
    assert_array_almost_equal(lons[:],lons_check)
except:
    raise ValueError('longitude data not what was expected')
# close the file.
ncfile.close()
print '*** SUCCESS reading example file sfc_pres_temp.nc!'
