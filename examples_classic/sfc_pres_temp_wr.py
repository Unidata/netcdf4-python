# the Scientific Python netCDF 3 interface
# http://dirac.cnrs-orleans.fr/ScientificPython/
#from Scientific.IO.NetCDF import NetCDFFile as Dataset
# the 'classic' version of the netCDF4 python interface
# http://code.google.com/p/netcdf4-python/
from netCDF4_classic import Dataset
from numpy import arange, dtype # array module from http://numpy.scipy.org
"""
This example writes some surface pressure and temperatures
The companion program sfc_pres_temp_rd.py shows how to read the netCDF
data file created by this program.

This example demonstrates the netCDF Python API.
It will work either with the Scientific Python NetCDF version 3 interface
(http://dirac.cnrs-orleans.fr/ScientificPython/)
of the 'classic' version of the netCDF4 interface. 
(http://netcdf4-python.googlecode.com/svn/trunk/docs/netCDF4_classic-module.html)
To switch from one to another, just comment/uncomment the appropriate
import statements at the beginning of this file.

Jeff Whitaker <jeffrey.s.whitaker@noaa.gov> 20070202
"""
# the output array to write will be nlats x nlons
nlats = 6; nlons = 12
# open a new netCDF file for writing.
ncfile = Dataset('sfc_pres_temp.nc','w') 
# latitudes and longitudes of grid
lats_out = -25.0 + 5.0*arange(nlats,dtype='float32')
lons_out = -125.0 + 5.0*arange(nlons,dtype='float32')
# output data.
press_out = 900. + arange(nlats*nlons,dtype='float32') # 1d array
press_out.shape = (nlats,nlons) # reshape to 2d array
temp_out = 9. + 0.25*arange(nlats*nlons,dtype='float32') # 1d array
temp_out.shape = (nlats,nlons) # reshape to 2d array
# create the lat and lon dimensions.
ncfile.createDimension('latitude',nlats)
ncfile.createDimension('longitude',nlons)
# Define the coordinate variables. They will hold the coordinate
# information, that is, the latitudes and longitudes.
lats = ncfile.createVariable('latitude',dtype('float32').char,('latitude',))
lons = ncfile.createVariable('longitude',dtype('float32').char,('longitude',))
# Assign units attributes to coordinate var data. This attaches a
# text attribute to each of the coordinate variables, containing the
# units.
lats.units = 'degrees_north'
lons.units = 'degrees_east'
# write data to coordinate vars.
lats[:] = lats_out
lons[:] = lons_out
# create the pressure and temperature variables 
press = ncfile.createVariable('pressure',dtype('float32').char,('latitude','longitude'))
temp = ncfile.createVariable('temperature',dtype('float32').char,('latitude','longitude'))
# set the units attribute.
press.units =  'hPa'
temp.units = 'celsius'
# write data to variables.
press[:] = press_out
temp[:] = temp_out
# close the file.
ncfile.close()
print '*** SUCCESS writing example file sfc_press_temp.nc!'
