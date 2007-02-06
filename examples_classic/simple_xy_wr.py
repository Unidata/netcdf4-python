# the Scientific Python netCDF 3 interface
# http://dirac.cnrs-orleans.fr/ScientificPython/
#from Scientific.IO.NetCDF import NetCDFFile as Dataset
# the 'classic' version of the netCDF4 python interface
# http://code.google.com/p/netcdf4-python/
from netCDF4_classic import Dataset
from numpy import arange, dtype # array module from http://numpy.scipy.org
"""
This is a very simple example which writes a 2D array of
sample data. To handle this in netCDF we create two shared
dimensions, "x" and "y", and a netCDF variable, called "data".

This example demonstrates the netCDF Python API.
It will work either with the Scientific Python NetCDF version 3 interface
(http://dirac.cnrs-orleans.fr/ScientificPython/)
of the 'classic' version of the netCDF4 interface. 
(http://netcdf4-python.googlecode.com/svn/trunk/docs/netCDF4_classic-module.html)
To switch from one to another, just comment/uncomment the appropriate
import statements at the beginning of this file.

Jeff Whitaker <jeffrey.s.whitaker@noaa.gov> 20070201
"""
# the output array to write will be nx x ny
nx = 6; ny = 12
# open a new netCDF file for writing.
ncfile = Dataset('simple_xy.nc','w') 
# create the output data.
data_out = arange(nx*ny) # 1d array
data_out.shape = (nx,ny) # reshape to 2d array
# create the x and y dimensions.
ncfile.createDimension('x',nx)
ncfile.createDimension('y',ny)
# create the variable (4 byte integer in this case)
# first argument is name of variable, second is datatype, third is
# a tuple with the names of dimensions.
data = ncfile.createVariable('data',dtype('int32').char,('x','y'))
# write data to variable.
data[:] = data_out
# close the file.
ncfile.close()
print '*** SUCCESS writing example file simple_xy.nc!'
