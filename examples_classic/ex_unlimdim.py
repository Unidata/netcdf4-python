import numpy as NP
# test creating variables with unlimited dimensions,
# writing to and retrieving data from such variables.
from numpy.random.mtrand import uniform
import netCDF4_classic as netCDF4
import sys, time, os

# create an n1dim by n2dim by n3dim random array.
n1dim = 10
n2dim = 50
n3dim = 10
array = 100.*uniform(size=(n1dim,n2dim,n3dim))
print array.shape
print array[5,15:20,3:8]

# test writing data to a variable with a single unlimited dimension.
filename = 'tst_unlimdim.nc'
print 'testing a single unlimited dimension ...'
file = netCDF4.Dataset(filename,'w')
file.createDimension('n1', n1dim)
file.createDimension('n2', None)
file.createDimension('n3', n3dim)
foo = file.createVariable('data', array.dtype.str[1:], ('n1','n2','n3'))
for key,dim in file.dimensions.iteritems():
    print key, len(dim), dim.isunlimited()
# either one of these will work.
#foo[:,0:n2dim,:] = array
for m in range(n2dim):
    foo[:,m,:] = array[:,m,:]
print foo.shape
file.close()

file = netCDF4.Dataset(filename)
foo = file.variables['data'][:]
print foo.shape
for key,dim in file.dimensions.iteritems():
    print key, len(dim), dim.isunlimited()
data = foo[:]
file.close()
print data.shape
print data[5,15:20,3:8]
