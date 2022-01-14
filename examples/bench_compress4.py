from __future__ import print_function
# benchmark reads and writes, with and without compression.
# tests all four supported file formats.
from numpy.random.mtrand import uniform
import netCDF4
from timeit import Timer
import os, sys

# use real data.
URL="http://www.esrl.noaa.gov/psd/thredds/dodsC/Datasets/ncep.reanalysis/pressure/hgt.1990.nc"
nc = netCDF4.Dataset(URL)

# use real 500 hPa geopotential height data.
n1dim = 100
n3dim = 73
n4dim = 144
ntrials = 10
sys.stdout.write('reading and writing a %s by %s by %s random array ..\n'%(n1dim,n3dim,n4dim))
sys.stdout.write('(average of %s trials)\n\n' % ntrials)
print(nc)
print(nc.variables['hgt'])
array = nc.variables['hgt'][0:n1dim,5,:,:]
print(array.min(), array.max(), array.shape, array.dtype)


def write_netcdf(filename,nsd):
    file = netCDF4.Dataset(filename,'w',format='NETCDF4')
    file.createDimension('n1', None)
    file.createDimension('n3', n3dim)
    file.createDimension('n4', n4dim)
    foo = file.createVariable('data',\
                              'f4',('n1','n3','n4'),\
                              zlib=True,shuffle=True,\
                              significant_digits=nsd)
    foo[:] = array
    file.close()

def read_netcdf(filename):
    file = netCDF4.Dataset(filename)
    data = file.variables['data'][:]
    file.close()

for sigdigits in range(1,6,1):
    sys.stdout.write('testing compression with significant_digits=%s...\n' %\
            sigdigits)
    write_netcdf('test.nc',sigdigits)
    read_netcdf('test.nc')
    # print out size of resulting files with standard quantization.
    sys.stdout.write('size of test.nc = %s\n'%repr(os.stat('test.nc').st_size))
    sigdigits_neg = -sigdigits
    sys.stdout.write('testing compression with significant_digits=%s...\n' %\
            sigdigits_neg)
    write_netcdf('test.nc',sigdigits_neg)
    read_netcdf('test.nc')
    # print out size of resulting files with alternate quantization.
    sys.stdout.write('size of test.nc = %s\n'%repr(os.stat('test.nc').st_size))