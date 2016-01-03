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


def write_netcdf(filename,complevel,lsd):
    file = netCDF4.Dataset(filename,'w',format='NETCDF4')
    file.createDimension('n1', None)
    file.createDimension('n3', n3dim)
    file.createDimension('n4', n4dim)
    foo = file.createVariable('data',\
                              'f4',('n1','n3','n4'),\
                              zlib=True,shuffle=True,complevel=complevel,\
                              least_significant_digit=lsd)
    foo[:] = array
    file.close()

def read_netcdf(filename):
    file = netCDF4.Dataset(filename)
    data = file.variables['data'][:]
    file.close()

lsd = None
sys.stdout.write('using least_significant_digit %s\n\n' % lsd)
for complevel in range(0,10,2):
    sys.stdout.write('testing compression with complevel %s...\n' % complevel)
    # writing.
    t = Timer("write_netcdf('test.nc',%s,%s)" % (complevel,lsd),"from __main__ import write_netcdf")
    sys.stdout.write('writing took %s seconds\n' %\
            repr(sum(t.repeat(ntrials,1))/ntrials))
    # test reading.
    t = Timer("read_netcdf('test.nc')","from __main__ import read_netcdf")
    sys.stdout.write('reading took %s seconds\n' %
            repr(sum(t.repeat(ntrials,1))/ntrials))
    # print out size of resulting files.
    sys.stdout.write('size of test.nc = %s\n'%repr(os.stat('test.nc').st_size))

complevel = 4
complevel = 4
sys.stdout.write('\nusing complevel %s\n\n' % complevel)
for lsd in range(0,6):
    sys.stdout.write('testing compression with least_significant_digit %s..\n'\
            % lsd)
    # writing.
    t = Timer("write_netcdf('test.nc',%s,%s)" % (complevel,lsd),"from __main__ import write_netcdf")
    sys.stdout.write('writing took %s seconds\n' %\
            repr(sum(t.repeat(ntrials,1))/ntrials))
    # test reading.
    t = Timer("read_netcdf('test.nc')","from __main__ import read_netcdf")
    sys.stdout.write('reading took %s seconds\n' %
            repr(sum(t.repeat(ntrials,1))/ntrials))
    # print out size of resulting files.
    sys.stdout.write('size of test.nc = %s\n'%repr(os.stat('test.nc').st_size))
