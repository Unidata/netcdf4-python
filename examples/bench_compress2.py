# benchmark reads and writes, with and without compression.
# tests all four supported file formats.
from numpy.random.mtrand import uniform
import netCDF4
from timeit import Timer
import os, sys

# create an n1dim by n2dim by n3dim random array.
n1dim = 30
n2dim = 15
n3dim = 73
n4dim = 144
ntrials = 10
sys.stdout.write('reading and writing a %s by %s by %s by %s random array ..\n'%(n1dim,n2dim,n3dim,n4dim))
sys.stdout.write('(average of %s trials)\n' % ntrials)
lsd = 4
if lsd is not None:
    array = netCDF4._quantize(uniform(size=(n1dim,n2dim,n3dim,n4dim)),4)
else:
    array = uniform(size=(n1dim,n2dim,n3dim,n4dim))


def write_netcdf(filename,complevel):
    file = netCDF4.Dataset(filename,'w',format='NETCDF4')
    file.createDimension('n1', n1dim)
    file.createDimension('n2', n2dim)
    file.createDimension('n3', n3dim)
    file.createDimension('n4', n4dim)
    foo = file.createVariable('data',\
                              'f8',('n1','n2','n3','n4'),zlib=True,shuffle=True,complevel=complevel)
    foo[:] = array
    file.close()

def read_netcdf(filename):
    file = netCDF4.Dataset(filename)
    data = file.variables['data'][:]
    file.close()

for complevel in range(0,10):
    sys.stdout.write('testing compression with complevel %s...\n' % complevel)
    # writing.
    t = Timer("write_netcdf('test.nc',%s)" % complevel,"from __main__ import write_netcdf")
    sys.stdout.write('writing took %s seconds\n' %\
            repr(sum(t.repeat(ntrials,1))/ntrials))
    # test reading.
    t = Timer("read_netcdf('test.nc')","from __main__ import read_netcdf")
    sys.stdout.write('reading took %s seconds\n' %
            repr(sum(t.repeat(ntrials,1))/ntrials))
    # print out size of resulting files.
    sys.stdout.write('size of test.nc = %s\n'%repr(os.stat('test.nc').st_size))
