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
sys.stdout.write('(average of %s trials)\n\n' % ntrials)
array = uniform(size=(n1dim,n2dim,n3dim,n4dim))


def write_netcdf(filename,complevel,lsd):
    file = netCDF4.Dataset(filename,'w',format='NETCDF4')
    file.createDimension('n1', n1dim)
    file.createDimension('n2', n2dim)
    file.createDimension('n3', n3dim)
    file.createDimension('n4', n4dim)
    foo = file.createVariable('data',\
                              'f8',('n1','n2','n3','n4'),\
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
sys.stdout.write('\nusing complevel %s\n\n' % complevel)
for lsd in range(1,6):
    sys.stdout.write('testing compression with least_significant_digit %s...\n' % lsd)
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
