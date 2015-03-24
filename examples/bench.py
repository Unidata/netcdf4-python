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
array = uniform(size=(n1dim,n2dim,n3dim,n4dim))

def write_netcdf(filename,zlib=False,least_significant_digit=None,format='NETCDF4'):
    file = netCDF4.Dataset(filename,'w',format=format)
    file.createDimension('n1', n1dim)
    file.createDimension('n2', n2dim)
    file.createDimension('n3', n3dim)
    file.createDimension('n4', n4dim)
    foo = file.createVariable('data', 'f8',('n1','n2','n3','n4'),zlib=zlib,least_significant_digit=least_significant_digit)
    foo[:] = array
    file.close()

def read_netcdf(filename):
    file = netCDF4.Dataset(filename)
    data = file.variables['data'][:]
    file.close()

for format in ['NETCDF3_CLASSIC','NETCDF3_64BIT','NETCDF4_CLASSIC','NETCDF4']:
    sys.stdout.write('testing file format %s ...\n' % format)
    # writing, no compression.
    t = Timer("write_netcdf('test1.nc',format='%s')" % format,"from __main__ import write_netcdf")
    sys.stdout.write('writing took %s seconds\n' %\
            repr(sum(t.repeat(ntrials,1))/ntrials))
    # test reading.
    t = Timer("read_netcdf('test1.nc')","from __main__ import read_netcdf")
    sys.stdout.write('reading took %s seconds\n' %
            repr(sum(t.repeat(ntrials,1))/ntrials))
