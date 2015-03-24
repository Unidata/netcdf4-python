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

def write_netcdf(filename,zlib=False,least_significant_digit=None,format='NETCDF4',closeit=False):
    file = netCDF4.Dataset(filename,'w',format=format,diskless=True,persist=True)
    file.createDimension('n1', n1dim)
    file.createDimension('n2', n2dim)
    file.createDimension('n3', n3dim)
    file.createDimension('n4', n4dim)
    foo = file.createVariable('data',\
          'f8',('n1','n2','n3','n4'),zlib=zlib,least_significant_digit=None)
    foo.testme="hi I am an attribute"
    foo.testme1="hi I am an attribute"
    foo.testme2="hi I am an attribute"
    foo.testme3="hi I am an attribute"
    foo.testme4="hi I am an attribute"
    foo.testme5="hi I am an attribute"
    foo[:] = array
    if closeit: file.close()
    return file

def read_netcdf(ncfile):
    data = ncfile.variables['data'][:]

for format in ['NETCDF4','NETCDF3_CLASSIC','NETCDF3_64BIT']:
    sys.stdout.write('testing file format %s ...\n' % format)
    # writing, no compression.
    t = Timer("write_netcdf('test1.nc',closeit=True,format='%s')" % format,"from __main__ import write_netcdf")
    sys.stdout.write('writing took %s seconds\n' %\
            repr(sum(t.repeat(ntrials,1))/ntrials))
    # test reading.
    ncfile = write_netcdf('test1.nc',format=format)
    t = Timer("read_netcdf(ncfile)","from __main__ import read_netcdf,ncfile")
    sys.stdout.write('reading took %s seconds\n' %
            repr(sum(t.repeat(ntrials,1))/ntrials))

# test diskless=True in nc_open
format='NETCDF3_CLASSIC'
trials=50
sys.stdout.write('test caching of file in memory on open for %s\n' % format)
sys.stdout.write('testing file format %s ...\n' % format)
write_netcdf('test1.nc',format=format,closeit=True)
ncfile = netCDF4.Dataset('test1.nc',diskless=False)
t = Timer("read_netcdf(ncfile)","from __main__ import read_netcdf,ncfile")
sys.stdout.write('reading (from disk) took %s seconds\n' %
            repr(sum(t.repeat(ntrials,1))/ntrials))
ncfile.close()
ncfile = netCDF4.Dataset('test1.nc',diskless=True)
# setting diskless=True should cache the file in memory,
# resulting in faster reads.
t = Timer("read_netcdf(ncfile)","from __main__ import read_netcdf,ncfile")
sys.stdout.write('reading (cached in memory) took %s seconds\n' %
            repr(sum(t.repeat(ntrials,1))/ntrials))
ncfile.close()
