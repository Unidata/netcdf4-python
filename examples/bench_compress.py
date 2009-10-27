# benchmark reads and writes, with and without compression.
# tests all four supported file formats.
from numpy.random.mtrand import uniform
import netCDF4
from timeit import Timer
import os

# create an n1dim by n2dim by n3dim random array.
n1dim = 30   
n2dim = 15
n3dim = 73
n4dim = 144
ntrials = 10
print 'reading and writing a %s by %s by %s by %s random array ..'%(n1dim,n2dim,n3dim,n4dim)
print '(average of %s trials)' % ntrials
array = netCDF4._quantize(uniform(size=(n1dim,n2dim,n3dim,n4dim)),4)


def write_netcdf(filename,zlib=False,shuffle=False):
    file = netCDF4.Dataset(filename,'w',format='NETCDF4')
    file.createDimension('n1', n1dim)
    file.createDimension('n2', n2dim)
    file.createDimension('n3', n3dim)
    file.createDimension('n4', n4dim)
    foo = file.createVariable('data',\
                              'f8',('n1','n2','n3','n4'),zlib=zlib,shuffle=shuffle)
    foo[:] = array
    file.close()

def read_netcdf(filename):
    file = netCDF4.Dataset(filename)
    data = file.variables['data'][:]
    file.close()

for compress_kwargs in ["zlib=False,shuffle=False","zlib=True,shuffle=False",
                        "zlib=True,shuffle=True"]:
    print 'testing compression ...',compress_kwargs
    # writing.
    t = Timer("write_netcdf('test.nc',%s)" % compress_kwargs,"from __main__ import write_netcdf")
    print 'writing took',sum(t.repeat(ntrials,1))/ntrials,'seconds'
    # test reading.
    t = Timer("read_netcdf('test.nc')","from __main__ import read_netcdf")
    print 'reading took',sum(t.repeat(ntrials,1))/ntrials,'seconds'
    # print out size of resulting files.
    print 'size of test.nc = ',os.stat('test.nc').st_size
