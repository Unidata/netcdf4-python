# benchmark reads and writes, compare netCDF4_classic to Scientific.IO.NetCDF
# (pynetcdf module - numpy version)
from numpy.random.mtrand import uniform
import pynetcdf
import netCDF4_classic
import sys, time, os

# create an n1dim by n2dim by n3dim random array.
n1dim = 30   
n2dim = 15
n3dim = 73
n4dim = 144
print 'reading and writing a %s by %s by %s by %s random array ..'%(n1dim,n2dim,n3dim,n4dim)
array = uniform(size=(n1dim,n2dim,n3dim,n4dim))

def write_pynetcdf(array,filename,n1dim,n2dim,n3dim,n4dim,zlib=False,complevel=0,shuffle=0,least_significant_digit=None):
    file = pynetcdf.NetCDFFile(filename,'w')
    file.createDimension('n1', n1dim)
    file.createDimension('n2', n2dim)
    file.createDimension('n3', n3dim)
    file.createDimension('n4', n4dim)
    foo = file.createVariable('data', 'd',('n1','n2','n3','n4'))
    foo[:] = array
    file.close()

def write_netcdf4(array,filename,n1dim,n2dim,n3dim,n4dim,zlib=False,complevel=0,shuffle=0,least_significant_digit=None):
    file = netCDF4_classic.Dataset(filename,'w')
    file.createDimension('n1', n1dim)
    file.createDimension('n2', n2dim)
    file.createDimension('n3', n3dim)
    file.createDimension('n4', n4dim)
    foo = file.createVariable('data', 'f8',('n1','n2','n3','n4'),zlib=False)
    foo[:] = array
    file.close()

def read_pynetcdf(filename):
    file = pynetcdf.NetCDFFile(filename)
    data = file.variables['data'][:]
    file.close()

def read_netcdf4(filename):
    file = netCDF4_classic.Dataset(filename)
    data = file.variables['data'][:]
    file.close()

# test writing.
t1 = time.time()
write_pynetcdf(array,'test_pynetcdf.nc',n1dim,n2dim,n3dim,n4dim)
print 'writing with pynetcdf took',time.time()-t1,'seconds'
t1 = time.time()
write_netcdf4(array,'test_netcdf4.nc',n1dim,n2dim,n3dim,n4dim)
print 'writing with netCDF4_classic took',time.time()-t1,'seconds'
# test reading.
t1 = time.time()
read_pynetcdf('test_pynetcdf.nc')
print 'reading with pynetcdf took',time.time()-t1,'seconds to read'
t1 = time.time()
read_netcdf4('test_netcdf4.nc')
print 'reading with netCDF4_classic file took',time.time()-t1,'seconds to read'
