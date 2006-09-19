# benchmark reads and writes, compare netCDF4_classic to Scientific.IO.NetCDF
# (pynetcdf module - numpy version)
from numpy.random.mtrand import uniform
import pynetcdf
import netCDF4_classic
from timeit import Timer

# create an n1dim by n2dim by n3dim random array.
n1dim = 30   
n2dim = 15
n3dim = 73
n4dim = 144
ntrials = 10
print 'reading and writing a %s by %s by %s by %s random array ..'%(n1dim,n2dim,n3dim,n4dim)
print '(average of %s trials)' % ntrials
array = uniform(size=(n1dim,n2dim,n3dim,n4dim))

def write_pynetcdf(filename):
    file = pynetcdf.NetCDFFile(filename,'w')
    file.createDimension('n1', n1dim)
    file.createDimension('n2', n2dim)
    file.createDimension('n3', n3dim)
    file.createDimension('n4', n4dim)
    foo = file.createVariable('data', 'd',('n1','n2','n3','n4'))
    foo[:] = array
    file.close()

def write_netcdf4(filename):
    file = netCDF4_classic.Dataset(filename,'w')
    file.createDimension('n1', n1dim)
    file.createDimension('n2', n2dim)
    file.createDimension('n3', n3dim)
    file.createDimension('n4', n4dim)
    foo = file.createVariable('data', 'd',('n1','n2','n3','n4'),zlib=False)
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
t = Timer("write_pynetcdf('test_pynetcdf.nc')","from __main__ import write_pynetcdf")
print 'writing with pynetcdf took',sum(t.repeat(ntrials,1))/ntrials,'seconds'
t = Timer("write_netcdf4('test_netcdf4.nc')","from __main__ import write_netcdf4")
print 'writing with netCDF4_classic took',sum(t.repeat(ntrials,1))/ntrials,'seconds'
# test reading.
t = Timer("read_pynetcdf('test_pynetcdf.nc')","from __main__ import read_pynetcdf")
print 'reading with pynetcdf took',sum(t.repeat(ntrials,1))/ntrials,'seconds'
t = Timer("read_netcdf4('test_netcdf4.nc')","from __main__ import read_netcdf4")
print 'reading with netCDF4_classic took',sum(t.repeat(ntrials,1))/ntrials,'seconds'
