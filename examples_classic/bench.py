# benchmark reads and writes, with and without compression.
from numpy.random.mtrand import uniform
import netCDF4_classic as netCDF4
import sys, time, os

# create an n1dim by n2dim by n3dim random array.
n1dim = 30   
n2dim = 15
n3dim = 73
n4dim = 144
print 'reading and writing a %s by %s by %s by %s random array ..'%(n1dim,n2dim,n3dim,n4dim)
array = uniform(size=(n1dim,n2dim,n3dim,n4dim))
print array.min(), array.max()

def write_netcdf(array,filename,n1dim,n2dim,n3dim,n4dim,zlib=False,complevel=0,shuffle=0,least_significant_digit=None):
    file = netCDF4.Dataset(filename,'w')
    file.createDimension('n1', n1dim)
    file.createDimension('n2', n2dim)
    file.createDimension('n3', n3dim)
    file.createDimension('n4', n4dim)
    foo = file.createVariable('data', 'f8',('n1','n2','n3','n4'),zlib=zlib,complevel=complevel,shuffle=shuffle,least_significant_digit=least_significant_digit)
    foo[:] = array
    file.close()

def write_netcdf_unlim(array,filename,n1dim,n2dim,n3dim,n4dim,zlib=False,complevel=0,shuffle=0,least_significant_digit=None):
    file = netCDF4.Dataset(filename,'w')
    #file.set_fill_off()
    file.createDimension('n1', None)
    file.createDimension('n2', n2dim)
    file.createDimension('n3', n3dim)
    file.createDimension('n4', n4dim)
    foo = file.createVariable('data', 'f8',('n1','n2','n3','n4'),zlib=zlib,complevel=complevel,shuffle=shuffle,least_significant_digit=least_significant_digit)
    # any of these should work.
    foo[0:n1dim] = array[0:n1dim]
    file.close()

def read_netcdf(filename):
    file = netCDF4.Dataset(filename)
    data = file.variables['data'][:]
    print data[10,10,10,10]
    print data.min(), data.max()
    file.close()

complevel = 6
shuffle = 1
least_significant_digit = 4

# create a file, put a random array in it.
# no compression is used.
t1 = time.time()
write_netcdf_unlim(array,'test1.nc',n1dim,n2dim,n3dim,n4dim)
print 'writing with no compression took',time.time()-t1,'seconds'
# create a file, put a random array in it.
# use compression.
t1 = time.time()
write_netcdf_unlim(array,'test2.nc',n1dim,n2dim,n3dim,n4dim,zlib=True,complevel=complevel,shuffle=shuffle,least_significant_digit=least_significant_digit)
print 'writing with compression took',time.time()-t1,'seconds'
print 'complevel = ',complevel
print 'shuffle = ',shuffle
print 'least_signficant_digit = ',least_significant_digit
# test reading.
t1 = time.time()
read_netcdf('test1.nc')
print 'reading uncompressed file took',time.time()-t1,'seconds to read'
t1 = time.time()
read_netcdf('test2.nc')
print 'reading compressed file took',time.time()-t1,'seconds to read'
# print out size of resulting files.
print 'size of test1.nc (no compression) = ',os.stat('test1.nc').st_size
print 'size of test2.nc (compression) = ',os.stat('test2.nc').st_size

# test interoperability with re-linked netCDF 3 clients.
#from Scientific.IO.NetCDF import *
#file = NetCDFFile('test2.nc') # note: this file is compressed
#data = file.variables['data'][:]
#print data[10;12,10:12,10:12]
#import Numeric as N
#print min(N.ravel(data)),max(N.ravel(data))
#file.close()

