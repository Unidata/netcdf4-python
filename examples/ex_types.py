import numpy as NP
# test all primitive data types.
from numpy.random.mtrand import uniform
import netCDF4, sys, time, os

# create an n1dim by n2dim random array.
n1dim = 10
n2dim = 1000
print 'reading and writing a %s by %s random array ..'%(n1dim,n2dim)
array = 100.*uniform(size=(n1dim,n2dim))
print array.shape
print array[5,100:103]

def write_netcdf_unlim(array,filename,n1dim,n2dim,zlib=False,complevel=0,shuffle=0,least_significant_digit=None):
    file = netCDF4.Dataset(filename,'w')
    file.createDimension('n1', None)
    file.createDimension('n2', n2dim)
    foo = file.createVariable('data', array.dtype.str[1:], ('n1','n2',),zlib=zlib,complevel=complevel,shuffle=shuffle,least_significant_digit=least_significant_digit)
    # test writing of _FillValue attribute for diff types
    # (should be cast to type of variable silently)
    foo._FillValue  = 1.0
    for n in range(n1dim):
        foo[n] = array[n]
    file.close()

def read_netcdf(filename):
    file = netCDF4.Dataset(filename)
    data = file.variables['data'][:]
    file.close()
    return data

# create a file, put a random array in it.
# no compression is used.
for dtype in ['f8','f4','i1','i2','i4','i8','u1','u2','u4','u8','S1']:
    dat1 = array.astype(dtype)
    write_netcdf_unlim(dat1,'test_'+dtype+'.nc',n1dim,n2dim)
    dat2 = read_netcdf('test_'+dtype+'.nc')
    print dtype,dat1[5,100:103],dat2[5,100:103]
