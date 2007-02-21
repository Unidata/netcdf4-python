from numpy.random.mtrand import uniform
from netCDF4_classic import Dataset
import os

ndim = 100000
array = uniform(size=(ndim,))

def write_netcdf(filename,zlib,least_significant_digit,data,dtype='f8',shuffle=False,chunking='seq',complevel=6,fletcher32=False):
    file = Dataset(filename,'w')
    file.createDimension('n', ndim)
    foo = file.createVariable('data', dtype,('n'),zlib=zlib,least_significant_digit=least_significant_digit,shuffle=shuffle,chunking=chunking,complevel=complevel,fletcher32=fletcher32)
    foo[:] = data
    file.close()
    file = Dataset(filename)
    var = file.variables['data']
    data = var[:]
    print var.filters()
    print var.__dict__
    print data[0:4]

# uncompressed.
filename = 'uncompressed.nc'
least_significant_digit = None
zlib = False
write_netcdf(filename,zlib,least_significant_digit,array)
print 'size of uncompressed.nc (no compression) = ',os.stat('uncompressed.nc').st_size

# compressed (lossless, no shuffle).
filename = 'lossless.nc'
least_significant_digit = None
zlib = True
write_netcdf(filename,zlib,least_significant_digit,array)
print 'size of lossless.nc (lossless compression, no shuffle) = ',os.stat('lossless.nc').st_size

# compressed (lossless, shuffle).
filename = 'lossless_s.nc'
least_significant_digit = None
zlib = True
write_netcdf(filename,zlib,least_significant_digit,array,shuffle=True)
print 'size of lossless_s.nc (lossless compression, with shuffle) = ',os.stat('lossless.nc').st_size

# compressed (lossy, 3 digits, no shuffle).
filename = 'lossy3.nc'
least_significant_digit = 3
zlib = True
write_netcdf(filename,zlib,least_significant_digit,array)
print 'size of lossy3.nc (lossy compression, 3 digits, no shuffle) = ',os.stat('lossy3.nc').st_size

# compressed (lossy, 3 digits, with shuffle).
filename = 'lossy3s.nc'
least_significant_digit = 3
zlib = True
write_netcdf(filename,zlib,least_significant_digit,array,shuffle=True)
print 'size of lossy3s.nc (lossy compression, 3 digits, shuffle) = ',os.stat('lossy3s.nc').st_size

# compressed (lossy, 3 digits, with shuffle and fletcher32 checksum).
filename = 'lossy3sf.nc'
least_significant_digit = 3
zlib = True
write_netcdf(filename,zlib,least_significant_digit,array,shuffle=True,fletcher32=True)
print 'size of lossy3sf.nc (lossy compression, 3 digits, shuffle, checksum) = ',os.stat('lossy3s.nc').st_size
