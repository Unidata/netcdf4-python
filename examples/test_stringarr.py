from netCDF4 import Dataset, stringtochar, chartostring
import random, numpy

# test utilities for converting arrays of fixed-length strings
# to arrays of characters (with an extra dimension), and vice-versa.

# netCDF does not have a fixed-length string data-type (only characters
# and variable length strings). The convenience function chartostring
# converts an array of characters to an array of fixed-length strings.
# The array of fixed length strings has one less dimension, and the
# length of the strings is equal to the rightmost dimension of the
# array of characters. The convenience function stringtochar goes
# the other way, converting an array of fixed-length strings to an
# array of characters with an extra dimension (the number of characters
# per string) appended on the right.


FILE_NAME = 'tst_stringarr.nc'
FILE_FORMAT = 'NETCDF4_CLASSIC'
chars = '1234567890aabcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

nc = Dataset(FILE_NAME,'w',format=FILE_FORMAT)
n2 = 10; nchar = 12; nrecs = 4
nc.createDimension('n1',None)
nc.createDimension('n2',n2)
nc.createDimension('nchar',nchar)
v = nc.createVariable('strings','S1',('n1','n2','nchar'))
for nrec in range(nrecs):
    data = []
    data = numpy.empty((n2,),'S'+repr(nchar))
    # fill data with random nchar character strings
    for n in range(n2):
        data[n] = ''.join([random.choice(chars) for i in range(nchar)])
    print(nrec,data)
    # convert data to array of characters with an extra dimension
    # (the number of characters per string) added to the right.
    datac = stringtochar(data)
    v[nrec] = datac
nc.close()

nc = Dataset(FILE_NAME)
v = nc.variables['strings']
print(v.shape, v.dtype)
for nrec in range(nrecs):
    # read character array back, convert to an array of strings
    # of length equal to the rightmost dimension.
    print(nrec, chartostring(v[nrec]))
nc.close()
