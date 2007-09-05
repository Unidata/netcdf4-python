from netCDF4 import Dataset, stringtochar, chartostring
import random, numpy

# test utilities for converting arrays of fixed-length strings
# to arrays of characters (with an extra dimension), and vice-versa.

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
    for n in range(n2):
        data[n] = ''.join([random.choice(chars) for i in range(nchar)])
    print nrec,data
    datac = stringtochar(data)
    v[nrec] = datac
nc.close()

nc = Dataset(FILE_NAME)
v = nc.variables['strings']
print v.shape, v.dtype
for nrec in range(nrecs):
    print nrec, chartostring(v[nrec])
nc.close()
