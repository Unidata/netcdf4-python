import netCDF4
# test reading a netCDF version 3 file (NETCDF_CLASSIC format)
f = netCDF4.Dataset('tst_classic.nc')
print f.dimensions
print f.file_format
for key,dim in f.dimensions.iteritems():
    print key, len(dim), dim.isunlimited()
print f.ncattrs()
print f.TITLE
print f.variables
x = f.variables['Float']
print x.ncattrs()
print x[0]
