import netCDF4_classic as netCDF4
# test changing dimension, variable names
# and deleting attributes.

FILE_NAME="tst_rename.nc"
LAT_NAME="lat"
LON_NAME="lon"
LON_NAME2 = "longitude"
LEVEL_NAME="level"
TIME_NAME="time"
VAR_NAME='temp'
VAR_NAME2='wind'

f  = netCDF4.Dataset(FILE_NAME, 'w')
f.createDimension(LAT_NAME,73)
f.createDimension(LON_NAME,145)
f.createDimension(LEVEL_NAME,10)
f.createDimension(TIME_NAME,None)
f.foo = 'bar'
f.goober = 2
f.createVariable(VAR_NAME,'f4',(LAT_NAME, LON_NAME, TIME_NAME))
v = f.variables[VAR_NAME]
v.bar = 'foo'
v.slobber = 3
print 'testing renaming dimensions:'
print '----------------------------'
print 'dimensions in dataset:'
print tuple(f.dimensions.keys())
print 'variable dimensions:'
print f.variables[VAR_NAME].dimensions
f.close()
print "change '%s' to '%s' ..." % (LON_NAME, LON_NAME2)
f  = netCDF4.Dataset(FILE_NAME, 'r+')
f.renameDimension(LON_NAME,LON_NAME2)
v = f.variables[VAR_NAME]
print 'dimensions in dataset:'
print tuple(f.dimensions.keys())
print 'variable dimensions:'
print v.dimensions
print
print 'testing renaming variables:'
print '---------------------------'
print f.variables
print "change '%s' to '%s' ..." % (VAR_NAME, VAR_NAME2)
f.renameVariable(VAR_NAME,VAR_NAME2)
f.close()

# make sure attributes can't be deleted when file is open read-only.
f  = netCDF4.Dataset(FILE_NAME)
print f.variables
print
print 'testing deletion of attributes'
print '------------------------------'
try:
    del f.foo
    raise IOError, 'should not be able to delete an attribute from a read-only file!'
except:
    print 'OK good - deleting an attribute when file is read-only raises an error'
f.close()
f  = netCDF4.Dataset(FILE_NAME, 'r+')
v = f.variables[VAR_NAME2]
print f.ncattrs()
print v.ncattrs()
print "remove 'goober' and 'slobber'"
del f.goober
del v.slobber
print f.ncattrs()
print v.ncattrs()
f.close()
