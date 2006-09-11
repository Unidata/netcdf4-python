import netCDF4
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
GROUP_NAME='subgroup'

f  = netCDF4.Dataset(FILE_NAME, 'w')
f.createDimension(LAT_NAME,73)
f.createDimension(LON_NAME,145)
f.createDimension(LEVEL_NAME,10)
f.createDimension(TIME_NAME,None)
g = f.createGroup(GROUP_NAME)
g.createDimension(LAT_NAME,145)
g.createDimension(LON_NAME,289)
g.createDimension(LEVEL_NAME,20)
g.createDimension(TIME_NAME,None)
f.foo = 'bar'
f.goober = 2
g.foo = 'bar'
g.goober = 2
f.createVariable(VAR_NAME,'f4',(LAT_NAME, LON_NAME, TIME_NAME))
v = f.variables[VAR_NAME]
v.bar = 'foo'
v.slobber = 3
g.createVariable(VAR_NAME,'f4',(LAT_NAME, LON_NAME, TIME_NAME))
v2 = g.variables[VAR_NAME]
v2.bar = 'foo'
v2.slobber = 3
print 'testing renaming dimensions:'
print '----------------------------'
print 'dimensions in dataset:'
print tuple(f.dimensions.keys())
print tuple(g.dimensions.keys())
print 'variable dimensions:'
print f.variables[VAR_NAME].dimensions
print g.variables[VAR_NAME].dimensions
f.close()
print "change '%s' to '%s' ..." % (LON_NAME, LON_NAME2)
f  = netCDF4.Dataset(FILE_NAME, 'r+')
g = f.groups[GROUP_NAME]
f.renameDimension(LON_NAME,LON_NAME2)
g.renameDimension(LON_NAME,LON_NAME2)
v = f.variables[VAR_NAME]
v2 = g.variables[VAR_NAME]
print 'dimensions in dataset:'
print tuple(f.dimensions.keys())
print tuple(g.dimensions.keys())
print 'variable dimensions:'
print v.dimensions
print v2.dimensions
print
print 'testing renaming variables:'
print '---------------------------'
print f.variables
print g.variables
print "change '%s' to '%s' ..." % (VAR_NAME, VAR_NAME2)
f.renameVariable(VAR_NAME,VAR_NAME2)
g.renameVariable(VAR_NAME,VAR_NAME2)
f.close()

# make sure attributes can't be deleted when file is open read-only.
f  = netCDF4.Dataset(FILE_NAME)
g = f.groups[GROUP_NAME]
print f.variables
print g.variables
print
print 'testing deletion of attributes'
print '------------------------------'
try:
    del f.foo
    del g.foo
    raise IOError, 'should not be able to delete an attribute from a read-only file!'
except:
    print 'OK good - deleting an attribute when file is read-only raises an error'
f.close()
f  = netCDF4.Dataset(FILE_NAME, 'r+')
g = f.groups[GROUP_NAME]
v = f.variables[VAR_NAME2]
v2 = g.variables[VAR_NAME2]
print f.ncattrs()
print g.ncattrs()
print v.ncattrs()
print v2.ncattrs()
print "remove 'goober' and 'slobber'"
del f.goober
del v.slobber
del g.goober
del v2.slobber
print f.ncattrs()
print g.ncattrs()
print v.ncattrs()
print v2.ncattrs()
f.close()
