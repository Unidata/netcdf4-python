import netCDF4, sys
# test creating variable dimensions.

FILE_NAME="tst_dims.nc"
LAT_NAME="lat"
LON_NAME="lon"
LEVEL_NAME="level"
TIME_NAME="time"
GROUP_NAME1='forecasts'
GROUP_NAME2='analyses'

f  = netCDF4.Dataset(FILE_NAME, 'w')
f.createDimension(LAT_NAME,73)
f.createDimension(LON_NAME,145)
f.createDimension(LEVEL_NAME,None)
f.createDimension(TIME_NAME,None)
f.createVariable('temp','f4',(LAT_NAME, LON_NAME, TIME_NAME))
print
print f.variables['temp'].dimensions
print
f.close()

f  = netCDF4.Dataset(FILE_NAME, 'a')
g1 = f.createGroup(GROUP_NAME1)
g2 = f.createGroup(GROUP_NAME2)
g1.createDimension(LAT_NAME,None)
g1.createDimension(LON_NAME,181)
g1.createDimension(LEVEL_NAME,1)
g1.createDimension(TIME_NAME,None)
g2.createDimension(LAT_NAME,181)
g2.createDimension(LON_NAME,361)
g2.createDimension(LEVEL_NAME,15)
g2.createDimension(TIME_NAME,None)
g2.createVariable('temp','f4',(LAT_NAME, LON_NAME, LEVEL_NAME, TIME_NAME))
print
print g2.variables['temp'].dimensions
print 
f.close()

f  = netCDF4.Dataset(FILE_NAME, 'r')
print 'root group'
print 'dimension name, unlimited?, length'
for name, dim in f.dimensions.iteritems():
    print name, dim.isunlimited(), len(dim)
g1 = f.groups[GROUP_NAME1]
print 
print 'group',GROUP_NAME1
print 'dimension name, unlimited?, length'
for name, dim in g1.dimensions.iteritems():
    print name, dim.isunlimited(), len(dim)
g2 = f.groups[GROUP_NAME2]
print
print 'group',GROUP_NAME2
print 'dimension name, unlimited?, length'
for name, dim in g2.dimensions.iteritems():
    print name, dim.isunlimited(), len(dim)
f.close()
