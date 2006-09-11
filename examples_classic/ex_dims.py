import netCDF4_classic as netCDF4
import sys
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
f.createDimension(LEVEL_NAME,10)
f.createDimension(TIME_NAME,None)
f.createVariable('temp','f4',(LAT_NAME, LON_NAME, TIME_NAME))
print
print f.variables['temp'].dimensions
print
f.close()

f  = netCDF4.Dataset(FILE_NAME, 'r')
print 'dimension name, unlimited?, length'
for name, dim in f.dimensions.iteritems():
    print name, dim.isunlimited(), len(dim)
f.close()
